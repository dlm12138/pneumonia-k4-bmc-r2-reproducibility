"""Actual study functions extracted without changes to their calculations.
See docs/code_provenance.tsv and docs/reproduction_limits.md.
"""
from __future__ import annotations
import math
import os
import re
import csv
import gzip
import io
import sys
import zipfile
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any
from types import SimpleNamespace
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import solve_triangular
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



VITALS = ["SBP", "DBP", "HR", "RR"]


FEATURES = [f"{vital}_h{hour}" for vital in VITALS for hour in range(12)]


PHENOTYPES = ["P1", "P2", "P3", "P4"]


DATABASES = ["MIMIC-IV", "MIMIC-III", "eICU"]


SEED = 20260903


class FixedMaskScorer:
    """Exact implementation retained from the 2026-08-01 analysis."""

    def __init__(self, raw: np.ndarray, means: np.ndarray, sds: np.ndarray, model):
        self.means = means
        self.sds = sds
        self.model = model
        packed = np.packbits(np.isfinite(raw), axis=1)
        unique, inverse = np.unique(packed, axis=0, return_inverse=True)
        self.groups = []
        log2pi = math.log(2.0 * math.pi)
        for number, packed_mask in enumerate(unique):
            rows = np.flatnonzero(inverse == number)
            columns = np.flatnonzero(
                np.unpackbits(packed_mask, count=raw.shape[1]).astype(bool)
            )
            components = []
            for component in range(model.n_components):
                covariance = model.covariances_[component][np.ix_(columns, columns)]
                cholesky = np.linalg.cholesky(covariance)
                constant = math.log(max(model.weights_[component], 1e-300)) - 0.5 * (
                    len(columns) * log2pi + 2.0 * np.log(np.diag(cholesky)).sum()
                )
                components.append(
                    (model.means_[component, columns], cholesky, constant)
                )
            self.groups.append((rows, columns, components))

    def posterior(self, raw: np.ndarray) -> np.ndarray:
        standardized = (raw - self.means[None, :]) / self.sds[None, :]
        likelihood = np.full((len(raw), self.model.n_components), -np.inf)
        for rows, columns, components in self.groups:
            values = standardized[np.ix_(rows, columns)]
            for component, (mean, cholesky, constant) in enumerate(components):
                difference = values - mean[None, :]
                solved = solve_triangular(
                    cholesky, difference.T, lower=True, check_finite=False
                )
                likelihood[rows, component] = constant - 0.5 * np.sum(
                    solved * solved, axis=0
                )
        return np.exp(likelihood - logsumexp(likelihood, axis=1, keepdims=True))


def transform_raw(
    raw: np.ndarray, mode: str, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Exact transform retained from the 2026-08-01 analysis."""
    output = raw.copy()
    for vital_index in range(4):
        start = vital_index * 12
        stop = start + 12
        block = output[:, start:stop]
        if mode == "mean_flattened":
            means = np.nanmean(block, axis=1)
            mask = np.isfinite(block)
            block[mask] = np.repeat(means[:, None], 12, axis=1)[mask]
        else:
            for row_index in range(len(block)):
                observed = np.flatnonzero(np.isfinite(block[row_index]))
                if len(observed) < 2:
                    continue
                values = block[row_index, observed].copy()
                if mode == "reversed":
                    values = values[::-1]
                elif mode == "permuted":
                    if rng is None:
                        raise ValueError("Permutation requires an RNG")
                    values = values[rng.permutation(len(values))]
                else:
                    raise ValueError(mode)
                block[row_index, observed] = values
        output[:, start:stop] = block
    return output


def effect_estimates(module, frame: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    data = frame.copy()
    data["phenotype_test"] = labels
    data = data.dropna(
        subset=["death_24h", "age", "male", "observed_dimensions_48d"]
    ).copy()
    data["age10"] = data["age"] / 10.0
    data["coverage10"] = data["observed_dimensions_48d"] / 10.0
    columns = [np.ones(len(data), float)]
    for phenotype in ["P1", "P3", "P4"]:
        columns.append(data["phenotype_test"].eq(phenotype).to_numpy(float))
    columns.extend(
        [
            data["age10"].to_numpy(float),
            data["male"].to_numpy(float),
            data["coverage10"].to_numpy(float),
        ]
    )
    x = np.column_stack(columns)
    y = data["death_24h"].to_numpy(float)
    if data["phenotype_test"].eq("P3").sum() == 0:
        return {
            "model_N": len(data),
            "events": int(y.sum()),
            "P3_events": 0,
            "p3_beta": np.nan,
            "P3_OR": np.nan,
            "OR_CI_low": np.nan,
            "OR_CI_high": np.nan,
            "p_value": np.nan,
            "P2_standardized_risk": np.nan,
            "P3_standardized_risk": np.nan,
            "P3_RD_fraction": np.nan,
            "RD_CI_low": np.nan,
            "RD_CI_high": np.nan,
        }
    beta, converged = module.fit_logistic(x, y)
    if not converged:
        raise RuntimeError("Time-order outcome model did not converge")
    clusters = (
        data["hospital_id"].astype(str).to_numpy()
        if data["database"].iloc[0] == "eICU"
        else None
    )
    covariance = module.robust_covariance(x, y, beta, clusters)
    standard_error = math.sqrt(max(float(covariance[2, 2]), 0.0))
    z_value = beta[2] / standard_error
    counterfactual = {}
    gradients = {}
    risks = {}
    for phenotype in ["P2", "P3"]:
        current = x.copy()
        current[:, 1:4] = 0.0
        if phenotype == "P3":
            current[:, 2] = 1.0
        probability = expit(current @ beta)
        counterfactual[phenotype] = current
        risks[phenotype] = float(probability.mean())
        gradients[phenotype] = np.mean(
            current * (probability * (1.0 - probability))[:, None], axis=0
        )
    rd = risks["P3"] - risks["P2"]
    rd_gradient = gradients["P3"] - gradients["P2"]
    rd_se = math.sqrt(max(float(rd_gradient @ covariance @ rd_gradient), 0.0))
    return {
        "model_N": len(data),
        "events": int(y.sum()),
        "P3_events": int(
            data.loc[data["phenotype_test"].eq("P3"), "death_24h"].sum()
        ),
        "p3_beta": float(beta[2]),
        "P3_OR": math.exp(beta[2]),
        "OR_CI_low": math.exp(beta[2] - 1.96 * standard_error),
        "OR_CI_high": math.exp(beta[2] + 1.96 * standard_error),
        "p_value": float(2.0 * norm.sf(abs(z_value))),
        "P2_standardized_risk": risks["P2"],
        "P3_standardized_risk": risks["P3"],
        "P3_RD_fraction": rd,
        "RD_CI_low": rd - 1.96 * rd_se,
        "RD_CI_high": rd + 1.96 * rd_se,
    }


def distribution(labels: np.ndarray) -> dict[str, int]:
    return {
        f"{phenotype}_N": int(np.sum(labels == phenotype))
        for phenotype in PHENOTYPES
    }


def audit_transform(
    database: str, raw: np.ndarray, transformed: np.ndarray, mode: str
) -> list[dict[str, object]]:
    rows = []
    for vital_index, vital in enumerate(VITALS):
        start = vital_index * 12
        stop = start + 12
        before = raw[:, start:stop]
        after = transformed[:, start:stop]
        rows.append(
            {
                "database": database,
                "transformation": mode,
                "vital": vital,
                "audit_expectation": (
                    "preserve_observed_value_multiset"
                    if mode != "mean_flattened"
                    else "preserve_mask_count_and_mean"
                ),
                "max_abs_count_difference": int(
                    np.max(
                        np.abs(
                            np.isfinite(before).sum(axis=1)
                            - np.isfinite(after).sum(axis=1)
                        )
                    )
                ),
                "max_abs_sum_difference": float(
                    np.nanmax(
                        np.abs(np.nansum(before, axis=1) - np.nansum(after, axis=1))
                    )
                ),
                "max_abs_sum_squares_difference": float(
                    np.nanmax(
                        np.abs(
                            np.nansum(before * before, axis=1)
                            - np.nansum(after * after, axis=1)
                        )
                    )
                ),
            }
        )
    return rows
