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



DATABASES = ["MIMIC-IV", "MIMIC-III", "eICU"]


PHENOTYPES = ["P1", "P2", "P3", "P4"]


VITALS = ["SBP", "DBP", "HR", "RR"]


FEATURES = [f"{vital}_h{hour}" for vital in VITALS for hour in range(12)]


def observed_component_log_likelihood(
    raw: np.ndarray,
    standardization_means: np.ndarray,
    standardization_sds: np.ndarray,
    model,
) -> np.ndarray:
    standardized = (raw - standardization_means[None, :]) / standardization_sds[None, :]
    masks = np.isfinite(standardized)
    packed = np.packbits(masks, axis=1)
    unique_masks, inverse = np.unique(packed, axis=0, return_inverse=True)
    output = np.full((len(raw), model.n_components), -np.inf)
    log_weights = np.log(np.clip(model.weights_, 1e-300, None))
    log_2pi = math.log(2.0 * math.pi)
    for mask_number, packed_mask in enumerate(unique_masks):
        rows = np.flatnonzero(inverse == mask_number)
        columns = np.flatnonzero(np.unpackbits(packed_mask, count=raw.shape[1]).astype(bool))
        values = standardized[np.ix_(rows, columns)]
        for component in range(model.n_components):
            covariance = model.covariances_[component][np.ix_(columns, columns)]
            mean = model.means_[component, columns]
            chol = np.linalg.cholesky(covariance)
            difference = values - mean[None, :]
            solved = solve_triangular(chol, difference.T, lower=True, check_finite=False)
            quadratic = np.sum(solved * solved, axis=0)
            log_determinant = float(2.0 * np.log(np.diag(chol)).sum())
            output[rows, component] = log_weights[component] - 0.5 * (
                len(columns) * log_2pi + log_determinant + quadratic
            )
    return output


def assigned_mahalanobis(
    raw: np.ndarray,
    assigned_component: np.ndarray,
    standardization_means: np.ndarray,
    standardization_sds: np.ndarray,
    model,
) -> np.ndarray:
    standardized = (raw - standardization_means[None, :]) / standardization_sds[None, :]
    masks = np.isfinite(standardized)
    packed = np.packbits(masks, axis=1)
    unique_masks, inverse = np.unique(packed, axis=0, return_inverse=True)
    output = np.full(len(raw), np.nan)
    for mask_number, packed_mask in enumerate(unique_masks):
        mask_rows = np.flatnonzero(inverse == mask_number)
        columns = np.flatnonzero(np.unpackbits(packed_mask, count=raw.shape[1]).astype(bool))
        for component in range(model.n_components):
            rows = mask_rows[assigned_component[mask_rows] == component]
            if len(rows) == 0:
                continue
            covariance = model.covariances_[component][np.ix_(columns, columns)]
            mean = model.means_[component, columns]
            chol = np.linalg.cholesky(covariance)
            difference = standardized[np.ix_(rows, columns)] - mean[None, :]
            solved = solve_triangular(chol, difference.T, lower=True, check_finite=False)
            output[rows] = np.sum(solved * solved, axis=0)
    return output


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    midpoint = 0.5 * (left + right)
    left_term = np.where(left > 0, left * np.log(left / midpoint), 0.0)
    right_term = np.where(right > 0, right * np.log(right / midpoint), 0.0)
    return float(0.5 * (left_term.sum() + right_term.sum()))


def feature_shift(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    reference = frame.loc[frame["database"].eq("MIMIC-IV")]
    for feature in FEATURES:
        reference_values = pd.to_numeric(reference[feature], errors="coerce").dropna()
        reference_mean = float(reference_values.mean())
        reference_sd = float(reference_values.std(ddof=1))
        for database in DATABASES:
            values = pd.to_numeric(
                frame.loc[frame["database"].eq(database), feature], errors="coerce"
            ).dropna()
            rows.append(
                {
                    "database": database,
                    "reference_database": "MIMIC-IV",
                    "feature": feature,
                    "domain": feature.split("_")[0],
                    "observed_N": len(values),
                    "missing_N": int(frame["database"].eq(database).sum()) - len(values),
                    "mean": float(values.mean()),
                    "SD": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "Q1": float(values.quantile(0.25)),
                    "Q3": float(values.quantile(0.75)),
                    "MIMIC_IV_reference_mean": reference_mean,
                    "MIMIC_IV_reference_SD": reference_sd,
                    "SMD_vs_MIMIC_IV": (float(values.mean()) - reference_mean) / reference_sd,
                    "SMD_definition": "difference in means divided by frozen MIMIC-IV feature SD",
                }
            )
    detailed = pd.DataFrame(rows)
    summary_rows = []
    for database in DATABASES:
        values = detailed.loc[detailed["database"].eq(database), "SMD_vs_MIMIC_IV"].abs()
        summary_rows.append(
            {
                "database": database,
                "feature_abs_SMD_median": float(values.median()),
                "feature_abs_SMD_P90": float(values.quantile(0.90)),
                "feature_abs_SMD_max": float(values.max()),
                "features_abs_SMD_gt_0_1": int(values.gt(0.1).sum()),
                "features_abs_SMD_gt_0_2": int(values.gt(0.2).sum()),
                "features_abs_SMD_gt_0_5": int(values.gt(0.5).sum()),
            }
        )
    return detailed, pd.DataFrame(summary_rows)


def prevalence_shift(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prevalence = (
        assignments.groupby("database")["phenotype"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(index=DATABASES, columns=PHENOTYPES, fill_value=0)
    )
    rows = []
    summary_rows = []
    reference = prevalence.loc["MIMIC-IV"].to_numpy(float)
    for database in DATABASES:
        current = prevalence.loc[database].to_numpy(float)
        divergence = jensen_shannon_divergence(reference, current)
        total_variation = float(0.5 * np.abs(reference - current).sum())
        summary_rows.append(
            {
                "database": database,
                "reference_database": "MIMIC-IV",
                "Jensen_Shannon_divergence": divergence,
                "total_variation_distance": total_variation,
            }
        )
        for phenotype in PHENOTYPES:
            rows.append(
                {
                    "database": database,
                    "reference_database": "MIMIC-IV",
                    "phenotype": phenotype,
                    "N": int(
                        (
                            assignments["database"].eq(database)
                            & assignments["phenotype"].eq(phenotype)
                        ).sum()
                    ),
                    "prevalence_fraction": float(prevalence.loc[database, phenotype]),
                    "MIMIC_IV_prevalence_fraction": float(prevalence.loc["MIMIC-IV", phenotype]),
                    "Jensen_Shannon_divergence": divergence,
                    "total_variation_distance": total_variation,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def distribution_summary(
    frame: pd.DataFrame,
    variable: str,
    threshold: float,
    threshold_name: str,
) -> pd.DataFrame:
    rows = []
    for database in DATABASES:
        values = pd.to_numeric(frame.loc[frame["database"].eq(database), variable], errors="coerce")
        rows.append(
            {
                "database": database,
                "N": int(values.notna().sum()),
                "median": float(values.median()),
                "Q1": float(values.quantile(0.25)),
                "Q3": float(values.quantile(0.75)),
                "P5": float(values.quantile(0.05)),
                "P10": float(values.quantile(0.10)),
                "P90": float(values.quantile(0.90)),
                "P95": float(values.quantile(0.95)),
                "P99": float(values.quantile(0.99)),
                threshold_name: threshold,
                "above_MIMIC_IV_P95_N": int(values.gt(threshold).sum()),
                "above_MIMIC_IV_P95_pct": 100.0 * float(values.gt(threshold).mean()),
            }
        )
    return pd.DataFrame(rows)


def posterior_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for database in DATABASES:
        current = frame.loc[frame["database"].eq(database)]
        for variable in ["max_posterior", "posterior_entropy_normalized"]:
            values = current[variable]
            rows.append(
                {
                    "database": database,
                    "metric": variable,
                    "N": len(values),
                    "median": float(values.median()),
                    "Q1": float(values.quantile(0.25)),
                    "Q3": float(values.quantile(0.75)),
                    "P10": float(values.quantile(0.10)),
                    "P90": float(values.quantile(0.90)),
                    "interpretation": "internal locked-model assignment certainty, not external model fit",
                }
            )
    return pd.DataFrame(rows)
