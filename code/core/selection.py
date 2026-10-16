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


OUTCOMES = ["death_24h", "hospital_mortality"]


TRIMS = {
    "untrimmed": (0.0, 1.0),
    "p01_p99": (0.01, 0.99),
    "p025_p975": (0.025, 0.975),
    "p05_p95": (0.05, 0.95),
}


def weighted_moments(values: pd.Series, weights: np.ndarray | None = None) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(numeric)
    numeric = numeric[valid]
    if len(numeric) < 2:
        return np.nan, np.nan
    if weights is None:
        return float(np.mean(numeric)), float(np.var(numeric, ddof=1))
    current_weights = np.asarray(weights, float)[valid]
    if current_weights.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.average(numeric, weights=current_weights))
    variance = float(np.average(np.square(numeric - mean), weights=current_weights))
    return mean, variance


def standardized_difference(
    target_values: pd.Series,
    selected_values: pd.Series,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    target_mean, target_variance = weighted_moments(target_values)
    selected_mean, selected_variance = weighted_moments(selected_values, weights)
    scale = math.sqrt((target_variance + selected_variance) / 2.0)
    smd = (selected_mean - target_mean) / scale if scale > 0 else 0.0
    return target_mean, selected_mean, smd


def fit_outcome_glm(
    frame: pd.DataFrame,
    formula: str,
    weights: pd.Series | None,
    cluster: pd.Series | None,
):
    model = smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Binomial(),
        freq_weights=weights,
    )
    if weights is None:
        if cluster is None:
            return model.fit(cov_type="HC1")
        groups = pd.Series(cluster, index=frame.index).reindex(model.data.row_labels)
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups.to_numpy()})

    if len(model.endog) != len(frame):
        raise ValueError("Weighted model contains implicit complete-case deletion")
    result = model.fit()
    exog = np.asarray(model.exog, float)
    endog = np.asarray(model.endog, float)
    fitted = np.asarray(result.fittedvalues, float)
    w = np.asarray(weights, float)
    information = exog.T @ (exog * (w * fitted * (1.0 - fitted))[:, None])
    bread = np.linalg.pinv(information)
    score = exog * (w * (endog - fitted))[:, None]
    n, p = exog.shape
    if cluster is None:
        meat = score.T @ score
        correction = n / (n - p) if n > p else 1.0
    else:
        group_values = np.asarray(cluster)
        codes, levels = pd.factorize(group_values, sort=False)
        group_score = np.zeros((len(levels), p), float)
        np.add.at(group_score, codes, score)
        meat = group_score.T @ group_score
        group_n = len(levels)
        correction = (
            (group_n / (group_n - 1.0)) * ((n - 1.0) / (n - p))
            if group_n > 1 and n > p
            else 1.0
        )
    covariance = bread @ (meat * correction) @ bread
    return SimpleNamespace(
        params=result.params,
        bse=pd.Series(np.sqrt(np.maximum(np.diag(covariance), 0.0)), index=result.params.index),
    )


def effect_row(result, term: str) -> dict[str, float]:
    beta = float(result.params[term])
    se = float(result.bse[term])
    return {
        "beta": beta,
        "se": se,
        "OR": math.exp(beta),
        "CI_low": math.exp(beta - 1.96 * se),
        "CI_high": math.exp(beta + 1.96 * se),
        "p_value": float(2.0 * norm.sf(abs(beta / se))),
    }


def summarize_distribution(values: pd.Series) -> dict[str, float]:
    quantiles = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "min": float(values.min()),
        "P1": float(quantiles.loc[0.01]),
        "P5": float(quantiles.loc[0.05]),
        "P10": float(quantiles.loc[0.10]),
        "P25": float(quantiles.loc[0.25]),
        "P50": float(quantiles.loc[0.50]),
        "P75": float(quantiles.loc[0.75]),
        "P90": float(quantiles.loc[0.90]),
        "P95": float(quantiles.loc[0.95]),
        "P99": float(quantiles.loc[0.99]),
        "max": float(values.max()),
    }
