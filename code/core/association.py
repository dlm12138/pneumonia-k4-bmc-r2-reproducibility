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



SEED = 20260903


BOOTSTRAP_N = 500


BOOTSTRAP_MAX_ATTEMPTS = 650


DATABASES = ["MIMIC-IV", "MIMIC-III", "eICU"]


PHENOTYPES = ["P1", "P2", "P3", "P4"]


REFERENCE = "P2"


COMORBIDITIES = [
    "myocardial_infarction",
    "heart_failure",
    "cerebrovascular",
    "chronic_pulmonary",
    "diabetes",
    "liver_disease",
    "renal_disease",
    "malignancy",
]


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    frequency_weights: np.ndarray | None = None,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    weights = np.ones(len(y), float) if frequency_weights is None else np.asarray(frequency_weights, float)
    beta = np.zeros(x.shape[1], float)
    converged = False
    for _ in range(max_iter):
        probability = expit(np.clip(x @ beta, -35.0, 35.0))
        variance_weight = weights * probability * (1.0 - probability)
        information = x.T @ (x * variance_weight[:, None])
        score = x.T @ (weights * (y - probability))
        step = np.linalg.pinv(information, rcond=1e-12) @ score
        beta = beta + step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break
        if not np.isfinite(beta).all() or np.max(np.abs(beta)) > 100:
            return beta, False
    return beta, converged


def robust_covariance(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cluster: np.ndarray | None,
) -> np.ndarray:
    probability = expit(np.clip(x @ beta, -35.0, 35.0))
    information = x.T @ (x * (probability * (1.0 - probability))[:, None])
    bread = np.linalg.pinv(information, rcond=1e-12)
    score = x * (y - probability)[:, None]
    n, p = x.shape
    if cluster is None:
        meat = score.T @ score
        correction = 1.0
    else:
        codes, levels = pd.factorize(cluster, sort=False)
        group_score = np.zeros((len(levels), p), float)
        np.add.at(group_score, codes, score)
        meat = group_score.T @ group_score
        groups = len(levels)
        correction = (
            (groups / (groups - 1.0)) * ((n - 1.0) / (n - p))
            if groups > 1 and n > p
            else 1.0
        )
    return bread @ (meat * correction) @ bread


def prepare_model_data(
    source: pd.DataFrame,
    outcome: str,
    require_apache: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    needed = [outcome, "phenotype", "age", "male", "observed_dimensions_48d"]
    if source["database"].iloc[0] == "eICU":
        needed.append("hospital_id")
    if require_apache:
        needed.append("acutephysiologyscore")
    frame = source.dropna(subset=needed).copy()
    parameters = {}
    for variable in ["SpO2_min_0_12h"]:
        median = float(pd.to_numeric(frame[variable], errors="coerce").median())
        if not np.isfinite(median):
            raise ValueError(f"No usable values for {variable}")
        parameters[variable] = median
        frame[f"{variable}_missing"] = frame[variable].isna().astype(float)
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce").fillna(median)
    frame["age10"] = pd.to_numeric(frame["age"], errors="coerce") / 10.0
    frame["coverage10"] = pd.to_numeric(frame["observed_dimensions_48d"], errors="coerce") / 10.0
    frame["SpO2_min10"] = frame["SpO2_min_0_12h"] / 10.0
    frame["APS10"] = pd.to_numeric(frame["acutephysiologyscore"], errors="coerce") / 10.0
    for variable in [
        "active_invasive_ventilation_at_12h",
        "active_vasopressor_at_12h",
        "active_RRT_at_12h",
    ]:
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    return frame, parameters


def design_matrix(frame: pd.DataFrame, covariates: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = [
        np.ones(len(frame), float),
        frame["phenotype"].eq("P1").astype(float).to_numpy(),
        frame["phenotype"].eq("P3").astype(float).to_numpy(),
        frame["phenotype"].eq("P4").astype(float).to_numpy(),
    ]
    names = ["intercept", "P1_vs_P2", "P3_vs_P2", "P4_vs_P2"]
    for covariate in covariates:
        columns.append(pd.to_numeric(frame[covariate], errors="coerce").to_numpy(float))
        names.append(covariate)
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite model matrix after declared missing-data handling")
    return matrix, names


def counterfactual_matrix(x: np.ndarray, phenotype: str) -> np.ndarray:
    result = x.copy()
    result[:, 1:4] = 0.0
    if phenotype == "P1":
        result[:, 1] = 1.0
    elif phenotype == "P3":
        result[:, 2] = 1.0
    elif phenotype == "P4":
        result[:, 3] = 1.0
    return result


def bootstrap_risks(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray | None,
    seed_offset: int,
) -> tuple[dict[str, np.ndarray], int]:
    rng = np.random.default_rng(SEED + seed_offset)
    counterfactual = {phenotype: counterfactual_matrix(x, phenotype) for phenotype in PHENOTYPES}
    samples = {phenotype: [] for phenotype in PHENOTYPES}
    if clusters is not None:
        codes, levels = pd.factorize(clusters, sort=False)
        group_probability = np.full(len(levels), 1.0 / len(levels))
    else:
        patient_probability = np.full(len(y), 1.0 / len(y))
    attempts = 0
    while len(samples["P1"]) < BOOTSTRAP_N and attempts < BOOTSTRAP_MAX_ATTEMPTS:
        attempts += 1
        if clusters is None:
            frequency = rng.multinomial(len(y), patient_probability).astype(float)
        else:
            group_frequency = rng.multinomial(len(levels), group_probability).astype(float)
            frequency = group_frequency[codes]
        beta, converged = fit_logistic(x, y, frequency_weights=frequency)
        if not converged or not np.isfinite(beta).all():
            continue
        denominator = frequency.sum()
        current = {
            phenotype: float(np.sum(frequency * expit(counterfactual[phenotype] @ beta)) / denominator)
            for phenotype in PHENOTYPES
        }
        if not np.isfinite(list(current.values())).all():
            continue
        for phenotype in PHENOTYPES:
            samples[phenotype].append(current[phenotype])
    if len(samples["P1"]) < BOOTSTRAP_N:
        raise RuntimeError(f"Only {len(samples['P1'])} successful bootstrap fits")
    return {key: np.asarray(values, float) for key, values in samples.items()}, attempts


def run_model(
    frame: pd.DataFrame,
    outcome: str,
    model_name: str,
    covariates: list[str],
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, terms = design_matrix(frame, covariates)
    y = pd.to_numeric(frame[outcome], errors="coerce").to_numpy(float)
    beta, converged = fit_logistic(x, y)
    if not converged:
        raise RuntimeError(f"Point model did not converge: {frame['database'].iloc[0]} {model_name}")
    clusters = (
        frame["hospital_id"].astype(str).to_numpy()
        if frame["database"].iloc[0] == "eICU"
        else None
    )
    covariance = robust_covariance(x, y, beta, clusters)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    counterfactual = {phenotype: counterfactual_matrix(x, phenotype) for phenotype in PHENOTYPES}
    point_risks = {
        phenotype: float(np.mean(expit(counterfactual[phenotype] @ beta)))
        for phenotype in PHENOTYPES
    }
    bootstrap, attempts = bootstrap_risks(x, y, clusters, seed_offset)
    p2_bootstrap = bootstrap[REFERENCE]
    rows = []
    for phenotype in PHENOTYPES:
        if phenotype == REFERENCE:
            odds_ratio = 1.0
            ci_low = 1.0
            ci_high = 1.0
            p_value = np.nan
        else:
            term = terms.index(f"{phenotype}_vs_P2")
            odds_ratio = math.exp(beta[term])
            ci_low = math.exp(beta[term] - 1.96 * standard_errors[term])
            ci_high = math.exp(beta[term] + 1.96 * standard_errors[term])
            p_value = float(2.0 * norm.sf(abs(beta[term] / standard_errors[term])))
        risk_values = bootstrap[phenotype]
        rd_values = risk_values - p2_bootstrap
        rows.append(
            {
                "database": frame["database"].iloc[0],
                "outcome": outcome,
                "model": model_name,
                "phenotype": phenotype,
                "reference": REFERENCE,
                "N": len(frame),
                "events": int(y.sum()),
                "OR": odds_ratio,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "p_value": p_value,
                "standardized_risk_fraction": point_risks[phenotype],
                "risk_CI_low": float(np.quantile(risk_values, 0.025)),
                "risk_CI_high": float(np.quantile(risk_values, 0.975)),
                "RD_vs_P2_fraction": point_risks[phenotype] - point_risks[REFERENCE],
                "RD_vs_P2_percentage_points": 100.0 * (point_risks[phenotype] - point_risks[REFERENCE]),
                "RD_CI_low": float(np.quantile(rd_values, 0.025)),
                "RD_CI_high": float(np.quantile(rd_values, 0.975)),
                "variance": "hospital-cluster robust" if clusters is not None else "HC0 sandwich robust",
                "standardization": "marginal g-computation over the model analysis cohort",
                "bootstrap": "hospital-cluster" if clusters is not None else "patient-level",
                "bootstrap_successful": len(risk_values),
                "bootstrap_attempts": attempts,
                "association_not_causal": True,
            }
        )
    details = pd.DataFrame(
        [
            {
                "database": frame["database"].iloc[0],
                "outcome": outcome,
                "model": model_name,
                "N": len(frame),
                "events": int(y.sum()),
                "covariates": "; ".join(covariates) if covariates else "none",
                "converged": converged,
                "bootstrap_successful": len(bootstrap["P1"]),
                "bootstrap_attempts": attempts,
            }
        ]
    )
    return pd.DataFrame(rows), details
