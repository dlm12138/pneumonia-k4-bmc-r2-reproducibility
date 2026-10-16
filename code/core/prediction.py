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

CANDIDATES = Path(os.environ.get('FEATURE_MATRIX', 'licensed_inputs/hourly_feature_matrix.parquet'))

DATABASES = ["MIMIC-IV", "MIMIC-III", "eICU"]


PHENOTYPES = ["P1", "P2", "P3", "P4"]


VITALS = ["SBP", "DBP", "HR", "RR"]


SEED = 20260903


BOOTSTRAP_N = 500


def trajectory_summaries() -> pd.DataFrame:
    candidate = pd.read_parquet(CANDIDATES)
    candidate["database"] = candidate["database"].replace({"MIMIC-III MetaVision": "MIMIC-III"})
    candidate["stay_id"] = candidate["stay_id"].astype(str)
    output = candidate[["database", "stay_id"]].copy()
    hours = np.arange(12, dtype=float)
    for vital in VITALS:
        values = candidate[[f"{vital}_h{hour}" for hour in range(12)]].to_numpy(float)
        observed = np.isfinite(values)
        count = observed.sum(axis=1)
        mean = np.divide(
            np.nansum(values, axis=1),
            count,
            out=np.full(len(values), np.nan),
            where=count > 0,
        )
        squared_deviation = np.nansum((values - mean[:, None]) ** 2, axis=1)
        standard_deviation = np.sqrt(
            np.divide(
                squared_deviation,
                count - 1,
                out=np.full(len(values), np.nan),
                where=count > 1,
            )
        )
        hour_mean = np.divide(
            (observed * hours[None, :]).sum(axis=1),
            count,
            out=np.full(len(values), np.nan),
            where=count > 0,
        )
        centered_hours = hours[None, :] - hour_mean[:, None]
        numerator = np.nansum(centered_hours * (values - mean[:, None]), axis=1)
        denominator = np.sum(np.where(observed, centered_hours**2, 0.0), axis=1)
        slope = np.divide(
            numerator,
            denominator,
            out=np.full(len(values), np.nan),
            where=denominator > 0,
        )
        output[f"{vital}_mean"] = mean
        output[f"{vital}_count"] = count
        output[f"{vital}_sd"] = standard_deviation
        output[f"{vital}_slope"] = slope
    output["fixed_HRRR_rule"] = (
        output["HR_mean"].gt(90.0) & output["RR_mean"].gt(20.0)
    ).astype(int)
    return output


def calibration_metrics(outcome: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit_probability = np.log(clipped / (1.0 - clipped))
    design = sm.add_constant(logit_probability)
    try:
        fitted = sm.Logit(outcome, design).fit(disp=False, maxiter=200)
        return float(fitted.params[0]), float(fitted.params[1])
    except Exception:
        return np.nan, np.nan


def metric_values(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    calibration_intercept, calibration_slope = calibration_metrics(outcome, probability)
    return {
        "AUROC": float(roc_auc_score(outcome, probability)),
        "AUPRC": float(average_precision_score(outcome, probability)),
        "Brier": float(brier_score_loss(outcome, probability)),
        "log_loss": float(log_loss(outcome, probability)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def prediction_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical,
            )
        )
    processor = ColumnTransformer(transformers)
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    return Pipeline([("processor", processor), ("model", classifier)])


def grouped_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_groups = pd.unique(groups)
    sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
    return np.concatenate([lookup[group] for group in sampled_groups])


def bootstrap_prediction_metrics(
    outcome: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    database: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + sum(ord(character) for character in database))
    metric_names = ["AUROC", "AUPRC", "Brier", "log_loss", "calibration_intercept", "calibration_slope"]
    draws = {model: {metric: [] for metric in metric_names} for model in predictions}
    comparisons = [
        ("B_base_plus_HRRR", "A_base_severity"),
        ("C_base_plus_K4", "A_base_severity"),
        ("E_base_conventional_plus_K4", "D_base_plus_conventional"),
    ]
    delta_draws = {
        comparison: {metric: [] for metric in ["AUROC", "AUPRC", "Brier"]}
        for comparison in comparisons
    }
    successful = 0
    attempts = 0
    unique_groups = pd.unique(groups)
    group_lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
    while successful < BOOTSTRAP_N and attempts < 650:
        attempts += 1
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_lookup[group] for group in sampled_groups])
        sampled_outcome = outcome[indices]
        if len(np.unique(sampled_outcome)) < 2:
            continue
        sampled_metrics = {}
        for model, probability in predictions.items():
            values = metric_values(sampled_outcome, probability[indices])
            sampled_metrics[model] = values
            for metric in metric_names:
                draws[model][metric].append(values[metric])
        for newer, reference in comparisons:
            for metric in ["AUROC", "AUPRC", "Brier"]:
                delta_draws[(newer, reference)][metric].append(
                    sampled_metrics[newer][metric] - sampled_metrics[reference][metric]
                )
        successful += 1
    if successful < BOOTSTRAP_N:
        raise RuntimeError(f"Only {successful} successful prediction bootstraps for {database}")

    model_rows = []
    for model, probability in predictions.items():
        point = metric_values(outcome, probability)
        row: dict[str, object] = {
            "row_type": "model_metric",
            "database": database,
            "outcome": "death_24h",
            "model": model,
            "reference_model": "",
            "N": len(outcome),
            "events": int(outcome.sum()),
            "validation": "5-fold grouped cross-validation; patient for MIMIC, hospital for eICU",
            "bootstrap": "grouped paired bootstrap",
            "bootstrap_successful": successful,
            "bootstrap_attempts": attempts,
        }
        for metric in metric_names:
            values = np.asarray(draws[model][metric], float)
            row[metric] = point[metric]
            row[f"{metric}_CI_low"] = float(np.nanquantile(values, 0.025))
            row[f"{metric}_CI_high"] = float(np.nanquantile(values, 0.975))
        model_rows.append(row)

    delta_rows = []
    for newer, reference in comparisons:
        point_new = metric_values(outcome, predictions[newer])
        point_reference = metric_values(outcome, predictions[reference])
        row = {
            "row_type": "paired_delta",
            "database": database,
            "outcome": "death_24h",
            "model": newer,
            "reference_model": reference,
            "N": len(outcome),
            "events": int(outcome.sum()),
            "validation": "same out-of-fold predictions and same bootstrap sample",
            "bootstrap": "grouped paired bootstrap",
            "bootstrap_successful": successful,
            "bootstrap_attempts": attempts,
        }
        for metric in ["AUROC", "AUPRC", "Brier"]:
            values = np.asarray(delta_draws[(newer, reference)][metric], float)
            row[f"delta_{metric}"] = point_new[metric] - point_reference[metric]
            row[f"delta_{metric}_CI_low"] = float(np.quantile(values, 0.025))
            row[f"delta_{metric}_CI_high"] = float(np.quantile(values, 0.975))
        delta_rows.append(row)
    return pd.DataFrame(model_rows), pd.DataFrame(delta_rows)


def marker_association(
    frame: pd.DataFrame,
    database: str,
    model_name: str,
    covariates: list[str],
    severity_module: Any,
    seed_offset: int,
) -> dict[str, object]:
    columns = [np.ones(len(frame)), frame["fixed_HRRR_rule"].to_numpy(float)]
    for covariate in covariates:
        columns.append(pd.to_numeric(frame[covariate], errors="coerce").to_numpy(float))
    design = np.column_stack(columns)
    outcome = frame["death_24h"].to_numpy(float)
    beta, converged = severity_module.fit_logistic(design, outcome)
    if not converged:
        raise RuntimeError(f"HR/RR association did not converge: {database}, {model_name}")
    clusters = frame["hospital_id"].astype(str).to_numpy() if database == "eICU" else None
    covariance = severity_module.robust_covariance(design, outcome, beta, clusters)
    standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    negative = design.copy()
    positive = design.copy()
    negative[:, 1] = 0.0
    positive[:, 1] = 1.0
    risk_negative = float(np.mean(expit(negative @ beta)))
    risk_positive = float(np.mean(expit(positive @ beta)))

    rng = np.random.default_rng(SEED + seed_offset)
    risk_negative_draws = []
    risk_positive_draws = []
    rd_draws = []
    groups = clusters if clusters is not None else frame["patient_id"].astype(str).to_numpy()
    group_codes, group_levels = pd.factorize(groups, sort=False)
    group_probability = np.full(len(group_levels), 1.0 / len(group_levels))
    attempts = 0
    while len(rd_draws) < BOOTSTRAP_N and attempts < 650:
        attempts += 1
        group_frequency = rng.multinomial(len(group_levels), group_probability).astype(float)
        frequency = group_frequency[group_codes]
        sampled_beta, sampled_converged = severity_module.fit_logistic(
            design,
            outcome,
            frequency_weights=frequency,
        )
        if not sampled_converged:
            continue
        denominator = frequency.sum()
        current_negative = float(
            np.sum(frequency * expit(negative @ sampled_beta)) / denominator
        )
        current_positive = float(
            np.sum(frequency * expit(positive @ sampled_beta)) / denominator
        )
        if not np.isfinite([current_negative, current_positive]).all():
            continue
        risk_negative_draws.append(current_negative)
        risk_positive_draws.append(current_positive)
        rd_draws.append(current_positive - current_negative)
    if len(rd_draws) < BOOTSTRAP_N:
        raise RuntimeError(f"Insufficient HR/RR bootstraps: {database}, {model_name}")
    positive_rows = frame["fixed_HRRR_rule"].eq(1)
    negative_rows = ~positive_rows
    return {
        "database": database,
        "marker": "fixed_HRRR_rule",
        "contrast": "positive_vs_negative",
        "model": model_name,
        "N": len(frame),
        "events": int(outcome.sum()),
        "positive_N": int(positive_rows.sum()),
        "positive_events": int(frame.loc[positive_rows, "death_24h"].sum()),
        "negative_or_reference_N": int(negative_rows.sum()),
        "negative_or_reference_events": int(frame.loc[negative_rows, "death_24h"].sum()),
        "OR": math.exp(beta[1]),
        "CI_low": math.exp(beta[1] - 1.96 * standard_error),
        "CI_high": math.exp(beta[1] + 1.96 * standard_error),
        "p_value": float(2.0 * norm.sf(abs(beta[1] / standard_error))),
        "standardized_risk_positive": risk_positive,
        "risk_positive_CI_low": float(np.quantile(risk_positive_draws, 0.025)),
        "risk_positive_CI_high": float(np.quantile(risk_positive_draws, 0.975)),
        "standardized_risk_negative_or_reference": risk_negative,
        "risk_negative_CI_low": float(np.quantile(risk_negative_draws, 0.025)),
        "risk_negative_CI_high": float(np.quantile(risk_negative_draws, 0.975)),
        "RD_fraction": risk_positive - risk_negative,
        "RD_percentage_points": 100.0 * (risk_positive - risk_negative),
        "RD_CI_low": float(np.quantile(rd_draws, 0.025)),
        "RD_CI_high": float(np.quantile(rd_draws, 0.975)),
        "bootstrap_successful": len(rd_draws),
        "variance": "hospital-cluster robust" if database == "eICU" else "HC0 sandwich robust",
    }
