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


def summary_continuous(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "NOT_AVAILABLE"
    return (
        f"{numeric.median():.2f} [{numeric.quantile(0.25):.2f}, "
        f"{numeric.quantile(0.75):.2f}] (n={len(numeric)})"
    )


def summary_binary(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "NOT_AVAILABLE"
    return f"{int(numeric.sum())}/{len(numeric)} ({100.0 * numeric.mean():.1f}%)"


def smd(left: pd.Series, right: pd.Series) -> float:
    left_numeric = pd.to_numeric(left, errors="coerce").dropna()
    right_numeric = pd.to_numeric(right, errors="coerce").dropna()
    if len(left_numeric) < 2 or len(right_numeric) < 2:
        return np.nan
    pooled = math.sqrt((left_numeric.var(ddof=1) + right_numeric.var(ddof=1)) / 2.0)
    return float((left_numeric.mean() - right_numeric.mean()) / pooled) if pooled > 0 else 0.0


def variable_specs() -> list[dict[str, str | None]]:
    specs: list[dict[str, str | None]] = [
        {"variable": "age", "column": "age", "type": "continuous", "role": "external descriptor", "time_window": "at ICU admission", "unit": "years"},
        {"variable": "male", "column": "male", "type": "binary", "role": "external descriptor", "time_window": "at ICU admission", "unit": "binary"},
    ]
    for comorbidity in [
        "myocardial_infarction",
        "heart_failure",
        "cerebrovascular",
        "chronic_pulmonary",
        "diabetes",
        "liver_disease",
        "renal_disease",
        "malignancy",
    ]:
        specs.append(
            {
                "variable": comorbidity,
                "column": f"comorb_{comorbidity}",
                "type": "binary",
                "role": "external descriptor",
                "time_window": "diagnosis-derived history; timing not resolved",
                "unit": "binary",
            }
        )
    def add_spec(
        variable: str,
        column: str,
        kind: str,
        role: str,
        time_window: str,
        unit: str,
    ) -> None:
        specs.append(
            {
                "variable": variable,
                "column": column,
                "type": kind,
                "role": role,
                "time_window": time_window,
                "unit": unit,
            }
        )

    add_spec("SpO2_mean", "SpO2_mean_0_12h", "continuous", "external descriptor", "0-12 h", "%")
    add_spec("SpO2_minimum", "SpO2_min_0_12h", "continuous", "external descriptor", "0-12 h", "%")
    add_spec("FiO2_mean", "FiO2_mean_0_12h", "continuous", "external descriptor", "0-12 h", "fraction")
    add_spec("S_F_ratio_minimum", "SF_ratio_min_0_12h", "continuous", "external descriptor", "0-12 h", "ratio")
    add_spec("invasive_mechanical_ventilation", "active_invasive_ventilation_at_12h", "binary", "external descriptor", "active at 12 h", "binary")
    add_spec("vasopressor", "active_vasopressor_at_12h", "binary", "external descriptor", "active at 12 h", "binary")
    add_spec("RRT", "active_RRT_at_12h", "binary", "external descriptor", "active at 12 h", "binary")
    add_spec("temperature_mean", "TEMP_mean_0_12h", "continuous", "external descriptor; characterization only", "0-12 h", "degrees C")
    for vital in VITALS:
        specs.append(
            {
                "variable": f"{vital}_mean",
                "column": f"{vital}_mean_0_12h",
                "type": "continuous",
                "role": "phenotype-defining variable",
                "time_window": "0-12 h",
                "unit": "mmHg" if vital in ["SBP", "DBP"] else "beats/min" if vital == "HR" else "breaths/min",
            }
        )
    unavailable = [
        ("Charlson_index", "points"),
        ("NIV", "binary"),
        ("HFNC", "binary"),
        ("norepinephrine_equivalent_dose", "dose"),
        ("lactate_maximum", "mmol/L"),
        ("creatinine", "mg/dL"),
        ("BUN", "mg/dL"),
        ("bilirubin", "mg/dL"),
        ("platelet_minimum", "10^9/L"),
        ("WBC", "10^9/L"),
        ("pH_minimum", "pH"),
        ("GCS_minimum", "points"),
        ("PaO2_FiO2_ratio", "ratio"),
    ]
    for variable, unit in unavailable:
        specs.append(
            {
                "variable": variable,
                "column": None,
                "type": "continuous",
                "role": "external descriptor",
                "time_window": "NOT_AVAILABLE_PRELANDMARK",
                "unit": unit,
            }
        )
    return specs


def characterize(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    by_database = {}
    ranked_rows = []
    for database in DATABASES:
        frame = data.loc[data["database"].eq(database)].copy()
        rows = []
        for spec in variable_specs():
            column = spec["column"]
            summaries = {}
            observed_counts = {}
            if column is None or column not in frame.columns:
                for phenotype in PHENOTYPES:
                    summaries[f"{phenotype}_summary"] = "NOT_AVAILABLE"
                    observed_counts[f"{phenotype}_observed_N"] = 0
                max_pairwise = np.nan
                status = "NOT_AVAILABLE_PRELANDMARK"
            else:
                for phenotype in PHENOTYPES:
                    values = frame.loc[frame["phenotype"].eq(phenotype), column]
                    summaries[f"{phenotype}_summary"] = (
                        summary_binary(values) if spec["type"] == "binary" else summary_continuous(values)
                    )
                    observed_counts[f"{phenotype}_observed_N"] = int(values.notna().sum())
                pairwise = [
                    abs(
                        smd(
                            frame.loc[frame["phenotype"].eq(left), column],
                            frame.loc[frame["phenotype"].eq(right), column],
                        )
                    )
                    for left, right in combinations(PHENOTYPES, 2)
                ]
                finite = [value for value in pairwise if np.isfinite(value)]
                max_pairwise = max(finite) if finite else np.nan
                status = "AVAILABLE" if sum(observed_counts.values()) > 0 else "NOT_AVAILABLE"
            row = {
                "database": database,
                **spec,
                "availability_status": status,
                "max_pairwise_SMD": max_pairwise,
                **summaries,
                **observed_counts,
            }
            rows.append(row)
            ranked_rows.append(row.copy())
        by_database[database] = pd.DataFrame(rows)
    ranked = pd.DataFrame(ranked_rows).sort_values(
        ["database", "max_pairwise_SMD"], ascending=[True, False], na_position="last"
    )
    return by_database, ranked


def stress_profile(data: pd.DataFrame) -> pd.DataFrame:
    requested = [
        "HR_mean",
        "RR_mean",
        "SBP_mean",
        "DBP_mean",
        "SpO2_mean",
        "SpO2_minimum",
        "temperature_mean",
        "invasive_mechanical_ventilation",
        "vasopressor",
        "RRT",
        "lactate_maximum",
        "creatinine",
        "PaO2_FiO2_ratio",
        "GCS_minimum",
    ]
    specs = {str(spec["variable"]): spec for spec in variable_specs()}
    rows = []
    for database in DATABASES:
        frame = data.loc[data["database"].eq(database)]
        p3 = frame.loc[frame["phenotype"].eq("P3")]
        p2 = frame.loc[frame["phenotype"].eq("P2")]
        for variable in requested:
            spec = specs[variable]
            column = spec["column"]
            if column is None or column not in frame.columns:
                p3_summary = "NOT_AVAILABLE"
                p2_summary = "NOT_AVAILABLE"
                difference = np.nan
                standardized = np.nan
                p3_observed = 0
                p2_observed = 0
            else:
                p3_values = pd.to_numeric(p3[column], errors="coerce")
                p2_values = pd.to_numeric(p2[column], errors="coerce")
                formatter = summary_binary if spec["type"] == "binary" else summary_continuous
                p3_summary = formatter(p3_values)
                p2_summary = formatter(p2_values)
                difference = float(p3_values.mean() - p2_values.mean())
                standardized = smd(p3_values, p2_values)
                p3_observed = int(p3_values.notna().sum())
                p2_observed = int(p2_values.notna().sum())
            rows.append(
                {
                    "database": database,
                    "variable": variable,
                    "role": spec["role"],
                    "time_window": spec["time_window"],
                    "P3_summary": p3_summary,
                    "P2_summary": p2_summary,
                    "P3_observed_N": p3_observed,
                    "P2_observed_N": p2_observed,
                    "mean_or_proportion_difference_P3_minus_P2": difference,
                    "SMD_P3_minus_P2": standardized,
                }
            )
    return pd.DataFrame(rows)


CANDIDATES = Path(os.environ.get('FEATURE_MATRIX', 'PRIVATE_FEATURE_MATRIX_REQUIRED.parquet'))

def append_vital_summaries(data: pd.DataFrame) -> pd.DataFrame:
    candidate = pd.read_parquet(CANDIDATES)
    candidate["database"] = candidate["database"].replace({"MIMIC-III MetaVision": "MIMIC-III"})
    candidate["stay_id"] = candidate["stay_id"].astype(str)
    columns = ["database", "stay_id"]
    for vital in VITALS:
        hourly = [f"{vital}_h{hour}" for hour in range(12)]
        candidate[f"{vital}_mean_0_12h"] = candidate[hourly].mean(axis=1, skipna=True)
        candidate[f"{vital}_min_0_12h"] = candidate[hourly].min(axis=1, skipna=True)
        candidate[f"{vital}_max_0_12h"] = candidate[hourly].max(axis=1, skipna=True)
        columns.extend([f"{vital}_mean_0_12h", f"{vital}_min_0_12h", f"{vital}_max_0_12h"])
    return data.merge(
        candidate[columns],
        on=["database", "stay_id"],
        how="left",
        validate="one_to_one",
    )
