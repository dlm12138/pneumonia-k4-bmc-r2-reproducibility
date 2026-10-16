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



SPO2_COLUMNS = """SpO2_mean_0_12h SpO2_max_0_12h SpO2_min_0_12h
SpO2_sd_0_12h SpO2_n_records_0_12h SpO2_burden_lt90_0_12h
SpO2_burden_lt92_0_12h SpO2_burden_lt94_0_12h""".split()


def aggregate_hourly_spo2(hourly: pd.DataFrame) -> pd.DataFrame:
    """Recreate the historical unweighted hourly-summary definition."""
    hourly = hourly.copy()
    hourly["SpO2"] = pd.to_numeric(hourly["SpO2"], errors="coerce")
    observed = hourly.dropna(subset=["SpO2"]).copy()
    grouped = observed.groupby(["database", "stay_id"], sort=False)["SpO2"]
    result = grouped.agg(
        SpO2_mean_0_12h="mean",
        SpO2_max_0_12h="max",
        SpO2_min_0_12h="min",
        SpO2_n_records_0_12h="count",
    ).reset_index()
    # The historical extractor used numpy.std with ddof=0.
    standard_deviation = grouped.apply(
        lambda values: float(np.std(values.to_numpy(float), ddof=0))
    ).rename("SpO2_sd_0_12h")
    result = result.merge(
        standard_deviation.reset_index(),
        on=["database", "stay_id"],
        how="left",
        validate="one_to_one",
    )
    for threshold in [90, 92, 94]:
        burden = grouped.apply(
            lambda values, cutoff=threshold: float(
                np.mean(values.to_numpy(float) < cutoff)
            )
        ).rename(f"SpO2_burden_lt{threshold}_0_12h")
        result = result.merge(
            burden.reset_index(),
            on=["database", "stay_id"],
            how="left",
            validate="one_to_one",
        )
    return result[["database", "stay_id"] + SPO2_COLUMNS]
