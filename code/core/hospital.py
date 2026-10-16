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



def site_effects(frame: pd.DataFrame) -> pd.DataFrame:
    cells = (
        frame.groupby(["hospital_id", "p3"])["death_24h"]
        .agg(["size", "sum"])
        .reset_index()
    )
    n_wide = cells.pivot(index="hospital_id", columns="p3", values="size").fillna(0)
    e_wide = cells.pivot(index="hospital_id", columns="p3", values="sum").fillna(0)
    rows = []
    for hospital in sorted(frame["hospital_id"].dropna().unique()):
        p2_n = float(n_wide.loc[hospital].get(0, 0))
        p3_n = float(n_wide.loc[hospital].get(1, 0))
        p2_events = float(e_wide.loc[hospital].get(0, 0))
        p3_events = float(e_wide.loc[hospital].get(1, 0))
        eligible = (
            p2_n + p3_n >= 50
            and p2_n >= 5
            and p3_n >= 5
            and p2_events + p3_events >= 5
        )
        a = p3_events
        b = p3_n - p3_events
        c = p2_events
        d = p2_n - p2_events
        continuity_correction = min(a, b, c, d) == 0
        if continuity_correction:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        log_or = math.log((a * d) / (b * c)) if min(a, b, c, d) > 0 else np.nan
        standard_error = (
            math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
            if min(a, b, c, d) > 0
            else np.nan
        )
        rows.append(
            {
                "hospital_id": hospital,
                "N": int(p2_n + p3_n),
                "P2_N": int(p2_n),
                "P3_N": int(p3_n),
                "P2_events": int(p2_events),
                "P3_events": int(p3_events),
                "eligible_for_hospital_specific": eligible,
                "OR": math.exp(log_or) if np.isfinite(log_or) else np.nan,
                "CI_low": math.exp(log_or - 1.96 * standard_error)
                if np.isfinite(log_or)
                else np.nan,
                "CI_high": math.exp(log_or + 1.96 * standard_error)
                if np.isfinite(log_or)
                else np.nan,
                "p_value": float(2.0 * norm.sf(abs(log_or / standard_error)))
                if np.isfinite(log_or)
                else np.nan,
                "log_OR": log_or,
                "SE_log_OR": standard_error,
                "continuity_correction_0_5": continuity_correction,
                "criterion": (
                    "P2+P3 N>=50; P2 N>=5; P3 N>=5; combined events>=5"
                ),
                "included_in_synthesis": False,
            }
        )
    return pd.DataFrame(rows)


def loho(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = ["death_24h", "p3", "age10", "male", "coverage10", "hospital_id"]
    for hospital in sorted(frame["hospital_id"].dropna().unique()):
        subset = frame.loc[frame["hospital_id"].ne(hospital)].dropna(subset=required)
        model = smf.glm(
            "death_24h ~ p3 + age10 + male + coverage10",
            data=subset,
            family=sm.families.Binomial(),
        )
        fit = model.fit(
            cov_type="cluster", cov_kwds={"groups": subset["hospital_id"].to_numpy()}
        )
        beta = float(fit.params["p3"])
        standard_error = float(fit.bse["p3"])
        rows.append(
            {
                "excluded_hospital_id": hospital,
                "N": len(subset),
                "beta": beta,
                "SE": standard_error,
                "OR": math.exp(beta),
                "CI_low": math.exp(beta - 1.96 * standard_error),
                "CI_high": math.exp(beta + 1.96 * standard_error),
                "p_value": float(2.0 * norm.sf(abs(beta / standard_error))),
            }
        )
    return pd.DataFrame(rows)
