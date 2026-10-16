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



SUPPORT_COLUMNS = ['first_post12_vasopressor_time', 'first_post12_invasive_ventilation_time', 'first_post12_RRT_time']


def normalize(frame):
    frame = frame.copy()
    frame['database'] = frame.database.replace({'MIMIC-III MetaVision': 'MIMIC-III'})
    frame['stay_id'] = frame.stay_id.astype(str)
    return frame


def evidence(frame):
    z = frame.copy()
    z['exit_hours'] = np.nan
    z['timestamp_death_hours'] = np.nan
    for db, start, end, death in [('MIMIC-IV', 'intime', 'outtime', 'deathtime'),
                                   ('MIMIC-III', 'INTIME', 'OUTTIME', 'DEATHTIME')]:
        mask = z.database.eq(db)
        origin = pd.to_datetime(z.loc[mask, start])
        z.loc[mask, 'exit_hours'] = (pd.to_datetime(z.loc[mask, end])-origin).dt.total_seconds()/3600
        z.loc[mask, 'timestamp_death_hours'] = (pd.to_datetime(z.loc[mask, death])-origin).dt.total_seconds()/3600
    z['first_support_hours'] = z[SUPPORT_COLUMNS].apply(pd.to_numeric, errors='coerce').min(axis=1)
    z['death_hours'] = pd.to_numeric(z.death_time_from_icu_admission, errors='coerce')
    z['target_hours'] = z[['first_support_hours', 'death_hours']].where(
        z[['first_support_hours', 'death_hours']].gt(12)).min(axis=1)
    return z


def classify_vector(z, horizon):
    target, discharge = z.target_hours, z.exit_hours
    is_target = target.le(12+horizon) & (discharge.isna() | target.le(discharge))
    is_discharge = ~is_target & discharge.le(12+horizon)
    status = np.select([is_target, is_discharge], [1, 2], default=0)
    time = np.select([is_target, is_discharge], [target, discharge], default=12+horizon)-12
    return status.astype(int), time.astype(float)


def classify_scalar(support, death, discharge, horizon):
    available = [v for v in (support, death) if np.isfinite(v) and v > 12]
    target = min(available) if available else np.inf
    end = 12+horizon
    if target <= end and (not np.isfinite(discharge) or target <= discharge):
        return 1, target-12
    if np.isfinite(discharge) and discharge <= end:
        return 2, discharge-12
    return 0, float(horizon)
