"""Historical outcome/coverage/DBSOURCE join, not current final effects.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
import hashlib
import math
import pickle
import subprocess
import warnings
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests
OUT = Path(os.environ['OUTPUT_ROOT'])
PROJECT = OUT.parent
MAIN = PROJECT / 'analysis_main12_clinical_actionability_20260730'
LANDMARK = PROJECT / 'analysis_landmark_external_upgrade_20260730'
RAW_STAYS = Path(os.environ['SOURCE_31F1D1CB18'])
CANDIDATE_PATH = LANDMARK / '14_landmark12_candidate_matrix.parquet'
ASSIGNMENT_PATH = MAIN / '07_common4_selected_assignments.parquet'
ENDPOINT_PATH = MAIN / '13_organ_support_escalation_patient_level.parquet'
MODEL_PATH = MAIN / '06_common4_selected_model.pkl'
HASH_MANIFEST = MAIN / '52_final_k4_file_hash_manifest.csv'
R_SCRIPT = OUT / 'run_k4_hierarchical_and_competing_models.R'
VITALS = ['SBP', 'DBP', 'HR', 'RR']
FEATURES = [f'{vital}_h{hour}' for vital in VITALS for hour in range(12)]
PHENOTYPES = ['P1', 'P2', 'P3', 'P4']
DATABASES = ['MIMIC-IV', 'MIMIC-III', 'eICU']
SEED = 20260801

def normalize_database(series: pd.Series) -> pd.Series:
    return series.replace({'MIMIC-III MetaVision': 'MIMIC-III'})

def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    observed = frame[FEATURES].notna()
    eligible = observed.sum(axis=1).ge(32)
    for vital in VITALS:
        columns = [f'{vital}_h{hour}' for hour in range(12)]
        eligible &= observed[columns].any(axis=1)
    return eligible

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    candidates = pd.read_parquet(CANDIDATE_PATH)
    assignments = pd.read_parquet(ASSIGNMENT_PATH)
    endpoints = pd.read_parquet(ENDPOINT_PATH)
    endpoints['database'] = normalize_database(endpoints['database'])
    with MODEL_PATH.open('rb') as handle:
        model_bundle = pickle.load(handle)
    return (candidates, assignments, endpoints, model_bundle)
