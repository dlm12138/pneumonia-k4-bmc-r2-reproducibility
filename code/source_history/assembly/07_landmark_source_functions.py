"""Historical candidates and source-tier matrix construction.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit, logsumexp
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score
OUT = Path(os.environ['OUTPUT_ROOT'])
PROJECT = OUT.parent
REBUILD = PROJECT / 'analysis_rebuild_20260729'
UPGRADE = PROJECT / 'analysis_topjournal_upgrade_20260730'
SOURCE_0529 = Path(os.environ['SOURCE_0C46C81942'])
SOURCE_0517 = Path(os.environ['SOURCE_C74F17AEDB'])
SOURCE_0601 = Path(os.environ['SOURCE_68000518F1'])
RAW = Path(os.environ['SOURCE_31F1D1CB18'])
DATABASE_ORDER = ['MIMIC-IV', 'MIMIC-III', 'eICU', 'HiRID', 'SICdb']
PHENOTYPES = ['P1', 'P2', 'P3', 'P4']
VITALS = ['SBP', 'DBP', 'HR', 'RR', 'TEMP']
FEATURES = [f'{vital}_h{hour}' for vital in VITALS for hour in range(12)]
THRESHOLDS = {'MIMIC-IV': 40, 'MIMIC-III': 30, 'eICU': 30, 'HiRID': 30, 'SICdb': 30}

def normalize_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').astype('Int64').astype('string').fillna('')

def load_hourly_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected = pd.read_parquet(SOURCE_0529 / '03_hourly_vitals_wide_cleaned.parquet')
    corrected = corrected.rename(columns={'stay_id_or_icustay_id': 'stay_id', 'hour_index': 'hour'})
    corrected['stay_id'] = normalize_id(corrected['stay_id'])
    corrected = corrected[['database', 'stay_id', 'hour', *VITALS]].copy()
    corrected['source_tier'] = 'corrected_20260529'
    legacy_frames = []
    specifications = [('MIMIC-IV', 'mimiciv_vital_12h_hourly_all.csv', 'stay_id'), ('MIMIC-III', 'mimiciii_vital_12h_hourly_all.csv', 'stay_id'), ('eICU', 'eicu_vital_12h_hourly_all.csv', 'patientunitstayid')]
    for database, filename, id_column in specifications:
        frame = pd.read_csv(SOURCE_0517 / filename, low_memory=False)
        frame = frame.rename(columns={id_column: 'stay_id'})
        frame['stay_id'] = normalize_id(frame['stay_id'])
        frame['database'] = database
        frame['source_tier'] = 'legacy_full_pneumonia_extract_20260517'
        legacy_frames.append(frame[['database', 'stay_id', 'hour', *VITALS, 'source_tier']])
    legacy = pd.concat(legacy_frames, ignore_index=True)
    overlap = corrected.merge(legacy, on=['database', 'stay_id', 'hour'], how='inner', suffixes=('_corrected', '_legacy'))
    audit_rows = []
    for database in ['MIMIC-IV', 'MIMIC-III', 'eICU']:
        selected = overlap.loc[overlap['database'].eq(database)]
        for vital in VITALS:
            left = pd.to_numeric(selected[f'{vital}_corrected'], errors='coerce')
            right = pd.to_numeric(selected[f'{vital}_legacy'], errors='coerce')
            paired = left.notna() & right.notna()
            difference = (left.loc[paired] - right.loc[paired]).abs()
            audit_rows.append({'database': database, 'vital': vital, 'paired_stay_hours': int(paired.sum()), 'median_absolute_difference': difference.median(), 'p95_absolute_difference': difference.quantile(0.95), 'exact_value_agreement_percent': 100 * difference.le(1e-08).mean() if len(difference) else np.nan})
    return (pd.concat([corrected, legacy], ignore_index=True), pd.DataFrame(audit_rows))

def build_landmark_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    flow_rows = []
    miv = pd.read_csv(RAW / 'mimiciv_pneumonia_stays_raw.csv', low_memory=False)
    miv['_intime'] = pd.to_datetime(miv['intime'], errors='coerce')
    miv['_deathtime'] = pd.to_datetime(miv['deathtime'], errors='coerce')
    miv = miv.sort_values(['subject_id', '_intime', 'stay_id']).drop_duplicates('subject_id', keep='first')
    miv['age_std'] = pd.to_numeric(miv['age'], errors='coerce')
    miv['los_hours_std'] = pd.to_numeric(miv['los_hours'], errors='coerce')
    miv['death_hours'] = (miv['_deathtime'] - miv['_intime']).dt.total_seconds() / 3600
    miv['alive_at_landmark'] = miv['death_hours'].isna() | miv['death_hours'].gt(12)
    miv['outcome_available'] = True
    miv['hospital_mortality'] = miv['_deathtime'].notna().astype(int)
    miv['male'] = miv['gender'].astype('string').str.upper().eq('M').astype(float)
    miv['stay_id_std'] = normalize_id(miv['stay_id'])
    miv['hospital_id'] = np.nan
    frames.append(('MIMIC-IV', miv))
    miii = pd.read_csv(RAW / 'mimiciii_pneumonia_stays_raw.csv', low_memory=False)
    miii['_intime'] = pd.to_datetime(miii['INTIME'], errors='coerce')
    miii['_deathtime'] = pd.to_datetime(miii['DEATHTIME'], errors='coerce')
    miii = miii.sort_values(['SUBJECT_ID', '_intime', 'ICUSTAY_ID']).drop_duplicates('SUBJECT_ID', keep='first')
    miii['age_std'] = pd.to_numeric(miii['age'], errors='coerce')
    miii['los_hours_std'] = pd.to_numeric(miii['los_hours'], errors='coerce')
    miii['death_hours'] = (miii['_deathtime'] - miii['_intime']).dt.total_seconds() / 3600
    miii['alive_at_landmark'] = miii['death_hours'].isna() | miii['death_hours'].gt(12)
    miii['outcome_available'] = True
    miii['hospital_mortality'] = miii['_deathtime'].notna().astype(int)
    miii['male'] = miii['GENDER'].astype('string').str.upper().eq('M').astype(float)
    miii['stay_id_std'] = normalize_id(miii['ICUSTAY_ID'])
    miii['hospital_id'] = np.nan
    frames.append(('MIMIC-III', miii))
    eicu = pd.read_csv(RAW / 'eicu_pneumonia_stays_raw.csv', low_memory=False)
    eicu = eicu.sort_values(['uniquepid', 'patienthealthsystemstayid', 'unitvisitnumber', 'patientunitstayid']).drop_duplicates('uniquepid', keep='first')
    eicu['age_std'] = pd.to_numeric(eicu['age_num'], errors='coerce')
    eicu['los_hours_std'] = pd.to_numeric(eicu['unitdischargeoffset'], errors='coerce') / 60
    eicu['death_hours'] = np.where(pd.to_numeric(eicu['hospital_mortality'], errors='coerce').eq(1), pd.to_numeric(eicu['hospitaldischargeoffset'], errors='coerce') / 60, np.nan)
    eicu['alive_at_landmark'] = pd.isna(eicu['death_hours']) | (eicu['death_hours'] > 12)
    eicu['outcome_available'] = eicu['hospitaldischargestatus'].notna()
    eicu['hospital_mortality'] = pd.to_numeric(eicu['hospital_mortality'], errors='coerce')
    eicu_gender = eicu['gender'].astype('string').str.lower()
    eicu['male'] = np.where(eicu_gender.eq('male').fillna(False), 1.0, np.where(eicu_gender.eq('female').fillna(False), 0.0, np.nan))
    eicu['stay_id_std'] = normalize_id(eicu['patientunitstayid'])
    eicu['hospital_id'] = eicu['hospitalid']
    frames.append(('eICU', eicu))
    candidates = []
    for database, frame in frames:
        adult = frame['age_std'].ge(18)
        at_12_hours = frame['los_hours_std'].ge(12)
        alive = frame['alive_at_landmark']
        outcome = frame['outcome_available']
        eligible = adult & at_12_hours & alive & outcome
        flow_rows.extend([{'database': database, 'step': 'first pneumonia-associated ICU stay', 'n': len(frame)}, {'database': database, 'step': 'adult age >=18', 'n': int(adult.sum())}, {'database': database, 'step': 'still in ICU at 12 hours', 'n': int((adult & at_12_hours).sum())}, {'database': database, 'step': 'alive at 12-hour landmark', 'n': int((adult & at_12_hours & alive).sum())}, {'database': database, 'step': 'post-landmark hospital mortality classifiable', 'n': int(eligible.sum())}, {'database': database, 'step': 'deaths at or before 12 hours', 'n': int((adult & at_12_hours & ~alive).sum())}, {'database': database, 'step': 'ICU discharge before 12 hours', 'n': int((adult & ~at_12_hours).sum())}])
        selected = frame.loc[eligible].copy()
        standardized = pd.DataFrame({'database': database, 'stay_id': selected['stay_id_std'], 'age': selected['age_std'], 'male': selected['male'], 'hospital_id': selected['hospital_id'], 'hospital_mortality': selected['hospital_mortality'], 'icu_los_hours': selected['los_hours_std']})
        candidates.append(standardized)
    return (pd.concat(candidates, ignore_index=True), pd.DataFrame(flow_rows))

def build_landmark_matrix(candidates: pd.DataFrame, hourly_sources: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected_ids = hourly_sources.loc[hourly_sources['source_tier'].eq('corrected_20260529'), ['database', 'stay_id']].drop_duplicates().assign(corrected_available=True)
    source = candidates[['database', 'stay_id']].merge(corrected_ids, on=['database', 'stay_id'], how='left')
    corrected_available = source['corrected_available'].astype('boolean').fillna(False)
    source['selected_source_tier'] = np.where(corrected_available, 'corrected_20260529', 'legacy_full_pneumonia_extract_20260517')
    selected_hourly = hourly_sources.merge(source[['database', 'stay_id', 'selected_source_tier']], left_on=['database', 'stay_id', 'source_tier'], right_on=['database', 'stay_id', 'selected_source_tier'], how='inner')
    selected_hourly['hour'] = pd.to_numeric(selected_hourly['hour'], errors='coerce')
    selected_hourly = selected_hourly.loc[selected_hourly['hour'].between(0, 11, inclusive='both')].copy()
    for vital in VITALS:
        selected_hourly[vital] = pd.to_numeric(selected_hourly[vital], errors='coerce')
        valid_min = float(SCHEMA.loc[SCHEMA['vital'].eq(vital), 'valid_min'].iloc[0])
        valid_max = float(SCHEMA.loc[SCHEMA['vital'].eq(vital), 'valid_max'].iloc[0])
        selected_hourly.loc[~selected_hourly[vital].between(valid_min, valid_max, inclusive='both'), vital] = np.nan
    matrices = []
    for database, cohort in candidates.groupby('database', sort=False):
        hourly = selected_hourly.loc[selected_hourly['database'].eq(database)]
        matrix = hourly.pivot_table(index='stay_id', columns='hour', values=VITALS, aggfunc='mean')
        matrix.columns = [f'{vital}_h{int(hour)}' for vital, hour in matrix.columns]
        matrix = matrix.reindex(columns=FEATURES).reset_index()
        joined = cohort.merge(matrix, on='stay_id', how='left')
        joined['source_tier'] = joined[['database', 'stay_id']].merge(source[['database', 'stay_id', 'selected_source_tier']], on=['database', 'stay_id'], how='left')['selected_source_tier'].to_numpy()
        joined['observed_dimensions_before_threshold'] = joined[FEATURES].notna().sum(axis=1)
        matrices.append(joined)
    matrix = pd.concat(matrices, ignore_index=True)
    source_audit = matrix.groupby(['database', 'source_tier'], dropna=False).agg(candidate_n=('stay_id', 'size'), with_any_trajectory_n=('observed_dimensions_before_threshold', lambda values: int((values > 0).sum())), median_observed_dimensions=('observed_dimensions_before_threshold', 'median')).reset_index()
    return (matrix, source_audit)
SCHEMA = pd.read_csv(os.environ['CLEANING_SCHEMA'], usecols=['vital', 'valid_min', 'valid_max'])
