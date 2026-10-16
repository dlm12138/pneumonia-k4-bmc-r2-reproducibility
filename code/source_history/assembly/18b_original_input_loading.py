"""Actual original endpoint/respiratory/APACHE/clinical input joins; intermediate old oxygen is replaced later.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
ROOT = Path(os.environ['SOURCE_C1E4D8E159'])
PROJECT = Path(os.environ['SOURCE_7E9AB01024'])
HISTORY = PROJECT / '01_k5-7.29_history/k5-7.29'
MAIN = HISTORY / 'analysis_main12_clinical_actionability_20260730'
LANDMARK = HISTORY / 'analysis_landmark_external_upgrade_20260730'
DEEP = HISTORY / 'analysis_k4_evidence_deepening_20260801'
RESPIRATORY = PROJECT / '02_STTT_K4_K5_analysis/2026-05-30/06_patient_level_oxygenation_respiratory_severity_dataset.parquet'
CLINICAL_ROOT = Path(os.environ['SOURCE_C74F17AEDB'])
APACHE = Path(os.environ['SOURCE_C950215C87'])
ENDPOINT = MAIN / '13_organ_support_escalation_patient_level.parquet'
CANDIDATES = LANDMARK / '14_landmark12_candidate_matrix.parquet'
OLD_SEVERITY = DEEP / '102_current_k4_severity_sensitivity_models.csv'
SEED = 20260903
BOOTSTRAP_N = 500
BOOTSTRAP_MAX_ATTEMPTS = 650
DATABASES = ['MIMIC-IV', 'MIMIC-III', 'eICU']
PHENOTYPES = ['P1', 'P2', 'P3', 'P4']
REFERENCE = 'P2'
COMORBIDITIES = ['myocardial_infarction', 'heart_failure', 'cerebrovascular', 'chronic_pulmonary', 'diabetes', 'liver_disease', 'renal_disease', 'malignancy']

def normalize_database(series: pd.Series) -> pd.Series:
    return series.replace({'MIMIC-III MetaVision': 'MIMIC-III'})

def load_analysis_data() -> pd.DataFrame:
    endpoint = pd.read_parquet(ENDPOINT)
    endpoint['database'] = normalize_database(endpoint['database'])
    endpoint['stay_id'] = endpoint['stay_id'].astype(str)
    candidates = pd.read_parquet(CANDIDATES)
    candidates['database'] = normalize_database(candidates['database'])
    candidates['stay_id'] = candidates['stay_id'].astype(str)
    temp_columns = [f'TEMP_h{hour}' for hour in range(12)]
    candidates['TEMP_mean_0_12h'] = candidates[temp_columns].mean(axis=1, skipna=True)
    candidates['TEMP_min_0_12h'] = candidates[temp_columns].min(axis=1, skipna=True)
    temp = candidates[['database', 'stay_id', 'TEMP_mean_0_12h', 'TEMP_min_0_12h']]
    respiratory = pd.read_parquet(RESPIRATORY)
    respiratory['database'] = normalize_database(respiratory['database'])
    respiratory['stay_id'] = respiratory['stay_id_or_icustay_id'].astype(str)
    respiratory_columns = [column for column in respiratory.columns if column.startswith(('SpO2_', 'FiO2_', 'SF_ratio_', 'PaO2_', 'PF_ratio_', 'lactate_', 'pH_', 'HCO3_'))]
    respiratory = respiratory[['database', 'stay_id'] + respiratory_columns]
    if respiratory.duplicated(['database', 'stay_id']).any():
        raise ValueError('Respiratory source has duplicate database/stay_id keys')
    result = endpoint.merge(temp, on=['database', 'stay_id'], how='left', validate='one_to_one')
    result = result.merge(respiratory, on=['database', 'stay_id'], how='left', validate='one_to_one')
    clinical_pieces = []
    clinical_specs = {'MIMIC-IV': CLINICAL_ROOT / 'mimiciv_clinical_features.csv', 'MIMIC-III': CLINICAL_ROOT / 'mimiciii_clinical_features.csv', 'eICU': CLINICAL_ROOT / 'eicu_clinical_features.csv'}
    common_labs = ['BUN', 'WBC', 'bilirubin', 'creatinine', 'hemoglobin', 'lactate', 'pH', 'platelet']
    for database, path in clinical_specs.items():
        raw = pd.read_csv(path, low_memory=False)
        if database == 'eICU':
            raw['stay_id'] = pd.to_numeric(raw['patientunitstayid'], errors='coerce').astype('Int64').astype(str)
            column_map = {'BUN': 'BUN', 'WBC': 'WBC', 'bilirubin': 'bilirubin_y', 'creatinine': 'creatinine_y', 'hemoglobin': 'hemoglobin', 'lactate': 'lactate', 'pH': 'pH', 'platelet': 'platelet'}
        else:
            raw['stay_id'] = pd.to_numeric(raw['stay_id'], errors='coerce').astype('Int64').astype(str)
            column_map = {variable: variable for variable in common_labs if variable in raw.columns}
        piece = pd.DataFrame({'stay_id': raw['stay_id'], 'database': database})
        for comorbidity in COMORBIDITIES:
            piece[f'comorb_{comorbidity}'] = pd.to_numeric(raw[comorbidity], errors='coerce') if comorbidity in raw.columns else np.nan
        for standardized, source_column in column_map.items():
            piece[f'clinical_{standardized}'] = pd.to_numeric(raw[source_column], errors='coerce')
        clinical_pieces.append(piece)
    clinical = pd.concat(clinical_pieces, ignore_index=True)
    if clinical.duplicated(['database', 'stay_id']).any():
        raise ValueError('Clinical source has duplicate database/stay_id keys')
    result = result.merge(clinical, on=['database', 'stay_id'], how='left', validate='one_to_one')
    apache = pd.read_csv(APACHE, usecols=['patientunitstayid', 'acutephysiologyscore', 'apachescore', 'apacheversion'], low_memory=False)
    apache = apache.loc[apache['apacheversion'].eq('IVa')].copy()
    duplicate = apache.groupby('patientunitstayid').agg(row_N=('patientunitstayid', 'size'), APS_unique_N=('acutephysiologyscore', 'nunique'), APACHE_unique_N=('apachescore', 'nunique'))
    if (duplicate['row_N'].gt(1) & duplicate[['APS_unique_N', 'APACHE_unique_N']].max(axis=1).gt(1)).any():
        raise ValueError('Conflicting duplicate APACHE IVa rows')
    apache = apache.drop_duplicates('patientunitstayid')
    apache['stay_id'] = pd.to_numeric(apache['patientunitstayid'], errors='coerce').astype('Int64').astype(str)
    apache['acutephysiologyscore'] = pd.to_numeric(apache['acutephysiologyscore'], errors='coerce')
    apache.loc[apache['acutephysiologyscore'].lt(0), 'acutephysiologyscore'] = np.nan
    apache = apache[['stay_id', 'acutephysiologyscore', 'apachescore']]
    eicu = result.loc[result['database'].eq('eICU')].merge(apache, on='stay_id', how='left', validate='one_to_one')
    other = result.loc[~result['database'].eq('eICU')].copy()
    other['acutephysiologyscore'] = np.nan
    other['apachescore'] = np.nan
    result = pd.concat([other, eicu], ignore_index=True, sort=False)
    return result

def variable_availability(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = [('age', 'age', 'age at ICU admission', 'at/before ICU admission', 'years', 'YES', 'available in all databases'), ('sex', 'male', 'recorded sex encoded as male indicator', 'at/before ICU admission', 'binary', 'YES', 'available in all databases; 3 eICU values missing'), ('recording_density', 'observed_dimensions_48d', 'observed locked K4 dimensions', '0-12 h', 'count of 48', 'YES', 'current-model covariate, not a severity marker'), ('Charlson_index', None, 'Charlson Comorbidity Index', 'history', 'points', 'NO', 'not present in the frozen project inputs')]
    for comorbidity in COMORBIDITIES:
        metadata.append((comorbidity, f'comorb_{comorbidity}', 'diagnosis-derived comorbidity indicator', 'history; diagnosis timing not resolved', 'binary', 'NO', 'available for MIMIC-IV/III but not eICU'))
    metadata.extend([('SpO2_mean', 'SpO2_mean_0_12h', 'mean peripheral oxygen saturation', '0-12 h', '%', 'YES', 'exact stay-ID linkage; median imputation plus missingness indicator used'), ('SpO2_minimum', 'SpO2_min_0_12h', 'minimum peripheral oxygen saturation', '0-12 h', '%', 'YES', 'exact stay-ID linkage; used in Model 1'), ('FiO2_mean', 'FiO2_mean_0_12h', 'mean inspired oxygen fraction', '0-12 h', 'fraction', 'NO', 'not available in all three databases in the retained extraction'), ('PaO2', 'PaO2_first_0_24h', 'first arterial oxygen tension', '0-24 h', 'mmHg', 'NO', 'window extends beyond landmark and MIMIC-III is unavailable'), ('PaO2_FiO2_ratio', 'PF_ratio_min_0_24h', 'minimum available P/F ratio', '0-24 h', 'ratio', 'NO', 'window extends beyond landmark and is not available in all databases'), ('mechanical_ventilation', 'active_invasive_ventilation_at_12h', 'active invasive ventilation at landmark', 'active at 12 h', 'binary', 'YES', 'interval-timed and available in all databases; Model 2'), ('NIV', None, 'non-invasive ventilation', 'before/at 12 h', 'binary', 'NO', 'not consistently separable in current frozen sources'), ('HFNC', None, 'high-flow nasal cannula', 'before/at 12 h', 'binary', 'NO', 'not consistently separable in current frozen sources'), ('vasopressor_use', 'active_vasopressor_at_12h', 'active vasopressor at landmark', 'active at 12 h', 'binary', 'YES', 'interval-timed and available in all databases; Model 2'), ('norepinephrine_equivalent_dose', None, 'norepinephrine-equivalent vasopressor dose', 'before/at 12 h', 'dose', 'NO', 'no harmonized dose variable in retained project'), ('GCS', None, 'Glasgow Coma Scale', '0-12 h', 'points', 'NO', 'not available in current frozen analytical inputs'), ('lactate', 'lactate_first_0_24h', 'first lactate in retained exploratory extraction', '0-24 h', 'mmol/L', 'NO', 'window extends beyond landmark and unavailable for MIMIC-III'), ('creatinine', 'clinical_creatinine', 'retained clinical-feature creatinine', 'timing not verified', 'mg/dL', 'NO', 'source timing cannot prove measurement by 12 h and MIMIC-III unavailable'), ('BUN', 'clinical_BUN', 'retained clinical-feature blood urea nitrogen', 'timing not verified', 'mg/dL', 'NO', 'source timing cannot prove measurement by 12 h and MIMIC-III unavailable'), ('RRT', 'active_RRT_at_12h', 'active renal replacement therapy at landmark', 'active at 12 h', 'binary', 'YES', 'interval-timed and available in all databases; Model 2'), ('bilirubin', 'clinical_bilirubin', 'retained clinical-feature bilirubin', 'timing not verified', 'mg/dL', 'NO', 'source timing cannot prove measurement by 12 h and MIMIC-III unavailable'), ('platelet', 'clinical_platelet', 'retained clinical-feature platelet count', 'timing not verified', '10^9/L', 'NO', 'source timing cannot prove measurement by 12 h and MIMIC-III unavailable'), ('WBC', 'clinical_WBC', 'retained clinical-feature white blood cell count', 'timing not verified', '10^9/L', 'NO', 'source timing cannot prove measurement by 12 h and MIMIC-III unavailable'), ('hemoglobin', 'clinical_hemoglobin', 'retained clinical-feature hemoglobin', 'timing not verified', 'g/dL', 'NO', 'source timing cannot prove measurement by 12 h and MIMIC-III unavailable'), ('pH', 'pH_first_0_24h', 'first pH in retained exploratory extraction', '0-24 h', 'pH', 'NO', 'window extends beyond landmark and MIMIC-III unavailable'), ('bicarbonate', 'HCO3_first_0_24h', 'first bicarbonate in retained exploratory extraction', '0-24 h', 'mmol/L', 'NO', 'window extends beyond landmark and MIMIC-III unavailable'), ('temperature', 'TEMP_mean_0_12h', 'mean hourly temperature', '0-12 h', 'degrees C', 'NO', 'pre-landmark but about 70% missing in eICU; characterization only, excluded from core model'), ('APACHE_IVa_APS', 'acutephysiologyscore', 'APACHE IVa Acute Physiology Score', 'first 24 h', 'points', 'NO', 'eICU only; prohibited for 12-36 h primary outcome because of temporal overlap')])
    rows = []
    for variable, column, definition, time_window, unit, harmonizable, reason in metadata:
        row: dict[str, object] = {'variable': variable, 'definition': definition, 'time_window': time_window, 'unit': unit, 'harmonizable_yes_no': harmonizable, 'reason': reason}
        for database in DATABASES:
            subset = frame.loc[frame['database'].eq(database)]
            observed = int(subset[column].notna().sum()) if column and column in subset.columns else 0
            row[f"{database.replace('-', '_')}_N"] = observed
            row[f"{database.replace('-', '_')}_missing_pct"] = 100.0 * (len(subset) - observed) / len(subset)
        rows.append(row)
    return pd.DataFrame(rows)
