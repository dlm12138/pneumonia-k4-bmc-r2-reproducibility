"""Historical candidate identification; current strict gate MUST follow.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'\nScript 01: Pneumonia Cohort Identification\nIdentifies pneumonia ICU patients in MIMIC-IV, eICU, and MIMIC-III\n'
import sys, os, gzip, csv, json
from pathlib import Path
from datetime import datetime
RUN_DIR = Path(os.environ['SOURCE_1B251D2246'])
MIMICIV_RAW = RUN_DIR / '10_temp/mimiciv_raw'
MIMICIII_RAW = RUN_DIR / '10_temp/mimiciii_raw'
EICU_DIR = Path(os.environ['SOURCE_B41C1FC8CF'])
RAW_EXTRACT = RUN_DIR / '02_raw_extract'
INVENTORY = RUN_DIR / '01_inventory'
LOGS = RUN_DIR / '00_logs'
import pandas as pd
import numpy as np
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)
log('=' * 60)
log('PNEUMONIA IDENTIFICATION REPORT')
log(f'Started: {datetime.now()}')
log('=' * 60)
ICD9_PNEUMONIA = ['4800', '4801', '4802', '4803', '4808', '4809', '481', '4820', '4821', '4822', '4823', '4824', '4825', '48241', '48242', '48249', '4828', '48281', '48282', '48283', '48284', '48289', '4829', '4830', '4831', '4838', '4840', '4841', '4843', '4845', '4846', '4847', '4848', '485', '486', '4870', '4880', '48801', '48811', '5070', '99731']
ICD10_PNEUMONIA = ['J120', 'J121', 'J122', 'J123', 'J1281', 'J1289', 'J129', 'J13', 'J14', 'J150', 'J151', 'J1520', 'J1521', 'J1529', 'J153', 'J154', 'J155', 'J156', 'J157', 'J158', 'J159', 'J160', 'J168', 'J170', 'J180', 'J181', 'J182', 'J188', 'J189', 'J690', 'J95851', 'U071', 'J100', 'J1000', 'J1001', 'J1008', 'J110', 'J1100', 'J1108']
ICD9_PREFIXES = ['480', '481', '482', '483', '484', '485', '486', '487', '488', '507', '997']
ICD10_PREFIXES = ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J69', 'J95', 'U07', 'J10', 'J11']

def icd9_match(code):
    code = str(code).strip().replace('.', '')
    for exact in ICD9_PNEUMONIA:
        if code == exact:
            return True
    for prefix in ICD9_PREFIXES:
        if code.startswith(prefix):
            if prefix in ['480', '481', '482', '483', '484', '485', '486']:
                return True
            if prefix == '487' and len(code) >= 4 and (code[3] == '0'):
                return True
            if prefix == '488' and len(code) >= 4 and (code[3] in ['0', '1']):
                return True
            if prefix == '507' and len(code) >= 4 and (code[3] == '0'):
                return True
            if prefix == '997' and code.startswith('99731'):
                return True
    return False

def icd10_match(code):
    code = str(code).strip().upper()
    for exact in ICD10_PNEUMONIA:
        if code == exact:
            return True
    for prefix in ICD10_PREFIXES:
        if code.startswith(prefix):
            if prefix in ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18']:
                return True
            if prefix == 'J69' and code.startswith('J690'):
                return True
            if prefix == 'J95' and code.startswith('J95851'):
                return True
            if prefix == 'U07' and code.startswith('U071'):
                return True
            if prefix == 'J10' and (code.startswith('J100') or code.startswith('J101')):
                return True
            if prefix == 'J11' and code.startswith('J110'):
                return True
    return False
log('\n' + '=' * 50)
log('STEP 1: MIMIC-IV Pneumonia Identification')
log('=' * 50)
log('\nLoading MIMIC-IV diagnoses_icd...')
diag_iv = pd.read_csv(MIMICIV_RAW / 'diagnoses_icd.csv.gz')
log(f'  Total diagnosis rows: {len(diag_iv):,}')
log(f'  Columns: {list(diag_iv.columns)}')
d_icd_iv = pd.read_csv(MIMICIV_RAW / 'd_icd_diagnoses.csv.gz')
log(f'  ICD dictionary rows: {len(d_icd_iv):,}')
matched_icd9 = []
matched_icd10 = []
for _, row in diag_iv.iterrows():
    code = str(row['icd_code']).strip()
    version = str(row['icd_version']).strip()
    if version == '9':
        if icd9_match(code):
            matched_icd9.append(row)
    elif version == '10':
        if icd10_match(code):
            matched_icd10.append(row)
df_icd9 = pd.DataFrame(matched_icd9)
df_icd10 = pd.DataFrame(matched_icd10)
log(f"\nICD-9 matches: {len(df_icd9):,} rows, {(df_icd9['hadm_id'].nunique() if len(df_icd9) > 0 else 0):,} unique admissions")
log(f"ICD-10 matches: {len(df_icd10):,} rows, {(df_icd10['hadm_id'].nunique() if len(df_icd10) > 0 else 0):,} unique admissions")
pneu_hadm_iv = set()
if len(df_icd9) > 0:
    pneu_hadm_iv.update(df_icd9['hadm_id'].unique())
if len(df_icd10) > 0:
    pneu_hadm_iv.update(df_icd10['hadm_id'].unique())
log(f'Total unique pneumonia hadm_ids (MIMIC-IV): {len(pneu_hadm_iv):,}')
audit_codes_iv = []
if len(df_icd9) > 0:
    for code, cnt in df_icd9['icd_code'].value_counts().head(30).items():
        desc_rows = d_icd_iv[d_icd_iv['icd_code'] == code]
        desc = desc_rows['long_title'].iloc[0] if len(desc_rows) > 0 else 'Unknown'
        audit_codes_iv.append({'database': 'MIMIC-IV', 'icd_version': 9, 'icd_code': code, 'description': desc, 'n_admissions': cnt, 'source_field': 'diagnoses_icd.icd_code'})
if len(df_icd10) > 0:
    for code, cnt in df_icd10['icd_code'].value_counts().head(30).items():
        desc_rows = d_icd_iv[d_icd_iv['icd_code'] == code]
        desc = desc_rows['long_title'].iloc[0] if len(desc_rows) > 0 else 'Unknown'
        audit_codes_iv.append({'database': 'MIMIC-IV', 'icd_version': 10, 'icd_code': code, 'description': desc, 'n_admissions': cnt, 'source_field': 'diagnoses_icd.icd_code'})
log('\nLoading MIMIC-IV ICU stays...')
icu_iv = pd.read_csv(MIMICIV_RAW / 'icustays.csv.gz')
log(f"  ICU stays: {len(icu_iv):,}, unique subjects: {icu_iv['subject_id'].nunique():,}")
log(f'  Columns: {list(icu_iv.columns)}')
log('Loading patients and admissions...')
pat_iv = pd.read_csv(MIMICIV_RAW / 'patients.csv.gz')
adm_iv = pd.read_csv(MIMICIV_RAW / 'admissions.csv.gz')
log(f'  Patients: {len(pat_iv):,}')
log(f'  Admissions: {len(adm_iv):,}')
icu_pneu_iv = icu_iv[icu_iv['hadm_id'].isin(pneu_hadm_iv)].copy()
log(f'\nICU stays with pneumonia diagnosis: {len(icu_pneu_iv):,}')
log(f"Unique patients with pneumonia ICU stays: {icu_pneu_iv['subject_id'].nunique():,}")
icu_pneu_iv = icu_pneu_iv.merge(pat_iv[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']], on='subject_id', how='left')
icu_pneu_iv = icu_pneu_iv.merge(adm_iv[['hadm_id', 'admittime', 'dischtime', 'deathtime', 'race']], on='hadm_id', how='left')
for col in ['intime', 'outtime', 'admittime', 'dischtime', 'deathtime']:
    if col in icu_pneu_iv.columns:
        icu_pneu_iv[col] = pd.to_datetime(icu_pneu_iv[col], errors='coerce')
icu_pneu_iv['icu_year'] = icu_pneu_iv['intime'].dt.year
icu_pneu_iv['age'] = icu_pneu_iv['anchor_age'] + (icu_pneu_iv['icu_year'] - icu_pneu_iv['anchor_year'])
icu_pneu_iv['los_hours'] = (icu_pneu_iv['outtime'] - icu_pneu_iv['intime']).dt.total_seconds() / 3600
icu_pneu_iv['dod'] = pd.to_datetime(icu_pneu_iv['dod'], errors='coerce')
icu_pneu_iv['deathtime'] = pd.to_datetime(icu_pneu_iv['deathtime'], errors='coerce')

def get_death_days(row):
    ref = row['intime']
    if pd.isna(ref):
        return np.nan
    candidates = []
    if not pd.isna(row['dod']):
        candidates.append((row['dod'] - ref).total_seconds() / 86400)
    if not pd.isna(row['deathtime']):
        candidates.append((row['deathtime'] - ref).total_seconds() / 86400)
    return min(candidates) if candidates else np.nan
icu_pneu_iv['days_to_death'] = icu_pneu_iv.apply(get_death_days, axis=1)
icu_pneu_iv['mortality_30d'] = ((icu_pneu_iv['days_to_death'] >= 0) & (icu_pneu_iv['days_to_death'] <= 30)).astype(int)
icu_pneu_iv['mortality_30d_known'] = icu_pneu_iv['days_to_death'].notna() | True
icu_pneu_iv['icu_mortality'] = 0
icu_pneu_iv.loc[icu_pneu_iv['deathtime'].notna() & (icu_pneu_iv['deathtime'] >= icu_pneu_iv['intime']) & (icu_pneu_iv['deathtime'] <= icu_pneu_iv['outtime']), 'icu_mortality'] = 1
icu_pneu_iv_sorted = icu_pneu_iv.sort_values(['subject_id', 'intime'])
icu_pneu_iv_first = icu_pneu_iv_sorted.drop_duplicates('subject_id', keep='first')
log(f'\nAfter keeping first ICU stay per patient: {len(icu_pneu_iv_first):,}')
log(f"Age distribution: median={icu_pneu_iv_first['age'].median():.1f}, range={icu_pneu_iv_first['age'].min():.0f}-{icu_pneu_iv_first['age'].max():.0f}")
log(f"ICU LOS distribution: median={icu_pneu_iv_first['los_hours'].median():.1f}h")
log(f"30-day mortality: {icu_pneu_iv_first['mortality_30d'].mean() * 100:.1f}%")
icu_pneu_iv_first.to_csv(RAW_EXTRACT / 'mimiciv_pneumonia_stays_raw.csv', index=False)
log(f'\nSaved: mimiciv_pneumonia_stays_raw.csv ({len(icu_pneu_iv_first):,} rows)')
log('\n' + '=' * 50)
log('STEP 2: MIMIC-III Pneumonia Identification')
log('=' * 50)
diag_iii = pd.read_csv(MIMICIII_RAW / 'DIAGNOSES_ICD.csv.gz')
log(f'  Total diagnosis rows: {len(diag_iii):,}')
log(f'  Columns: {list(diag_iii.columns)}')
d_icd_iii = pd.read_csv(MIMICIII_RAW / 'D_ICD_DIAGNOSES.csv.gz')
matched_iii = []
for _, row in diag_iii.iterrows():
    code = str(row['ICD9_CODE']).strip()
    if icd9_match(code):
        matched_iii.append(row)
df_pneu_iii = pd.DataFrame(matched_iii)
log(f'ICD-9 pneumonia matches: {len(df_pneu_iii):,} rows')
pneu_hadm_iii = set(df_pneu_iii['HADM_ID'].unique()) if len(df_pneu_iii) > 0 else set()
log(f'Unique hadm_ids: {len(pneu_hadm_iii):,}')
audit_codes_iii = []
if len(df_pneu_iii) > 0:
    for code, cnt in df_pneu_iii['ICD9_CODE'].value_counts().head(30).items():
        desc_rows = d_icd_iii[d_icd_iii['ICD9_CODE'] == code]
        desc = desc_rows['LONG_TITLE'].iloc[0] if len(desc_rows) > 0 else 'Unknown'
        audit_codes_iii.append({'database': 'MIMIC-III', 'icd_version': 9, 'icd_code': code, 'description': desc, 'n_admissions': cnt, 'source_field': 'DIAGNOSES_ICD.ICD9_CODE'})
icu_iii = pd.read_csv(MIMICIII_RAW / 'ICUSTAYS.csv.gz')
pat_iii = pd.read_csv(MIMICIII_RAW / 'PATIENTS.csv.gz')
adm_iii = pd.read_csv(MIMICIII_RAW / 'ADMISSIONS.csv.gz')
log(f"  ICU stays: {len(icu_iii):,}, unique subjects: {icu_iii['SUBJECT_ID'].nunique():,}")
icu_pneu_iii = icu_iii[icu_iii['HADM_ID'].isin(pneu_hadm_iii)].copy()
log(f'ICU stays with pneumonia: {len(icu_pneu_iii):,}')
icu_pneu_iii = icu_pneu_iii.merge(pat_iii[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN']], on='SUBJECT_ID', how='left')
icu_pneu_iii = icu_pneu_iii.merge(adm_iii[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY']], on='HADM_ID', how='left')
for col in ['INTIME', 'OUTTIME', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME']:
    if col in icu_pneu_iii.columns:
        icu_pneu_iii[col] = pd.to_datetime(icu_pneu_iii[col], errors='coerce')

def calc_age_iii(row):
    try:
        if pd.isna(row['DOB']) or pd.isna(row['INTIME']):
            return np.nan
        dob = row['DOB']
        intime = row['INTIME']
        year_diff = intime.year - dob.year
        if year_diff > 120:
            return 91.4
        age = (intime - dob).total_seconds() / (365.25 * 24 * 3600)
        return max(0, min(age, 120))
    except:
        return np.nan
icu_pneu_iii['age'] = icu_pneu_iii.apply(calc_age_iii, axis=1)
icu_pneu_iii['los_hours'] = (icu_pneu_iii['OUTTIME'] - icu_pneu_iii['INTIME']).dt.total_seconds() / 3600

def get_dod_iii(row):
    candidates = []
    for col in ['DOD', 'DOD_HOSP', 'DOD_SSN']:
        if not pd.isna(row.get(col)):
            candidates.append(row[col])
    if not candidates:
        return pd.NaT
    return min(candidates)
icu_pneu_iii['dod_best'] = icu_pneu_iii.apply(get_dod_iii, axis=1)
icu_pneu_iii['days_to_death'] = (icu_pneu_iii['dod_best'] - icu_pneu_iii['INTIME']).dt.total_seconds() / 86400
icu_pneu_iii['mortality_30d'] = ((icu_pneu_iii['days_to_death'] >= 0) & (icu_pneu_iii['days_to_death'] <= 30)).astype(int)
icu_pneu_iii['icu_mortality'] = 0
icu_pneu_iii.loc[icu_pneu_iii['DEATHTIME'].notna() & (icu_pneu_iii['DEATHTIME'] >= icu_pneu_iii['INTIME']) & (icu_pneu_iii['DEATHTIME'] <= icu_pneu_iii['OUTTIME']), 'icu_mortality'] = 1
icu_pneu_iii_sorted = icu_pneu_iii.sort_values(['SUBJECT_ID', 'INTIME'])
icu_pneu_iii_first = icu_pneu_iii_sorted.drop_duplicates('SUBJECT_ID', keep='first')
log(f'After first ICU stay: {len(icu_pneu_iii_first):,}')
log(f"30-day mortality: {icu_pneu_iii_first['mortality_30d'].mean() * 100:.1f}%")
icu_pneu_iii_first.to_csv(RAW_EXTRACT / 'mimiciii_pneumonia_stays_raw.csv', index=False)
log(f'Saved: mimiciii_pneumonia_stays_raw.csv')
log('\n' + '=' * 50)
log('STEP 3: eICU Pneumonia Identification')
log('=' * 50)
pat_eicu = pd.read_csv(EICU_DIR / 'patient.csv.gz')
log(f"  eICU patients: {len(pat_eicu):,}, unique hospitalizations: {pat_eicu['patienthealthsystemstayid'].nunique():,}")
log(f'  Columns: {list(pat_eicu.columns[:10])}')
diag_eicu = pd.read_csv(EICU_DIR / 'diagnosis.csv.gz')
log(f'  diagnosis.csv.gz: {len(diag_eicu):,} rows')
log(f'  Columns: {list(diag_eicu.columns)}')
admdx_eicu = pd.read_csv(EICU_DIR / 'admissionDx.csv.gz')
log(f'  admissionDx.csv.gz: {len(admdx_eicu):,} rows')
log(f'  Columns: {list(admdx_eicu.columns)}')
PNEUMONIA_KEYWORDS = ['pneumonia', 'community acquired pneumonia', 'hospital acquired pneumonia', 'healthcare associated pneumonia', 'ventilator associated pneumonia', 'aspiration pneumonia', 'bacterial pneumonia', 'viral pneumonia', 'influenza pneumonia', 'covid pneumonia', 'lower respiratory tract infection', 'bronchopneumonia', 'cap', 'hap', 'vap', 'lrti']

def keyword_match_pneumonia(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    for kw in PNEUMONIA_KEYWORDS:
        if kw in text_lower:
            return True
    return False
log('\nSearching diagnosis.diagnosisstring...')
diag_eicu_pneu_mask = diag_eicu['diagnosisstring'].apply(keyword_match_pneumonia)
diag_eicu_pneu = diag_eicu[diag_eicu_pneu_mask].copy()
log(f'  Matches in diagnosis.diagnosisstring: {len(diag_eicu_pneu):,}')
log('Searching admissionDx...')
admdx_cols = [c for c in admdx_eicu.columns if 'dx' in c.lower() or 'diagnosis' in c.lower() or 'admit' in c.lower()]
log(f'  admissionDx columns to search: {admdx_cols}')
admdx_pneu_ids = set()
for col in admdx_cols:
    mask = admdx_eicu[col].apply(keyword_match_pneumonia)
    matched = admdx_eicu[mask]
    if len(matched) > 0:
        admdx_pneu_ids.update(matched['patientunitstayid'].unique())
        log(f'  Column {col}: {len(matched):,} rows')
log(f'  Total unique patientunitstayid from admissionDx: {len(admdx_pneu_ids):,}')
diag_pneu_ids = set(diag_eicu_pneu['patientunitstayid'].unique())
all_pneu_ids_eicu = diag_pneu_ids | admdx_pneu_ids
log(f'\nTotal unique pneumonia patientunitstayids: {len(all_pneu_ids_eicu):,}')
audit_strings_eicu = []
if len(diag_eicu_pneu) > 0:
    for txt, cnt in diag_eicu_pneu['diagnosisstring'].value_counts().head(50).items():
        audit_strings_eicu.append({'database': 'eICU', 'field': 'diagnosis.diagnosisstring', 'matched_text': txt, 'n_stays': cnt})
eicu_pneu = pat_eicu[pat_eicu['patientunitstayid'].isin(all_pneu_ids_eicu)].copy()
log(f'eICU patient rows with pneumonia: {len(eicu_pneu):,}')
eicu_pneu['age_num'] = pd.to_numeric(eicu_pneu['age'].replace('> 89', '90'), errors='coerce')
eicu_pneu['los_hours'] = eicu_pneu['hospitaldischargeoffset'] / 60
log(f"  Unique unitdischargestatus: {eicu_pneu['unitdischargestatus'].unique()[:10]}")
log(f"  Unique hospitaldischargestatus: {eicu_pneu['hospitaldischargestatus'].unique()[:10]}")
eicu_pneu['icu_mortality'] = (eicu_pneu['unitdischargestatus'] == 'Expired').astype(int)
eicu_pneu['hospital_mortality'] = (eicu_pneu['hospitaldischargestatus'] == 'Expired').astype(int)
eicu_pneu['mortality_30d'] = np.nan
log('  NOTE: eICU 30-day mortality not available; will use ICU/hospital mortality')
eicu_pneu_sorted = eicu_pneu.sort_values(['patienthealthsystemstayid', 'unitvisitnumber'])
eicu_pneu_first = eicu_pneu_sorted.drop_duplicates('patienthealthsystemstayid', keep='first')
log(f'After first ICU stay per patient: {len(eicu_pneu_first):,}')
log(f"ICU mortality: {eicu_pneu_first['icu_mortality'].mean() * 100:.1f}%")
log(f"Hospital mortality: {eicu_pneu_first['hospital_mortality'].mean() * 100:.1f}%")
eicu_pneu_first.to_csv(RAW_EXTRACT / 'eicu_pneumonia_stays_raw.csv', index=False)
log(f'Saved: eicu_pneumonia_stays_raw.csv')
log('\n' + '=' * 50)
log('STEP 4: Writing Audit Tables')
log('=' * 50)
all_audit = audit_codes_iv + audit_codes_iii
audit_df = pd.DataFrame(all_audit)
if len(audit_strings_eicu) > 0:
    eicu_audit_df = pd.DataFrame(audit_strings_eicu)
    eicu_audit_df['icd_version'] = 'NA'
    eicu_audit_df['icd_code'] = 'keyword'
    eicu_audit_df['description'] = eicu_audit_df['matched_text']
    eicu_audit_df['n_admissions'] = eicu_audit_df['n_stays']
    eicu_audit_df['source_field'] = eicu_audit_df['field']
    audit_df = pd.concat([audit_df, eicu_audit_df[['database', 'icd_version', 'icd_code', 'description', 'n_admissions', 'source_field']]], ignore_index=True)
audit_df.to_csv(INVENTORY / 'pneumonia_case_definition_terms.csv', index=False)
log(f'Saved: pneumonia_case_definition_terms.csv ({len(audit_df):,} rows)')
counts_df = pd.DataFrame([{'database': 'MIMIC-IV', 'initial_pneumonia_stays': len(icu_pneu_iv), 'first_icu_stay': len(icu_pneu_iv_first), 'notes': 'ICD-9+ICD-10 from diagnoses_icd'}, {'database': 'MIMIC-III', 'initial_pneumonia_stays': len(icu_pneu_iii), 'first_icu_stay': len(icu_pneu_iii_first), 'notes': 'ICD-9 from DIAGNOSES_ICD'}, {'database': 'eICU', 'initial_pneumonia_stays': len(eicu_pneu), 'first_icu_stay': len(eicu_pneu_first), 'notes': 'keyword match from diagnosis.diagnosisstring + admissionDx'}])
counts_df.to_csv(INVENTORY / 'pneumonia_case_counts_by_database.csv', index=False)
log('Saved: pneumonia_case_counts_by_database.csv')
log('\n' + '=' * 50)
log('PNEUMONIA IDENTIFICATION SUMMARY')
log('=' * 50)
for _, row in counts_df.iterrows():
    log(f"  {row['database']}: {row['initial_pneumonia_stays']:,} ICU stays → {row['first_icu_stay']:,} first stays")
with open(LOGS / '03_pneumonia_identification_report.md', 'w') as f:
    f.write('# Pneumonia Identification Report\n\n')
    f.write('\n'.join(log_lines))
print('\n=== Script 01 COMPLETE ===')
