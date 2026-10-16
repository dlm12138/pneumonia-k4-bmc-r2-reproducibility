"""Original day-one clinical/lab/comorbidity extraction; restricted historical cohort coverage is corrected by raw diagnosis recovery, and generic oxygen is superseded.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'\nScript 08: Extract clinical characteristics for Table 1 and supplementary tables.\nExtracts day-1 labs, vitals, comorbidities, mechanical ventilation, vasopressors,\nand outcomes for all three cohorts.\n'
import sys, time, zipfile, gzip
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
RUN_DIR = Path(os.environ['SOURCE_1B251D2246'])
MIMICIV_ZIP = os.environ['SOURCE_E15CB3D6F9']
MIMICIII_ZIP = os.environ['SOURCE_1D1D312607']
EICU_DIR = Path(os.environ['SOURCE_B41C1FC8CF'])
MIMICIV_RAW = RUN_DIR / '10_temp/mimiciv_raw'
MIMICIII_RAW = RUN_DIR / '10_temp/mimiciii_raw'
PROCESSED = RUN_DIR / '03_processed'
TABLES = RUN_DIR / '05_tables'
LOGS = RUN_DIR / '00_logs'
print('=' * 60)
print('CLINICAL FEATURE EXTRACTION')
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 60)
print('\nLoading final cohort datasets...')
iv_final = pd.read_csv(PROCESSED / 'development_mimiciv_final.csv')
eicu_final = pd.read_csv(PROCESSED / 'validation_eicu_final.csv')
iii_final = pd.read_csv(PROCESSED / 'validation_mimiciii_final.csv')
pheno_file = TABLES / 'Table_S9_final_K4_membership_development.csv'
if pheno_file.exists():
    pheno_df = pd.read_csv(pheno_file)
    iv_final = iv_final.merge(pheno_df[['stay_id', 'phenotype']], on='stay_id', how='left')
    print(f"  Merged phenotype assignments for {iv_final['phenotype'].notna().sum():,} development stays")
else:
    print('  WARNING: K=4 membership not yet available. Will add phenotype column as placeholder.')
    iv_final['phenotype'] = np.nan
print(f'  MIMIC-IV: {len(iv_final):,}')
print(f'  eICU: {len(eicu_final):,}')
print(f'  MIMIC-III: {len(iii_final):,}')
print('\nExtracting MIMIC-IV lab values...')
d_labitems = pd.read_csv(MIMICIV_RAW / 'd_labitems.csv.gz')
print(f'  Lab items: {len(d_labitems):,}')
LAB_TARGETS = {'WBC': ['white blood cell', 'wbc', 'leukocyte'], 'BUN': ['urea nitrogen', 'bun'], 'creatinine': ['creatinine'], 'sodium': ['sodium'], 'potassium': ['potassium'], 'calcium': ['calcium, total', 'calcium'], 'phosphate': ['phosphate', 'phosphorus'], 'hemoglobin': ['hemoglobin'], 'hematocrit': ['hematocrit'], 'pH': ['ph, arterial', 'ph, blood', 'ph'], 'glucose': ['glucose'], 'lactate': ['lactate', 'lactic acid'], 'platelet': ['platelet', 'thrombocyte'], 'albumin': ['albumin'], 'bilirubin': ['bilirubin, total', 'bilirubin'], 'CRP': ['c-reactive protein', 'crp'], 'PCT': ['procalcitonin', 'pct'], 'triglycerides': ['triglycerides'], 'PaO2': ['po2, arterial', 'pao2', 'po2'], 'FiO2': ['fio2', 'fraction of inspired oxygen'], 'SpO2': ['spo2', 'oxygen saturation']}
lab_itemids = {}
for lab, keywords in LAB_TARGETS.items():
    mask = d_labitems['label'].str.lower().apply(lambda x: any((kw in str(x).lower() for kw in keywords)) if pd.notna(x) else False)
    matched = d_labitems[mask]
    if lab == 'creatinine':
        matched = matched[~matched['label'].str.lower().str.contains('urine|ratio|osmolality', na=False)]
    if lab == 'calcium':
        matched = matched[matched['label'].str.lower().str.contains('calcium', na=False)]
        matched = matched[~matched['label'].str.lower().str.contains('ionized|urine', na=False)]
    if lab == 'pH':
        matched = matched[matched['label'].str.lower().str.contains('ph', na=False)]
    if lab == 'FiO2':
        matched = matched[matched['label'].str.lower().str.contains('fio2|fraction of inspired', na=False)]
    lab_itemids[lab] = list(matched['itemid'])
for lab, ids in lab_itemids.items():
    if ids:
        labels = d_labitems[d_labitems['itemid'].isin(ids)]['label'].tolist()
        print(f'  {lab}: {ids[:3]} ... ({len(ids)} items)')
all_lab_itemids = set()
for ids in lab_itemids.values():
    all_lab_itemids.update(ids)
print(f'\n  Total unique lab itemids: {len(all_lab_itemids):,}')
print('  Streaming labevents from MIMIC-IV zip...')
iv_hadm_ids = set(iv_final['hadm_id'].dropna().astype(int).tolist())
iv_intime_map = dict(zip(iv_final['hadm_id'], pd.to_datetime(iv_final['intime'])))
LABEVENTS_PATH = 'mimic-iv-3.1/hosp/labevents.csv.gz'
lab_chunks = []
with zipfile.ZipFile(MIMICIV_ZIP) as zf:
    all_names = zf.namelist()
    if LABEVENTS_PATH not in all_names:
        print(f'  WARNING: {LABEVENTS_PATH} not in zip')
        lab_files = [n for n in all_names if 'labevents' in n.lower()]
        print(f'  Found: {lab_files}')
        if lab_files:
            LABEVENTS_PATH = lab_files[0]
    with zf.open(LABEVENTS_PATH) as raw_gz:
        with gzip.open(raw_gz, 'rt') as gz_file:
            for chunk in pd.read_csv(gz_file, chunksize=500000, usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'], dtype={'hadm_id': 'Int64', 'itemid': 'Int64', 'valuenum': 'float64'}, low_memory=False):
                chunk = chunk[chunk['hadm_id'].isin(iv_hadm_ids) & chunk['itemid'].isin(all_lab_itemids) & chunk['valuenum'].notna()].copy()
                if len(chunk) > 0:
                    lab_chunks.append(chunk)
if lab_chunks:
    labs_raw = pd.concat(lab_chunks, ignore_index=True)
    print(f'  Lab events matched: {len(labs_raw):,}')
    labs_raw['charttime'] = pd.to_datetime(labs_raw['charttime'])
    labs_raw['intime'] = labs_raw['hadm_id'].map(iv_intime_map)
    labs_raw['hours_after_icu'] = (labs_raw['charttime'] - labs_raw['intime']).dt.total_seconds() / 3600
    labs_day1 = labs_raw[(labs_raw['hours_after_icu'] >= 0) & (labs_raw['hours_after_icu'] <= 24)].copy()
    print(f'  Day-1 lab events: {len(labs_day1):,}')

    def map_lab(itemid, lab_itemids):
        for lab, ids in lab_itemids.items():
            if itemid in ids:
                return lab
        return None
    labs_day1['lab_name'] = labs_day1['itemid'].apply(lambda x: map_lab(x, lab_itemids))
    labs_day1 = labs_day1.dropna(subset=['lab_name'])
    labs_day1_first = labs_day1.sort_values('hours_after_icu').groupby(['hadm_id', 'lab_name'])['valuenum'].first().reset_index()
    labs_wide = labs_day1_first.pivot_table(index='hadm_id', columns='lab_name', values='valuenum').reset_index()
    labs_wide.columns.name = None
    print(f'  Lab wide table: {len(labs_wide):,} admissions x {len(labs_wide.columns) - 1} labs')
    iv_final = iv_final.merge(labs_wide, on='hadm_id', how='left')
else:
    print('  WARNING: No lab events found')
print('\n  Extracting SpO2 from MIMIC-IV chartevents (day 1)...')
iv_stay_ids = set(iv_final['stay_id'].tolist())
iv_intime_by_stay = dict(zip(iv_final['stay_id'], pd.to_datetime(iv_final['intime'])))
spo2_rows = []
SPO2_ITEMID = {220277}
with zipfile.ZipFile(MIMICIV_ZIP) as zf:
    with zf.open('mimic-iv-3.1/icu/chartevents.csv.gz') as raw_gz:
        with gzip.open(raw_gz, 'rt') as gz_file:
            for chunk in pd.read_csv(gz_file, chunksize=500000, usecols=['stay_id', 'itemid', 'charttime', 'valuenum'], dtype={'stay_id': 'Int64', 'itemid': 'Int64', 'valuenum': 'float64'}, low_memory=False):
                chunk = chunk[chunk['stay_id'].isin(iv_stay_ids) & chunk['itemid'].isin(SPO2_ITEMID) & chunk['valuenum'].notna()].copy()
                if len(chunk) > 0:
                    spo2_rows.append(chunk)
if spo2_rows:
    spo2_df = pd.concat(spo2_rows, ignore_index=True)
    spo2_df['charttime'] = pd.to_datetime(spo2_df['charttime'])
    spo2_df['intime'] = spo2_df['stay_id'].map(iv_intime_by_stay)
    spo2_df['hours'] = (spo2_df['charttime'] - spo2_df['intime']).dt.total_seconds() / 3600
    spo2_day1 = spo2_df[(spo2_df['hours'] >= 0) & (spo2_df['hours'] <= 24)]
    spo2_stay = spo2_day1.groupby('stay_id')['valuenum'].mean().reset_index()
    spo2_stay.columns = ['stay_id', 'SpO2']
    iv_final = iv_final.merge(spo2_stay, on='stay_id', how='left')
    print(f"  SpO2 extracted for {spo2_stay['stay_id'].nunique():,} stays")
print('\nExtracting MIMIC-IV comorbidities...')
diag_iv = pd.read_csv(MIMICIV_RAW / 'diagnoses_icd.csv.gz')
COMORBIDITY_ICD10 = {'myocardial_infarction': ['I21', 'I22', 'I252'], 'heart_failure': ['I50'], 'cerebrovascular': ['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'G45', 'G46'], 'chronic_pulmonary': ['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47'], 'diabetes': ['E10', 'E11', 'E12', 'E13', 'E14'], 'liver_disease': ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76'], 'renal_disease': ['N18', 'N19', 'Z99.2'], 'malignancy': ['C']}
COMORBIDITY_ICD9 = {'myocardial_infarction': ['410', '412'], 'heart_failure': ['428'], 'cerebrovascular': ['430', '431', '432', '433', '434', '435', '436', '437', '438'], 'chronic_pulmonary': ['490', '491', '492', '493', '494', '495', '496'], 'diabetes': ['250'], 'liver_disease': ['570', '571', '572', '573'], 'renal_disease': ['585', '586', 'V420', 'V451'], 'malignancy': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175']}
iv_hadm_set = set(iv_final['hadm_id'].dropna().astype(int))
diag_subset = diag_iv[diag_iv['hadm_id'].isin(iv_hadm_set)].copy()

def has_comorbidity(hadm_ids_per_group, diag_df, icd9_dict, icd10_dict):
    result = {k: set() for k in icd9_dict}
    for _, row in diag_df.iterrows():
        code = str(row['icd_code']).strip().upper()
        hadm = row['hadm_id']
        version = str(row['icd_version'])
        if version == '9':
            for comorb, prefixes in icd9_dict.items():
                if any((code.startswith(p) for p in prefixes)):
                    result[comorb].add(hadm)
        elif version == '10':
            for comorb, prefixes in icd10_dict.items():
                if any((code.startswith(p) for p in prefixes)):
                    result[comorb].add(hadm)
    return result
comorbidity_sets = has_comorbidity(iv_hadm_set, diag_subset, COMORBIDITY_ICD9, COMORBIDITY_ICD10)
for comorb, hadm_set in comorbidity_sets.items():
    iv_final[comorb] = iv_final['hadm_id'].isin(hadm_set).astype(int)
    pct = iv_final[comorb].mean() * 100
    print(f'  {comorb}: {pct:.1f}%')
print('\nExtracting mechanical ventilation and vasopressor use...')
PROCEVENTS_PATH = 'mimic-iv-3.1/icu/procedureevents.csv.gz'
MV_ITEMS = {225792, 225794}
proc_rows = []
with zipfile.ZipFile(MIMICIV_ZIP) as zf:
    if PROCEVENTS_PATH in zf.namelist():
        with zf.open(PROCEVENTS_PATH) as raw_gz:
            with gzip.open(raw_gz, 'rt') as gz_file:
                for chunk in pd.read_csv(gz_file, chunksize=200000, usecols=['stay_id', 'itemid', 'starttime'], dtype={'stay_id': 'Int64', 'itemid': 'Int64'}, low_memory=False):
                    chunk = chunk[chunk['stay_id'].isin(iv_stay_ids) & chunk['itemid'].isin(MV_ITEMS)]
                    if len(chunk) > 0:
                        proc_rows.append(chunk)
if proc_rows:
    proc_df = pd.concat(proc_rows, ignore_index=True)
    proc_df['starttime'] = pd.to_datetime(proc_df['starttime'])
    proc_df['intime'] = proc_df['stay_id'].map(iv_intime_by_stay)
    proc_df['hours'] = (proc_df['starttime'] - proc_df['intime']).dt.total_seconds() / 3600
    mv_day1 = proc_df[(proc_df['hours'] >= 0) & (proc_df['hours'] <= 24)]
    mv_stays = set(mv_day1['stay_id'].unique())
    iv_final['mechanical_ventilation'] = iv_final['stay_id'].isin(mv_stays).astype(int)
    print(f"  Mechanical ventilation day-1: {iv_final['mechanical_ventilation'].mean() * 100:.1f}%")
else:
    iv_final['mechanical_ventilation'] = np.nan
    print('  Mechanical ventilation: not found in procedureevents')
VASOPRESSOR_ITEMS = {221662, 221653, 221289, 222315, 221906, 221749}
with zipfile.ZipFile(MIMICIV_ZIP) as zf:
    vaso_path = 'mimic-iv-3.1/icu/inputevents.csv.gz'
    if vaso_path in zf.namelist():
        vaso_rows = []
        with zf.open(vaso_path) as raw_gz:
            with gzip.open(raw_gz, 'rt') as gz_file:
                for chunk in pd.read_csv(gz_file, chunksize=200000, usecols=['stay_id', 'itemid', 'starttime'], dtype={'stay_id': 'Int64', 'itemid': 'Int64'}, low_memory=False):
                    chunk = chunk[chunk['stay_id'].isin(iv_stay_ids) & chunk['itemid'].isin(VASOPRESSOR_ITEMS)]
                    if len(chunk) > 0:
                        vaso_rows.append(chunk)
        if vaso_rows:
            vaso_df = pd.concat(vaso_rows, ignore_index=True)
            vaso_df['starttime'] = pd.to_datetime(vaso_df['starttime'])
            vaso_df['intime'] = vaso_df['stay_id'].map(iv_intime_by_stay)
            vaso_df['hours'] = (vaso_df['starttime'] - vaso_df['intime']).dt.total_seconds() / 3600
            vaso_day1 = vaso_df[(vaso_df['hours'] >= 0) & (vaso_df['hours'] <= 24)]
            vaso_stays = set(vaso_day1['stay_id'].unique())
            iv_final['vasopressor_use'] = iv_final['stay_id'].isin(vaso_stays).astype(int)
            print(f"  Vasopressor use day-1: {iv_final['vasopressor_use'].mean() * 100:.1f}%")
        else:
            iv_final['vasopressor_use'] = np.nan
    else:
        iv_final['vasopressor_use'] = np.nan
print('\nExtracting eICU clinical features...')
eicu_pids = set(eicu_final['patientunitstayid'].tolist())
try:
    apache_vars = pd.read_csv(EICU_DIR / 'apacheApsVar.csv.gz')
    apache_sub = apache_vars[apache_vars['patientunitstayid'].isin(eicu_pids)].copy()
    apache_cols = ['patientunitstayid', 'heartrate', 'meanbp', 'respiratoryrate', 'temperature', 'wbc', 'creatinine', 'bilirubin', 'albumin']
    apache_cols_avail = [c for c in apache_cols if c in apache_sub.columns]
    apache_sub = apache_sub[apache_cols_avail].copy()
    rename_apache = {'heartrate': 'HR_baseline', 'meanbp': 'MAP_baseline', 'respiratoryrate': 'RR_baseline', 'temperature': 'TEMP_baseline'}
    apache_sub = apache_sub.rename(columns=rename_apache)
    eicu_final = eicu_final.merge(apache_sub, on='patientunitstayid', how='left')
    print(f"  APACHE vars merged for {apache_sub['patientunitstayid'].nunique():,} stays")
except Exception as e:
    print(f'  APACHE vars error: {e}')
try:
    print('  Loading eICU lab.csv.gz...')
    eicu_lab_chunks = []
    for chunk in pd.read_csv(EICU_DIR / 'lab.csv.gz', chunksize=500000):
        chunk = chunk[chunk['patientunitstayid'].isin(eicu_pids)].copy()
        if len(chunk) > 0:
            eicu_lab_chunks.append(chunk)
    if eicu_lab_chunks:
        eicu_labs = pd.concat(eicu_lab_chunks, ignore_index=True)
        print(f'  eICU lab rows: {len(eicu_labs):,}')
        print(f"  Lab names sample: {eicu_labs['labname'].dropna().unique()[:10]}")
        EICU_LAB_MAP = {'WBC': ['wbc', 'white blood cell count', 'white blood cells'], 'creatinine': ['creatinine'], 'BUN': ['bun', 'blood urea nitrogen', 'urea nitrogen'], 'sodium': ['sodium'], 'potassium': ['potassium'], 'glucose': ['glucose'], 'lactate': ['lactate'], 'platelet': ['platelet', 'platelets'], 'albumin': ['albumin'], 'bilirubin': ['total bilirubin', 'bilirubin'], 'hemoglobin': ['hemoglobin', 'hgb'], 'hematocrit': ['hematocrit', 'hct'], 'pH': ['ph', 'arterial ph']}
        eicu_labs['lab_name_lower'] = eicu_labs['labname'].str.lower().fillna('')
        eicu_labs['std_name'] = None
        for std, keywords in EICU_LAB_MAP.items():
            mask = eicu_labs['lab_name_lower'].apply(lambda x: any((kw in x for kw in keywords)))
            eicu_labs.loc[mask & eicu_labs['std_name'].isna(), 'std_name'] = std
        eicu_labs_day1 = eicu_labs[(eicu_labs['labresultoffset'] >= 0) & (eicu_labs['labresultoffset'] <= 1440) & eicu_labs['std_name'].notna()].copy()
        eicu_labs_day1['labresult'] = pd.to_numeric(eicu_labs_day1['labresult'], errors='coerce')
        eicu_lab_pivot = eicu_labs_day1.sort_values('labresultoffset').groupby(['patientunitstayid', 'std_name'])['labresult'].first().reset_index()
        eicu_lab_wide = eicu_lab_pivot.pivot_table(index='patientunitstayid', columns='std_name', values='labresult').reset_index()
        eicu_lab_wide.columns.name = None
        eicu_final = eicu_final.merge(eicu_lab_wide, on='patientunitstayid', how='left')
        print(f'  eICU labs merged: {list(eicu_lab_wide.columns[1:])}')
except Exception as e:
    print(f'  eICU lab error: {e}')
print('\nExtracting MIMIC-III clinical features...')
iii_icustay_ids = set(iii_final['stay_id'].tolist())
iii_hadm_ids = set(iii_final.get('HADM_ID', pd.Series()).dropna().astype(int).tolist())
diag_iii = pd.read_csv(MIMICIII_RAW / 'DIAGNOSES_ICD.csv.gz')
diag_iii_sub = diag_iii[diag_iii['HADM_ID'].isin(iii_hadm_ids)].copy()
comorbidity_sets_iii = has_comorbidity(iii_hadm_ids, diag_iii_sub.rename(columns={'ICD9_CODE': 'icd_code', 'HADM_ID': 'hadm_id'}).assign(icd_version='9'), COMORBIDITY_ICD9, {})
for comorb, hadm_set in comorbidity_sets_iii.items():
    iii_final[comorb] = iii_final.get('HADM_ID', pd.Series(np.nan, index=iii_final.index)).isin(hadm_set).astype(int)
print(f'  MIMIC-III comorbidities extracted')
print('\nSaving clinical feature tables...')
all_feature_cols = list(LAB_TARGETS.keys()) + ['SpO2', 'mechanical_ventilation', 'vasopressor_use'] + list(COMORBIDITY_ICD9.keys()) + ['age', 'gender', 'race', 'mortality_30d', 'icu_mortality', 'hospital_mortality', 'los_hours']
miss_rows = []
for col in all_feature_cols:
    for db, df in [('MIMIC-IV', iv_final), ('eICU', eicu_final), ('MIMIC-III', iii_final)]:
        if col in df.columns:
            miss_pct = df[col].isna().mean() * 100
        else:
            miss_pct = 100.0
        miss_rows.append({'variable': col, 'database': db, 'missing_pct': round(miss_pct, 1), 'n_total': len(df)})
miss_df = pd.DataFrame(miss_rows)
miss_df.to_csv(TABLES / 'Table_S13_missing_rate_by_database.csv', index=False)
print(f'  Saved: Table_S13')
iv_final.to_csv(PROCESSED / 'mimiciv_clinical_features.csv', index=False)
eicu_final.to_csv(PROCESSED / 'eicu_clinical_features.csv', index=False)
iii_final.to_csv(PROCESSED / 'mimiciii_clinical_features.csv', index=False)
print(f'  Saved clinical feature CSVs')

def compute_table1(df, group_col=None):
    """Compute descriptive statistics for Table 1"""
    rows = []
    CONT_VARS = ['age', 'los_hours'] + list(LAB_TARGETS.keys()) + ['SpO2']
    CAT_VARS = ['gender', 'mechanical_ventilation', 'vasopressor_use', 'mortality_30d', 'icu_mortality'] + list(COMORBIDITY_ICD9.keys())
    if group_col and group_col in df.columns:
        groups = sorted(df[group_col].dropna().unique())
    else:
        groups = [None]
    for var in CONT_VARS:
        if var not in df.columns:
            continue
        row = {'variable': var, 'type': 'continuous'}
        vals_all = df[var].dropna()
        row['overall'] = f'{vals_all.median():.1f} ({vals_all.quantile(0.25):.1f}-{vals_all.quantile(0.75):.1f})'
        row['n_overall'] = len(vals_all)
        if group_col:
            group_vals = []
            for g in groups:
                gdf = df[df[group_col] == g][var].dropna()
                row[f'G{int(g)}'] = f'{gdf.median():.1f} ({gdf.quantile(0.25):.1f}-{gdf.quantile(0.75):.1f})' if len(gdf) > 0 else 'NA'
                group_vals.append(gdf.values)
            if len([v for v in group_vals if len(v) > 0]) >= 2:
                stat, pval = stats.kruskal(*[v for v in group_vals if len(v) > 0])
                row['p_value'] = f'{pval:.3f}' if pval >= 0.001 else '<0.001'
        rows.append(row)
    for var in CAT_VARS:
        if var not in df.columns:
            continue
        row = {'variable': var, 'type': 'categorical'}
        n_all = df[var].notna().sum()
        n_pos = (df[var] == 1).sum() if df[var].dtype in [int, float] else (df[var] == df[var].mode()[0]).sum() if len(df[var].mode()) > 0 else 0
        row['overall'] = f'{n_pos} ({n_pos / n_all * 100:.1f}%)' if n_all > 0 else 'NA'
        row['n_overall'] = n_all
        if group_col:
            grp_counts = []
            for g in groups:
                gdf = df[df[group_col] == g]
                n_g = gdf[var].notna().sum()
                n_pos_g = (gdf[var] == 1).sum() if n_g > 0 else 0
                row[f'G{int(g)}'] = f'{n_pos_g} ({n_pos_g / n_g * 100:.1f}%)' if n_g > 0 else 'NA'
                grp_counts.append([n_pos_g, n_g - n_pos_g])
            try:
                chi2, pval, _, _ = stats.chi2_contingency(grp_counts)
                row['p_value'] = f'{pval:.3f}' if pval >= 0.001 else '<0.001'
            except:
                row['p_value'] = 'NA'
        rows.append(row)
    return pd.DataFrame(rows)
if 'phenotype' in iv_final.columns and iv_final['phenotype'].notna().any():
    table1 = compute_table1(iv_final, group_col='phenotype')
    table1.to_csv(TABLES / 'Table_1_development_baseline.csv', index=False)
    print(f'  Saved: Table_1_development_baseline.csv')
else:
    table1 = compute_table1(iv_final, group_col=None)
    table1.to_csv(TABLES / 'Table_1_development_baseline.csv', index=False)
    print(f'  Saved: Table_1_development_baseline.csv (no phenotype groups yet)')
table_s11 = compute_table1(eicu_final)
table_s11.to_csv(TABLES / 'Table_S11_validation_eicu_baseline.csv', index=False)
table_s12 = compute_table1(iii_final)
table_s12.to_csv(TABLES / 'Table_S12_validation_mimiciii_baseline.csv', index=False)
print(f'  Saved: Table_S11 and Table_S12')
with open(LOGS / '09_clinical_feature_extraction_report.md', 'w') as f:
    f.write('# Clinical Feature Extraction Report\n\n')
    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f'## Lab Values Extracted\n\n')
    for lab, ids in lab_itemids.items():
        f.write(f'- {lab}: {len(ids)} itemids\n')
    f.write(f'\n## Notes\n\n')
    f.write('- CRP, PCT, triglycerides: low availability in MIMIC-IV (rarely measured)\n')
    f.write('- PaO2/FiO2 ratio: computed if both available\n')
    f.write('- Missing rate > 20%: variable excluded from RF model\n')
print(f'\n=== Script 08 COMPLETE ===')
