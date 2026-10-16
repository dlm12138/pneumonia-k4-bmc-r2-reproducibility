"""Original corrected-source hourly generator; generic oxygen excluded from current use.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
import csv
import gzip
import importlib.util
import io
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd
ROOT = Path(os.environ['SOURCE_9A17FFCF9A'])
SCAN = ROOT / '00_scan_reports'
COUNT_DIR = ROOT / '01_pneumonia_cohort_counts'
VITAL_DIR = ROOT / '02_vital_completeness'
OUT = ROOT / '03_hourly_vital_trajectory_dataset'
LOG = OUT / '03_run_log.txt'
PREV_SCRIPT = COUNT_DIR / '01_build_pneumonia_cohort_feasibility_counts.py'
DATABASES = ['MIMIC-IV', 'MIMIC-III', 'eICU']
VARS = ['SBP', 'DBP', 'MAP', 'HR', 'RR', 'TEMP', 'SpO2']
BASE_5 = ['SBP', 'DBP', 'HR', 'RR', 'TEMP']
RANGES = {'SBP': (40, 300), 'DBP': (20, 200), 'MAP': (30, 250), 'HR': (20, 250), 'RR': (3, 80), 'TEMP': (25, 45), 'SpO2': (50, 100)}
EICU_FIELD_MAP = {'vitalPeriodic': {'SBP': ['systemicsystolic'], 'DBP': ['systemicdiastolic'], 'MAP': ['systemicmean'], 'HR': ['heartrate'], 'RR': ['respiration'], 'TEMP': ['temperature'], 'SpO2': ['sao2']}, 'vitalAperiodic': {'SBP': ['noninvasivesystolic'], 'DBP': ['noninvasivediastolic'], 'MAP': ['noninvasivemean']}}
VERSIONS = {'A_strict_5var_2h': {'label': 'strict_5var_2h', 'vars': BASE_5, 'hour_min_present': 5, 'required_hours': 2, 'description': '0-12 h 内至少 2 个小时同时具备 SBP、DBP、HR、RR、TEMP。'}, 'B_strict_5var_4h': {'label': 'strict_5var_4h', 'vars': BASE_5, 'hour_min_present': 5, 'required_hours': 4, 'description': '0-12 h 内至少 4 个小时同时具备 SBP、DBP、HR、RR、TEMP。'}, 'C_relaxed_5var_4of5_6h': {'label': 'relaxed_5var_4of5_6h', 'vars': BASE_5, 'hour_min_present': 4, 'required_hours': 6, 'description': '每小时 5 个主变量中至少 4 个可用，且至少 6 个小时满足。'}, 'D_core_4var_no_temp_6h': {'label': 'core_4var_no_temp_6h', 'vars': ['SBP', 'DBP', 'HR', 'RR'], 'hour_min_present': 4, 'required_hours': 6, 'description': 'SBP、DBP、HR、RR 每小时均可用，且至少 6 个小时完整。'}, 'E_map_core_4var_6h': {'label': 'map_core_4var_6h', 'vars': ['MAP', 'HR', 'RR', 'TEMP'], 'hour_min_present': 4, 'required_hours': 6, 'description': 'MAP、HR、RR、TEMP 每小时均可用，且至少 6 个小时完整。'}, 'F_spo2_sensitivity_6var_2h': {'label': 'spo2_sensitivity_6var_2h', 'vars': ['SBP', 'DBP', 'HR', 'RR', 'TEMP', 'SpO2'], 'hour_min_present': 6, 'required_hours': 2, 'description': '0-12 h 内至少 2 个小时同时具备 SBP、DBP、HR、RR、TEMP、SpO2。'}}

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def read_required_inputs():
    required = [SCAN / '03_table_schema_inventory.csv', SCAN / '04_key_table_candidates.csv', COUNT_DIR / '01_database_table_paths_used.csv', COUNT_DIR / '09_candidate_ids_preview.csv', VITAL_DIR / '01_vital_table_paths_used.csv', VITAL_DIR / '02_vital_itemid_or_field_candidates.csv', VITAL_DIR / '03_vital_record_value_audit.csv', VITAL_DIR / '08_candidate_vital_completeness_summary.csv', VITAL_DIR / '10_variable_level_availability.csv']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('缺少必须读取的上一步结果: ' + '; '.join(missing))
    loaded = {}
    for path in required:
        loaded[path.name] = pd.read_csv(path, dtype=str)
        log(f'已读取上一步结果: {path} rows={len(loaded[path.name])}')
    return loaded

def read_scan(name):
    with (SCAN / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def source_parts(path):
    if str(path).startswith('zip://'):
        rest = str(path)[len('zip://'):]
        z, member = rest.split('!', 1)
        return (Path(z), member)
    return (Path(path), None)

def open_source(path):
    p, member = source_parts(path)
    if member:
        zf = zipfile.ZipFile(p)
        raw = zf.open(member)
        if member.lower().endswith('.gz'):
            return (gzip.open(raw, 'rt', encoding='utf-8', errors='replace', newline=''), zf)
        return (io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline=''), zf)
    if str(p).lower().endswith('.gz'):
        return (gzip.open(p, 'rt', encoding='utf-8', errors='replace', newline=''), None)
    return (open(p, 'r', encoding='utf-8', errors='replace', newline=''), None)

def available_columns(path):
    stream, holder = open_source(path)
    try:
        header = stream.readline().strip('\n\r')
        return next(csv.reader([header])) if header else []
    finally:
        stream.close()
        if holder:
            holder.close()

def actual_usecols(path, usecols):
    if not usecols:
        return None
    cols = available_columns(path)
    lower_map = {str(c).strip().lower(): c for c in cols}
    return [lower_map[c.lower()] for c in usecols if c.lower() in lower_map]

def read_csv_chunks(path, usecols=None, chunksize=600000):
    actual = actual_usecols(path, usecols)
    stream, holder = open_source(path)
    try:
        for chunk in pd.read_csv(stream, usecols=actual, dtype=str, chunksize=chunksize, low_memory=False):
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            yield chunk
    finally:
        stream.close()
        if holder:
            holder.close()

def read_csv_full(path, usecols=None):
    actual = actual_usecols(path, usecols)
    stream, holder = open_source(path)
    try:
        df = pd.read_csv(stream, usecols=actual, dtype=str, low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    finally:
        stream.close()
        if holder:
            holder.close()

def load_prev_module():
    spec = importlib.util.spec_from_file_location('prev_counts', PREV_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG = LOG
    mod.log = log
    return mod

def rebuild_candidates(schema_rows):
    prev = load_prev_module()
    selected = prev.choose_tables(schema_rows)
    log('重建三数据库完整候选 ICU stay；preview 文件仅用于核对，不作为最终数据来源')
    results = {'MIMIC-IV': prev.mimic_counts('MIMIC-IV', selected['MIMIC-IV']), 'MIMIC-III': prev.mimic_counts('MIMIC-III', selected['MIMIC-III']), 'eICU': prev.eicu_counts(selected['eICU'])}
    return (results, selected)

def choose_vital_paths(schema_rows):

    def by_path(db, contains, prefer_local=False):
        candidates = []
        for row in schema_rows:
            if row['database_inferred'] != db:
                continue
            if contains in row['path'].lower():
                candidates.append(row)
        if prefer_local:
            for row in candidates:
                if not row['path'].startswith('zip://'):
                    return row
        return candidates[0] if candidates else None
    return {'MIMIC-IV': {'chartevents': by_path('MIMIC-IV', 'chartevents.csv.gz'), 'd_items': by_path('MIMIC-IV', 'd_items.csv.gz', prefer_local=True)}, 'MIMIC-III': {'chartevents': by_path('MIMIC-III', 'chartevents.csv.gz'), 'd_items': by_path('MIMIC-III', 'd_items.csv.gz', prefer_local=True)}, 'eICU': {'vitalPeriodic': by_path('eICU', 'vitalperiodic.csv.gz'), 'vitalAperiodic': by_path('eICU', 'vitalaperiodic.csv.gz'), 'nurseCharting': by_path('eICU', 'nursecharting.csv.gz'), 'respiratoryCharting': by_path('eICU', 'respiratorycharting.csv.gz')}}

def label_to_var(label):
    low = str(label).lower().strip()
    if not low:
        return None
    excluded_fragments = ['alarm', 'set ', ' setting', 'resistance', 'vital capacity', 'temporary', 'chromium', 'ingr', 'diet', 'nutrition', 'minute volume', 'pulmonary artery', 'central venous', 'intra aortic']
    if any((x in low for x in excluded_fragments)):
        return None
    if 'o2 saturation' in low or 'oxygen saturation' in low or 'spo2' in low or ('sao2' in low):
        return 'SpO2'
    if 'temperature' in low or low == 'temp' or low.startswith('temp '):
        return 'TEMP'
    if 'respiratory rate' in low or low == 'rr' or low == 'respiration':
        return 'RR'
    if 'heart rate' in low or low == 'hr' or low == 'pulse':
        return 'HR'
    if 'mean arterial pressure' in low or 'blood pressure mean' in low or 'nibp mean' in low or (low == 'map'):
        return 'MAP'
    if 'blood pressure systolic' in low or 'systolic blood pressure' in low or 'nibp systolic' in low or (low == 'sbp'):
        return 'SBP'
    if 'blood pressure diastolic' in low or 'diastolic blood pressure' in low or 'nibp diastolic' in low or (low == 'dbp'):
        return 'DBP'
    return None

def clean_value(var, value, uom=''):
    raw = pd.to_numeric(value, errors='coerce')
    if pd.isna(raw):
        return (None, False, False)
    val = float(raw)
    converted = False
    unit = str(uom).lower()
    if var == 'TEMP' and ('f' in unit or val > 60):
        val = (val - 32) * 5 / 9
        converted = True
    lo, hi = RANGES[var]
    if val < lo or val > hi:
        return (None, True, converted)
    return (val, False, converted)

def add_agg(agg, db, stay_id, hour, var, val, source_table, source_item):
    key = (db, str(stay_id), int(hour), var)
    rec = agg[key]
    rec['sum'] += val
    rec['n'] += 1
    rec['source_tables'][source_table] += 1
    rec['source_items'][str(source_item)] += 1

def scan_mimic_items(db, d_items_path):
    items = read_csv_full(d_items_path)
    unit_col = 'unitname' if 'unitname' in items.columns else None
    mapping = {}
    rows = []
    for _, row in items.iterrows():
        itemid = str(row.get('itemid', ''))
        label = str(row.get('label', ''))
        var = label_to_var(label)
        if var:
            mapping[itemid] = {'variable': var, 'label': label, 'unit': row.get(unit_col, '') if unit_col else ''}
        rows.append({'database': db, 'table': 'chartevents/d_items', 'variable': var or '', 'itemid_or_field': itemid, 'label': label, 'unit': row.get(unit_col, '') if unit_col else '', 'included_in_step03': 'yes' if var else 'no'})
    return (mapping, rows)

def scan_mimic_vitals(db, paths, candidates, item_map):
    id_col = 'stay_id' if db == 'MIMIC-IV' else 'icustay_id'
    cand = candidates[['subject_id', 'hadm_id', id_col, 'intime_dt']].copy()
    cand[id_col] = cand[id_col].astype(str)
    intime_map = dict(zip(cand[id_col], cand['intime_dt']))
    stay_set = set(intime_map)
    item_set = set(item_map)
    agg = defaultdict(lambda: {'sum': 0.0, 'n': 0, 'source_tables': Counter(), 'source_items': Counter()})
    audit = {v: {'raw': 0, 'valid': 0, 'abnormal': 0, 'converted': 0, 'stays': set()} for v in VARS}
    item_counts = Counter()
    item_stays = defaultdict(set)
    usecols = [id_col, 'itemid', 'charttime', 'valuenum', 'valueuom']
    log(f'{db}: 分块扫描 chartevents；候选 stays={len(stay_set)}，生命体征 itemids={len(item_set)}')
    for idx, chunk in enumerate(read_csv_chunks(paths['chartevents']['path'], usecols=usecols, chunksize=600000), 1):
        if idx % 20 == 0:
            log(f'{db}: chartevents 已处理 {idx} 个分块')
        chunk[id_col] = chunk[id_col].astype(str)
        chunk['itemid'] = chunk['itemid'].astype(str)
        sub = chunk[chunk[id_col].isin(stay_set) & chunk['itemid'].isin(item_set)].copy()
        if sub.empty:
            continue
        sub['chart_dt'] = pd.to_datetime(sub['charttime'], errors='coerce')
        sub['intime_dt'] = sub[id_col].map(intime_map)
        sub['hours'] = (sub['chart_dt'] - sub['intime_dt']).dt.total_seconds() / 3600
        sub = sub[(sub['hours'] >= 0) & (sub['hours'] < 12)]
        if sub.empty:
            continue
        sub['hour'] = sub['hours'].astype(int)
        for row in sub.itertuples(index=False):
            itemid = str(getattr(row, 'itemid'))
            meta = item_map[itemid]
            var = meta['variable']
            stay_id = str(getattr(row, id_col))
            audit[var]['raw'] += 1
            item_counts[var, itemid] += 1
            item_stays[var, itemid].add(stay_id)
            val, abnormal, converted = clean_value(var, getattr(row, 'valuenum'), getattr(row, 'valueuom', ''))
            if converted:
                audit[var]['converted'] += 1
            if abnormal:
                audit[var]['abnormal'] += 1
            if val is None:
                continue
            audit[var]['valid'] += 1
            audit[var]['stays'].add(stay_id)
            add_agg(agg, db, stay_id, getattr(row, 'hour'), var, val, 'chartevents', itemid)
    return (agg, audit, item_counts, item_stays)

def scan_eicu_vitals(paths, candidates):
    cand = candidates[['patientunitstayid']].copy()
    cand['patientunitstayid'] = cand['patientunitstayid'].astype(str)
    stay_set = set(cand['patientunitstayid'])
    agg = defaultdict(lambda: {'sum': 0.0, 'n': 0, 'source_tables': Counter(), 'source_items': Counter()})
    audit = {v: {'raw': 0, 'valid': 0, 'abnormal': 0, 'converted': 0, 'stays': set()} for v in VARS}
    field_counts = Counter()
    field_stays = defaultdict(set)
    for table, varmap in EICU_FIELD_MAP.items():
        path = paths[table]['path']
        fields = ['patientunitstayid', 'observationoffset']
        for fs in varmap.values():
            fields.extend(fs)
        fields = sorted(set(fields))
        log(f'eICU: 分块扫描 {table}')
        for idx, chunk in enumerate(read_csv_chunks(path, usecols=fields, chunksize=600000), 1):
            if idx % 20 == 0:
                log(f'eICU: {table} 已处理 {idx} 个分块')
            chunk['patientunitstayid'] = chunk['patientunitstayid'].astype(str)
            sub = chunk[chunk['patientunitstayid'].isin(stay_set)].copy()
            if sub.empty:
                continue
            sub['offset_min'] = pd.to_numeric(sub['observationoffset'], errors='coerce')
            sub = sub[(sub['offset_min'] >= 0) & (sub['offset_min'] < 720)]
            if sub.empty:
                continue
            sub['hour'] = (sub['offset_min'] // 60).astype(int)
            for var, fs in varmap.items():
                for field in fs:
                    if field not in sub.columns:
                        continue
                    nonmiss = sub[sub[field].notna()]
                    if nonmiss.empty:
                        continue
                    for row in nonmiss.itertuples(index=False):
                        stay_id = str(getattr(row, 'patientunitstayid'))
                        audit[var]['raw'] += 1
                        field_counts[var, field] += 1
                        field_stays[var, field].add(stay_id)
                        val, abnormal, converted = clean_value(var, getattr(row, field), 'C' if var == 'TEMP' else '')
                        if converted:
                            audit[var]['converted'] += 1
                        if abnormal:
                            audit[var]['abnormal'] += 1
                        if val is None:
                            continue
                        audit[var]['valid'] += 1
                        audit[var]['stays'].add(stay_id)
                        add_agg(agg, 'eICU', stay_id, getattr(row, 'hour'), var, val, table, field)
    return (agg, audit, field_counts, field_stays)

def normalize_candidate_frame(db, final, id_col):
    frame = final.copy()
    if db == 'eICU':
        out = pd.DataFrame({'database': db, 'subject_id_or_patientunitstayid': frame['patientunitstayid'].astype(str), 'hadm_id': '', 'stay_id_or_icustay_id': frame['patientunitstayid'].astype(str), 'intime': frame.get('unitadmittime24', ''), 'outtime': frame.get('unitdischargetime24', ''), 'age': frame.get('age_num', ''), 'sex': frame.get('gender', ''), 'hospital_mortality': frame.get('hospital_mortality', ''), 'mortality_28d_available': False, 'mortality_28d': '', 'icu_los_hours': pd.to_numeric(frame.get('unitdischargeoffset', ''), errors='coerce') / 60})
    else:
        out = pd.DataFrame({'database': db, 'subject_id_or_patientunitstayid': frame['subject_id'].astype(str), 'hadm_id': frame['hadm_id'].astype(str), 'stay_id_or_icustay_id': frame[id_col].astype(str), 'intime': frame.get('intime', ''), 'outtime': frame.get('outtime', ''), 'age': frame.get('age', ''), 'sex': frame.get('gender', ''), 'hospital_mortality': frame.get('hospital_mortality', ''), 'mortality_28d_available': frame.get('mortality_28d_available', True), 'mortality_28d': frame.get('mortality_28d', ''), 'icu_los_hours': pd.to_numeric(frame.get('los_days', ''), errors='coerce') * 24})
    return out

def build_hourly_tables(candidates_all, agg):
    meta = candidates_all.set_index(['database', 'stay_id_or_icustay_id']).to_dict('index')
    rows = []
    for (db, stay_id, hour, var), rec in agg.items():
        if rec['n'] <= 0:
            continue
        m = meta.get((db, stay_id), {})
        intime = m.get('intime', '')
        if db == 'eICU':
            hour_start = f'offset_{hour * 60}_min'
            hour_end = f'offset_{(hour + 1) * 60}_min'
        else:
            dt = pd.to_datetime(intime, errors='coerce')
            hour_start_dt = dt + pd.Timedelta(hours=hour) if pd.notna(dt) else pd.NaT
            hour_end_dt = dt + pd.Timedelta(hours=hour + 1) if pd.notna(dt) else pd.NaT
            hour_start = hour_start_dt.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(hour_start_dt) else ''
            hour_end = hour_end_dt.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(hour_end_dt) else ''
        rows.append({'database': db, 'subject_id_or_patientunitstayid': m.get('subject_id_or_patientunitstayid', ''), 'hadm_id': m.get('hadm_id', ''), 'stay_id_or_icustay_id': stay_id, 'intime': intime, 'hour_index': int(hour), 'hour_start': hour_start, 'hour_end': hour_end, 'variable': var, 'value': rec['sum'] / rec['n'], 'n_raw_records': rec['n'], 'source_table': '|'.join(sorted(rec['source_tables'].keys())), 'source_itemid_or_field': '|'.join(sorted(rec['source_items'].keys()))})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError('没有生成任何小时级生命体征记录')
    long_df = long_df.sort_values(['database', 'stay_id_or_icustay_id', 'hour_index', 'variable'])
    value_wide = long_df.pivot_table(index=['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime', 'hour_index'], columns='variable', values='value', aggfunc='first').reset_index()
    n_wide = long_df.pivot_table(index=['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime', 'hour_index'], columns='variable', values='n_raw_records', aggfunc='first').reset_index()
    n_wide = n_wide.rename(columns={v: f'{v}_n' for v in VARS if v in n_wide.columns})
    wide_df = candidates_all[['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime']].merge(pd.DataFrame({'hour_index': range(12)}), how='cross')
    wide_df = wide_df.merge(value_wide, on=['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime', 'hour_index'], how='left')
    wide_df = wide_df.merge(n_wide, on=['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime', 'hour_index'], how='left')
    for var in VARS:
        if var not in wide_df.columns:
            wide_df[var] = pd.NA
        ncol = f'{var}_n'
        if ncol not in wide_df.columns:
            wide_df[ncol] = pd.NA
    return (long_df, wide_df)

def build_patient_matrix(candidates_all, wide_df):
    matrix = candidates_all.copy()
    base_cols = ['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime']
    for var in VARS:
        p = wide_df.pivot_table(index=base_cols, columns='hour_index', values=var, aggfunc='first')
        p = p.rename(columns={h: f'{var}_h{int(h)}' for h in p.columns}).reset_index()
        matrix = matrix.merge(p, on=base_cols, how='left')
    ordered = ['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime', 'outtime', 'age', 'sex', 'hospital_mortality', 'mortality_28d_available', 'mortality_28d', 'icu_los_hours']
    value_cols = [f'{var}_h{h}' for var in VARS for h in range(12)]
    for c in value_cols:
        if c not in matrix.columns:
            matrix[c] = pd.NA
    return matrix[ordered + value_cols]

def version_hour_counts(row, spec):
    count = 0
    for h in range(12):
        present = sum((pd.notna(row.get(f'{v}_h{h}')) for v in spec['vars']))
        if present >= spec['hour_min_present']:
            count += 1
    return count

def classify_versions(matrix):
    ids = matrix[['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id']].copy()
    for key, spec in VERSIONS.items():
        hour_col = f"{spec['label']}_available_hours"
        flag_col = f"in_{spec['label']}"
        counts = matrix.apply(lambda row: version_hour_counts(row, spec), axis=1)
        ids[hour_col] = counts
        ids[flag_col] = counts >= spec['required_hours']
    return ids

def make_version_counts(matrix, version_ids):
    rows = []
    total_by_db = matrix.groupby('database')['stay_id_or_icustay_id'].nunique().to_dict()
    meta = matrix[['database', 'stay_id_or_icustay_id', 'hospital_mortality', 'mortality_28d_available', 'mortality_28d']]
    flags = version_ids[['database', 'stay_id_or_icustay_id'] + [f"in_{s['label']}" for s in VERSIONS.values()]]
    joined = meta.merge(flags, on=['database', 'stay_id_or_icustay_id'], how='left')
    for key, spec in VERSIONS.items():
        flag = f"in_{spec['label']}"
        for db in DATABASES:
            sub = joined[(joined['database'] == db) & joined[flag]]
            n = len(sub)
            total = total_by_db.get(db, 0)
            mort28_avail = sub['mortality_28d_available'].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
            rows.append({'version': spec['label'], 'version_letter': key.split('_', 1)[0], 'database': db, 'included_n': n, 'candidate_n': total, 'included_percent': round(100 * n / total, 2) if total else 0, 'hospital_mortality_available_n': sub['hospital_mortality'].notna().sum(), 'mortality_28d_available_n': int(mort28_avail), 'hospital_deaths': int(pd.to_numeric(sub['hospital_mortality'], errors='coerce').fillna(0).sum()) if n else 0, 'mortality_28d_deaths': int(pd.to_numeric(sub['mortality_28d'], errors='coerce').fillna(0).sum()) if n else ''})
    return rows

def make_missingness(matrix, version_ids):
    rows = []
    m = matrix.merge(version_ids, on=['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id'], how='left')
    for key, spec in VERSIONS.items():
        label = spec['label']
        sub_all = m[m[f'in_{label}']].copy()
        for db in ['ALL'] + DATABASES:
            sub = sub_all if db == 'ALL' else sub_all[sub_all['database'] == db]
            denom = len(sub)
            for var in spec['vars']:
                total_cells = denom * 12
                miss = int(sub[[f'{var}_h{h}' for h in range(12)]].isna().sum().sum()) if denom else 0
                rows.append({'version': label, 'database': db, 'missingness_type': 'variable', 'item': var, 'denominator_cells': total_cells, 'missing_cells': miss, 'missing_rate': round(miss / total_cells, 4) if total_cells else ''})
            for h in range(12):
                cols = [f'{var}_h{h}' for var in spec['vars']]
                total_cells = denom * len(cols)
                miss = int(sub[cols].isna().sum().sum()) if denom else 0
                rows.append({'version': label, 'database': db, 'missingness_type': 'hour', 'item': f'hour{h}', 'denominator_cells': total_cells, 'missing_cells': miss, 'missing_rate': round(miss / total_cells, 4) if total_cells else ''})
            dist = sub[f'{label}_available_hours'].value_counts(dropna=False).sort_index().to_dict() if denom else {}
            rows.append({'version': label, 'database': db, 'missingness_type': 'available_hour_distribution', 'item': str(dist), 'denominator_cells': denom, 'missing_cells': '', 'missing_rate': ''})
    return rows

def make_preview(matrix, version_ids):
    rows = []
    m = matrix.merge(version_ids, on=['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id'], how='left')
    base_cols = ['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id']
    for key, spec in VERSIONS.items():
        label = spec['label']
        cols = base_cols + [f'{var}_h{h}' for var in spec['vars'] for h in range(12)]
        sub = m[m[f'in_{label}']][cols].head(20).copy()
        sub.insert(0, 'dataset_version', label)
        rows.append(sub)
    if rows:
        return pd.concat(rows, ignore_index=True, sort=False)
    return pd.DataFrame()

def write_report(counts, missingness, value_audit, paths_used, item_rows, recommendation):
    count_df = pd.DataFrame(counts)
    miss_df = pd.DataFrame(missingness)
    lines = ['# 0-12 h 小时级生命体征轨迹数据集构建审计报告', '', f'生成时间：{now()}', '', '## 候选队列重建', '', '- 本步骤通过导入第二步脚本并重新运行三数据库候选队列筛选逻辑，完整重建候选 ICU stay。', '- `09_candidate_ids_preview.csv` 已读取用于审计，但没有作为最终轨迹矩阵的候选来源。', '', '## 小时级轨迹构建方法', '', '- time zero 为 ICU 入科时间；MIMIC-IV/MIMIC-III 使用绝对 `intime` 和 `charttime`，纳入 `[intime, intime + 12 hours)`。', '- eICU 使用 ICU 入科后 offset，纳入 `[0, 720)` 分钟。由于 eICU 缺少统一绝对入科日期，`hour_start/hour_end` 用 offset 标签记录。', '- 同一 ICU stay、同一小时、同一变量多条记录取平均值，并保留 `n_raw_records`。', '- 本次只做轨迹数据构建和候选版本比较，没有 GBMTM、聚类、作图、死亡模型、液体、机械通气或治疗异质性分析。', '', '## 使用的生命体征表和字段', '']
    for row in paths_used:
        lines.append(f"- {row['database']} / {row['table_type']}: `{row['path']}`")
    lines.extend(['', '## 值域过滤和温度转换', ''])
    for row in value_audit:
        lines.append(f"- {row['database']} {row['variable']}: raw={row['raw_record_count']}, valid={row['valid_record_count']}, abnormal={row['abnormal_record_count']}, temperature_converted={row['temperature_converted_count']}。")
    lines.extend(['', '## A-F 候选版本样本量', ''])
    for label in [spec['label'] for spec in VERSIONS.values()]:
        sub = count_df[count_df['version'] == label]
        parts = [f'{r.database}={int(r.included_n)} ({r.included_percent}%)' for r in sub.itertuples()]
        lines.append(f'- {label}: ' + '; '.join(parts))
    lines.extend(['', '## 缺失模式和跨库差异', ''])
    for label in [spec['label'] for spec in VERSIONS.values()]:
        sub = miss_df[(miss_df['version'] == label) & (miss_df['database'] != 'ALL') & (miss_df['missingness_type'] == 'variable')]
        if sub.empty:
            continue
        summary = []
        for db in DATABASES:
            db_sub = sub[sub['database'] == db]
            if not db_sub.empty:
                worst = db_sub.sort_values('missing_rate', ascending=False).iloc[0]
                summary.append(f"{db} 最高缺失变量 {worst['item']}={worst['missing_rate']}")
        lines.append(f'- {label}: ' + '；'.join(summary))
    lines.extend(['', '## 版本解释', '', '- 严格 5/5 不能直接作为唯一主模型，因为 TEMP 在 eICU 覆盖偏低，SBP/DBP 在 MIMIC-III 覆盖也明显低于 MIMIC-IV/eICU；若强制每小时 5/5 且高小时阈值，会把队列推向局部可测量人群。', '- relaxed_5var_4of5_6h 保留 5 个核心变量体系，同时允许每小时缺 1 个变量，适合做第一版主模型候选。', '- core_4var_no_temp_6h 用于检查 TEMP 是否造成样本损失；如果它明显扩大样本量，应作为敏感性或备选主模型候选，而不是直接替代含体温方案。', '- map_core_4var_6h 检查 MAP 替代 SBP/DBP 的跨库一致性；若 MIMIC-III 血压字段稀疏，MAP 方案可作为备选。', '- SpO2 在第三步已显示加入后对 MIMIC-IV/MIMIC-III 6 小时阈值损失严重，因此本步只输出敏感性候选，不推荐进入主模型。', '', '## 推荐', '', f'- 推荐进入下一步 GBMTM 前预处理的数据集版本：`{recommendation}`。', '- 下一步脚本建议：`04_prepare_gbmtm_input_and_standardization.py`。', '- 下一步可以考虑在标准化前设计插补方案，但本步骤未生成最终插补建模数据。'])
    (OUT / '08_dataset_recommendation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f'[{now()}] 开始构建 0-12 h 小时级生命体征轨迹数据集\n', encoding='utf-8')
    loaded = read_required_inputs()
    schema_rows = read_scan('03_table_schema_inventory.csv')
    results, _selected = rebuild_candidates(schema_rows)
    vital_paths = choose_vital_paths(schema_rows)
    paths_used = []
    for db, tabs in vital_paths.items():
        for table_type, meta in tabs.items():
            if meta:
                paths_used.append({'database': db, 'table_type': table_type, 'path': meta['path'], 'columns': meta['columns']})
    candidate_frames = []
    cohort_rows = []
    all_aggs = defaultdict(lambda: {'sum': 0.0, 'n': 0, 'source_tables': Counter(), 'source_items': Counter()})
    item_rows_all = []
    value_audit_rows = []
    for db in ['MIMIC-IV', 'MIMIC-III']:
        final = results[db]['final'].copy()
        id_col = results[db]['id_col']
        candidate_frames.append(normalize_candidate_frame(db, final, id_col))
        cohort_rows.append({'database': db, 'rebuilt_candidate_n': len(final), 'preview_rows_read': int((loaded['09_candidate_ids_preview.csv']['database'] == db).sum()), 'used_preview_as_source': 'no', 'id_column': id_col})
        item_map, item_rows = scan_mimic_items(db, vital_paths[db]['d_items']['path'])
        agg, audit, item_counts, item_stays = scan_mimic_vitals(db, vital_paths[db], final, item_map)
        for key, rec in agg.items():
            all_aggs[key]['sum'] += rec['sum']
            all_aggs[key]['n'] += rec['n']
            all_aggs[key]['source_tables'].update(rec['source_tables'])
            all_aggs[key]['source_items'].update(rec['source_items'])
        for row in item_rows:
            key = (row['variable'], row['itemid_or_field'])
            row['record_count_step03'] = item_counts.get(key, 0)
            row['stay_count_step03'] = len(item_stays.get(key, set()))
            item_rows_all.append(row)
        for var in VARS:
            value_audit_rows.append({'database': db, 'variable': var, 'raw_record_count': audit[var]['raw'], 'valid_record_count': audit[var]['valid'], 'abnormal_record_count': audit[var]['abnormal'], 'temperature_converted_count': audit[var]['converted'], 'stay_count_with_valid_record': len(audit[var]['stays'])})
    db = 'eICU'
    final = results[db]['final'].copy()
    candidate_frames.append(normalize_candidate_frame(db, final, 'patientunitstayid'))
    cohort_rows.append({'database': db, 'rebuilt_candidate_n': len(final), 'preview_rows_read': int((loaded['09_candidate_ids_preview.csv']['database'] == db).sum()), 'used_preview_as_source': 'no', 'id_column': 'patientunitstayid'})
    agg, audit, field_counts, field_stays = scan_eicu_vitals(vital_paths[db], final)
    for key, rec in agg.items():
        all_aggs[key]['sum'] += rec['sum']
        all_aggs[key]['n'] += rec['n']
        all_aggs[key]['source_tables'].update(rec['source_tables'])
        all_aggs[key]['source_items'].update(rec['source_items'])
    for table, varmap in EICU_FIELD_MAP.items():
        for var, fields in varmap.items():
            for field in fields:
                item_rows_all.append({'database': db, 'table': table, 'variable': var, 'itemid_or_field': field, 'label': field, 'unit': 'eICU native field', 'included_in_step03': 'yes', 'record_count_step03': field_counts.get((var, field), 0), 'stay_count_step03': len(field_stays.get((var, field), set()))})
    for table in ['nurseCharting', 'respiratoryCharting']:
        if vital_paths[db].get(table):
            item_rows_all.append({'database': db, 'table': table, 'variable': '', 'itemid_or_field': 'label/value text fields', 'label': '文本型候选来源，本步骤未纳入结构化轨迹', 'unit': '', 'included_in_step03': 'no', 'record_count_step03': 0, 'stay_count_step03': 0})
    for var in VARS:
        value_audit_rows.append({'database': db, 'variable': var, 'raw_record_count': audit[var]['raw'], 'valid_record_count': audit[var]['valid'], 'abnormal_record_count': audit[var]['abnormal'], 'temperature_converted_count': audit[var]['converted'], 'stay_count_with_valid_record': len(audit[var]['stays'])})
    candidates_all = pd.concat(candidate_frames, ignore_index=True)
    candidates_all['stay_id_or_icustay_id'] = candidates_all['stay_id_or_icustay_id'].astype(str)
    log('开始构建长表、小时宽表和患者级 12 小时矩阵')
    long_df, wide_df = build_hourly_tables(candidates_all, all_aggs)
    matrix = build_patient_matrix(candidates_all, wide_df)
    version_ids = classify_versions(matrix)
    counts = make_version_counts(matrix, version_ids)
    missingness = make_missingness(matrix, version_ids)
    preview = make_preview(matrix, version_ids)
    write_csv(OUT / '01_candidate_cohort_rebuilt_summary.csv', cohort_rows, ['database', 'rebuilt_candidate_n', 'preview_rows_read', 'used_preview_as_source', 'id_column'])
    long_df.to_parquet(OUT / '02_hourly_vitals_long_cleaned.parquet', index=False)
    wide_df.to_parquet(OUT / '03_hourly_vitals_wide_cleaned.parquet', index=False)
    matrix.to_parquet(OUT / '04_patient_level_trajectory_matrix_all_variables.parquet', index=False)
    write_csv(OUT / '05_dataset_version_counts.csv', counts, ['version', 'version_letter', 'database', 'included_n', 'candidate_n', 'included_percent', 'hospital_mortality_available_n', 'mortality_28d_available_n', 'hospital_deaths', 'mortality_28d_deaths'])
    write_csv(OUT / '06_missingness_by_dataset_version.csv', missingness, ['version', 'database', 'missingness_type', 'item', 'denominator_cells', 'missing_cells', 'missing_rate'])
    version_ids.to_csv(OUT / '07_dataset_version_patient_ids.csv', index=False, encoding='utf-8-sig')
    preview.to_csv(OUT / '09_gbmtm_input_format_preview.csv', index=False, encoding='utf-8-sig')
    write_csv(OUT / '10_step03_vital_itemid_or_field_audit.csv', item_rows_all, ['database', 'table', 'variable', 'itemid_or_field', 'label', 'unit', 'included_in_step03', 'record_count_step03', 'stay_count_step03'])
    write_csv(OUT / '11_step03_value_cleaning_audit.csv', value_audit_rows, ['database', 'variable', 'raw_record_count', 'valid_record_count', 'abnormal_record_count', 'temperature_converted_count', 'stay_count_with_valid_record'])
    count_df = pd.DataFrame(counts)
    c_counts = count_df[count_df['version'] == 'relaxed_5var_4of5_6h'].set_index('database')['included_n'].to_dict()
    recommendation = 'C_relaxed_5var_4of5_6h'
    if sum(c_counts.values()) == 0:
        recommendation = 'D_core_4var_no_temp_6h'
    write_report(counts, missingness, value_audit_rows, paths_used, item_rows_all, recommendation)
    log('所有第四步输出文件已生成')

    def get_counts(version):
        sub = count_df[count_df['version'] == version].set_index('database')['included_n'].to_dict()
        return {db: int(sub.get(db, 0)) for db in DATABASES}
    print('\n========== 终端最终摘要 ==========')
    print('1. 运行是否成功：成功')
    print('2. 是否完整重建三数据库候选 ICU stay，而不是只使用 preview：是，已完整重建；preview 仅用于核对')
    for idx, label in enumerate([spec['label'] for spec in VERSIONS.values()], 3):
        vals = get_counts(label)
        letter = list(VERSIONS.keys())[idx - 3].split('_', 1)[0]
        print(f"{idx}. 版本 {letter} {label} 人数：MIMIC-IV={vals['MIMIC-IV']}；MIMIC-III={vals['MIMIC-III']}；eICU={vals['eICU']}")
    print(f'9. 推荐作为 GBMTM 第一版候选：{recommendation}')
    print('10. 是否建议进入下一步 04_prepare_gbmtm_input_and_standardization.py：建议进入；下一步仍需只做标准化和建模输入准备，不在本步建模')
if __name__ == '__source_reference_run__':
    try:
        main()
    except Exception as exc:
        log(f'运行失败: {exc}')
        raise
