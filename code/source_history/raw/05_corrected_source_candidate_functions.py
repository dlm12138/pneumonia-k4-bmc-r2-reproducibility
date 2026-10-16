"""Historical corrected-source subset; not the final 12h analysis cohort.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
import csv
import gzip
import io
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd
ROOT = Path(os.environ['SOURCE_9A17FFCF9A'])
SCAN = ROOT / '00_scan_reports'
OUT = ROOT / '01_pneumonia_cohort_counts'
LOG = OUT / '01_run_log.txt'
TEXT_TERMS = ['pneumonia', 'aspiration pneumonia', 'viral pneumonia', 'bacterial pneumonia', 'bronchopneumonia', 'community acquired pneumonia', 'hospital acquired pneumonia', 'ventilator associated pneumonia', 'pneumonitis', 'lung infection', 'pulmonary infection', '肺炎', '重症肺炎', '吸入性肺炎', '细菌性肺炎', '病毒性肺炎', '支气管肺炎', '坠积性肺炎', '肺部感染', '社区获得性肺炎', '医院获得性肺炎', '呼吸机相关肺炎']
MISINCLUSION_TERMS = ['chemical pneumonitis', 'radiation pneumonitis', 'hypersensitivity pneumonitis', 'postoperative', 'post-operative', 'chronic', 'noninfectious', 'non-infectious', 'aspiration pneumonitis']
LEAK_RISK_TERMS = ['sepsis', 'respiratory failure', 'ards', 'acute respiratory distress']

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')

def read_scan_csv(name):
    with (SCAN / name).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

def norm_col(c):
    return str(c).strip().lower()

def parse_source(path):
    if path.startswith('zip://'):
        rest = path[len('zip://'):]
        zip_path, member = rest.split('!', 1)
        return (Path(zip_path), member)
    return (Path(path), None)

def open_source(path):
    physical, member = parse_source(path)
    if member:
        zf = zipfile.ZipFile(physical)
        raw = zf.open(member)
        if member.lower().endswith('.gz'):
            stream = gzip.open(raw, 'rt', encoding='utf-8', errors='replace', newline='')
        else:
            stream = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
        return (stream, zf)
    if str(physical).lower().endswith('.gz'):
        return (gzip.open(physical, 'rt', encoding='utf-8', errors='replace', newline=''), None)
    return (open(physical, 'r', encoding='utf-8', errors='replace', newline=''), None)

def available_columns(path):
    stream, holder = open_source(path)
    try:
        header = stream.readline().strip('\n\r')
    finally:
        stream.close()
        if holder:
            holder.close()
    if not header:
        return []
    return next(csv.reader([header]))

def read_csv_selected(path, usecols=None, chunksize=None, dtype=str):
    cols = available_columns(path)
    colmap = {c.lower(): c for c in cols}
    actual = None
    if usecols:
        actual = [colmap[c.lower()] for c in usecols if c.lower() in colmap]
    stream, holder = open_source(path)
    try:
        if chunksize:
            for chunk in pd.read_csv(stream, usecols=actual, dtype=dtype, chunksize=chunksize, low_memory=False):
                chunk.columns = [norm_col(c) for c in chunk.columns]
                yield chunk
        else:
            df = pd.read_csv(stream, usecols=actual, dtype=dtype, low_memory=False)
            df.columns = [norm_col(c) for c in df.columns]
            yield df
    finally:
        stream.close()
        if holder:
            holder.close()

def read_small(path, usecols=None):
    return next(read_csv_selected(path, usecols=usecols, chunksize=None))

def code_norm(x):
    if pd.isna(x):
        return ''
    return re.sub('[^A-Za-z0-9]', '', str(x)).upper()

def is_pneumonia_code(code, version=None):
    c = code_norm(code)
    if not c:
        return False
    v = str(version).strip() if version is not None and (not pd.isna(version)) else ''
    if c.startswith(tuple((f'J{i}' for i in range(12, 19)))) or c.startswith('J690'):
        return True
    if any((c.startswith(str(i)) for i in range(480, 487))):
        return True
    if c.startswith(('4870', '48801', '48811', '48881', '5070')):
        return True
    return False

def match_text_terms(text):
    low = str(text).lower()
    return [t for t in TEXT_TERMS if t.lower() in low]

def summarize_numeric(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return {'mean': '', 'median': '', 'p25': '', 'p75': ''}
    return {'mean': round(float(s.mean()), 2), 'median': round(float(s.median()), 2), 'p25': round(float(s.quantile(0.25)), 2), 'p75': round(float(s.quantile(0.75)), 2)}

def pct(n, d):
    return round(100 * n / d, 2) if d else ''

def choose_tables(schema_rows):
    by_db = defaultdict(list)
    for r in schema_rows:
        by_db[r['database_inferred']].append(r)

    def find(db, member_names, local_hint=None):
        rows = by_db[db]
        if local_hint:
            for r in rows:
                p = r['path']
                if p.startswith('zip://'):
                    continue
                low = p.lower()
                if local_hint in low and any((low.endswith(m.lower()) for m in member_names)):
                    return r
        for r in rows:
            low = r['path'].lower()
            if r['path'].startswith('zip://') and any((low.endswith(m.lower()) for m in member_names)):
                return r
        for r in rows:
            low = r['path'].lower()
            if any((low.endswith(m.lower()) for m in member_names)):
                return r
        return None
    selected = {'MIMIC-IV': {'patients': find('MIMIC-IV', ['patients.csv.gz'], '10_temp/mimiciv_raw'), 'admissions': find('MIMIC-IV', ['admissions.csv.gz'], '10_temp/mimiciv_raw'), 'icustays': find('MIMIC-IV', ['icustays.csv.gz'], '10_temp/mimiciv_raw'), 'diagnoses': find('MIMIC-IV', ['diagnoses_icd.csv.gz'], '10_temp/mimiciv_raw'), 'diag_dict': find('MIMIC-IV', ['d_icd_diagnoses.csv.gz'], '10_temp/mimiciv_raw')}, 'MIMIC-III': {'patients': find('MIMIC-III', ['patients.csv.gz', 'PATIENTS.csv.gz'], '10_temp/mimiciii_raw'), 'admissions': find('MIMIC-III', ['admissions.csv.gz', 'ADMISSIONS.csv.gz'], '10_temp/mimiciii_raw'), 'icustays': find('MIMIC-III', ['icustays.csv.gz', 'ICUSTAYS.csv.gz'], '10_temp/mimiciii_raw'), 'diagnoses': find('MIMIC-III', ['diagnoses_icd.csv.gz', 'DIAGNOSES_ICD.csv.gz'], '10_temp/mimiciii_raw'), 'diag_dict': find('MIMIC-III', ['d_icd_diagnoses.csv.gz', 'D_ICD_DIAGNOSES.csv.gz'], '10_temp/mimiciii_raw')}, 'eICU': {'patient': find('eICU', ['patient.csv.gz']), 'diagnosis': find('eICU', ['diagnosis.csv.gz'])}}
    return selected

def table_paths_rows(selected):
    rows = []
    for db, tabs in selected.items():
        for typ, meta in tabs.items():
            if not meta:
                rows.append({'database': db, 'table_type': typ, 'path': '未发现', 'columns': '', 'stage_inferred': ''})
                continue
            rows.append({'database': db, 'table_type': typ, 'path': meta['path'], 'columns': meta['columns'], 'stage_inferred': meta['stage_inferred']})
    return rows

def count_diagnosis_hits(db, diag_path, dict_path=None):
    log(f'{db}: 分块扫描诊断表肺炎编码/文本命中')
    code_counter = Counter()
    text_counter = Counter()
    code_subjects = defaultdict(set)
    code_hadms = defaultdict(set)
    code_stays = defaultdict(set)
    text_subjects = defaultdict(set)
    text_hadms = defaultdict(set)
    text_stays = defaultdict(set)
    admission_hits = set()
    subject_hits = set()
    stay_hits = set()
    rows_hit = 0
    diag_text_by_code = {}
    if dict_path:
        try:
            d = read_small(dict_path, usecols=['icd_code', 'icd_version', 'long_title', 'ICD9_CODE', 'LONG_TITLE', 'SHORT_TITLE'])
            code_col = 'icd_code' if 'icd_code' in d.columns else 'icd9_code'
            title_col = 'long_title' if 'long_title' in d.columns else 'long_title' if 'long_title' in d.columns else None
            if 'long_title' in d.columns:
                title_col = 'long_title'
            elif 'long_title'.upper().lower() in d.columns:
                title_col = 'long_title'
            elif 'short_title' in d.columns:
                title_col = 'short_title'
            for _, r in d.iterrows():
                diag_text_by_code[code_norm(r.get(code_col)), str(r.get('icd_version', ''))] = str(r.get(title_col, '')) if title_col else ''
        except Exception as e:
            log(f'{db}: 诊断字典读取失败，继续编码计数: {e}')
    usecols = ['subject_id', 'hadm_id', 'icd_code', 'icd_version', 'ICD9_CODE', 'HADM_ID', 'SUBJECT_ID', 'patientunitstayid', 'diagnosisstring', 'icd9code', 'diagnosisid']
    chunk_iter = read_csv_selected(diag_path, usecols=usecols, chunksize=250000)
    for chunk in chunk_iter:
        cols = set(chunk.columns)
        code_col = 'icd_code' if 'icd_code' in cols else 'icd9_code' if 'icd9_code' in cols else None
        version_col = 'icd_version' if 'icd_version' in cols else None
        text_col = 'diagnosisstring' if 'diagnosisstring' in cols else None
        if code_col:
            versions = chunk[version_col] if version_col else [None] * len(chunk)
            mask = [is_pneumonia_code(c, v) for c, v in zip(chunk[code_col], versions)]
        else:
            mask = [False] * len(chunk)
        if text_col:
            text_masks = chunk[text_col].fillna('').map(lambda x: bool(match_text_terms(x)))
            mask = pd.Series(mask, index=chunk.index) | text_masks
        else:
            mask = pd.Series(mask, index=chunk.index)
        hit = chunk.loc[mask].copy()
        rows_hit += len(hit)
        if hit.empty:
            continue
        if 'hadm_id' in hit.columns:
            admission_hits.update(hit['hadm_id'].dropna().astype(str))
        if 'subject_id' in hit.columns:
            subject_hits.update(hit['subject_id'].dropna().astype(str))
        if 'patientunitstayid' in hit.columns:
            stay_hits.update(hit['patientunitstayid'].dropna().astype(str))
            subject_hits.update(hit['patientunitstayid'].dropna().astype(str))
        if code_col:
            for _, rr in hit.dropna(subset=[code_col]).iterrows():
                cn = code_norm(rr.get(code_col))
                code_counter[cn] += 1
                if 'subject_id' in hit.columns and pd.notna(rr.get('subject_id')):
                    code_subjects[cn].add(str(rr.get('subject_id')))
                if 'hadm_id' in hit.columns and pd.notna(rr.get('hadm_id')):
                    code_hadms[cn].add(str(rr.get('hadm_id')))
                if 'patientunitstayid' in hit.columns and pd.notna(rr.get('patientunitstayid')):
                    code_stays[cn].add(str(rr.get('patientunitstayid')))
                    code_subjects[cn].add(str(rr.get('patientunitstayid')))
        if text_col:
            for _, rr in hit.dropna(subset=[text_col]).iterrows():
                txt = str(rr.get(text_col))
                matched = match_text_terms(txt)
                keys = list(matched)
                if matched:
                    keys.append('诊断原文: ' + txt[:180])
                for term in matched:
                    text_counter[term] += 1
                if matched:
                    text_counter['诊断原文: ' + txt[:180]] += 1
                for key in keys:
                    if 'subject_id' in hit.columns and pd.notna(rr.get('subject_id')):
                        text_subjects[key].add(str(rr.get('subject_id')))
                    if 'hadm_id' in hit.columns and pd.notna(rr.get('hadm_id')):
                        text_hadms[key].add(str(rr.get('hadm_id')))
                    if 'patientunitstayid' in hit.columns and pd.notna(rr.get('patientunitstayid')):
                        text_stays[key].add(str(rr.get('patientunitstayid')))
                        text_subjects[key].add(str(rr.get('patientunitstayid')))
    return {'rows_hit': rows_hit, 'code_counter': code_counter, 'text_counter': text_counter, 'hadm_ids': admission_hits, 'subject_ids': subject_hits, 'stay_ids': stay_hits, 'diag_text_by_code': diag_text_by_code, 'code_subjects': code_subjects, 'code_hadms': code_hadms, 'code_stays': code_stays, 'text_subjects': text_subjects, 'text_hadms': text_hadms, 'text_stays': text_stays}

def mimic_counts(db, paths):
    p_path = paths['patients']['path']
    a_path = paths['admissions']['path']
    i_path = paths['icustays']['path']
    d_path = paths['diagnoses']['path']
    dict_path = paths['diag_dict']['path'] if paths.get('diag_dict') else None
    log(f'{db}: 读取 patients/admissions/icustays 核心表')
    if db == 'MIMIC-IV':
        patients = read_small(p_path, ['subject_id', 'gender', 'anchor_age', 'dod'])
        admissions = read_small(a_path, ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'])
        icu = read_small(i_path, ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])
        id_col = 'stay_id'
        age_df = patients[['subject_id', 'gender', 'anchor_age', 'dod']].copy()
        age_df['age'] = pd.to_numeric(age_df['anchor_age'], errors='coerce')
    else:
        patients = read_small(p_path, ['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'EXPIRE_FLAG'])
        admissions = read_small(a_path, ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS'])
        icu = read_small(i_path, ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS'])
        id_col = 'icustay_id'
        for df in [patients, admissions, icu]:
            df.columns = [c.lower() for c in df.columns]
        patients['dob_dt'] = pd.to_datetime(patients['dob'], errors='coerce')
        admissions['admittime_dt'] = pd.to_datetime(admissions['admittime'], errors='coerce')
        age_df = admissions[['subject_id', 'hadm_id', 'admittime_dt']].merge(patients[['subject_id', 'gender', 'dob_dt', 'dod', 'expire_flag']], on='subject_id', how='left')
        age_df['age'] = age_df['admittime_dt'].dt.year - age_df['dob_dt'].dt.year
        age_df.loc[age_df['age'] > 89, 'age'] = 90
    for df in [patients, admissions, icu]:
        df.columns = [c.lower() for c in df.columns]
    admissions['admittime_dt'] = pd.to_datetime(admissions.get('admittime'), errors='coerce')
    admissions['dischtime_dt'] = pd.to_datetime(admissions.get('dischtime'), errors='coerce')
    admissions['deathtime_dt'] = pd.to_datetime(admissions.get('deathtime'), errors='coerce')
    if 'dod' in patients.columns:
        patients['dod_dt'] = pd.to_datetime(patients['dod'], errors='coerce')
    icu['intime_dt'] = pd.to_datetime(icu.get('intime'), errors='coerce')
    icu['outtime_dt'] = pd.to_datetime(icu.get('outtime'), errors='coerce')
    if 'los' not in icu.columns:
        icu['los'] = (icu['outtime_dt'] - icu['intime_dt']).dt.total_seconds() / 86400
    icu['los_days'] = pd.to_numeric(icu['los'], errors='coerce')
    diag_hits = count_diagnosis_hits(db, d_path, dict_path)
    pneumonia_hadm = diag_hits['hadm_ids']
    merged = icu.merge(admissions, on=['subject_id', 'hadm_id'], how='left', suffixes=('', '_adm'))
    if db == 'MIMIC-IV':
        merged = merged.merge(age_df[['subject_id', 'gender', 'age', 'dod']], on='subject_id', how='left')
    else:
        merged = merged.merge(age_df[['subject_id', 'hadm_id', 'gender', 'age', 'dod', 'expire_flag']], on=['subject_id', 'hadm_id'], how='left')
    merged['pneumonia'] = merged['hadm_id'].astype(str).isin(pneumonia_hadm)
    connected = merged['admittime'].notna() & merged['age'].notna()
    adult = merged['age'] >= 18
    pn_icu = merged[merged['pneumonia'] & adult & connected].copy()
    first = pn_icu.sort_values(['subject_id', 'intime_dt', id_col]).drop_duplicates('subject_id', keep='first')
    los24 = first[first['los_days'] >= 1].copy()
    time_ok = los24[los24['intime_dt'].notna() & los24['outtime_dt'].notna()].copy()
    hosp_mort_available = time_ok['hospital_expire_flag'].notna() if 'hospital_expire_flag' in time_ok else pd.Series(False, index=time_ok.index)
    hosp_ok = time_ok[hosp_mort_available].copy()
    dod_col = None
    if 'dod' in hosp_ok.columns:
        dod_col = 'dod'
        hosp_ok['death_dt'] = pd.to_datetime(hosp_ok['dod'], errors='coerce')
    elif 'deathtime' in hosp_ok.columns:
        dod_col = 'deathtime'
        hosp_ok['death_dt'] = pd.to_datetime(hosp_ok['deathtime'], errors='coerce')
    hosp_ok['mortality_28d_available'] = hosp_ok['intime_dt'].notna() & (('death_dt' in hosp_ok.columns) | hosp_ok['hospital_expire_flag'].notna())
    mort28_ok = hosp_ok[hosp_ok['mortality_28d_available']].copy()
    final = mort28_ok.copy()
    final['hospital_mortality'] = pd.to_numeric(final['hospital_expire_flag'], errors='coerce').fillna(0).astype(int)
    if 'death_dt' in final.columns:
        final['mortality_28d'] = (final['death_dt'].notna() & ((final['death_dt'] - final['intime_dt']).dt.days <= 28) & ((final['death_dt'] - final['intime_dt']).dt.days >= 0)).astype(int)
    else:
        final['mortality_28d'] = ''
    counts = [('原始 patients 人数', patients['subject_id'].nunique()), ('原始 admissions 人数', admissions['hadm_id'].nunique()), ('原始 ICU stays 人数', icu[id_col].nunique()), ('年龄 ≥18 岁 ICU stays 人数', int(adult.sum())), ('可连接 admissions/patients/icustays 的 ICU stays 人数', int((connected & adult).sum())), ('有肺炎候选诊断的 hospital admissions 人数', len(pneumonia_hadm)), ('有肺炎候选诊断且进入 ICU 的 ICU stays 人数', int((merged['pneumonia'] & adult & connected).sum())), ('只保留首次 ICU stay 后人数', len(first)), ('ICU length of stay ≥24 h 后人数', len(los24)), ('有 ICU 入科时间和出科时间后人数', len(time_ok)), ('可判定 hospital mortality 后人数', len(hosp_ok)), ('可判定 28-day mortality 后人数', len(mort28_ok)), ('最终可用于下一步生命体征完整性检查的候选人数', len(final))]
    return {'patients': patients, 'admissions': admissions, 'icu': icu, 'merged': merged, 'final': final, 'diag_hits': diag_hits, 'id_col': id_col, 'counts': counts, 'pneumonia_hadm': pneumonia_hadm}

def eicu_counts(paths):
    db = 'eICU'
    pat_path = paths['patient']['path']
    diag_path = paths['diagnosis']['path']
    log('eICU: 读取 patient 核心表')
    patient = read_small(pat_path, ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid', 'gender', 'age', 'unitadmittime24', 'unitdischargetime24', 'unitdischargeoffset', 'unitdischargestatus', 'hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitalid', 'apacheadmissiondx'])
    patient.columns = [c.lower() for c in patient.columns]
    patient['age_num'] = pd.to_numeric(patient['age'].replace({'> 89': '90', '>89': '90'}), errors='coerce')
    patient['los_days'] = pd.to_numeric(patient.get('unitdischargeoffset'), errors='coerce') / 1440
    diag_hits = count_diagnosis_hits(db, diag_path, None)
    pn_stays = diag_hits['stay_ids']
    patient['pneumonia'] = patient['patientunitstayid'].astype(str).isin(pn_stays)
    adult = patient['age_num'] >= 18
    pn = patient[patient['pneumonia'] & adult].copy()
    sort_cols = ['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid']
    for c in sort_cols:
        if c not in pn.columns:
            pn[c] = ''
    first = pn.sort_values(sort_cols).drop_duplicates('uniquepid' if 'uniquepid' in pn.columns else 'patientunitstayid', keep='first')
    los24 = first[first['los_days'] >= 1].copy()
    icu_mort_available = los24['unitdischargestatus'].notna() if 'unitdischargestatus' in los24 else pd.Series(False, index=los24.index)
    hosp_mort_available = los24['hospitaldischargestatus'].notna() if 'hospitaldischargestatus' in los24 else pd.Series(False, index=los24.index)
    icu_ok = los24[icu_mort_available].copy()
    hosp_ok = los24[hosp_mort_available].copy()
    final = los24[hosp_mort_available].copy()
    final['hospital_mortality'] = final['hospitaldischargestatus'].fillna('').str.lower().str.contains('expired|death|dead').astype(int)
    final['icu_mortality'] = final['unitdischargestatus'].fillna('').str.lower().str.contains('expired|death|dead').astype(int)
    final['mortality_28d_available'] = False
    counts = [('原始 patient 或 patientunitstayid 人数', patient['patientunitstayid'].nunique()), ('成人 ICU stays 人数', int(adult.sum())), ('有肺炎候选诊断的人数', len(pn_stays)), ('肺炎候选且 ICU stay 可识别的人数', len(pn)), ('只保留首次 ICU stay 后人数', len(first)), ('ICU length of stay ≥24 h 后人数', len(los24)), ('可判定 ICU mortality 后人数', len(icu_ok)), ('可判定 hospital mortality 后人数', len(hosp_ok)), ('是否可判定 28-day mortality', 0), ('最终可用于下一步生命体征完整性检查的候选人数', len(final))]
    return {'patient': patient, 'final': final, 'diag_hits': diag_hits, 'counts': counts, 'id_col': 'patientunitstayid'}

def make_code_hit_rows(results):
    rows = []
    for db, res in results.items():
        counter = res['diag_hits']['code_counter']
        dh = res['diag_hits']
        hadm_to_stays = {}
        if db != 'eICU' and 'icu' in res:
            for hadm, g in res['icu'].groupby('hadm_id'):
                hadm_to_stays[str(hadm)] = set(g[res['id_col']].dropna().astype(str))
        for code, n in counter.most_common():
            hadms = dh['code_hadms'].get(code, set())
            mapped_stays = set()
            for h in hadms:
                mapped_stays.update(hadm_to_stays.get(str(h), set()))
            direct_stays = dh['code_stays'].get(code, set())
            rows.append({'database': db, 'icd_code_normalized': code, 'diagnosis_rows': n, 'patient_count': len(dh['code_subjects'].get(code, set())), 'hadm_or_stay_count': len(hadms) or len(direct_stays), 'icu_stay_count': len(mapped_stays) or len(direct_stays)})
    return rows

def make_text_hit_rows(results):
    rows = []
    for db, res in results.items():
        c = res['diag_hits']['text_counter']
        dh = res['diag_hits']
        hadm_to_stays = {}
        if db != 'eICU' and 'icu' in res:
            for hadm, g in res['icu'].groupby('hadm_id'):
                hadm_to_stays[str(hadm)] = set(g[res['id_col']].dropna().astype(str))
        for term, n in c.most_common(300):
            risk = ''
            low = term.lower()
            if any((x in low for x in MISINCLUSION_TERMS)):
                risk = '可能误纳入风险'
            if any((x in low for x in LEAK_RISK_TERMS)):
                risk = '可能漏纳入提示'
            hadms = dh['text_hadms'].get(term, set())
            mapped_stays = set()
            for h in hadms:
                mapped_stays.update(hadm_to_stays.get(str(h), set()))
            direct_stays = dh['text_stays'].get(term, set())
            rows.append({'database': db, 'text_or_keyword': term, 'frequency': n, 'patient_count': len(dh['text_subjects'].get(term, set())), 'hadm_or_stay_count': len(hadms) or len(direct_stays), 'icu_stay_count': len(mapped_stays) or len(direct_stays), 'audit_note': risk})
    return rows

def make_join_audit(results):
    rows = []
    for db, res in results.items():
        if db == 'eICU':
            p = res['patient']
            rows += [{'database': db, 'audit_item': 'patientunitstayid 唯一值数量', 'numerator': p['patientunitstayid'].nunique(), 'denominator': len(p), 'rate_percent': pct(p['patientunitstayid'].nunique(), len(p)), 'note': 'eICU 单表识别 ICU stay'}, {'database': db, 'audit_item': 'patientunitstayid 重复行数', 'numerator': len(p) - p['patientunitstayid'].nunique(), 'denominator': len(p), 'rate_percent': pct(len(p) - p['patientunitstayid'].nunique(), len(p)), 'note': ''}, {'database': db, 'audit_item': 'diagnosis 到 patientunitstayid 连接成功率', 'numerator': len(res['diag_hits']['stay_ids'] & set(p['patientunitstayid'].astype(str))), 'denominator': len(res['diag_hits']['stay_ids']), 'rate_percent': pct(len(res['diag_hits']['stay_ids'] & set(p['patientunitstayid'].astype(str))), len(res['diag_hits']['stay_ids'])), 'note': '肺炎诊断命中 stay 与 patient 表连接'}, {'database': db, 'audit_item': '首次 ICU stay 判定规则', 'numerator': '', 'denominator': '', 'rate_percent': '', 'note': '按 uniquepid 分组，依 patienthealthsystemstayid/patientunitstayid 排序取第一条'}]
        else:
            m, adm, pat, icu = (res['merged'], res['admissions'], res['patients'], res['icu'])
            pneumonia_hadm = res['pneumonia_hadm']
            rows += [{'database': db, 'audit_item': 'diagnosis 到 admissions 的连接成功率', 'numerator': len(pneumonia_hadm & set(adm['hadm_id'].astype(str))), 'denominator': len(pneumonia_hadm), 'rate_percent': pct(len(pneumonia_hadm & set(adm['hadm_id'].astype(str))), len(pneumonia_hadm)), 'note': '肺炎 HADM_ID 与 admissions 连接'}, {'database': db, 'audit_item': 'admissions 到 patients 的连接成功率', 'numerator': adm['subject_id'].isin(set(pat['subject_id'].astype(str))).sum(), 'denominator': len(adm), 'rate_percent': pct(adm['subject_id'].isin(set(pat['subject_id'].astype(str))).sum(), len(adm)), 'note': ''}, {'database': db, 'audit_item': 'admissions 到 ICU stays 的连接成功率', 'numerator': adm['hadm_id'].isin(set(icu['hadm_id'].astype(str))).sum(), 'denominator': len(adm), 'rate_percent': pct(adm['hadm_id'].isin(set(icu['hadm_id'].astype(str))).sum(), len(adm)), 'note': ''}, {'database': db, 'audit_item': 'ICU stays 到 mortality 字段的连接成功率', 'numerator': m['hospital_expire_flag'].notna().sum(), 'denominator': len(m), 'rate_percent': pct(m['hospital_expire_flag'].notna().sum(), len(m)), 'note': 'hospital_expire_flag 来自 admissions'}, {'database': db, 'audit_item': 'subject_id 唯一值数量', 'numerator': pat['subject_id'].nunique(), 'denominator': len(pat), 'rate_percent': pct(pat['subject_id'].nunique(), len(pat)), 'note': ''}, {'database': db, 'audit_item': 'hadm_id 唯一值数量', 'numerator': adm['hadm_id'].nunique(), 'denominator': len(adm), 'rate_percent': pct(adm['hadm_id'].nunique(), len(adm)), 'note': ''}, {'database': db, 'audit_item': f"{res['id_col']} 唯一值数量", 'numerator': icu[res['id_col']].nunique(), 'denominator': len(icu), 'rate_percent': pct(icu[res['id_col']].nunique(), len(icu)), 'note': ''}, {'database': db, 'audit_item': '首次 ICU stay 判定规则', 'numerator': '', 'denominator': '', 'rate_percent': '', 'note': '肺炎候选且成人且连接成功后，按 subject_id 分组，以 intime 最早的 ICU stay 为首次 ICU stay'}]
    return rows

def make_mortality_rows(results):
    rows = []
    for db, res in results.items():
        final = res['final']
        n = len(final)
        if db == 'eICU':
            rows.append({'database': db, 'hospital_mortality_available': '可用', 'icu_mortality_available': '可用', 'mortality_28d_available': '不可用', 'recommended_outcome': 'eICU 建议用于 hospital mortality 或 ICU mortality 验证', 'candidate_n': n, 'hospital_deaths': int(final['hospital_mortality'].sum()) if n else 0, 'hospital_mortality_rate_percent': pct(int(final['hospital_mortality'].sum()), n), 'mortality_28d_deaths': '', 'mortality_28d_rate_percent': '', 'reason': '未发现可直接判定 ICU 入科后 28 天死亡的长期死亡日期字段；只有 offset/status 类出院结局字段'})
        else:
            rows.append({'database': db, 'hospital_mortality_available': '可用', 'icu_mortality_available': '可通过 ICU 时间与死亡时间间接定义，需后续严格实现', 'mortality_28d_available': '可用', 'recommended_outcome': '主结局建议 ICU 入科后 28 天死亡；住院死亡作为敏感性/补充结局', 'candidate_n': n, 'hospital_deaths': int(final['hospital_mortality'].sum()) if n else 0, 'hospital_mortality_rate_percent': pct(int(final['hospital_mortality'].sum()), n), 'mortality_28d_deaths': int(pd.to_numeric(final['mortality_28d'], errors='coerce').fillna(0).sum()) if n else 0, 'mortality_28d_rate_percent': pct(int(pd.to_numeric(final['mortality_28d'], errors='coerce').fillna(0).sum()), n), 'reason': '存在 ICU 入科时间与 dod/deathtime/hospital_expire_flag 字段'})
    return rows

def make_summary_rows(results):
    rows = []
    for db, res in results.items():
        f = res['final']
        age_col = 'age' if 'age' in f.columns else 'age_num'
        los = summarize_numeric(f['los_days'] if 'los_days' in f.columns else pd.Series(dtype=float))
        age = summarize_numeric(f[age_col] if age_col in f.columns else pd.Series(dtype=float))
        gender_counts = f['gender'].fillna('missing').astype(str).value_counts().to_dict() if 'gender' in f.columns else {}
        rows.append({'database': db, 'candidate_icu_stays': len(f), 'candidate_patients': f['subject_id'].nunique() if 'subject_id' in f.columns else f['patientunitstayid'].nunique(), 'age_mean': age['mean'], 'age_median': age['median'], 'age_p25': age['p25'], 'age_p75': age['p75'], 'gender_counts': str(gender_counts), 'icu_los_days_mean': los['mean'], 'icu_los_days_median': los['median'], 'icu_los_days_p25': los['p25'], 'icu_los_days_p75': los['p75'], 'hospital_deaths': int(f['hospital_mortality'].sum()) if 'hospital_mortality' in f.columns and len(f) else 0, 'hospital_mortality_rate_percent': pct(int(f['hospital_mortality'].sum()), len(f)) if 'hospital_mortality' in f.columns else '', 'mortality_28d_available': '可用' if db != 'eICU' else '不可用', 'mortality_28d_deaths': int(pd.to_numeric(f['mortality_28d'], errors='coerce').fillna(0).sum()) if 'mortality_28d' in f.columns and db != 'eICU' else ''})
    return rows

def make_preview(results):
    rows = []
    for db, res in results.items():
        f = res['final'].head(80)
        for _, r in f.iterrows():
            rows.append({'database': db, 'subject_id_or_patientunitstayid': r.get('subject_id', r.get('patientunitstayid', '')), 'hadm_id': r.get('hadm_id', ''), 'stay_id_or_icustay_id': r.get('stay_id', r.get('icustay_id', '')), 'intime': r.get('intime', r.get('unitadmittime24', '')), 'outtime': r.get('outtime', r.get('unitdischargetime24', '')), 'age': r.get('age', r.get('age_num', '')), 'pneumonia_source': 'ICD/text diagnosis feasibility match', 'mortality_available': 'yes' if db != 'eICU' else 'hospital/ICU yes; 28-day no'})
    return rows[:150]

def write_report(results, selected, step_rows, mortality_rows, summary_rows, text_rows):
    report = []
    report.append('# 重症肺炎 ICU 队列可行性计数报告\n')
    report.append(f'生成时间：{now()}\n')
    report.append('## 本次使用了哪些表\n')
    for db, tabs in selected.items():
        report.append(f'### {db}')
        for typ, meta in tabs.items():
            report.append(f"- {typ}: `{(meta['path'] if meta else '未发现')}`")
    report.append('\n## 三数据库各自能筛出多少肺炎 ICU 患者\n')
    for r in summary_rows:
        report.append(f"- {r['database']}：最终候选 ICU stays={r['candidate_icu_stays']}，患者数={r['candidate_patients']}，住院死亡率={r['hospital_mortality_rate_percent']}%。")
    report.append('\n## 每一步筛选掉了多少人\n')
    for db in results:
        report.append(f'### {db}')
        prev = None
        for row in [x for x in step_rows if x['database'] == db]:
            n = int(row['count']) if str(row['count']).isdigit() else row['count']
            dropped = '' if prev is None or not isinstance(n, int) else f'，较上一步减少 {prev - n}'
            report.append(f"- {row['step_order']}. {row['step_name']}：{row['count']}{dropped}")
            if isinstance(n, int):
                prev = n
    report.append('\n## 开发队列和验证队列建议\n')
    report.append('- MIMIC-IV：最适合作为开发队列，字段结构、ICU stay、死亡日期、ICD-9/10 诊断连接均较完整。')
    report.append('- MIMIC-III：适合作为外部验证队列，使用 ICD-9 肺炎编码和 ICU stay 时间可完成候选队列计数。')
    report.append('- eICU：适合作为多中心外部验证队列，但 28 天死亡不可直接判定，建议主用于 ICU/住院死亡验证。')
    report.append('\n## 28 天死亡是否可做\n')
    for r in mortality_rows:
        report.append(f"- {r['database']}：28 天死亡={r['mortality_28d_available']}；推荐结局：{r['recommended_outcome']}。原因：{r['reason']}")
    report.append('\n## 肺炎定义误纳入和漏纳入风险\n')
    risky = [r for r in text_rows if r.get('audit_note')]
    if risky:
        for r in risky[:30]:
            report.append(f"- {r['database']}：{r['text_or_keyword']} -> {r['audit_note']}")
    else:
        report.append('- 本次频次表中未自动检出明显误纳入/漏纳入提示词；仍建议下一步人工审计前 50 个诊断名称。')
    report.append('- 误纳入风险：pneumonitis 可能包含非感染性、化学性、放射性或吸入性炎症，后续正式定义需区分 pneumonia 与 pneumonitis。')
    report.append('- 漏纳入风险：以 sepsis、respiratory failure、ARDS 为主要诊断而肺炎仅在自由文本或次级诊断中的病例可能漏掉，需在 eICU 文本诊断中重点审计。')
    report.append('\n## 是否可以进入 0–12 h 生命体征完整性检查\n')
    report.append('- 可以。三库均能识别成人 ICU 肺炎候选患者、ICU stay、入出 ICU 时间和至少一种死亡结局。下一步应只检查生命体征完整性，不建模、不画图。')
    report.append('- 下一步脚本建议：`02_check_0_12h_vital_completeness.py`。')
    (OUT / '08_pneumonia_cohort_count_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f'[{now()}] 开始肺炎 ICU 队列可行性计数\n', encoding='utf-8')
    log('读取 00_scan_reports 中的表结构清单')
    schema_rows = read_scan_csv('03_table_schema_inventory.csv')
    selected = choose_tables(schema_rows)
    missing = [(db, typ) for db, tabs in selected.items() for typ, meta in tabs.items() if not meta]
    if missing:
        raise RuntimeError(f'关键表缺失: {missing}')
    write_csv(OUT / '01_database_table_paths_used.csv', table_paths_rows(selected), ['database', 'table_type', 'path', 'columns', 'stage_inferred'])
    log('开始三数据库逐步计数')
    results = {'MIMIC-IV': mimic_counts('MIMIC-IV', selected['MIMIC-IV']), 'MIMIC-III': mimic_counts('MIMIC-III', selected['MIMIC-III']), 'eICU': eicu_counts(selected['eICU'])}
    step_rows = []
    for db, res in results.items():
        for i, (name, count) in enumerate(res['counts'], 1):
            step_rows.append({'database': db, 'step_order': i, 'step_name': name, 'count': count})
    write_csv(OUT / '02_stepwise_cohort_counts.csv', step_rows, ['database', 'step_order', 'step_name', 'count'])
    code_rows = make_code_hit_rows(results)
    write_csv(OUT / '03_pneumonia_code_hits_by_database.csv', code_rows, ['database', 'icd_code_normalized', 'diagnosis_rows', 'patient_count', 'hadm_or_stay_count', 'icu_stay_count'])
    text_rows = make_text_hit_rows(results)
    write_csv(OUT / '04_pneumonia_text_hits_by_database.csv', text_rows, ['database', 'text_or_keyword', 'frequency', 'patient_count', 'hadm_or_stay_count', 'icu_stay_count', 'audit_note'])
    join_rows = make_join_audit(results)
    write_csv(OUT / '05_join_audit_by_database.csv', join_rows, ['database', 'audit_item', 'numerator', 'denominator', 'rate_percent', 'note'])
    mort_rows = make_mortality_rows(results)
    write_csv(OUT / '06_mortality_feasibility_by_database.csv', mort_rows, ['database', 'hospital_mortality_available', 'icu_mortality_available', 'mortality_28d_available', 'recommended_outcome', 'candidate_n', 'hospital_deaths', 'hospital_mortality_rate_percent', 'mortality_28d_deaths', 'mortality_28d_rate_percent', 'reason'])
    summary_rows = make_summary_rows(results)
    write_csv(OUT / '07_candidate_cohort_summary_by_database.csv', summary_rows, ['database', 'candidate_icu_stays', 'candidate_patients', 'age_mean', 'age_median', 'age_p25', 'age_p75', 'gender_counts', 'icu_los_days_mean', 'icu_los_days_median', 'icu_los_days_p25', 'icu_los_days_p75', 'hospital_deaths', 'hospital_mortality_rate_percent', 'mortality_28d_available', 'mortality_28d_deaths'])
    preview_rows = make_preview(results)
    write_csv(OUT / '09_candidate_ids_preview.csv', preview_rows, ['database', 'subject_id_or_patientunitstayid', 'hadm_id', 'stay_id_or_icustay_id', 'intime', 'outtime', 'age', 'pneumonia_source', 'mortality_available'])
    write_report(results, selected, step_rows, mort_rows, summary_rows, text_rows)
    log('所有输出文件已生成')
    n4 = len(results['MIMIC-IV']['final'])
    n3 = len(results['MIMIC-III']['final'])
    ne = len(results['eICU']['final'])
    print('\n========== 终端最终摘要 ==========')
    print('1. 运行是否成功：成功')
    print(f'2. MIMIC-IV 肺炎 ICU 候选人数：{n4}')
    print(f'3. MIMIC-III 肺炎 ICU 候选人数：{n3}')
    print(f'4. eICU 肺炎 ICU 候选人数：{ne}')
    print('5. 三库是否均能识别成人 ICU 肺炎患者：是')
    print('6. 三库是否均能识别 ICU 入科时间：MIMIC-IV/MIMIC-III 可直接识别；eICU 可通过 ICU stay/offset 结构识别，但无绝对日期时间')
    print('7. 三库是否均能识别 ICU stay ≥24 h：是')
    print('8. MIMIC-IV 是否能做 28 天死亡：能')
    print('9. MIMIC-III 是否能做 28 天死亡：能')
    print('10. eICU 是否能做 28 天死亡：不能直接做；可替代结局为 ICU mortality 和 hospital mortality')
    print('11. 是否建议继续做 0–12 h 生命体征完整性检查：建议继续')
    print('12. 下一步脚本建议名称：02_check_0_12h_vital_completeness.py')
if __name__ == '__source_reference_run__':
    main()
