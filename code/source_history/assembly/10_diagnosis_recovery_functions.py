"""Actual flow, raw diagnosis recovery and pneumonia evidence calculations.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
from pathlib import Path
from datetime import datetime
import argparse
import ast
import hashlib
import io
import json
import math
import subprocess
import sys
import numpy as np
import pandas as pd
ROOT = Path(os.environ['SOURCE_7E9AB01024'])
OUT = Path(os.environ['OUTPUT_ROOT'])
TABLE = OUT / 'tables'
PRIVATE = OUT / 'private'
HIST = ROOT / '01_k5-7.29_history/k5-7.29'
MAIN = HIST / 'analysis_main12_clinical_actionability_20260730'
REV = ROOT / 'revision_R1_R2_data'
CAND = HIST / 'analysis_landmark_external_upgrade_20260730/14_landmark12_candidate_matrix.parquet'
END = MAIN / '13_organ_support_escalation_patient_level.parquet'
ASSIGN = MAIN / '07_common4_selected_assignments.parquet'
RAW_EXTRACT = Path(os.environ['SOURCE_31F1D1CB18'])
CLIN = Path(os.environ['SOURCE_C74F17AEDB'])
RAW = Path(os.environ['SOURCE_550D92AA9D'])
IV = RAW / 'MIMIC-IV_3.1/mimic-iv-3.1'
III = RAW / 'MIMIC-III_1.4/mimic-iii-clinical-database-1.4'
EU = RAW / 'eICU-CRD_2.0/EICU 2.0数据'
DUCK = os.environ.get('DUCKDB', 'duckdb')
DBS = ['MIMIC-IV', 'MIMIC-III', 'eICU']
VITALS = ['SBP', 'DBP', 'HR', 'RR']
FEATURES = [f'{v}_h{h}' for v in VITALS for h in range(12)]
COMORB = ['myocardial_infarction', 'heart_failure', 'cerebrovascular', 'chronic_pulmonary', 'diabetes', 'liver_disease', 'renal_disease', 'malignancy']
CHECKS = []

def log(text):
    print(datetime.now().isoformat(timespec='seconds'), text, flush=True)

def quote(path):
    return "'" + str(path).replace("'", "''") + "'"

def query(sql):
    p = subprocess.run([DUCK, ':memory:', '-csv', '-c', sql], text=True, capture_output=True, check=True)
    return pd.read_csv(io.StringIO(p.stdout), low_memory=False)

def parquet(path):
    return query('SELECT * FROM read_parquet(' + quote(path) + ')')

def save(frame, name):
    frame.to_csv(TABLE / (name + '.tsv'), sep='\t', index=False)

def private_save(frame, name):
    staging = OUT / 'tmp' / (name + '.staging.tsv')
    target = PRIVATE / (name + '.parquet')
    frame.to_csv(staging, sep='\t', index=False)
    sql = f"COPY (SELECT * FROM read_csv({quote(staging)}, delim='\t', header=true, sample_size=-1)) TO {quote(target)} (FORMAT PARQUET);"
    subprocess.run([DUCK, ':memory:', '-c', sql], check=True, capture_output=True)
    assert int(query(f'SELECT count(*) AS n FROM read_parquet({quote(target)})').iloc[0, 0]) == len(frame)
    staging.unlink()

def check(module, name, actual, expected, tolerance=0):
    passed = bool(abs(actual - expected) <= tolerance)
    CHECKS.append(dict(module=module, check=name, actual=actual, expected=expected, tolerance=tolerance, passed=passed))
    save(pd.DataFrame(CHECKS), '00_independent_checks')
    if not passed:
        raise RuntimeError(f'{module}: {name}: {actual} != {expected}')

def normalise(frame):
    frame = frame.copy()
    frame['database'] = frame.database.replace({'MIMIC-III MetaVision': 'MIMIC-III'})
    frame['stay_id'] = pd.to_numeric(frame.stay_id, errors='raise').astype('int64')
    assert not frame.duplicated(['database', 'stay_id']).any()
    return frame

def keys(frame):
    return set(zip(frame.database, frame.stay_id))

def model_fit(frame, extra=(), outcome='death_24h'):
    f = frame.dropna(subset=['phenotype', 'age', 'male', 'observed_dimensions_48d', outcome] + list(extra)).copy()
    if f.database.eq('eICU').all():
        f = f.dropna(subset=['hospital_id'])
    columns = ['const', 'P1', 'P3', 'P4', 'age10', 'male', 'coverage10'] + list(extra)
    x = np.column_stack([np.ones(len(f)), *[f.phenotype.eq(p).to_numpy(float) for p in ['P1', 'P3', 'P4']], f.age.to_numpy(float) / 10, f.male.to_numpy(float), f.observed_dimensions_48d.to_numpy(float) / 10, *[f[c].to_numpy(float) for c in extra]])
    y = f[outcome].to_numpy(float)
    b = np.zeros(x.shape[1])
    for iteration in range(100):
        eta = np.clip(x @ b, -40, 40)
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        h = x.T @ (x * w[:, None])
        score = x.T @ (y - p)
        step = np.linalg.solve(h, score)
        ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
        scale = 1.0
        while scale > 1e-08:
            trial = b + step * scale
            et = x @ trial
            if np.sum(y * et - np.logaddexp(0, et)) >= ll - 1e-10:
                break
            scale /= 2
        b = trial
        if np.max(np.abs(step * scale)) < 1e-10:
            break
    else:
        raise RuntimeError('Logistic model did not converge')
    p = 1 / (1 + np.exp(-np.clip(x @ b, -40, 40)))
    scores = x * (y - p)[:, None]
    bread = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    covariance_type = 'HC0 sandwich robust'
    if f.database.eq('eICU').all():
        _, inverse = np.unique(f.hospital_id, return_inverse=True)
        cluster_scores = np.zeros((inverse.max() + 1, x.shape[1]))
        np.add.at(cluster_scores, inverse, scores)
        g, n, k = (len(cluster_scores), len(f), x.shape[1])
        meat = cluster_scores.T @ cluster_scores * (g / (g - 1)) * ((n - 1) / (n - k))
        covariance_type = 'hospital-cluster sandwich, finite-sample correction'
    else:
        meat = scores.T @ scores
    cov = bread @ meat @ bread
    se = math.sqrt(cov[2, 2])
    max_score = float(np.max(np.abs(x.T @ (y - p))))
    if max_score > 1e-05:
        raise RuntimeError(f'Unsolved likelihood score {max_score}')
    return dict(N=len(f), events=int(y.sum()), OR=float(np.exp(b[2])), CI_low=float(np.exp(b[2] - 1.95996398454 * se)), CI_high=float(np.exp(b[2] + 1.95996398454 * se)), variance=covariance_type, maximum_score=max_score)

def flow(candidate):
    log('阶段2：队列逐人排除原因与原始年龄核对')
    flows = []
    parts = []
    for db, fn, patient, stay, age_col in [('MIMIC-IV', 'mimiciv_pneumonia_stays_raw.csv', 'subject_id', 'stay_id', 'age'), ('MIMIC-III', 'mimiciii_pneumonia_stays_raw.csv', 'SUBJECT_ID', 'ICUSTAY_ID', 'age'), ('eICU', 'eicu_pneumonia_stays_raw.csv', 'uniquepid', 'patientunitstayid', 'age_num')]:
        raw = pd.read_csv(RAW_EXTRACT / fn, low_memory=False)
        sort = [patient, 'patienthealthsystemstayid', 'unitvisitnumber', stay] if db == 'eICU' else [patient, 'INTIME' if db == 'MIMIC-III' else 'intime', stay]
        first = raw.sort_values(sort).drop_duplicates(patient).copy()
        age = pd.to_numeric(first[age_col], errors='coerce')
        if db == 'eICU':
            los = pd.to_numeric(first.unitdischargeoffset, errors='coerce') / 60
            death = pd.Series(np.where(first.hospital_mortality.eq(1), pd.to_numeric(first.hospitaldischargeoffset, errors='coerce') / 60, np.nan), index=first.index)
            available = first.hospitaldischargestatus.notna()
            source = pd.read_csv(EU / 'patient.csv.gz', usecols=['patientunitstayid', 'age', 'unitdischargeoffset'], low_memory=False).set_index('patientunitstayid')
            rawage = first[stay].map(pd.to_numeric(source.age.replace({'> 89': 90, '>89': 90}), errors='coerce'))
            check('flow', db + '_raw_age_exclusion_mismatch', int((age.ge(18) != rawage.ge(18)).sum()), 0)
            check('flow', db + '_raw_los_mismatch', int((first.unitdischargeoffset.to_numpy() != first[stay].map(source.unitdischargeoffset).to_numpy()).sum()), 0)
        else:
            icu_in = 'INTIME' if db == 'MIMIC-III' else 'intime'
            died = 'DEATHTIME' if db == 'MIMIC-III' else 'deathtime'
            los = pd.to_numeric(first.los_hours, errors='coerce')
            death = (pd.to_datetime(first[died]) - pd.to_datetime(first[icu_in])).dt.total_seconds() / 3600
            available = pd.Series(True, index=first.index)
            if db == 'MIMIC-III':
                patients = pd.read_csv(III / 'PATIENTS.csv.gz', usecols=['SUBJECT_ID', 'DOB']).set_index('SUBJECT_ID')
                rawage = (pd.to_datetime(first[icu_in]).dt.as_unit('s') - pd.to_datetime(first[patient].map(patients.DOB)).dt.as_unit('s')).dt.total_seconds() / (365.25 * 86400)
                check('flow', db + '_raw_age_exclusion_mismatch', int((age.ge(18) != rawage.ge(18)).sum()), 0)
        adult = age.ge(18)
        inicu = los.ge(12)
        alive = death.isna() | death.gt(12)
        reason = np.select([age.isna(), age.lt(18), adult & ~inicu, adult & inicu & ~alive, adult & inicu & alive & ~available], ['年龄缺失', '年龄小于18岁', 'ICU停留不足12小时', '停留至少12小时但12小时内死亡', '住院结局不可用'], default='纳入12小时landmark')
        p = pd.DataFrame(dict(database=db, stay_id=first[stay].astype('int64'), age=age, los_hours=los, death_hours=death, exclusion_reason=reason))
        actual = set(p.loc[p.exclusion_reason.eq('纳入12小时landmark'), 'stay_id'])
        expected = set(candidate.loc[candidate.database.eq(db), 'stay_id'])
        check('flow', db + '_landmark_key_difference', len(actual ^ expected), 0)
        stages = {'提取的肺炎ICU记录': len(raw), '每位患者首次ICU': len(first), '年龄小于18岁': int(age.lt(18).sum()), '年龄缺失': int(age.isna().sum()), '成年人': int(adult.sum()), 'ICU停留不足12小时': int((adult & ~inicu).sum()), '停留至少12小时但12小时内死亡': int((adult & inicu & ~alive).sum()), '住院结局不可用': int((adult & inicu & alive & ~available).sum()), '12小时landmark队列': len(actual), 'landmark队列ICU出科恰为12小时': int((p.exclusion_reason.eq('纳入12小时landmark') & los.eq(12)).sum())}
        flows.extend((dict(database=db, stage=k, N=v) for k, v in stages.items()))
        parts.append(p)
    save(pd.DataFrame(flows), '02_cohort_flow_reconciled')
    private_save(pd.concat(parts), '02_cohort_exclusion_reasons')
    log('阶段2通过：三库landmark成员集合与原始排除原因对账一致')

def literals(path, names):
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != set(names):
        raise RuntimeError('Required historical definitions missing')
    return found

def mimic_evidence(candidate, endpoint):
    definitions = literals(CLIN / '08_clinical_features.py', ['COMORBIDITY_ICD9', 'COMORBIDITY_ICD10'])
    pneumonia = literals(CLIN / '01_pneumonia_identification.py', ['ICD9_PNEUMONIA', 'ICD10_PNEUMONIA'])
    sources = []
    phenotypes = []
    missingness = []
    recovery = []
    codesummary = []
    models = []
    dictionary = []
    for version in [9, 10]:
        for variable, prefixes in definitions[f'COMORBIDITY_ICD{version}'].items():
            for prefix in prefixes:
                dictionary.append(dict(variable=variable, ICD_version=version, prefix=prefix, matching='原历史规则starts-with，保持标点，不擅自改变编码表'))
    save(pd.DataFrame(dictionary), '03_comorbidity_code_dictionary')
    for db, prefix, rawfile, hadmcol, codecol, versioncol, extracthadm in [('MIMIC-IV', 'mimiciv', IV / 'hosp/diagnoses_icd.csv.gz', 'hadm_id', 'icd_code', 'icd_version', 'hadm_id'), ('MIMIC-III', 'mimiciii', III / 'DIAGNOSES_ICD.csv.gz', 'HADM_ID', 'ICD9_CODE', None, 'HADM_ID')]:
        log('阶段3：' + db + '共病缺失与肺炎诊断证据原表核验')
        f = endpoint.loc[endpoint.database.eq(db)].copy()
        f['analysis_complete'] = f[['age', 'male', 'observed_dimensions_48d', 'death_24h']].notna().all(axis=1)
        f = f.loc[f.analysis_complete].copy()
        if db == 'MIMIC-III':
            f['hadm_id'] = f.HADM_ID_1
        f['hadm_id'] = pd.to_numeric(f.hadm_id, errors='raise').astype('int64')
        old = pd.read_csv(CLIN / (prefix + '_clinical_features.csv'), usecols=['stay_id'] + COMORB, low_memory=False)
        old['stay_id'] = pd.to_numeric(old.stay_id, errors='raise').astype('int64')
        old['old_source_row_present'] = True
        old = old.rename(columns={c: 'old_' + c for c in COMORB})
        f = f.merge(old, on='stay_id', how='left', validate='one_to_one')
        f['old_complete'] = f[['old_' + c for c in COMORB]].notna().all(axis=1)
        f['old_source_row_present'] = f.old_source_row_present.eq(True)
        extract = pd.read_csv(RAW_EXTRACT / (prefix + '_pneumonia_stays_raw.csv'), low_memory=False)
        relevant = set(pd.to_numeric(extract[extracthadm], errors='raise').astype('int64'))
        diag_parts = []
        use = [hadmcol, codecol] + ([versioncol] if versioncol else [])
        for chunk in pd.read_csv(rawfile, usecols=use, dtype={codecol: 'string'}, chunksize=400000, low_memory=False):
            diag_parts.append(chunk.loc[chunk[hadmcol].isin(relevant)].copy())
        diag = pd.concat(diag_parts, ignore_index=True).rename(columns={hadmcol: 'hadm_id', codecol: 'code'})
        diag['version'] = pd.to_numeric(diag[versioncol], errors='coerce') if versioncol else 9
        diag['code'] = diag.code.fillna('').str.strip().str.upper()
        counts = diag.groupby('hadm_id').size()
        valid = diag.loc[diag.code.ne('') & diag.version.isin([9, 10])].groupby('hadm_id').size()
        f['raw_diagnosis_rows'] = f.hadm_id.map(counts).fillna(0).astype(int)
        f['raw_valid_diagnosis_rows'] = f.hadm_id.map(valid).fillna(0).astype(int)
        f['missing_reason'] = np.select([~f.old_source_row_present, f.old_source_row_present & ~f.old_complete], ['历史临床特征表无该次ICU记录', '历史特征行存在但共病字段缺失'], default='旧共病完整')
        for c in COMORB:
            hit = pd.Series(False, index=diag.index)
            for version in [9, 10]:
                prefixes = tuple(definitions[f'COMORBIDITY_ICD{version}'][c])
                hit |= diag.version.eq(version) & diag.code.str.startswith(prefixes)
            ids = set(diag.loc[hit, 'hadm_id'])
            f['raw_' + c] = np.where(f.raw_valid_diagnosis_rows.gt(0), f.hadm_id.isin(ids).astype(float), np.nan)
            common = f.old_complete & f['raw_' + c].notna()
            recovery.append(dict(database=db, variable=c, comparable_N=int(common.sum()), old_positive_N=int(f.loc[common, 'old_' + c].sum()), raw_positive_N=int(f.loc[common, 'raw_' + c].sum()), discordance_N=int(f.loc[common, 'old_' + c].ne(f.loc[common, 'raw_' + c]).sum())))
        for group, sub in f.groupby('missing_reason'):
            missingness.append(dict(database=db, group=group, N=len(sub), deaths=int(sub.death_24h.sum()), mortality_percent=100 * sub.death_24h.mean(), raw_diagnosis_available_N=int(sub.raw_diagnosis_rows.gt(0).sum()), raw_valid_diagnosis_available_N=int(sub.raw_valid_diagnosis_rows.gt(0).sum()), median_age=sub.age.median(), P3_N=int(sub.phenotype.eq('P3').sum()), mean_coverage=sub.observed_dimensions_48d.mean()))
        for label, mask in [('旧共病完整', f.old_complete), ('旧共病缺失', ~f.old_complete)]:
            sub = f.loc[mask]
            for phenotype, part in sub.groupby('phenotype'):
                phenotypes.append(dict(database=db, missingness_group=label, phenotype=phenotype, N=len(part), events=int(part.death_24h.sum()), mortality_percent=100 * part.death_24h.mean()))
        for label, sub, extras in [('全队列基础', f, []), ('旧完整病例基础', f.loc[f.old_complete], []), ('旧完整病例加共病', f.loc[f.old_complete], ['old_' + c for c in COMORB]), ('原始诊断可核查全队列基础', f.loc[f.raw_valid_diagnosis_rows.gt(0)], []), ('原始诊断可核查全队列加共病', f.loc[f.raw_valid_diagnosis_rows.gt(0)], ['raw_' + c for c in COMORB])]:
            models.append(dict(database=db, model=label, comparison='P3 vs P2', analysis_status='审稿后定向敏感性核查', **model_fit(sub, extras)))
        sources.append(f[['database', 'stay_id', 'hadm_id', 'phenotype', 'death_24h', 'age', 'male', 'observed_dimensions_48d', 'old_source_row_present', 'old_complete', 'missing_reason', 'raw_diagnosis_rows', 'raw_valid_diagnosis_rows'] + ['old_' + c for c in COMORB] + ['raw_' + c for c in COMORB]])
        fullhit = pd.Series(False, index=diag.index)
        for version, prefixes in [(9, tuple([str(x) for x in range(480, 487)] + ['4870', '4880', '4881', '5070', '99731'])), (10, tuple(['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J690', 'J95851', 'U071', 'J100', 'J101', 'J110']))]:
            fullhit |= diag.version.eq(version) & (diag.code.isin(pneumonia[f'ICD{version}_PNEUMONIA']) | diag.code.str.startswith(prefixes))
        covid = diag.version.eq(10) & diag.code.str.startswith('U071')
        hitids = set(diag.loc[fullhit, 'hadm_id'])
        noncovid = set(diag.loc[fullhit & ~covid, 'hadm_id'])
        covidids = set(diag.loc[fullhit & covid, 'hadm_id'])
        staycol = 'stay_id' if db == 'MIMIC-IV' else 'ICUSTAY_ID'
        ev = extract[[staycol, extracthadm]].rename(columns={staycol: 'stay_id', extracthadm: 'hadm_id'}).copy()
        ev['database'] = db
        ev['qualifying_code'] = ev.hadm_id.isin(hitids)
        ev['non_covid_qualifying_code'] = ev.hadm_id.isin(noncovid)
        ev['U071_code'] = ev.hadm_id.isin(covidids)
        ev['U071_only'] = ev.U071_code & ~ev.non_covid_qualifying_code
        check('pneumonia', db + '_frozen_initial_no_qualifying_code', int((~ev.qualifying_code).sum()), 0)
        ev = ev.merge(endpoint.loc[endpoint.database.eq(db), ['stay_id', 'phenotype', 'death_24h']], on='stay_id', how='left', validate='one_to_one')
        candidateids = set(candidate.loc[candidate.database.eq(db), 'stay_id'])
        for stage, mask in [('原始首ICU提取', pd.Series(True, index=ev.index)), ('12小时landmark', ev.stay_id.isin(candidateids)), ('可分型', ev.phenotype.notna())]:
            s = ev.loc[mask]
            for label, flag in [('全部', pd.Series(True, index=s.index)), ('仅U071符合原纳入规则', s.U071_only), ('至少另一肺炎合格码', s.non_covid_qualifying_code)]:
                part = s.loc[flag]
                codesummary.append(dict(database=db, stage=stage, evidence_category=label, N=len(part), assignable_N=int(part.phenotype.notna().sum()), events_assigned=int(part.death_24h.sum())))
        private_save(ev, '03_' + prefix + '_pneumonia_code_evidence')
        log('阶段3：' + db + '完成；共病来源与单一代码证据已保存')
    save(pd.DataFrame(missingness), '03_comorbidity_missing_source_summary')
    save(pd.DataFrame(phenotypes), '03_comorbidity_missingness_by_phenotype')
    save(pd.DataFrame(recovery), '03_comorbidity_old_raw_concordance')
    save(pd.DataFrame(models), '03_comorbidity_targeted_models')
    save(pd.DataFrame(codesummary), '04_mimic_pneumonia_evidence')
    private_save(pd.concat(sources), '03_comorbidity_source_recovery')

def eicu_evidence(candidate, endpoint):
    log('阶段4：eICU原始文本逐行匹配特异性及记录时点审计')
    definitions = literals(CLIN / '01_pneumonia_identification.py', ['PNEUMONIA_KEYWORDS'])
    terms = [str(t).lower() for t in definitions['PNEUMONIA_KEYWORDS']]
    short = ['cap', 'hap', 'vap', 'lrti']
    long = [t for t in terms if t not in short]
    raw = pd.read_csv(RAW_EXTRACT / 'eicu_pneumonia_stays_raw.csv', low_memory=False)
    first = raw.sort_values(['uniquepid', 'patienthealthsystemstayid', 'unitvisitnumber', 'patientunitstayid']).drop_duplicates('uniquepid')
    selected = set(first.patientunitstayid)
    evidence = []
    contexts = []
    for filename, timecol in [('diagnosis.csv.gz', 'diagnosisoffset'), ('admissionDx.csv.gz', 'admitdxenteredoffset')]:
        for chunk in pd.read_csv(EU / filename, chunksize=250000, low_memory=False):
            chunk = chunk.loc[chunk.patientunitstayid.isin(selected)]
            cols = ['diagnosisstring'] if filename.startswith('diagnosis') else [c for c in chunk.columns if any((x in c.lower() for x in ['dx', 'diagnosis', 'admit']))]
            for column in cols:
                text = chunk[column].fillna('').astype(str).str.lower()
                original = pd.Series(False, index=text.index)
                islong = original.copy()
                isshort = original.copy()
                bounded = original.copy()
                for term in terms:
                    hit = text.str.contains(term, regex=False)
                    original |= hit
                    if term in long:
                        islong |= hit
                    else:
                        isshort |= hit
                for term in short:
                    bounded |= text.str.contains('\\b' + term + '\\b', regex=True)
                hit = chunk.loc[original, ['patientunitstayid', timecol]].copy().rename(columns={'patientunitstayid': 'stay_id', timecol: 'offset'})
                if len(hit) == 0:
                    continue
                hit['source'] = filename + ':' + column
                hit['long_term'] = islong.loc[original].to_numpy()
                hit['short_term'] = isshort.loc[original].to_numpy()
                hit['bounded_short'] = bounded.loc[original].to_numpy()
                evidence.append(hit)
                problematic = original & ~islong
                review = chunk.loc[problematic, ['patientunitstayid', timecol, column]].copy()
                review.columns = ['stay_id', 'offset', 'text']
                review['source'] = filename + ':' + column
                contexts.append(review)
    evidence = pd.concat(evidence, ignore_index=True)
    evidence['offset'] = pd.to_numeric(evidence.offset, errors='coerce')
    evidence['long_early'] = evidence.long_term & evidence.offset.le(720)
    per = evidence.groupby('stay_id').agg(qualifying_rows=('offset', 'size'), earliest_offset=('offset', 'min'), any_long=('long_term', 'any'), any_short=('short_term', 'any'), any_bounded_short=('bounded_short', 'any'), any_long_by_12h=('long_early', 'any')).reset_index()
    per['short_only'] = ~per.any_long & per.any_short
    per['incidental_short_only'] = per.short_only & ~per.any_bounded_short
    per['timing_category'] = np.select([per.earliest_offset.isna(), per.earliest_offset.lt(0), per.earliest_offset.le(720)], ['时间不明', 'ICU前记录', '0至12小时记录'], default='仅12小时后记录')
    per = first[['patientunitstayid']].rename(columns={'patientunitstayid': 'stay_id'}).merge(per, on='stay_id', how='left', validate='one_to_one')
    check('eicu_evidence', 'first_stay_no_qualifying_row', int(per.qualifying_rows.isna().sum()), 0)
    per['database'] = 'eICU'
    per = per.merge(endpoint.loc[endpoint.database.eq('eICU'), ['stay_id', 'phenotype', 'death_24h', 'age', 'male', 'hospital_id', 'observed_dimensions_48d']], on='stay_id', how='left', validate='one_to_one')
    candidateids = set(candidate.loc[candidate.database.eq('eICU'), 'stay_id'])
    rows = []
    for stage, mask in [('首次ICU记录', pd.Series(True, index=per.index)), ('12小时landmark', per.stay_id.isin(candidateids)), ('可分型', per.phenotype.notna()), ('主要模型', per[['phenotype', 'age', 'male', 'hospital_id', 'observed_dimensions_48d']].notna().all(axis=1))]:
        s = per.loc[mask]
        for label, flag in [('全部', pd.Series(True, index=s.index)), ('至少一条长词证据', s.any_long), ('仅短词命中无长词证据', s.short_only), ('仅嵌入式短词无独立词边界', s.incidental_short_only), ('12小时前有长词证据', s.any_long_by_12h)]:
            part = s.loc[flag]
            rows.append(dict(stage=stage, evidence_category=label, N=len(part), death24h_assigned=int(part.death_24h.sum()), P2_N=int(part.phenotype.eq('P2').sum()), P3_N=int(part.phenotype.eq('P3').sum())))
    timing = []
    landmark = per.loc[per.stay_id.isin(candidateids)]
    for category, s in landmark.groupby('timing_category'):
        timing.append(dict(timing_category=category, N=len(s), percent=100 * len(s) / len(landmark), assigned_N=int(s.phenotype.notna().sum()), events=int(s.death_24h.sum())))
    old = {'ICU前记录': 513, '0至12小时记录': 13561, '仅12小时后记录': 3117}
    for label, n in old.items():
        check('eicu_evidence', 'timing_' + label, int(landmark.timing_category.eq(label).sum()), n)
    save(pd.DataFrame(rows), '04_eicu_pneumonia_specificity')
    save(pd.DataFrame(timing), '04_eicu_pneumonia_timing')
    modelrows = []
    for label, mask in [('原始冻结主分析', pd.Series(True, index=per.index)), ('排除仅短词命中', per.any_long), ('仅记录在12小时前', per.earliest_offset.le(720)), ('仅12小时前长词证据', per.any_long_by_12h)]:
        s = per.loc[mask & per.phenotype.notna()]
        modelrows.append(dict(database='eICU', subset=label, analysis_status='审稿后探索性定义敏感性，不重新拟合K4', **model_fit(s)))
    save(pd.DataFrame(modelrows), '04_eicu_definition_sensitivity_models')
    private_save(per, '04_eicu_patient_evidence')
    private_save(pd.concat(contexts, ignore_index=True).drop_duplicates(), '04_eicu_short_term_contexts_for_manual_review')
    source_summary = evidence.groupby('source').agg(matching_rows=('stay_id', 'size'), unique_stays=('stay_id', 'nunique'), long_term_rows=('long_term', 'sum'), short_term_rows=('short_term', 'sum')).reset_index()
    save(source_summary, '04_eicu_matching_fields')
    log('阶段4通过：eICU文本证据、时点及冻结模型定义敏感性已完成')
