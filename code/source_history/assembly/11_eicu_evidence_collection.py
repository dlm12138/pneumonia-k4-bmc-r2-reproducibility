"""Actual conflict-evidence reconstruction; outputs private only.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'只读收集eICU结局证据；不拟合模型，不改变原始或冻结文件。'
import argparse
import hashlib
import io
import itertools
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

def quoted(path):
    return "'" + str(path).replace("'", "''") + "'"

def query(sql):
    result = subprocess.run(['duckdb', ':memory:', '-csv', '-c', sql], check=True, text=True, capture_output=True)
    return pd.read_csv(io.StringIO(result.stdout), low_memory=False)

def parquet(frame, target, temporary):
    staging = temporary / (target.stem + '.staging.tsv')
    frame.to_csv(staging, sep='\t', index=False)
    subprocess.run(['duckdb', ':memory:', '-c', f"COPY (SELECT * FROM read_csv({quoted(staging)}, delim='\t', header=true, sample_size=-1)) TO {quoted(target)} (FORMAT PARQUET)"], check=True, capture_output=True)
    actual = query(f'SELECT count(*) AS n FROM read_parquet({quoted(target)})').iloc[0, 0]
    assert int(actual) == len(frame), '导出行数不一致'
    staging.unlink()

def normal_status(series):
    return series.astype('string').str.strip().str.upper().replace({'DEAD': 'EXPIRED'})

def consensus(series):
    values = sorted(set(series.dropna()) & {'ALIVE', 'EXPIRED'})
    return values[0] if len(values) == 1 else 'CONFLICT' if len(values) > 1 else 'UNKNOWN'

def collect(args):
    root, raw, out = (Path(args.project), Path(args.raw), Path(args.output))
    hist = root / '01_k5-7.29_history/k5-7.29'
    endpoint = hist / 'analysis_main12_clinical_actionability_20260730/13_organ_support_escalation_patient_level.parquet'
    candidate = hist / 'analysis_landmark_external_upgrade_20260730/14_landmark12_candidate_matrix.parquet'
    old_fg = root / 'revision_R1_R2_data/20_competing_risk_input.csv'
    old_models = root / 'revision_R1_R2_data/21_competing_risk_models.csv'
    evidence = root / '10_second_round_revision_20261014/data_audit/private/04_eicu_patient_evidence.parquet'
    table, private, temporary = (out / 'tables', out / 'private', out / 'tmp')
    sources = [endpoint, candidate, old_fg, old_models, evidence, raw / 'patient.csv.gz', raw / 'apachePatientResult.csv.gz']
    sources += sorted((root / 'revision_R1_R2_data/locked_K4_model').glob('*.csv'))
    sources += [p for p in [Path(os.environ['SOURCE_2F37D9D736']), Path(os.environ['SOURCE_7C5C0EEAE0'])] if p.exists()]
    hashes = []
    for path in sources:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        hashes.append({'source': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
    pd.DataFrame(hashes).to_csv(table / '00_只读输入校验值.tsv', sep='\t', index=False)
    f = query(f"SELECT * EXCLUDE (subject_id,hadm_id,intime,outtime,deathtime,SUBJECT_ID_1,HADM_ID_1,DBSOURCE,INTIME_1,OUTTIME_1,DEATHTIME_1) FROM read_parquet({quoted(endpoint)}) WHERE database='eICU'")
    f['stay_id'] = pd.to_numeric(f.stay_id).astype('int64')
    assert f.stay_id.is_unique, '冻结结局键重复'
    cols = ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid', 'hospitalid', 'hospitaladmitoffset', 'hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitaldischargelocation', 'unitdischargeoffset', 'unitdischargestatus', 'unitdischargelocation', 'unitvisitnumber', 'unitstaytype']
    p = pd.read_csv(raw / 'patient.csv.gz', usecols=cols, low_memory=False)
    assert p.patientunitstayid.is_unique, '原始patient表键重复'
    f = f.merge(p.rename(columns={'patientunitstayid': 'stay_id'}), on='stay_id', validate='one_to_one', suffixes=('_frozen', ''))
    for col in ['unitdischargeoffset', 'hospitaldischargeoffset', 'unitdischargestatus', 'hospitaldischargestatus']:
        if col.endswith('offset'):
            assert np.isclose(f[col], f[col + '_frozen'], equal_nan=True).all(), col + '来源不一致'
        else:
            assert f[col].fillna('').eq(f[col + '_frozen'].fillna('')).all(), col + '来源不一致'
    f['unit_status'] = normal_status(f.unitdischargestatus)
    f['hospital_status'] = normal_status(f.hospitaldischargestatus)
    for prefix in ['unit', 'hospital']:
        loc = f[prefix + 'dischargelocation'].astype('string').str.strip().str.upper()
        f[prefix + '_location_death'] = loc.str.contains('DEATH|MORGUE|EXPIRED', na=False)
        f[prefix + '_location_live'] = loc.str.contains('HOME|FLOOR|TELEMETRY|STEP.DOWN|REHAB|NURSING|OTHER ICU|OTHER HOSPITAL', na=False) & ~f[prefix + '_location_death']
        f[prefix + '_location_conflict'] = (f[prefix + '_status'].eq('ALIVE') & f[prefix + '_location_death'] | f[prefix + '_status'].eq('EXPIRED') & f[prefix + '_location_live']).fillna(False)
    a = pd.read_csv(raw / 'apachePatientResult.csv.gz', usecols=['patientunitstayid', 'apacheversion', 'actualicumortality', 'actualhospitalmortality', 'unabridgedunitlos', 'unabridgedhosplos'], low_memory=False)
    a = a.loc[a.patientunitstayid.isin(f.stay_id)].copy()
    a['apache_unit'] = normal_status(a.actualicumortality)
    a['apache_hospital'] = normal_status(a.actualhospitalmortality)
    aa = a.groupby('patientunitstayid').agg(apache_rows=('apacheversion', 'size'), apache_versions=('apacheversion', lambda s: '|'.join(sorted(s.astype(str).unique()))), apache_unit=('apache_unit', consensus), apache_hospital=('apache_hospital', consensus), apache_unit_los_min=('unabridgedunitlos', 'min'), apache_unit_los_max=('unabridgedunitlos', 'max'), apache_hospital_los_min=('unabridgedhosplos', 'min'), apache_hospital_los_max=('unabridgedhosplos', 'max')).reset_index()
    f = f.merge(aa.rename(columns={'patientunitstayid': 'stay_id'}), on='stay_id', how='left', validate='one_to_one')
    for prefix in ['unit', 'hospital']:
        f['apache_' + prefix] = f['apache_' + prefix].fillna('UNKNOWN')
        f['apache_' + prefix + '_conflict'] = f['apache_' + prefix].isin(['ALIVE', 'EXPIRED']) & f['apache_' + prefix].ne(f[prefix + '_status']) | f['apache_' + prefix].eq('CONFLICT')
    related = p.loc[p.patienthealthsystemstayid.isin(f.patienthealthsystemstayid)].copy()
    related['icu_start_from_hospital'] = -related.hospitaladmitoffset
    related['icu_end_from_hospital'] = related.unitdischargeoffset - related.hospitaladmitoffset
    related['hospital_end_from_hospital'] = related.hospitaldischargeoffset - related.hospitaladmitoffset
    related['hospital_status'] = normal_status(related.hospitaldischargestatus)
    related['unit_status'] = normal_status(related.unitdischargestatus)
    groups = {key: group for key, group in related.groupby('patienthealthsystemstayid')}
    chronology = []
    for row in f.itertuples(index=False):
        group = groups[row.patienthealthsystemstayid]
        other = group.loc[group.patientunitstayid.ne(row.stay_id)]
        start, end = (-row.hospitaladmitoffset, row.unitdischargeoffset - row.hospitaladmitoffset)
        later = other.loc[other.icu_start_from_hospital.ge(end)]
        statuses = group.hospital_status.dropna().unique()
        chronology.append(dict(stay_id=row.stay_id, same_hospital_icu_records=len(group), next_icu_after_exit_N=len(later), next_icu_after_expired=bool(row.unit_status == 'EXPIRED' and len(later)), later_icu_alive_after_expired=bool(row.unit_status == 'EXPIRED' and later.unit_status.eq('ALIVE').any()), same_hospital_status_conflict=len(statuses) > 1, same_hospital_end_spread_minutes=float(group.hospital_end_from_hospital.max() - group.hospital_end_from_hospital.min()), later_icu_admission_before_unit_exit_N=int((other.icu_start_from_hospital.gt(start) & other.icu_start_from_hospital.lt(end)).sum())))
    f = f.merge(pd.DataFrame(chronology), on='stay_id', validate='one_to_one')
    f['unit_hospital_status_conflict'] = f.unit_status.eq('EXPIRED') & f.hospital_status.eq('ALIVE')
    f['hospital_exit_before_unit'] = f.hospitaldischargeoffset.lt(f.unitdischargeoffset)
    f['both_dead_hospital_exit_before_unit'] = f.hospital_exit_before_unit & f.unit_status.eq('EXPIRED') & f.hospital_status.eq('EXPIRED')
    f['unit_hospital_death_lag_minutes'] = (f.hospitaldischargeoffset - f.unitdischargeoffset).where(f.unit_status.eq('EXPIRED') & f.hospital_status.eq('EXPIRED'))
    f['strict_landmark'] = f.unitdischargeoffset.gt(720)
    f['unknown_status'] = ~f.unit_status.isin(['ALIVE', 'EXPIRED']) | ~f.hospital_status.isin(['ALIVE', 'EXPIRED'])
    f['core_conflict'] = f.unit_hospital_status_conflict | f.hospital_exit_before_unit | f.next_icu_after_expired | f.same_hospital_status_conflict | f.unknown_status
    f['corroboration_conflict'] = f.apache_unit_conflict | f.apache_hospital_conflict | f.unit_location_conflict | f.hospital_location_conflict
    f['evidence_grade'] = np.select([f.core_conflict, f.corroboration_conflict, f.apache_unit.isin(['ALIVE', 'EXPIRED']) & f.apache_hospital.isin(['ALIVE', 'EXPIRED'])], ['E4_核心冲突_不可视为临床真值', 'E3_内部一致但佐证来源不一致', 'E1_原表与佐证一致_仍非人工判定'], default='E2_原表内部一致_缺少部分佐证')
    f['suggested_unit_status'] = f.unit_status.where(~f.core_conflict, 'INDETERMINATE')
    f['suggested_hospital_status'] = f.hospital_status.where(~f.core_conflict, 'INDETERMINATE')
    f['suggested_clock'] = np.where(f.core_conflict, '时间或状态冲突_不补造死亡时刻', '使用各结局对应出院时点作为记录代理_非精确死亡时刻')
    fg = pd.read_csv(old_fg)
    fg = fg.loc[fg.database.eq('eICU')].copy()
    fg['stay_id'] = fg.stay_id.astype('int64')
    fg = fg.merge(f[['stay_id', 'unit_status']], on='stay_id', validate='many_to_one')
    sets = {'old_FG24_competing_ICU_expired': set(fg.loc[fg.horizon_hours.eq(24) & fg.status.eq(2) & fg.unit_status.eq('EXPIRED'), 'stay_id']), 'old_FG48_competing_ICU_expired': set(fg.loc[fg.horizon_hours.eq(48) & fg.status.eq(2) & fg.unit_status.eq('EXPIRED'), 'stay_id']), 'ICU_expired_hospital_alive': set(f.loc[f.unit_hospital_status_conflict, 'stay_id']), 'both_dead_hospital_before_ICU': set(f.loc[f.both_dead_hospital_exit_before_unit, 'stay_id']), 'unit_exit_exact_720': set(f.loc[f.unitdischargeoffset.eq(720), 'stay_id'])}
    unique = set.union(*sets.values())
    membership = []
    for names, group in itertools.groupby(sorted(unique), key=lambda key: tuple((name for name, ids in sets.items() if key in ids))):
        membership.append(dict(set_combination='|'.join(names), unique_stays=len(list(group))))
    membership = pd.DataFrame(membership).groupby('set_combination', as_index=False).unique_stays.sum()
    membership.to_csv(table / '01_冲突集合互斥分解.tsv', sep='\t', index=False)
    intersections = [dict(set_A=name, set_B=name, intersection_N=len(ids)) for name, ids in sets.items()]
    intersections += [dict(set_A=a_name, set_B=b_name, intersection_N=len(a_ids & b_ids)) for (a_name, a_ids), (b_name, b_ids) in itertools.combinations(sets.items(), 2)]
    intersections.append(dict(set_A='全部五类并集', set_B='唯一患者', intersection_N=len(unique)))
    pd.DataFrame(intersections).to_csv(table / '01_冲突集合交集及并集.tsv', sep='\t', index=False)
    for name, ids in sets.items():
        f[name] = f.stay_id.isin(ids)
    for horizon in [24, 48]:
        old = fg.loc[fg.horizon_hours.eq(horizon), ['stay_id', 'status', 'time_from_landmark']].rename(columns={'status': f'old_FG{horizon}_status', 'time_from_landmark': f'old_FG{horizon}_time_hours'})
        f = f.merge(old, on='stay_id', how='left', validate='one_to_one')
    f['any_target_set'] = f.stay_id.isin(unique)
    f['conflict_type'] = f.apply(lambda row: '|'.join((col for col in ['unit_hospital_status_conflict', 'hospital_exit_before_unit', 'next_icu_after_expired', 'same_hospital_status_conflict', 'apache_unit_conflict', 'apache_hospital_conflict', 'unit_location_conflict', 'hospital_location_conflict'] if bool(row[col]))) or '无上述冲突', axis=1)
    flags = query(f'SELECT stay_id,earliest_offset,any_long,any_long_by_12h,short_only FROM read_parquet({quoted(evidence)})')
    f = f.merge(flags, on='stay_id', validate='one_to_one')
    parquet(f, private / '全部可分型患者结局证据.parquet', temporary)
    selected = f.any_target_set | f.core_conflict | f.corroboration_conflict
    parquet(f.loc[selected], private / '死亡和出科冲突逐人审定结果.parquet', temporary)
    parquet(related, private / '相关同次住院全部ICU时序证据.parquet', temporary)
    parquet(a, private / '相关APACHE原始佐证.parquet', temporary)
    summaries = []
    for population, subset in [('全部可分型', f), ('五类目标并集', f.loc[f.any_target_set])]:
        for col in ['unit_hospital_status_conflict', 'both_dead_hospital_exit_before_unit', 'hospital_exit_before_unit', 'next_icu_after_expired', 'same_hospital_status_conflict', 'apache_unit_conflict', 'apache_hospital_conflict', 'unit_location_conflict', 'hospital_location_conflict', 'core_conflict', 'corroboration_conflict']:
            summaries.append(dict(population=population, flag=col, N=int(subset[col].sum()), denominator=len(subset)))
    pd.DataFrame(summaries).to_csv(table / '02_逐项来源核查汇总.tsv', sep='\t', index=False)
    f.groupby(['any_target_set', 'unit_status', 'hospital_status', 'apache_unit', 'apache_hospital'], dropna=False).agg(N=('stay_id', 'size')).reset_index().to_csv(table / '02_APACHE与原表状态交叉.tsv', sep='\t', index=False)
    f.groupby(['any_target_set', 'evidence_grade']).agg(N=('stay_id', 'size')).reset_index().to_csv(table / '02_证据等级.tsv', sep='\t', index=False)
    c = query(f"SELECT stay_id FROM read_parquet({quoted(candidate)}) WHERE database='eICU'")
    c['stay_id'] = c.stay_id.astype('int64')
    c = c.merge(p.rename(columns={'patientunitstayid': 'stay_id'}), on='stay_id', validate='one_to_one')
    assert len(c) == 17191 and len(f) == 15112, '冻结队列分母改变'
    pd.DataFrame([dict(population='landmark', old_N=len(c), strict_N=int(c.unitdischargeoffset.gt(720).sum())), dict(population='assigned', old_N=len(f), strict_N=int(f.strict_landmark.sum()))]).to_csv(table / '03_严格边界分母.tsv', sep='\t', index=False)
    print('证据收集完成：可分型N=', len(f), '；五类目标唯一人数=', len(unique), '；核心冲突=', int(f.core_conflict.sum()), '；佐证冲突=', int(f.corroboration_conflict.sum()), flush=True)
if __name__ == '__source_reference_run__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    collect(parser.parse_args())
