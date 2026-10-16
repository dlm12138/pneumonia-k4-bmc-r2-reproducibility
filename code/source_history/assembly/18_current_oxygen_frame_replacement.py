"""Actual full oxygen replacement including >=24-dimension additions.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'只补取低覆盖阈值新增者的外周血氧，并整体替换八个既有血氧汇总列。'
from pathlib import Path
import importlib.util, subprocess, hashlib, json, math
import numpy as np
import pandas as pd
P = Path(os.environ['SOURCE_7E9AB01024'])
O = P / '20_spo2_targeted_repair_20261015'
R = P / '14_pneumonia_specificity_revision_20261014'
A = P / '19_targeted_methods_source_check_20261015'

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def sql(x):
    return "'" + str(x).replace("'", "''") + "'"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
checks = []

def check(name, ok, detail=''):
    checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    pd.DataFrame(checks).to_csv(O / 'qc/01_血氧替换与不变性核对.tsv', sep='\t', index=False)
    if not ok:
        raise RuntimeError(name)
protected = list((P / '18_author_review_revision_20261014/deliverables').glob('*')) + list((P / '18_author_review_revision_20261014/figures').glob('*')) + list((P / 'revision_R1_R2_data/locked_K4_model').glob('*'))
protected += [R / 'private/修订三库全模块分析输入.parquet', A / 'private/仅当前应用队列_已纠正外周血氧替换列_未重算下游.parquet']
hp = O / 'qc/只读保护SHA256.tsv'
if not hp.exists():
    pd.DataFrame([dict(path=str(p), sha256=sha(p)) for p in protected if p.is_file()]).to_csv(hp, sep='\t', index=False)
src = pd.read_parquet(R / 'private/修订三库全模块分析输入.parquet')
src.stay_id = src.stay_id.astype(str)
members = pd.read_parquet(R / 'private/三库修订肺炎资格与纳入标记.parquet')
members.stay_id = members.stay_id.astype(str)
cp = P / '01_k5-7.29_history/k5-7.29/analysis_k4_evidence_deepening_20260801/01_candidate_outcomes_and_coverage.parquet'
c = pd.read_parquet(cp)
c.database = c.database.replace({'MIMIC-III MetaVision': 'MIMIC-III'})
c.stay_id = c.stay_id.astype(str)
features = pd.read_csv(P / 'revision_R1_R2_data/locked_K4_model/feature_order.csv').sort_values('feature_position').feature.tolist()
c = c.merge(members.loc[members.revised_landmark, ['database', 'stay_id']], on=['database', 'stay_id'], validate='one_to_one')
obs = c[features].notna()
each = np.column_stack([obs[[v + '_h' + str(h) for h in range(12)]].any(axis=1) for v in ['SBP', 'DBP', 'HR', 'RR']]).all(axis=1)
c = c.loc[obs.sum(axis=1).ge(24) & each].copy()
c['observed_dimensions_48d'] = c[features].notna().sum(axis=1)
check('24维候选键唯一', not c.duplicated(['database', 'stay_id']).any())
c.to_parquet(O / 'private/24维以上严格肺炎候选_原特征结局不变.parquet', index=False)
root = Path(os.environ['SOURCE_550D92AA9D'])
loc = {'MIMIC-IV': root / 'MIMIC-IV_3.1/mimic-iv-3.1/icu', 'MIMIC-III': root / 'MIMIC-III_1.4/mimic-iii-clinical-database-1.4', 'eICU': root / 'eICU-CRD_2.0/EICU 2.0数据'}
hourly = []
counts = []
for db in ['MIMIC-IV', 'MIMIC-III', 'eICU']:
    keys = set(c.loc[c.database.eq(db), 'stay_id'])
    oldkeys = set(src.loc[src.database.eq(db), 'stay_id'])
    added = keys - oldkeys
    kp = O / 'private' / (db + '_仅新增阈值患者键.parquet')
    pd.DataFrame({'stay_id': sorted(added)}).to_parquet(kp, index=False)
    hpnew = O / 'private' / (db + '_新增者外周血氧小时项目汇总.parquet')
    pure = [0] if db == 'eICU' else [220277] if db == 'MIMIC-IV' else [646, 220277]
    if added and (not hpnew.exists()):
        if db == 'eICU':
            query = f"SELECT CAST(v.patientunitstayid AS VARCHAR) stay_id, CAST(floor(try_cast(v.observationoffset AS DOUBLE)/60.0) AS INTEGER) hour_index,0 itemid,sum(try_cast(v.sao2 AS DOUBLE)) value_sum,count(*) n\n            FROM read_csv_auto({sql(loc[db] / 'vitalPeriodic.csv.gz')},all_varchar=true) v JOIN read_parquet({sql(kp)}) k ON v.patientunitstayid=k.stay_id\n            WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<720 AND try_cast(v.sao2 AS DOUBLE) BETWEEN 50 AND 100 GROUP BY 1,2,3"
        else:
            upper = db == 'MIMIC-III'
            idc = 'icustay_id' if upper else 'stay_id'
            query = f"WITH selected AS (SELECT cast(s.{idc} AS BIGINT) stay_key,s.intime FROM read_csv_auto({sql(loc[db] / ('ICUSTAYS.csv.gz' if upper else 'icustays.csv.gz'))},sample_size=-1) s JOIN read_parquet({sql(kp)}) k ON cast(s.{idc} AS VARCHAR)=k.stay_id)\n            SELECT cast(v.{idc} AS VARCHAR) stay_id,cast(floor(epoch(try_cast(v.charttime AS TIMESTAMP)-s.intime)/3600.0) AS INTEGER) hour_index,try_cast(v.itemid AS BIGINT) itemid,sum(try_cast(v.valuenum AS DOUBLE)) value_sum,count(*) n\n            FROM read_csv_auto({sql(loc[db] / ('CHARTEVENTS.csv.gz' if upper else 'chartevents.csv.gz'))},all_varchar=true) v JOIN selected s ON try_cast(v.{idc} AS BIGINT)=s.stay_key\n            WHERE try_cast(v.itemid AS BIGINT) IN ({','.join(map(str, pure))}) AND try_cast(v.charttime AS TIMESTAMP)>=s.intime AND try_cast(v.charttime AS TIMESTAMP)<s.intime+INTERVAL '12 hours' AND try_cast(v.valuenum AS DOUBLE) BETWEEN 50 AND 100 GROUP BY 1,2,3"
        pend = hpnew.with_suffix('.inprogress.parquet')
        command = f".bail on\nSET threads=4; SET memory_limit='3GB'; SET temp_directory={sql(O / 'tmp')}; COPY ({query}) TO {sql(pend)} (FORMAT PARQUET);"
        print('只提取新增患者外周血氧', db, len(added), flush=True)
        z = subprocess.run([os.environ.get('DUCKDB', 'duckdb'), '-batch'], input=command, text=True, capture_output=True)
        (O / 'qc' / (db + '_新增血氧扫描.log')).write_text(z.stdout + '\n' + z.stderr)
        if z.returncode or not pend.exists():
            raise RuntimeError(db + '血氧扫描失败')
        pd.read_parquet(pend, columns=['stay_id']).head(0)
        pend.replace(hpnew)
    h = pd.read_parquet(A / 'private' / (db + '_raw_oxygen_by_hour_item.parquet'))
    h.stay_id = h.stay_id.astype(str)
    h = h.loc[h.itemid.isin(pure) & h.stay_id.isin(keys)].copy()
    if added:
        extra = pd.read_parquet(hpnew)
        extra.stay_id = extra.stay_id.astype(str)
        check(db + '_新增提取仅指定患者', set(extra.stay_id).issubset(added))
        check(db + '_新增提取仅外周血氧', set(extra.itemid).issubset(pure))
        h = pd.concat([h, extra], ignore_index=True)
    check(db + '_小时项目键唯一', not h.duplicated(['stay_id', 'hour_index', 'itemid']).any())
    check(db + '_小时位于0至不足12', h.hour_index.between(0, 11).all())
    h = h.groupby(['stay_id', 'hour_index'], sort=True).agg(value_sum=('value_sum', 'sum'), n=('n', 'sum')).reset_index()
    h['SpO2'] = h.value_sum / h.n
    h['database'] = db
    check(db + '_小时均值有效范围', h.SpO2.between(50, 100).all())
    counts.append(dict(database=db, coverage24_assignable_N=len(keys), already_rebuilt32_N=len(keys & oldkeys), additional_oxygen_only_N=len(added), additional_observed_N=int(h.loc[h.stay_id.isin(added), 'stay_id'].nunique())))
    hourly.append(h)
    print('血氧补取完成', db, counts[-1], flush=True)
hourly = pd.concat(hourly, ignore_index=True)
hourly.to_parquet(O / 'private/外周血氧每小时均值_所有所需覆盖阈值.parquet', index=False)
agg = load(P / 'revision_R1_R2_data/spo2_source_correction_20260903/scripts/01_build_corrected_spo2_source.py', 'original_hourly_aggregation')
new = agg.aggregate_hourly_spo2(hourly)
new = c[['database', 'stay_id']].merge(new, on=['database', 'stay_id'], how='left', validate='one_to_one')
check('所需血氧来源键唯一', not new.duplicated(['database', 'stay_id']).any())
scalar = hourly.groupby(['database', 'stay_id']).SpO2.apply(lambda v: min(v.tolist()))
ind = new.set_index(['database', 'stay_id']).SpO2_min_0_12h.reindex(scalar.index)
check('独立标量最小小时均值一致', np.allclose(ind, scalar, rtol=0, atol=1e-12))
replacement = pd.read_parquet(A / 'private/仅当前应用队列_已纠正外周血氧替换列_未重算下游.parquet')
replacement.stay_id = replacement.stay_id.astype(str)
z = replacement.merge(new[['database', 'stay_id', 'SpO2_min_0_12h']], on=['database', 'stay_id'], validate='one_to_one', suffixes=('_19', '_20'))
check('当前应用者与19已核替换列逐人一致', np.allclose(z.SpO2_min_0_12h_19, z.SpO2_min_0_12h_20, rtol=0, atol=1e-10, equal_nan=True))
cols = [x for x in src if x.startswith('SpO2_')]
updated = src.drop(columns=cols).merge(new, on=['database', 'stay_id'], how='left', validate='one_to_one', sort=False)[src.columns]
for col in [x for x in src if x not in cols]:
    check('非血氧输入不变_' + col, src[col].equals(updated[col]))
check('MIMICIII当前应用血氧无缺失', updated.loc[updated.database.eq('MIMIC-III'), 'SpO2_min_0_12h'].notna().all())
updated.to_parquet(O / 'private/修订三库全模块分析输入.parquet', index=False)
new.to_parquet(O / 'private/纠正外周血氧汇总_覆盖所有所需阈值.parquet', index=False)
pd.DataFrame(counts).to_csv(O / 'tables/01_低覆盖新增患者血氧补取.tsv', sep='\t', index=False)
availability = []
for db in ['MIMIC-IV', 'MIMIC-III', 'eICU']:
    a = src[src.database.eq(db)]
    b = updated[updated.database.eq(db)]
    for scope in ['assigned', 'primary_model']:
        mask = a[['death_24h', 'age', 'male', 'observed_dimensions_48d'] + (['hospital_id'] if db == 'eICU' else [])].notna().all(axis=1) if scope == 'primary_model' else pd.Series(True, index=a.index)
        aa, bb = (a.loc[mask], b.loc[mask])
        availability.append(dict(database=db, population=scope, N=len(aa), events=int(aa.death_24h.sum()), old_missing=int(aa.SpO2_min_0_12h.isna().sum()), new_missing=int(bb.SpO2_min_0_12h.isna().sum()), old_observed_replaced_N=int((aa.SpO2_min_0_12h.notna() & ~np.isclose(aa.SpO2_min_0_12h, bb.SpO2_min_0_12h, rtol=0, atol=1e-09, equal_nan=True)).sum())))
pd.DataFrame(availability).to_csv(O / 'tables/01_血氧可用性新旧对照.tsv', sep='\t', index=False)
(O / '血氧替换检查点.json').write_text(json.dumps(dict(status='血氧整体替换和低阈值新增提取完成', checks=len(checks), fail=0), ensure_ascii=False, indent=2))
print('全部血氧替换和不变性检查通过', len(checks), flush=True)
