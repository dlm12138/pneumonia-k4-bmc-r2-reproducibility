"""Actual MIMIC ordinary-vs-competing event definitions; current oxygen effects use existing current workflow.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'只补两版MIMIC受严格肺炎筛选影响的次要复合模型；所有原文件只读。'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
import statsmodels.api as sm
SUPPORT_COLUMNS = ['first_post12_vasopressor_time', 'first_post12_invasive_ventilation_time', 'first_post12_RRT_time']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def normalize(frame):
    frame = frame.copy()
    frame['database'] = frame.database.replace({'MIMIC-III MetaVision': 'MIMIC-III'})
    frame['stay_id'] = frame.stay_id.astype(str)
    return frame

def evidence(frame):
    z = frame.copy()
    z['exit_hours'] = np.nan
    z['timestamp_death_hours'] = np.nan
    for db, start, end, death in [('MIMIC-IV', 'intime', 'outtime', 'deathtime'), ('MIMIC-III', 'INTIME', 'OUTTIME', 'DEATHTIME')]:
        mask = z.database.eq(db)
        origin = pd.to_datetime(z.loc[mask, start])
        z.loc[mask, 'exit_hours'] = (pd.to_datetime(z.loc[mask, end]) - origin).dt.total_seconds() / 3600
        z.loc[mask, 'timestamp_death_hours'] = (pd.to_datetime(z.loc[mask, death]) - origin).dt.total_seconds() / 3600
    z['first_support_hours'] = z[SUPPORT_COLUMNS].apply(pd.to_numeric, errors='coerce').min(axis=1)
    z['death_hours'] = pd.to_numeric(z.death_time_from_icu_admission, errors='coerce')
    z['target_hours'] = z[['first_support_hours', 'death_hours']].where(z[['first_support_hours', 'death_hours']].gt(12)).min(axis=1)
    return z

def classify_vector(z, horizon):
    target, discharge = (z.target_hours, z.exit_hours)
    is_target = target.le(12 + horizon) & (discharge.isna() | target.le(discharge))
    is_discharge = ~is_target & discharge.le(12 + horizon)
    status = np.select([is_target, is_discharge], [1, 2], default=0)
    time = np.select([is_target, is_discharge], [target, discharge], default=12 + horizon) - 12
    return (status.astype(int), time.astype(float))

def classify_scalar(support, death, discharge, horizon):
    available = [v for v in (support, death) if np.isfinite(v) and v > 12]
    target = min(available) if available else np.inf
    end = 12 + horizon
    if target <= end and (not np.isfinite(discharge) or target <= discharge):
        return (1, target - 12)
    if np.isfinite(discharge) and discharge <= end:
        return (2, discharge - 12)
    return (0, float(horizon))

def main(args):
    project, out = (Path(args.project), Path(args.output))
    tables, private = (out / 'tables', out / 'private')
    for directory in [tables, private, out / 'tmp']:
        directory.mkdir(parents=True, exist_ok=True)
    history = project / '01_k5-7.29_history/k5-7.29/analysis_main12_clinical_actionability_20260730/13_organ_support_escalation_patient_level.parquet'
    sourcepath = project / '14_pneumonia_specificity_revision_20261014/private/修订三库全模块分析输入.parquet'
    oldfgpath = project / 'revision_R1_R2_data/20_competing_risk_input.csv'
    modulepath = project / 'revision_R1_R2_data/scripts/05_harmonized_severity_and_apache.py'
    docs = [project.parent / '02_k4_Revised_Clean (3).docx', project.parent / '04_Supplementary_Material_Revised.docx']
    protectedpaths = [history, sourcepath, oldfgpath, modulepath] + docs + sorted((project / 'revision_R1_R2_data/locked_K4_model').glob('*.csv'))
    protectionpath = tables / '01_只读文件保护.tsv'
    if protectionpath.exists():
        hashes = pd.read_csv(protectionpath, sep='\t').set_index('path').SHA256_before.to_dict()
    else:
        hashes = {str(p): digest(p) for p in protectedpaths}
        pd.DataFrame([dict(path=p, SHA256_before=h) for p, h in hashes.items()]).to_csv(protectionpath, sep='\t', index=False)
    checks = []

    def check(name, error, tolerance=0):
        passed = bool(np.isfinite(error) and error <= tolerance)
        checks.append(dict(check=name, error=error, tolerance=tolerance, passed=passed))
        pd.DataFrame(checks).to_csv(tables / '02_Python事件与数值独立检查.tsv', sep='\t', index=False)
        if not passed:
            raise RuntimeError('核查失败：' + name + '；仅报告聚合误差=' + str(error))
    old = normalize(pd.read_parquet(history))
    old = evidence(old[old.database.isin(['MIMIC-IV', 'MIMIC-III'])])
    source = normalize(pd.read_parquet(sourcepath))
    source = evidence(source[source.database.isin(['MIMIC-IV', 'MIMIC-III']) & source.revised_assigned])
    check('修订人群键唯一', int(source.duplicated(['database', 'stay_id']).sum()))
    check('全部事件单位是小时', int(source.event_time_unit.ne('hours').sum()))
    check('出科时间戳缺失', int(source.exit_hours.isna().sum()))
    check('出科严格晚于界标', int(source.exit_hours.le(12).sum()))
    for column in ['intime', 'outtime', 'deathtime', 'INTIME', 'OUTTIME', 'DEATHTIME']:
        ts = pd.to_datetime(source[column]).dropna()
        check(column + '_原始时间戳无亚秒值', int(((ts.dt.microsecond != 0) | (ts.dt.nanosecond != 0)).sum()))
    check('模型标签历史未改变', int(source[['database', 'stay_id', 'phenotype']].merge(old[['database', 'stay_id', 'phenotype']], on=['database', 'stay_id'], validate='one_to_one', suffixes=('', '_old')).eval('phenotype != phenotype_old').sum()))
    for db in ['MIMIC-IV', 'MIMIC-III']:
        z = source[source.database.eq(db)]
        finite = z.death_hours.notna() & z.timestamp_death_hours.notna()
        check(db + '_死亡小时与时间戳一致', float((z.loc[finite, 'death_hours'] - z.loc[finite, 'timestamp_death_hours']).abs().max()), 1e-10)
        check(db + '_院内死亡记录无缺失死亡时刻', int((z.hospital_mortality.eq(1) & z.death_hours.isna()).sum()))
    boundary = [(12, np.nan, 20, 24, 2, 8), (13, np.nan, 13, 24, 1, 1), (14, np.nan, 13, 24, 2, 1), (np.nan, 13, 13, 24, 1, 1), (np.nan, 14, 13, 24, 2, 1), (36, np.nan, 40, 24, 1, 24), (36 + 1e-05, np.nan, 40, 24, 0, 24), (np.nan, np.nan, 36, 24, 2, 24), (np.nan, np.nan, 37, 24, 0, 24), (60, np.nan, 80, 48, 1, 48)]
    for i, (s, d, e, h, expectedstatus, expectedtime) in enumerate(boundary):
        status, time = classify_scalar(s, d, e, h)
        check(f'合成边界{i}_状态', abs(status - expectedstatus))
        check(f'合成边界{i}_时间', abs(time - expectedtime), 1e-10)
    oldfg = normalize(pd.read_csv(oldfgpath, dtype={'stay_id': str}))
    oldfg = oldfg[oldfg.database.isin(['MIMIC-IV', 'MIMIC-III'])]
    oldfree = old[old.support_free_at_12h.eq(True) & old.exit_hours.gt(12)].copy()
    keys = source[['database', 'stay_id']].assign(revised=True)
    rebuilt, counts, phenotypecounts = ([], [], [])
    for horizon in [24, 48]:
        frame = oldfree.copy()
        frame['status'], frame['time_from_landmark'] = classify_vector(frame, horizon)
        expected = oldfg[oldfg.horizon_hours.eq(horizon)]
        check(f'旧竞争风险{horizon}_人数', abs(len(frame) - len(expected)))
        match = frame.merge(expected[['database', 'stay_id', 'status', 'time_from_landmark']], on=['database', 'stay_id'], validate='one_to_one', suffixes=('', '_old'))
        check(f'旧竞争风险{horizon}_逐人状态', int(match.status.ne(match.status_old).sum()))
        check(f'旧竞争风险{horizon}_逐人时间', float((match.time_from_landmark - match.time_from_landmark_old).abs().max()), 1e-10)
        current = source[source.support_free_at_12h.eq(True)].copy()
        current['status'], current['time_from_landmark'] = classify_vector(current, horizon)
        current['horizon_hours'] = horizon
        current['ordinary_composite'] = current[f'organ_support_or_death_{horizon}h'].astype(int)
        recovered = current.target_hours.le(12 + horizon).astype(int)
        check(f'普通复合{horizon}_逐人原始标记复原', int(recovered.ne(current.ordinary_composite).sum()))
        check(f'支持起始未早于界标{horizon}', int(current.first_support_hours.le(12).sum()))
        check(f'无支持者支持时刻不晚于出科{horizon}', int(current.first_support_hours.gt(current.exit_hours).sum()))
        scalar = [classify_scalar(s, d, e, horizon) for s, d, e in current[['first_support_hours', 'death_hours', 'exit_hours']].itertuples(index=False, name=None)]
        check(f'独立标量分类{horizon}_状态', int(np.sum(current.status.to_numpy() != np.array([a[0] for a in scalar]))))
        check(f'独立标量分类{horizon}_时间', float(np.max(np.abs(current.time_from_landmark.to_numpy() - np.array([a[1] for a in scalar])))), 1e-10)
        for db in ['MIMIC-IV', 'MIMIC-III']:
            o = frame[frame.database.eq(db)]
            n = current[current.database.eq(db)]
            difference = n.ordinary_composite.eq(1) & n.status.ne(1)
            explained = n.death_hours.gt(n.exit_hours) & n.death_hours.le(12 + horizon) & n.death_hours.gt(12)
            check(f'{db}_{horizon}_结局差异全部由出科后死亡解释', int((difference & ~explained).sum()))
            counts.append(dict(database=db, horizon_hours=horizon, old_N=len(o), revised_N=len(n), removed_N=len(o) - len(n), old_FG_events=int(o.status.eq(1).sum()), revised_FG_events=int(n.status.eq(1).sum()), revised_competing=int(n.status.eq(2).sum()), window_end=int(n.status.eq(0).sum()), ordinary_composite_events=int(n.ordinary_composite.sum()), ordinary_minus_FG=int(difference.sum()), post_exit_hospital_deaths_in_window=int(explained.sum()), exit_death_ties=int(n.death_hours.eq(n.exit_hours).sum()), SpO2_missing=int(n.SpO2_min_0_12h.isna().sum())))
            for p, g in n.groupby('phenotype'):
                phenotypecounts.append(dict(database=db, horizon_hours=horizon, phenotype=p, N=len(g), ordinary_events=int(g.ordinary_composite.sum()), FG_events=int(g.status.eq(1).sum()), competing=int(g.status.eq(2).sum())))
        rebuilt.append(current)
    pd.DataFrame(counts).to_csv(tables / '03_新旧样本及结局定义差异.tsv', sep='\t', index=False)
    pd.DataFrame(phenotypecounts).to_csv(tables / '04_各组人数与事件.tsv', sep='\t', index=False)
    events = pd.concat(rebuilt, ignore_index=True)
    eventcols = ['database', 'stay_id', 'phenotype', 'age', 'male', 'observed_dimensions_48d', 'horizon_hours', 'status', 'time_from_landmark', 'ordinary_composite', 'exit_hours', 'first_support_hours', 'death_hours', 'intime', 'outtime', 'deathtime', 'INTIME', 'OUTTIME', 'DEATHTIME'] + SUPPORT_COLUMNS
    events[eventcols].to_parquet(private / 'MIMIC修订逐人复合与竞争事件.parquet', index=False)
    print('旧状态与时间逐人复现、新人群定义差异及边界检查通过；开始四个普通回归', flush=True)
    spec = importlib.util.spec_from_file_location('historical_functions_only', modulepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = ['age10', 'male', 'coverage10']
    oxygen = base + ['SpO2_min10', 'SpO2_min_0_12h_missing']
    resultpath, detailpath, diagpath = (tables / '05_修订普通复合模型.tsv', tables / '06_模型与重抽样说明.tsv', tables / '07_数值与分离诊断.tsv')
    designpath = private / 'MIMIC普通回归独立设计矩阵.parquet'
    results = [pd.read_csv(resultpath, sep='\t')] if resultpath.exists() else []
    details = [pd.read_csv(detailpath, sep='\t')] if detailpath.exists() else []
    diagnostics = pd.read_csv(diagpath, sep='\t').to_dict('records') if diagpath.exists() else []
    designs = [pd.read_parquet(designpath)] if designpath.exists() else []
    for db, offsets in [('MIMIC-IV', [23000, 24000]), ('MIMIC-III', [27000, 28000])]:
        sourcefree = source[source.database.eq(db) & source.support_free_at_12h.eq(True)].copy()
        prepared, imputation = module.prepare_model_data(sourcefree, 'organ_support_or_death_24h')
        for label, covars, offset in [('current_adjusted', base, offsets[0]), ('harmonized_severity_adjusted', oxygen, offsets[1])]:
            key = db + '_' + label
            cached = bool(results and pd.concat(results).model_key.eq(key).any())
            x, terms = module.design_matrix(prepared, covars)
            y = prepared.organ_support_or_death_24h.to_numpy(float)
            check(key + '_矩阵满秩', x.shape[1] - np.linalg.matrix_rank(x))
            signed = x * (2 * y - 1)[:, None]
            separation = linprog(-signed.sum(axis=0), A_ub=-signed, b_ub=np.zeros(len(y)), bounds=[(-1, 1)] * x.shape[1], method='highs')
            check(key + '_分离优化成功', int(not separation.success))
            check(key + '_不存在完全或准完全分离', max(0.0, float(-separation.fun)), 1e-07)
            beta, converged = module.fit_logistic(x, y)
            check(key + '_点模型收敛', int(not converged))
            vc = module.robust_covariance(x, y, beta, None)
            reference = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-12, cov_type='HC0')
            check(key + '_独立GLM系数', float(np.max(np.abs(beta - reference.params))), 1e-07)
            check(key + '_独立GLM完整HC0协方差', float(np.max(np.abs(vc - reference.cov_params()))), 1e-07)
            check(key + '_最大归一化得分', float(np.max(np.abs(x.T @ (y - expit(x @ beta)))) / len(y)), 1e-08)
            if cached:
                saved = pd.concat(results)
                saved_p3 = saved[saved.model_key.eq(key) & saved.phenotype.eq('P3')].iloc[0]
                check(key + '_续用效应与当前核验一致', abs(float(saved_p3.OR) - float(np.exp(beta[2]))), 1e-10)
                print('续用完成的500次重抽样，并重新核验点估计：' + key, flush=True)
                continue
            outcome, detail = module.run_model(prepared, 'organ_support_or_death_24h', label, covars, offset)
            outcome['model_key'] = key
            outcome['variance'] = 'HC0 sandwich robust'
            outcome['horizon_hours'] = 24
            outcome['phenotype_N'] = outcome.phenotype.map(prepared.groupby('phenotype').size())
            outcome['phenotype_events'] = outcome.phenotype.map(prepared.groupby('phenotype').organ_support_or_death_24h.sum())
            outcome['raw_risk_fraction'] = outcome.phenotype.map(prepared.groupby('phenotype').organ_support_or_death_24h.mean())
            outcome['seed'] = module.SEED + offset
            outcome['RD_CI_low_percentage_points'] = 100 * outcome.RD_CI_low
            outcome['RD_CI_high_percentage_points'] = 100 * outcome.RD_CI_high
            p3 = outcome[outcome.phenotype.eq('P3')].iloc[0]
            diagnostics.append(dict(model_key=key, database=db, model=label, N=len(y), events=int(y.sum()), OR=p3.OR, SE=float(np.sqrt(vc[2, 2])), separation_max_margin_sum=float(-separation.fun), SpO2_median=imputation['SpO2_min_0_12h'], rank=int(np.linalg.matrix_rank(x)), parameters=x.shape[1], condition_number=float(np.linalg.cond(x)), converged=converged))
            design = pd.DataFrame(x, columns=[f'x{i}' for i in range(x.shape[1])])
            design['y'], design['model_key'], design['term_names'] = (y, key, ';'.join(terms))
            designs.append(design)
            results.append(outcome)
            detail['model_key'], detail['seed'], detail['variance'] = (key, module.SEED + offset, 'HC0 sandwich robust')
            details.append(detail)
            pd.concat(results, ignore_index=True).to_csv(resultpath, sep='\t', index=False)
            pd.concat(details, ignore_index=True).to_csv(detailpath, sep='\t', index=False)
            pd.DataFrame(diagnostics).to_csv(diagpath, sep='\t', index=False)
            pd.concat(designs, ignore_index=True).to_parquet(designpath, index=False)
            (out / '计算检查点.json').write_text(json.dumps({'普通模型已完成': pd.concat(results).model_key.unique().tolist(), '每模型成功重抽样次数': 500, 'R独立核验和FineGray': '尚待执行', '稿件修改': False}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'完成 {key}；N={len(y)}，事件={int(y.sum())}，P3比值比={p3.OR:.6f}；重抽样={int(p3.bootstrap_successful)}', flush=True)
    protection = []
    for p, h in hashes.items():
        now = digest(Path(p))
        check('只读SHA256_' + Path(p).name, int(h != now))
        protection.append(dict(path=p, SHA256_before=h, SHA256_after=now, unchanged=h == now))
    pd.DataFrame(protection).to_csv(protectionpath, sep='\t', index=False)
if __name__ == '__source_reference_run__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
