"""Current unchanged eICU endpoint-scenario sensitivity source.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'按已审定结局规则重算修订人群；主要终点保持记录的院内死亡。'
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main(args):
    project, out = (Path(args.project), Path(args.output))
    tables, private = (out / 'tables', out / 'private')
    severity = load(project / 'revision_R1_R2_data/scripts/05_harmonized_severity_and_apache.py', 'severity')
    engine = load(project / '11_eicu_outcome_adjudication_20261014/scripts/02_event_engine_and_models.py', 'event_engine')
    f = pd.read_parquet(private / '修订eICU逐人原始结局证据.parquet')
    source = pd.read_parquet(private / '修订三库全模块分析输入.parquet')
    f = f.merge(source.loc[source.database.eq('eICU'), ['stay_id', 'SpO2_min_0_12h', 'acutephysiologyscore']], on='stay_id', validate='one_to_one')
    hospital_time = f.hospitaldischargeoffset.where(f.hospital_status.eq('EXPIRED'), np.inf)
    unit_time = f.unitdischargeoffset.where(f.unit_status.eq('EXPIRED'), np.inf)
    f['hospital_proxy_24'] = (hospital_time.gt(720) & hospital_time.le(2160)).astype(float)
    f['icu_proxy_24'] = (unit_time.gt(720) & unit_time.le(2160)).astype(float)
    early_time = pd.Series(np.minimum(hospital_time, unit_time), index=f.index)
    f['earliest_proxy_24'] = (early_time.gt(720) & early_time.le(2160)).astype(float)
    f['full_hospital_death'] = f.hospital_status.eq('EXPIRED').astype(float)
    f['full_icu_death'] = f.unit_status.eq('EXPIRED').astype(float)
    f['clock_window_disagreement'] = f.hospital_proxy_24.ne(f.earliest_proxy_24)
    valid = f.strict_landmark & ~f.core_conflict
    masks = [('修订主队列_医院记录时点', 'hospital_proxy_24', pd.Series(True, index=f.index)), ('核心一致_医院记录时点', 'hospital_proxy_24', valid), ('核心一致_ICU记录死亡', 'icu_proxy_24', valid), ('核心一致_较早死亡记录时点', 'earliest_proxy_24', valid), ('核心一致_排除窗口归属分歧', 'hospital_proxy_24', valid & ~f.clock_window_disagreement), ('核心一致_APACHE佐证可用', 'hospital_proxy_24', valid & f.apache_unit.isin(['ALIVE', 'EXPIRED']) & f.apache_hospital.isin(['ALIVE', 'EXPIRED'])), ('核心一致_整个住院死亡', 'full_hospital_death', valid), ('核心一致_本次ICU住院死亡', 'full_icu_death', valid)]
    base = ['age10', 'male', 'coverage10']
    oxygen = base + ['SpO2_min10', 'SpO2_min_0_12h_missing']
    rows, qc, aligned = ([], [], [])

    def check(name, actual, expected=0, tolerance=1e-07):
        error = abs(float(actual) - float(expected))
        passed = bool(np.isfinite(error) and error <= tolerance)
        qc.append(dict(check=name, actual=actual, expected=expected, difference=error, tolerance=tolerance, passed=passed))
        pd.DataFrame(qc).to_csv(tables / '14_事件敏感性独立核查.tsv', sep='\t', index=False)
        if not passed:
            raise RuntimeError(name)

    def fit(frame, outcome, name, covars, seed):
        prepared, _ = severity.prepare_model_data(frame, outcome)
        x, terms = severity.design_matrix(prepared, covars)
        y = prepared[outcome].to_numpy(float)
        b, converged = severity.fit_logistic(x, y)
        check(name + '_收敛', int(converged), 1, 0)
        independent = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-12, cov_type='cluster', cov_kwds={'groups': prepared.hospital_id.astype(str), 'use_correction': True})
        check(name + '_独立系数', np.max(np.abs(b - independent.params)))
        check(name + '_独立协方差', np.max(np.abs(severity.robust_covariance(x, y, b, prepared.hospital_id.astype(str)) - independent.cov_params())))
        result, _ = severity.run_model(prepared, outcome, name, covars, seed)
        check(name + '_实际重抽样', result.bootstrap_successful.min(), 500, 0)
        return (result, prepared)
    check('主要结局逐人重建一致', int(f.death_24h.ne(f.hospital_proxy_24).sum()), 0, 0)
    tests = engine.boundary_tests()
    check('历史独立事件引擎边界测试', int((~tests.passed).sum()) if 'passed' in tests else int((~tests['pass']).sum()), 0, 0)
    tests.to_csv(tables / '14_事件边界测试.tsv', sep='\t', index=False)
    for index, (name, outcome, mask) in enumerate(masks):
        result, prepared = fit(f.loc[mask], outcome, name, base, 100000 + index * 1000)
        independent, _ = engine.logistic(f.loc[mask], outcome)
        primary = result.loc[result.phenotype.eq('P3')].iloc[0]
        check(name + '_独立事件引擎OR', primary.OR, independent['OR'])
        check(name + '_独立事件引擎事件', primary.events, independent['events'], 0)
        result['target_semantics'] = name
        rows.append(result)
        pd.concat(rows).to_csv(tables / '14_修订结局语义敏感性.tsv', sep='\t', index=False)
        print(name, '完成', flush=True)
    events = pd.read_parquet(private / '修订eICU竞争风险逐人事件.parquet')
    fg = pd.read_csv(tables / '13_修订FineGray实际重拟合.tsv', sep='\t')
    for horizon in [24, 48]:
        e = events.loc[events.scenario.eq('FG_CI') & events.horizon_hours.eq(horizon) & events.status.isin([0, 1, 2])]
        frame = f.merge(e[['stay_id', 'status', 'time_from_landmark']], on='stay_id', validate='one_to_one')
        frame['aligned_composite'] = frame.status.eq(1).astype(float)
        expected = fg.loc[fg.scenario.eq('FG_CI') & fg.horizon_hours.eq(horizon)].iloc[0]
        for j, (name, covars) in enumerate([('current_adjusted', base), ('oxygenation_adjusted', oxygen)]):
            modelname = 'ICU范围复合_' + str(horizon) + 'h_' + name
            result, prepared = fit(frame, 'aligned_composite', modelname, covars, 110000 + horizon * 1000 + j * 100)
            check(modelname + '_与FineGray同一分母', len(prepared), expected.N, 0)
            check(modelname + '_与FineGray同一事件', int(prepared.aligned_composite.sum()), expected.events, 0)
            result['horizon_hours'] = horizon
            result['target_semantics'] = 'ICU范围内新器官支持或记录的ICU死亡；存活ICU出科为竞争事件'
            aligned.append(result)
            pd.concat(aligned).to_csv(tables / '14_修订同范围复合普通回归.tsv', sep='\t', index=False)
    state = dict(status='修订结局语义敏感性和同范围复合模型完成', checks=len(qc), failed=0, primary='仍是12小时界标后24小时内记录的院内死亡，不因ICU结果而更换', clinical_truth='仍有229例核心记录冲突；程序一致不等于临床真值')
    (out / '结局敏感性检查点.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False), flush=True)
if __name__ == '__source_reference_run__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
