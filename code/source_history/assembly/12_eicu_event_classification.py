"""Actual event scenarios and proxy handling.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'按已保存规则生成事件、定向拟合；历史资料只读。'
import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location('evidence_io', Path(__file__).with_name('01_collect_evidence.py'))
io_module = module_from_spec(spec)
spec.loader.exec_module(io_module)
query, quoted, parquet = (io_module.query, io_module.quoted, io_module.parquet)
LANDMARK = 720.0
SUPPORT = ['first_post12_vasopressor_time', 'first_post12_invasive_ventilation_time', 'first_post12_RRT_time']

def classify(unit_status, hospital_status, unit_exit, hospital_exit, support_time, core_conflict, horizon, scenario, support_free=True):
    """单病例状态；返回代码、时间（小时）、原因。未知和冲突不当作零事件。"""
    end = LANDMARK + horizon * 60
    composite = scenario in ['FG0', 'FG_CI', 'FG_CH']
    if composite and (not support_free):
        return (6, np.nan, '12小时已有器官支持_不属于复合终点风险集')
    if not np.isfinite(unit_exit) or unit_exit <= LANDMARK:
        return (7, np.nan, '不符合严格12小时ICU界标')
    if scenario != 'FG0':
        if unit_status not in ['ALIVE', 'EXPIRED'] or hospital_status not in ['ALIVE', 'EXPIRED']:
            return (8, np.nan, '状态未知')
        if core_conflict:
            return (9, np.nan, '状态或时间核心冲突')
    hospital_scope = scenario in ['FG_H', 'FG_HE', 'FG_CH']
    if scenario == 'FG0':
        death = hospital_exit if hospital_status == 'EXPIRED' else math.inf
        exit_time, live = (unit_exit, True)
    elif hospital_scope:
        death = hospital_exit if hospital_status == 'EXPIRED' else math.inf
        if scenario == 'FG_HE' and unit_status == 'EXPIRED':
            death = unit_exit
        exit_time, live = (hospital_exit, hospital_status == 'ALIVE')
    else:
        death = unit_exit if unit_status == 'EXPIRED' else math.inf
        exit_time, live = (unit_exit, unit_status == 'ALIVE')
    if not np.isfinite(exit_time) or exit_time <= LANDMARK:
        return (9, np.nan, '退出时点缺失或不晚于界标')
    if death <= LANDMARK:
        return (9, np.nan, '死亡不晚于界标')
    earliest = death
    if composite and np.isfinite(support_time) and (support_time > LANDMARK):
        if scenario == 'FG0' or support_time <= unit_exit:
            earliest = min(earliest, support_time)
    if earliest <= end and earliest <= exit_time:
        return (1, (earliest - LANDMARK) / 60, '目标事件')
    if live and exit_time <= end:
        return (2, (exit_time - LANDMARK) / 60, '明确存活退出')
    if exit_time <= end:
        return (9, np.nan, '窗口内退出但状态和事件无法形成合法分类')
    return (0, horizon, '窗口结束')

def boundary_tests():
    cases = []

    def add(name, us, hs, u, h, s, conf, horizon, scenario, expected):
        result = classify(us, hs, u, h, s, conf, horizon, scenario)
        assert result[0] == expected, name + '分类错误'
        if result[0] in [0, 1, 2]:
            assert 0 < result[1] <= horizon
        cases.append(dict(check=name, passed=True))
    for horizon in [24, 48]:
        end = 720 + horizon * 60
        for exit_min, expected in [(719, 7), (720, 7), (721, 2)]:
            add(f'{horizon}_界标_{exit_min}', 'ALIVE', 'ALIVE', exit_min, 9999, math.inf, False, horizon, 'FG_I', expected)
        for death_min, expected in [(end - 1, 1), (end, 1), (end + 1, 0)]:
            add(f'{horizon}_末端_{death_min}', 'EXPIRED', 'EXPIRED', death_min, death_min, math.inf, False, horizon, 'FG_I', expected)
        add(f'{horizon}_存活ICU退出后医院死亡_医院目标', 'ALIVE', 'EXPIRED', 800, 900, math.inf, False, horizon, 'FG_H', 1)
        add(f'{horizon}_存活ICU退出后医院死亡_ICU目标', 'ALIVE', 'EXPIRED', 800, 900, math.inf, False, horizon, 'FG_I', 2)
        add(f'{horizon}_未知', 'UNKNOWN', 'ALIVE', 1000, 2000, math.inf, False, horizon, 'FG_I', 8)
        add(f'{horizon}_冲突', 'EXPIRED', 'ALIVE', 1000, 2000, math.inf, True, horizon, 'FG_I', 9)
        add(f'{horizon}_支持退出同刻', 'ALIVE', 'ALIVE', 1000, 2000, 1000, False, horizon, 'FG_CI', 1)
        add(f'{horizon}_支持在退出后', 'ALIVE', 'ALIVE', 1000, 2000, 1001, False, horizon, 'FG_CI', 2)
        add(f'{horizon}_医院时点较晚但ICU已记录死亡_早时点', 'EXPIRED', 'EXPIRED', 1000, end + 1, math.inf, False, horizon, 'FG_HE', 1)
        add(f'{horizon}_医院时点较晚但ICU已记录死亡_原时点', 'EXPIRED', 'EXPIRED', 1000, end + 1, math.inf, False, horizon, 'FG_H', 0)
    return pd.DataFrame(cases)

def logistic(frame, outcome):
    d = frame.dropna(subset=['phenotype', 'age', 'male', 'hospital_id', 'observed_dimensions_48d', outcome]).copy()
    x = np.column_stack([np.ones(len(d)), *[d.phenotype.eq(p).to_numpy(float) for p in ['P1', 'P3', 'P4']], d.age.to_numpy() / 10, d.male.to_numpy(), d.observed_dimensions_48d.to_numpy() / 10])
    y = d[outcome].to_numpy(float)
    assert np.isin(y, [0, 1]).all(), '二分类终点非法'
    b = np.zeros(x.shape[1])
    for iteration in range(120):
        eta = x @ b
        p = 1 / (1 + np.exp(-np.clip(eta, -40, 40)))
        h = x.T @ (x * (p * (1 - p))[:, None])
        step = np.linalg.solve(h, x.T @ (y - p))
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        scale = 1.0
        while scale > 1e-09:
            trial = b + scale * step
            t = x @ trial
            if np.sum(y * t - np.logaddexp(0, t)) >= ll - 1e-10:
                break
            scale /= 2
        b = trial
        if np.max(np.abs(scale * step)) < 1e-10:
            break
    else:
        raise RuntimeError('回归未收敛')
    p = 1 / (1 + np.exp(-x @ b))
    score = x * (y - p)[:, None]
    solved = float(np.max(np.abs(score.sum(axis=0))))
    assert solved < 1e-05, '回归似然未解'
    bread = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    _, group = np.unique(d.hospital_id, return_inverse=True)
    sums = np.zeros((group.max() + 1, x.shape[1]))
    np.add.at(sums, group, score)
    n, k, g = (len(d), x.shape[1], len(sums))
    covariance = bread @ (sums.T @ sums * (g / (g - 1)) * ((n - 1) / (n - k))) @ bread
    se = math.sqrt(covariance[2, 2])
    z = 1.95996398454
    pred = []
    grad = []
    for ph in ['P2', 'P3']:
        xx = x.copy()
        xx[:, 1:4] = 0
        xx[:, 2] = float(ph == 'P3')
        pp = 1 / (1 + np.exp(-xx @ b))
        pred.append(float(pp.mean()))
        grad.append((xx * (pp * (1 - pp))[:, None]).mean(axis=0))
    rd = pred[1] - pred[0]
    rd_se = math.sqrt((grad[1] - grad[0]) @ covariance @ (grad[1] - grad[0]))
    result = dict(N=n, events=int(y.sum()), hospitals=g, OR=float(np.exp(b[2])), CI_low=float(np.exp(b[2] - z * se)), CI_high=float(np.exp(b[2] + z * se)), beta=float(b[2]), SE=se, standardized_risk_P2=pred[0], standardized_risk_P3=pred[1], RD=rd, RD_CI_low=rd - z * rd_se, RD_CI_high=rd + z * rd_se, maximum_score=solved, OR_CI_method='医院聚类sandwich稳健方差_历史有限样本修正_正态95%分位数', risk_standardization='模型完整病例共同协变量分布上的边际标准化_非因果效应', RD_CI_method='delta方法诊断区间_不是历史500次聚类重抽样区间_不得静默替换投稿结果')
    for ph in ['P1', 'P2', 'P3', 'P4']:
        subset = d.loc[d.phenotype.eq(ph), outcome]
        result[ph + '_N'] = len(subset)
        result[ph + '_events'] = int(subset.sum())
    return (result, d)

def run(args):
    out, project = (Path(args.output), Path(args.project))
    table, private, tmp = (out / 'tables', out / 'private', out / 'tmp')
    f = query(f"SELECT * FROM read_parquet({quoted(private / '全部可分型患者结局证据.parquet')})")
    f['stay_id'] = f.stay_id.astype('int64')
    tests = boundary_tests()
    tests.to_csv(table / '04_边界单元测试.tsv', sep='\t', index=False)
    old = pd.read_csv(project / 'revision_R1_R2_data/20_competing_risk_input.csv')
    old = old.loc[old.database.eq('eICU')].copy()
    old['stay_id'] = old.stay_id.astype('int64')
    valid = f.strict_landmark & ~f.core_conflict
    hospital_time = f.hospitaldischargeoffset.astype(float).where(f.hospital_status.eq('EXPIRED'), np.inf)
    unit_time = f.unitdischargeoffset.astype(float).where(f.unit_status.eq('EXPIRED'), np.inf)
    early_time = pd.Series(np.minimum(hospital_time, unit_time), index=f.index)
    inside = lambda t: (t > 720) & (t <= 2160)
    f['hospital_proxy_24'] = inside(hospital_time).astype(float)
    f['icu_proxy_24'] = inside(unit_time).astype(float)
    f['earliest_proxy_24'] = inside(early_time).astype(float)
    f['full_hospital_death'] = f.hospital_status.eq('EXPIRED').astype(float)
    f['full_icu_death'] = f.unit_status.eq('EXPIRED').astype(float)
    assert f.death_24h.eq(f.hospital_proxy_24).all(), '历史主结局重建失败'
    f['clock_window_disagreement'] = f.hospital_proxy_24.ne(f.earliest_proxy_24)
    definitions = [('H0', '历史医院时点_原队列', 'hospital_proxy_24', pd.Series(True, index=f.index)), ('H1', '仅严格720分钟边界', 'hospital_proxy_24', f.strict_landmark), ('H2', '核心一致_医院时点', 'hospital_proxy_24', valid), ('I2', '核心一致_ICU死亡', 'icu_proxy_24', valid), ('H3', '核心一致_较早记录死亡时点', 'earliest_proxy_24', valid), ('H4', '核心一致_排除窗口归属分歧', 'hospital_proxy_24', valid & ~f.clock_window_disagreement), ('H5', '核心一致_APACHE佐证可用', 'hospital_proxy_24', valid & f.apache_unit.isin(['ALIVE', 'EXPIRED']) & f.apache_hospital.isin(['ALIVE', 'EXPIRED'])), ('F2', '核心一致_整个住院死亡', 'full_hospital_death', valid), ('FI2', '核心一致_本次ICU住院死亡', 'full_icu_death', valid)]
    model_rows = []
    model_inputs = []
    counts = []
    for name, label, outcome, mask in definitions:
        result, d = logistic(f.loc[mask], outcome)
        model_rows.append(dict(model_id=name, description=label, outcome=outcome, subset='原肺炎证据', **result))
        d = d[['stay_id', 'hospital_id', 'phenotype', 'age', 'male', 'observed_dimensions_48d', outcome]].rename(columns={outcome: 'y'})
        d['model_id'] = name
        model_inputs.append(d)
        counts.append(dict(model_id=name, assigned_N=int(mask.sum()), excluded_from_assigned_N=int((~mask).sum()), original_primary_events_excluded=int(f.loc[~mask, 'death_24h'].sum()), model_N=result['N'], events=result['events']))
    expected = pd.read_csv(project / '10_second_round_revision_20261014/data_audit/tables/01_frozen_primary_reproduction.tsv', sep='\t')
    expected = expected.loc[expected.database.eq('eICU')].iloc[0]
    for col in ['N', 'events', 'OR', 'CI_low', 'CI_high']:
        assert abs(model_rows[0][col] - expected[col]) < 1e-08, '历史主模型不复现_' + col
    pneumonia = [('排除仅短字符串', ~f.short_only), ('任意肺炎记录不晚于12小时', f.earliest_offset.le(720)), ('完整肺炎词不晚于12小时', f.any_long_by_12h)]
    for name, label, outcome, mask in definitions[2:5]:
        for j, (subset, eligibility) in enumerate(pneumonia, 1):
            mid = name + '_P' + str(j)
            result, d = logistic(f.loc[mask & eligibility], outcome)
            model_rows.append(dict(model_id=mid, description=label, outcome=outcome, subset=subset, **result))
            d = d[['stay_id', 'hospital_id', 'phenotype', 'age', 'male', 'observed_dimensions_48d', outcome]].rename(columns={outcome: 'y'})
            d['model_id'] = mid
            model_inputs.append(d)
    pd.DataFrame(model_rows).to_csv(table / '定向重算结果.tsv', sep='\t', index=False)
    pd.DataFrame(counts).to_csv(table / '新旧人数和事件对账.tsv', sep='\t', index=False)
    parquet(pd.concat(model_inputs, ignore_index=True), private / '定向死亡回归输入.parquet', tmp)
    selection = []
    for population, mask in [('纳入核心一致集', valid), ('排除核心冲突', f.strict_landmark & f.core_conflict), ('720分钟边界排除', ~f.strict_landmark), ('核心一致但缺APACHE佐证', valid & ~f.apache_unit.isin(['ALIVE', 'EXPIRED'])), ('两种死亡时点窗口归属不同', valid & f.clock_window_disagreement)]:
        for ph in ['全部', 'P1', 'P2', 'P3', 'P4']:
            d = f.loc[mask & (True if ph == '全部' else f.phenotype.eq(ph))]
            selection.append(dict(population=population, phenotype=ph, N=len(d), old_primary_events=int(d.death_24h.sum()), unit_recorded_deaths=int(d.unit_status.eq('EXPIRED').sum()), hospital_recorded_deaths=int(d.hospital_status.eq('EXPIRED').sum()), age_median=d.age.median(), coverage_median=d.observed_dimensions_48d.median(), male_percent=100 * d.male.mean()))
    pd.DataFrame(selection).to_csv(table / '05_排除及佐证缺失选择影响.tsv', sep='\t', index=False)
    supports = f[SUPPORT].to_numpy(float)
    first = np.where(np.isfinite(supports) & (supports > 720), supports, np.inf).min(axis=1)
    valid_supports = np.where(np.isfinite(supports) & (supports > 720) & (supports <= f.unitdischargeoffset.to_numpy()[:, None]), supports, np.inf)
    first_valid = valid_supports.min(axis=1)
    fg_parts = []
    summaries = []
    for scenario in ['FG0', 'FG_I', 'FG_H', 'FG_HE', 'FG_CI', 'FG_CH']:
        for horizon in [24, 48]:
            statuses = []
            times = []
            reasons = []
            for i, row in enumerate(f.itertuples(index=False)):
                code, time, reason = classify(row.unit_status, row.hospital_status, row.unitdischargeoffset, row.hospitaldischargeoffset, first[i] if scenario == 'FG0' else first_valid[i], row.core_conflict, horizon, scenario, bool(row.support_free_at_12h))
                statuses.append(code)
                times.append(time)
                reasons.append(reason)
            d = f[['stay_id', 'hospital_id', 'phenotype', 'age', 'male', 'observed_dimensions_48d']].copy()
            d['scenario'] = scenario
            d['horizon_hours'] = horizon
            d['status'] = statuses
            d['time_from_landmark'] = times
            d['reason'] = reasons
            included = d.status.isin([0, 1, 2])
            complete = included & d[['age', 'male', 'hospital_id', 'phenotype', 'observed_dimensions_48d']].notna().all(axis=1)
            if scenario == 'FG0':
                o = old.loc[old.horizon_hours.eq(horizon)]
                compared = d.loc[included].merge(o, on='stay_id', validate='one_to_one', suffixes=('_new', '_old'))
                assert len(compared) == len(o) == int(included.sum()), '历史竞争风险风险集不复现'
                assert compared.status_new.eq(compared.status_old).all(), '历史竞争风险状态不复现'
                assert np.isclose(compared.time_from_landmark_new, compared.time_from_landmark_old, atol=1e-10).all(), '历史竞争风险时间不复现'
            assert (d.loc[complete, 'time_from_landmark'] > 0).all()
            if scenario != 'FG0':
                exit_status = f.hospital_status if scenario in ['FG_H', 'FG_HE', 'FG_CH'] else f.unit_status
                assert exit_status.loc[d.status.eq(2)].eq('ALIVE').all(), '非存活退出被误编码竞争事件'
            summaries.append(dict(scenario=scenario, horizon_hours=horizon, assigned_N=len(f), risk_set_N=int(included.sum()), model_N=int(complete.sum()), events=int((complete & d.status.eq(1)).sum()), competing=int((complete & d.status.eq(2)).sum()), window_end=int((complete & d.status.eq(0)).sum()), unknown_status_N=int(d.status.eq(8).sum()), indeterminate_N=int(d.status.eq(9).sum()), boundary_excluded_N=int(d.status.eq(7).sum()), baseline_support_excluded_N=int(d.status.eq(6).sum()), covariate_missing_N=int((included & ~complete).sum())))
            fg_parts.append(d)
    pd.DataFrame(summaries).to_csv(table / '06_竞争风险事件及未知状态.tsv', sep='\t', index=False)
    parquet(pd.concat(fg_parts, ignore_index=True), private / '竞争风险各方案逐人事件.parquet', tmp)
    parquet(f, private / '结局时点与选择标记.parquet', tmp)
    with (out / '事件判定与定向分析规则.md').open('rb') as stream:
        rule_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    pd.DataFrame([dict(rules_sha256=rule_hash, models_N=len(model_rows), event_scenarios_N=len(summaries), boundary_tests_N=len(tests))]).to_csv(table / '04_规则锁定及运行检查点.tsv', sep='\t', index=False)
    print('定向死亡回归完成；历史死亡与竞争事件逐人复现通过；模型数=', len(model_rows), '；事件方案数=', len(summaries), flush=True)
if __name__ == '__source_reference_run__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
