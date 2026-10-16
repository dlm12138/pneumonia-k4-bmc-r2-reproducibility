"""Original strict frame and saved-fold assembly; old oxygen NEVER final.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'冻结分配和历史主效应先通过，再更新识别修正人群的主要定向分析。'
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import linprog
import statsmodels.api as sm

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main(args):
    project, out = (Path(args.project), Path(args.output))
    tables, private = (out / 'tables', out / 'private')
    revision = project / 'revision_R1_R2_data'
    correction = revision / 'spo2_source_correction_20260903'
    severity = load(revision / 'scripts/05_harmonized_severity_and_apache.py', 'actual_severity_functions')
    prediction = load(revision / 'scripts/08_hrrr_and_incremental_performance.py', 'actual_comparator_functions')
    severity.RESPIRATORY = correction / 'private/corrected_respiratory_source.parquet'
    membershippath = private / '三库修订肺炎资格与纳入标记.parquet'
    membership = pd.read_parquet(membershippath)
    membership.stay_id = membership.stay_id.astype(str)
    modelroot = revision / 'locked_K4_model'
    manifest = pd.read_csv(project / '10_second_round_revision_20261014/data_audit/tables/01_frozen_parameter_source_hashes.tsv', sep='\t')
    protected = {str(p): sha256(p) for p in [membershippath, severity.ENDPOINT, severity.CANDIDATES, severity.RESPIRATORY, project.parent / '02_k4_Revised_Clean (3).docx', project.parent / '04_Supplementary_Material_Revised.docx']}
    protected.update({str(p): sha256(p) for p in sorted(modelroot.glob('*.csv'))})
    qpath = tables / '02_主效应与严重度独立核查.tsv'
    if qpath.exists() and (not (tables / '02_首次独立检查异常保留.tsv').exists()):
        previous = pd.read_csv(qpath, sep='\t')
        if (~previous.passed).any():
            shutil.copy2(qpath, tables / '02_首次独立检查异常保留.tsv')
    qc = []
    designpath = private / '修订回归独立核查矩阵.parquet'
    designs = [pd.read_parquet(designpath)] if designpath.exists() else []
    diagpath = tables / '02_回归数值与稀疏性诊断.tsv'
    diagnostic = pd.read_csv(diagpath, sep='\t').to_dict('records') if diagpath.exists() else []
    separated = []

    def existing(filename):
        path = tables / filename
        if not path.exists():
            return []
        saved = pd.read_csv(path, sep='\t')
        if 'variance' in saved:
            saved.loc[saved.variance.eq('HC1 robust') & saved.database.ne('eICU'), 'variance'] = 'HC0 sandwich robust'
        return [saved]

    def cached(rows, db, name, outcome=None):
        if not rows:
            return False
        combined = pd.concat([r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in rows], ignore_index=True)
        mask = combined.database.eq(db) & combined.model.eq(name)
        if outcome is not None:
            mask &= combined.outcome.eq(outcome)
        return bool(mask.any())

    def check(name, actual, expected=0, tolerance=1e-08):
        error = 0 if pd.isna(actual) and pd.isna(expected) else abs(float(actual) - float(expected))
        passed = bool(np.isfinite(error) and error <= tolerance)
        qc.append(dict(check=name, actual=actual, expected=expected, difference=error, tolerance=tolerance, passed=passed))
        pd.DataFrame(qc).to_csv(tables / '02_主效应与严重度独立核查.tsv', sep='\t', index=False)
        if not passed:
            raise RuntimeError(f'核查失败：{name}，误差={error}')
    for row in manifest.itertuples(index=False):
        check('冻结参数校验_' + Path(row.source).name, int(sha256(Path(row.source)) == row.SHA256), 1, 0)
    features = pd.read_csv(modelroot / 'feature_order.csv').sort_values('feature_position').feature.tolist()
    check('冻结特征维数', len(features), 48, 0)
    standard = pd.read_csv(modelroot / 'standardization_parameters.csv').set_index('feature').loc[features]
    weights = pd.read_csv(modelroot / 'component_weights.csv').sort_values('raw_component_zero_based')
    mean_table = pd.read_csv(modelroot / 'component_means.csv')
    model = SimpleNamespace(n_components=4, weights_=weights.weight.to_numpy(float), means_=np.asarray([mean_table.loc[mean_table.raw_component_zero_based.eq(i)].set_index('feature').loc[features].component_mean_z for i in range(4)]), covariances_=np.asarray([pd.read_csv(modelroot / f'covariance_{p}.csv', index_col=0).loc[features, features].to_numpy(float) for p in weights.phenotype]))
    assignment = load(modelroot / 'assignment_code_clean.py', 'locked_assignment_only')
    mapping = dict(zip(weights.raw_component_zero_based.astype(int), weights.phenotype))
    candidates = pd.read_parquet(severity.CANDIDATES)
    candidates.stay_id = candidates.stay_id.astype(str)
    candidates.database = candidates.database.replace({'MIMIC-III MetaVision': 'MIMIC-III'})
    eligible = membership.loc[membership.revised_assigned, ['database', 'stay_id', 'phenotype']]
    checked = eligible.merge(candidates[['database', 'stay_id'] + features], on=['database', 'stay_id'], validate='one_to_one')
    labels, posterior = assignment.assign_locked_k4(checked[features].to_numpy(float), standard.MIMIC_IV_observed_mean.to_numpy(float), standard.MIMIC_IV_observed_sd_ddof0.to_numpy(float), model, mapping)
    check('修订保留者冻结标签逐人一致', int(np.sum(labels != checked.phenotype.to_numpy())), 0, 0)
    checked = checked[['database', 'stay_id', 'phenotype']].copy()
    for index in range(4):
        checked['posterior_' + mapping[index]] = posterior[:, index]
    checked['max_posterior_recomputed'] = posterior.max(axis=1)
    checked['posterior_entropy_normalized'] = -np.sum(posterior * np.log(np.clip(posterior, 1e-300, 1)), axis=1) / np.log(4)
    checked.to_parquet(private / '修订保留者冻结后验概率.parquet', index=False)
    print('修订保留者冻结参数和标签门控通过', flush=True)
    data = severity.load_analysis_data()
    trajectory = prediction.trajectory_summaries()
    data = data.merge(trajectory, on=['database', 'stay_id'], how='left', validate='one_to_one')
    old_predictions = pd.read_csv(correction / 'outputs/11_incremental_performance_predictions.csv', dtype={'stay_id': str, 'patient_id': str})
    data = data.merge(old_predictions[['database', 'stay_id', 'patient_id', 'fold']], on=['database', 'stay_id'], how='left', validate='one_to_one')
    data['HR_mean10'], data['RR_mean10'] = (data.HR_mean / 10, data.RR_mean / 10)
    data = data.merge(membership[['database', 'stay_id', 'revised_assigned', 'aspiration_only', 'early_specific', 'earliest_specific_offset']], on=['database', 'stay_id'], how='left', validate='one_to_one')
    check('所有冻结分析行均有资格标记', int(data.revised_assigned.isna().sum()), 0, 0)
    independent_original = pd.read_csv(project / '10_second_round_revision_20261014/data_audit/tables/01_frozen_primary_reproduction.tsv', sep='\t').set_index('database')
    base = ['age10', 'male', 'coverage10']
    oxygen = base + ['SpO2_min10', 'SpO2_min_0_12h_missing']
    full = oxygen + ['active_invasive_ventilation_at_12h', 'active_vasopressor_at_12h', 'active_RRT_at_12h']

    def audit_fit(frame, outcome, model_name, covars, marker=False):
        if marker:
            x = np.column_stack([np.ones(len(frame)), frame.fixed_HRRR_rule, *[frame[c].to_numpy(float) for c in covars]])
            terms = ['intercept', 'fixed_HRRR_rule'] + covars
            term = 1
        else:
            x, terms = severity.design_matrix(frame, covars)
            term = 2
        y = frame[outcome].to_numpy(float)
        groups = frame.hospital_id.astype(str).to_numpy() if frame.database.iloc[0] == 'eICU' else None
        beta, converged = severity.fit_logistic(x, y)
        prefix = frame.database.iloc[0] + '_' + model_name
        separation = linprog(-((2 * y - 1)[:, None] * x).sum(axis=0), A_ub=-((2 * y - 1)[:, None] * x), b_ub=np.zeros(len(y)), bounds=[(-1, 1)] * x.shape[1], method='highs')
        check(prefix + '_分离线性规划成功', int(separation.success), 1, 0)
        quasi_separated = bool(separation.fun < -1e-07)
        if quasi_separated:
            direction = separation.x
            identifying = [terms[i] for i in range(len(terms)) if abs(direction[i]) > 1e-07]
            separated.append(dict(database=frame.database.iloc[0], model=model_name, outcome=outcome, N=len(frame), events=int(y.sum()), separation_objective=float(separation.fun), separating_terms=';'.join(identifying), status='准完全分离：无有限常规最大似然估计；不输出正式OR、CI或风险'))
            pd.DataFrame(separated).to_csv(tables / '02_准完全分离与不可正式估计模型.tsv', sep='\t', index=False)
            diagnostic[:] = [r for r in diagnostic if not (r['database'] == frame.database.iloc[0] and r['model'] == model_name and (r['outcome'] == outcome))]
            diagnostic.append(dict(database=frame.database.iloc[0], model=model_name, outcome=outcome, N=len(frame), events=int(y.sum()), parameters=x.shape[1], rank=np.linalg.matrix_rank(x), events_per_nonintercept_parameter=float(y.sum() / (x.shape[1] - 1)), OR=np.nan, SE=np.nan, converged=False, sparse_warning=True, estimation_status='quasi_separation_no_finite_MLE'))
            pd.DataFrame(diagnostic).to_csv(diagpath, sep='\t', index=False)
            print(prefix, '准完全分离，常规模型不作为正式估计', flush=True)
            return (None, None)
        check(prefix + '_收敛', int(converged), 1, 0)
        check(prefix + '_得分方程', float(np.max(np.abs(x.T @ (y - expit(x @ beta))))), 0, 1e-05)
        kwargs = dict(cov_type='cluster', cov_kwds={'groups': groups, 'use_correction': True}) if groups is not None else dict(cov_type='HC0')
        independent = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-12, **kwargs)
        check(prefix + '_独立GLM系数', float(np.max(np.abs(beta - independent.params))), 0, 1e-07)
        covariance = severity.robust_covariance(x, y, beta, groups)
        check(prefix + '_独立稳健协方差', float(np.max(np.abs(covariance - independent.cov_params()))), 0, 1e-07)
        diagnostic[:] = [r for r in diagnostic if not (r['database'] == frame.database.iloc[0] and r['model'] == model_name and (r['outcome'] == outcome))]
        diagnostic.append(dict(database=frame.database.iloc[0], model=model_name, outcome=outcome, N=len(frame), events=int(y.sum()), parameters=x.shape[1], rank=np.linalg.matrix_rank(x), events_per_nonintercept_parameter=float(y.sum() / (x.shape[1] - 1)), OR=float(np.exp(beta[term])), SE=float(independent.bse[term]), converged=converged, sparse_warning=bool(y.sum() / (x.shape[1] - 1) < 10), estimation_status='regular_MLE_numerically_verified'))
        export = pd.DataFrame(x, columns=['x' + str(i) for i in range(x.shape[1])])
        export['y'], export['model_key'] = (y, prefix)
        export['cluster'] = groups if groups is not None else np.arange(len(frame)).astype(str)
        export['term_names'] = ';'.join(terms)
        if designs:
            designs[0] = designs[0].loc[designs[0].model_key.ne(prefix)]
        designs.append(export)
        pd.DataFrame(diagnostic).to_csv(tables / '02_回归数值与稀疏性诊断.tsv', sep='\t', index=False)
        pd.concat(designs, ignore_index=True).to_parquet(private / '修订回归独立核查矩阵.parquet', index=False)
        return (beta, covariance)
    for db in severity.DATABASES:
        historical, _ = severity.prepare_model_data(data.loc[data.database.eq(db)], 'death_24h')
        beta, _ = audit_fit(historical, 'death_24h', 'historical_gate', base)
        check(db + '_历史主N', len(historical), independent_original.loc[db, 'N'], 0)
        check(db + '_历史主事件', int(historical.death_24h.sum()), independent_original.loc[db, 'events'], 0)
        check(db + '_历史主OR', float(np.exp(beta[2])), independent_original.loc[db, 'OR'], 1e-08)
    print('原三库分母和主OR独立复现门控通过', flush=True)
    revised = data.loc[data.revised_assigned].copy()
    revised = revised.merge(checked.drop(columns='phenotype'), on=['database', 'stay_id'], how='left', validate='one_to_one')
    check('修订后最大后验逐人一致', float(np.abs(revised.max_posterior - revised.max_posterior_recomputed).max()), 0, 1e-10)
    recovery = pd.read_parquet(project / '10_second_round_revision_20261014/data_audit/private/03_comorbidity_source_recovery.parquet')
    recovery.stay_id = recovery.stay_id.astype(str)
    recovered = ['raw_' + c for c in severity.COMORBIDITIES]
    revised = revised.merge(recovery[['database', 'stay_id', 'raw_valid_diagnosis_rows'] + recovered], on=['database', 'stay_id'], how='left', validate='one_to_one')
    revised.to_parquet(private / '修订三库全模块分析输入.parquet', index=False)
    result_rows = existing('02_修订主要效应与严重度.tsv')
    hrrr_rows = existing('03_修订固定HRRR规则.tsv')
    apache_rows = existing('05_修订APACHE同样本敏感性.tsv')
    comorb_rows = existing('06_修订共病同样本敏感性.tsv')
    sensitivity_rows = existing('07_修订肺炎时点与吸入编码敏感性.tsv')
    hospital_rows = existing('04_修订整个住院死亡.tsv')

    def run(frame, outcome, name, covars, seed):
        beta, _ = audit_fit(frame, outcome, name, covars)
        if beta is None:
            return pd.DataFrame([dict(database=frame.database.iloc[0], outcome=outcome, model=name, phenotype=p, reference='P2', N=len(frame), events=int(frame[outcome].sum()), OR=np.nan, CI_low=np.nan, CI_high=np.nan, bootstrap_successful=0, estimation_status='quasi_separation_no_finite_MLE') for p in severity.PHENOTYPES])
        rows, _ = severity.run_model(frame, outcome, name, covars, seed)
        rows['estimation_status'] = 'regular_MLE_numerically_verified'
        rows.loc[rows.variance.eq('HC1 robust'), 'variance'] = 'HC0 sandwich robust'
        check(frame.database.iloc[0] + '_' + name + '_实际500次重抽样', int(rows.bootstrap_successful.min()), 500, 0)
        return rows
    for dbindex, db in enumerate(severity.DATABASES):
        source = revised.loc[revised.database.eq(db)].copy()
        prepared, parameters = severity.prepare_model_data(source, 'death_24h')
        for i, (name, covars) in enumerate([('Model_0_current', base), ('Model_1_harmonized_pre_landmark', oxygen), ('Model_2_extended_landmark_support', full), ('C_current_plus_conventional_HR_RR_means', base + ['HR_mean10', 'RR_mean10'])]):
            if cached(result_rows, db, name, 'death_24h'):
                print(db, name, '已完成结果续用，不重复重抽样', flush=True)
                continue
            rows = run(prepared, 'death_24h', name, covars, 7000 + dbindex * 10000 + i * 1000)
            result_rows.append(rows)
            pd.concat(result_rows).to_csv(tables / '02_修订主要效应与严重度.tsv', sep='\t', index=False)
            print(db, name, '完成', flush=True)
        for i, (name, covars) in enumerate([('unadjusted', []), ('current_adjusted', base), ('harmonized_severity_adjusted', full)]):
            if cached(hrrr_rows, db, name):
                continue
            beta, _ = audit_fit(prepared, 'death_24h', 'HRRR_' + name, covars, marker=True)
            if beta is None:
                row = dict(database=db, model=name, N=len(prepared), events=int(prepared.death_24h.sum()), OR=np.nan, CI_low=np.nan, CI_high=np.nan, bootstrap_successful=0, estimation_status='quasi_separation_no_finite_MLE')
            else:
                row = prediction.marker_association(prepared, db, name, covars, severity, 32000 + dbindex * 1000 + i * 100)
                if db != 'eICU' and row.get('variance') == 'HC1 robust':
                    row['variance'] = 'HC0 sandwich robust'
                row['estimation_status'] = 'regular_MLE_numerically_verified'
            hrrr_rows.append(pd.DataFrame([row]))
            pd.concat(hrrr_rows).to_csv(tables / '03_修订固定HRRR规则.tsv', sep='\t', index=False)
        hospital, _ = severity.prepare_model_data(source, 'hospital_mortality')
        for i, (name, covars) in enumerate([('hospital_current', base), ('hospital_harmonized', full)]):
            if cached(hospital_rows, db, name, 'hospital_mortality'):
                continue
            hospital_rows.append(run(hospital, 'hospital_mortality', name, covars, 60000 + dbindex * 2000 + i * 1000))
            pd.concat(hospital_rows).to_csv(tables / '04_修订整个住院死亡.tsv', sep='\t', index=False)
        if db == 'eICU':
            apache, _ = severity.prepare_model_data(source, 'hospital_mortality', require_apache=True)
            for i, (name, covars) in enumerate([('APACHE_same_sample_baseline', base), ('APACHE_IVa_adjusted_existing_specification', base + ['APS10'])]):
                if cached(apache_rows, db, name, 'hospital_mortality'):
                    continue
                apache_rows.append(run(apache, 'hospital_mortality', name, covars, 16000 + i * 1000))
                pd.concat(apache_rows).to_csv(tables / '05_修订APACHE同样本敏感性.tsv', sep='\t', index=False)
            sensitivity = source.loc[source.early_specific.eq(True)].copy()
            sensitivity, _ = severity.prepare_model_data(sensitivity, 'death_24h')
            if not cached(sensitivity_rows, db, 'explicit_pneumonia_documented_by12', 'death_24h'):
                sensitivity_rows.append(run(sensitivity, 'death_24h', 'explicit_pneumonia_documented_by12', base, 51000))
        else:
            complete = prepared.dropna(subset=recovered).copy()
            check(db + '_恢复共病不额外损失主要样本', len(complete), len(prepared), 0)
            for i, (name, covars) in enumerate([('recovered_same_sample_baseline', base), ('recovered_full_comorbidity', base + recovered)]):
                if cached(comorb_rows, db, name, 'death_24h'):
                    continue
                comorb_rows.append(run(complete, 'death_24h', name, covars, 65000 + dbindex * 2000 + i * 1000))
                pd.concat(comorb_rows).to_csv(tables / '06_修订共病同样本敏感性.tsv', sep='\t', index=False)
            sensitivity, _ = severity.prepare_model_data(source.loc[~source.aspiration_only], 'death_24h')
            if not cached(sensitivity_rows, db, 'exclude_aspiration_only', 'death_24h'):
                sensitivity_rows.append(run(sensitivity, 'death_24h', 'exclude_aspiration_only', base, 55000 + dbindex * 1000))
        pd.concat(sensitivity_rows).to_csv(tables / '07_修订肺炎时点与吸入编码敏感性.tsv', sep='\t', index=False)
        print(db, '主效应/严重度/固定规则及规定敏感性完成', flush=True)
    protected_rows = [dict(path=p, sha256_before=h, sha256_after=sha256(Path(p)), passed=h == sha256(Path(p))) for p, h in protected.items()]
    pd.DataFrame(protected_rows).to_csv(tables / '02_主要分析只读保护.tsv', sep='\t', index=False)
    check('原始输入冻结参数和现稿未变', int(sum((not row['passed'] for row in protected_rows))), 0, 0)
    state = dict(stage='识别修正主效应与固定敏感性', status='常规模型已完成；分离模型明确标记不可正式估计', checks=len(qc), quasi_separation_models=separated, remaining=['预测性能', '覆盖阈值', '临床表征', '分布与时间顺序', '竞争风险和复合', '医院层与后验', '稿件同步'])
    (out / '主效应检查点.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False), flush=True)
if __name__ == '__source_reference_run__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
