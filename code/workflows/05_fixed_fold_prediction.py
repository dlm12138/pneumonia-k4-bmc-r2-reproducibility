#!/usr/bin/env python3
"""筛选修订人群后保留原折号；折内插补及标准化，配对重抽样500次。"""
from __future__ import annotations

import os, sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from release_runtime import load_actual, init_output


import argparse
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def load(path):
    return load_actual(path, name if "name" in locals() else "actual_functions")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    project, out = Path(args.project), Path(args.output)
    init_output(out)
    table, private = out / 'tables', out / 'private'
    sourcepath = private / '修订三库全模块分析输入.parquet'
    scriptpath = project / 'revision_R1_R2_data/scripts/08_hrrr_and_incremental_performance.py'
    pred = load(scriptpath)
    sourcehash, scripthash = sha(sourcepath), sha(scriptpath)
    data = pd.read_parquet(sourcepath)
    qc, results, fold_rows = [], [], []

    def check(name, actual, expected=0, tolerance=1e-10):
        error = abs(float(actual) - float(expected))
        passed = bool(np.isfinite(error) and error <= tolerance)
        qc.append(dict(check=name, actual=actual, expected=expected, difference=error,
                       tolerance=tolerance, passed=passed))
        pd.DataFrame(qc).to_csv(table / '08_预测独立核查.tsv', sep='\t', index=False)
        if not passed:
            raise RuntimeError('预测检查失败：' + name)

    for db in ['MIMIC-IV', 'MIMIC-III', 'eICU']:
        src = data.loc[data.database.eq(db)].copy()
        needed = ['death_24h', 'age', 'male', 'observed_dimensions_48d', 'patient_id', 'fold']
        if db == 'eICU':
            needed.append('hospital_id')
        frame = src.dropna(subset=needed).reset_index(drop=True)
        groupname = 'hospital_id' if db == 'eICU' else 'patient_id'
        groups = frame[groupname].astype(str).to_numpy()
        folds = frame.fold.to_numpy(int)
        y = frame.death_24h.to_numpy(int)
        check(db + '_既有折号数量', len(np.unique(folds)), 5, 0)
        check(db + '_同一患者或医院仅一折', frame.groupby(groupname).fold.nunique().max(), 1, 0)
        numeric = ['age', 'male', 'observed_dimensions_48d', 'SpO2_min_0_12h',
                   'active_invasive_ventilation_at_12h', 'active_vasopressor_at_12h', 'active_RRT_at_12h']
        conventional = [f'{v}_{s}' for v in pred.VITALS for s in ['mean', 'count', 'sd', 'slope']]
        models = {'A_base_severity': (numeric, []),
                  'B_base_plus_HRRR': (numeric + ['fixed_HRRR_rule'], []),
                  'C_base_plus_K4': (numeric, ['phenotype']),
                  'D_base_plus_conventional': (numeric + conventional, []),
                  'E_base_conventional_plus_K4': (numeric + conventional, ['phenotype'])}
        probabilities = {}
        for model, (num, cat) in models.items():
            probability = np.full(len(y), np.nan)
            for fold in sorted(np.unique(folds)):
                train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                check(db + '_' + model + '_组无泄漏_' + str(fold), len(set(groups[train]) & set(groups[test])), 0, 0)
                check(db + '_' + model + '_训练集两个结局类别_' + str(fold), len(np.unique(y[train])), 2, 0)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*penalty.*', category=FutureWarning)
                    pipeline = pred.prediction_pipeline(num, cat)
                    pipeline.fit(frame.iloc[train][num + cat], y[train])
                    probability[test] = pipeline.predict_proba(frame.iloc[test][num + cat])[:, 1]
                estimator = pipeline.steps[-1][1]
                check(db + '_' + model + '_迭代未达上限_' + str(fold), int(np.max(estimator.n_iter_) < estimator.max_iter), 1, 0)
            check(db + '_' + model + '_概率有限且在零一内', int((~np.isfinite(probability) | (probability < 0) | (probability > 1)).sum()), 0, 0)
            point = pred.metric_values(y, probability)
            n1, n0 = int(y.sum()), int(len(y) - y.sum())
            independent_auc = (rankdata(probability)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            check(db + '_' + model + '_独立秩和AUROC', independent_auc, point['AUROC'])
            check(db + '_' + model + '_独立Brier', np.mean((y - probability) ** 2), point['Brier'])
            clipped = np.clip(probability, np.finfo(float).eps, 1 - np.finfo(float).eps)
            check(db + '_' + model + '_独立对数损失', -np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped)), point['log_loss'])
            probabilities[model] = probability
            print(db, model, '固定折号预测完成', flush=True)
        export = frame[['database', 'stay_id', 'patient_id', 'hospital_id', 'death_24h', 'fold']].copy()
        for model, probability in probabilities.items():
            export['prediction_' + model] = probability
        export.to_parquet(private / (db + '_修订固定折号折外预测.parquet'), index=False)
        metrics, deltas = pred.bootstrap_prediction_metrics(y, probabilities, groups, db)
        performance = pd.concat([metrics, deltas], ignore_index=True)
        check(db + '_真实配对重抽样次数', performance.bootstrap_successful.min(), 500, 0)
        performance['fold_provenance'] = '原研究保存折号；筛选后同折重拟合；折内插补和标准化'
        results.append(performance)
        pd.concat(results).to_csv(table / '08_修订预测性能与配对增量.tsv', sep='\t', index=False)
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            fold_rows.append(dict(database=db, fold=fold, N=int(mask.sum()), events=int(y[mask].sum()),
                                  groups=len(np.unique(groups[mask]))))
        pd.DataFrame(fold_rows).to_csv(table / '08_修订各折人数与事件.tsv', sep='\t', index=False)
        print(db, '预测及500次配对重抽样完成', flush=True)
    check('预测输入未改动', int(sha(sourcepath) == sourcehash), 1, 0)
    check('真实预测函数未改动', int(sha(scriptpath) == scripthash), 1, 0)
    state = dict(status='三库修订固定分折预测及独立点值核查完成', checks=len(qc), failed=0,
                 source_sha256=sourcehash, real_script_sha256=scripthash,
                 limitation='重抽样使用既有折外预测，不代表全嵌套重抽样重新训练；不证明临床效用')
    (out / '预测检查点.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
