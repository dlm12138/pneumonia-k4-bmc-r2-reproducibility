"""Actual recovered comorbidity and strict event subset assembly.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'生成修订临床表征、诊断时间/选择证据及竞争风险输入；不修改稿件。'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(args):
    project, out = (Path(args.project), Path(args.output))
    table, private = (out / 'tables', out / 'private')
    membership = pd.read_parquet(private / '三库修订肺炎资格与纳入标记.parquet')
    source = pd.read_parquet(private / '修订三库全模块分析输入.parquet')
    recoverypath = project / '10_second_round_revision_20261014/data_audit/private/03_comorbidity_source_recovery.parquet'
    recovery = pd.read_parquet(recoverypath)
    recovery.stay_id = recovery.stay_id.astype(str)
    qc = []

    def check(name, actual, expected=0, tol=1e-10):
        error = abs(float(actual) - float(expected))
        passed = bool(np.isfinite(error) and error <= tol)
        qc.append(dict(check=name, actual=actual, expected=expected, difference=error, tolerance=tol, passed=passed))
        pd.DataFrame(qc).to_csv(table / '09_临床选择和事件输入独立核查.tsv', sep='\t', index=False)
        if not passed:
            raise RuntimeError(name)
    selection = []
    selected = source.loc[source.database.isin(['MIMIC-IV', 'MIMIC-III'])].merge(recovery[['database', 'stay_id', 'old_source_row_present', 'old_complete', 'missing_reason']], on=['database', 'stay_id'], validate='one_to_one')
    for db, frame in selected.groupby('database', sort=False):
        check(db + '_原始有效诊断至少一行', int(frame.raw_valid_diagnosis_rows.le(0).sum()), 0, 0)
        complete = frame.old_complete.astype(bool)
        for name, mask in [('修订人群全部', pd.Series(True, index=frame.index)), ('原受限来源完整', complete), ('原受限来源缺失', ~complete)]:
            part = frame.loc[mask]
            selection.append(dict(database=db, selection=name, N=len(part), events=int(part.death_24h.sum()), death_pct=100 * part.death_24h.mean(), raw_diagnosis_available_N=int(part.raw_valid_diagnosis_rows.gt(0).sum())))
        for reason, part in frame.loc[~complete].groupby('missing_reason', dropna=False):
            selection.append(dict(database=db, selection=str(reason), N=len(part), events=int(part.death_24h.sum()), death_pct=100 * part.death_24h.mean(), raw_diagnosis_available_N=int(part.raw_valid_diagnosis_rows.gt(0).sum())))
        check(db + '_旧完整和缺失人数对账', int(complete.sum() + (~complete).sum()), len(frame), 0)
    pd.DataFrame(selection).to_csv(table / '09_修订共病来源选择与死亡流失.tsv', sep='\t', index=False)
    for c in [c for c in source if c.startswith('raw_') and c != 'raw_valid_diagnosis_rows']:
        newcolumn = 'comorb_' + c[4:]
        ix = source.database.isin(['MIMIC-IV', 'MIMIC-III'])
        check(c + '_恢复编码仅零一', int((~source.loc[ix, c].isin([0, 1])).sum()), 0, 0)
        source.loc[ix, newcolumn] = source.loc[ix, c]
    clinical = load(project / 'revision_R1_R2_data/scripts/07_clinical_characterization_and_stress.py', 'real_characterization')
    source = clinical.append_vital_summaries(source)
    clinical_tables, _ = clinical.characterize(source)
    for db, results in clinical_tables.items():
        results['comorbidity_source_note'] = 'MIMIC使用原始诊断表恢复的编码；有有效诊断且无该病编码记0，不等同临床确认无病；eICU沿用原可得描述'
        results.to_csv(table / ('10_' + db + '_修订四型临床表征.tsv'), sep='\t', index=False)
        for p in ['P1', 'P2', 'P3', 'P4']:
            row = results.loc[results.variable.eq('age')].iloc[0]
            check(db + '_' + p + '_年龄表征分母', int(row[p + '_observed_N']), int((source.database.eq(db) & source.phenotype.eq(p)).sum()), 0)
    clinical.stress_profile(source).to_csv(table / '10_修订P3与P2临床描述.tsv', sep='\t', index=False)
    timing = []
    for name, mask in [('严格肺炎界标人群', membership.revised_landmark), ('严格肺炎可分型人群', membership.revised_assigned)]:
        frame = membership.loc[membership.database.eq('eICU') & mask]
        check(name + '_首次合格诊断时点无缺失', int(frame.earliest_specific_offset.isna().sum()), 0, 0)
        strata = [('ICU前', frame.earliest_specific_offset.lt(0)), ('ICU入科至12小时', frame.earliest_specific_offset.between(0, 720, inclusive='both')), ('仅12小时后', frame.earliest_specific_offset.gt(720))]
        check(name + '_诊断时间分层完备', sum((int(m.sum()) for _, m in strata)), len(frame), 0)
        for stratum, sub in strata:
            timing.append(dict(population=name, timing=stratum, N=int(sub.sum()), denominator=len(frame), pct=100 * sub.mean()))
    pd.DataFrame(timing).to_csv(table / '11_修订明确肺炎记录时点.tsv', sep='\t', index=False)
    eventroot = project / '11_eicu_outcome_adjudication_20261014/private'
    epath, vpath = (eventroot / '全部可分型患者结局证据.parquet', eventroot / '竞争风险各方案逐人事件.parquet')
    hashes = {str(p): sha(p) for p in [epath, vpath, recoverypath]}
    evidence, events = (pd.read_parquet(epath), pd.read_parquet(vpath))
    keep = set(membership.loc[membership.database.eq('eICU') & membership.revised_assigned, 'stay_id'].astype(str))
    evidence.stay_id, events.stay_id = (evidence.stay_id.astype(str), events.stay_id.astype(str))
    evidence, events = (evidence.loc[evidence.stay_id.isin(keep)].copy(), events.loc[events.stay_id.isin(keep)].copy())
    check('修订可分型病例结局证据齐全', len(evidence), len(keep), 0)
    check('修订可分型病例严格大于12小时', int((evidence.unitdischargeoffset <= 720).sum()), 0, 0)
    evidence.to_parquet(private / '修订eICU逐人原始结局证据.parquet', index=False)
    events.to_parquet(private / '修订eICU竞争风险逐人事件.parquet', index=False)
    statuses = events.groupby(['scenario', 'horizon_hours', 'status']).size().rename('N').reset_index()
    statuses.to_csv(table / '12_修订竞争风险分类对账.tsv', sep='\t', index=False)
    conflict_rows = []
    for name, mask in [('全部', np.ones(len(evidence), bool)), ('原始字段一致', ~evidence.core_conflict), ('原始核心冲突', evidence.core_conflict)]:
        part = evidence.loc[mask]
        conflict_rows.append(dict(selection=name, N=len(part), primary_events=int(part.death_24h.sum()), main_complete_N=int(part[['age', 'male', 'hospital_id']].notna().all(axis=1).sum())))
    pd.DataFrame(conflict_rows).to_csv(table / '12_修订结局冲突与选择.tsv', sep='\t', index=False)
    protected = pd.DataFrame([dict(path=p, sha256_before=h, sha256_after=sha(Path(p)), passed=h == sha(Path(p))) for p, h in hashes.items()])
    protected.to_csv(table / '09_临床与事件输入只读保护.tsv', sep='\t', index=False)
    check('诊断恢复和原始结局证据未改动', int((~protected.passed).sum()), 0, 0)
    state = dict(status='临床、选择和时间证据完成；竞争风险仅完成输入，尚待实际重拟合', checks=len(qc), failed=0)
    (out / '临床与事件输入检查点.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False), flush=True)
if __name__ == '__source_reference_run__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
