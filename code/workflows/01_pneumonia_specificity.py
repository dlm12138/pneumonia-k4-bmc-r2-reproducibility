#!/usr/bin/env python3
"""结局盲识别修正；先生成资格证据、分母与冻结分配门控，不计算新效应。"""
from __future__ import annotations

import os, sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from release_runtime import load_actual, init_output


import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    project, out, raw = Path(args.project), Path(args.output), Path(args.raw)
    init_output(out)
    tables, private = out / 'tables', out / 'private'
    tables.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (out / 'tmp').mkdir(exist_ok=True)
    history = project / '01_k5-7.29_history/k5-7.29'
    audit = project / '10_second_round_revision_20261014/data_audit'
    frozenpath = history / 'analysis_main12_clinical_actionability_20260730/07_common4_selected_assignments.parquet'
    matrixpath = history / 'analysis_landmark_external_upgrade_20260730/14_landmark12_candidate_matrix.parquet'
    frozen = pd.read_parquet(frozenpath)
    frozen.stay_id = frozen.stay_id.astype(str)
    matrix = pd.read_parquet(matrixpath)
    matrix.database = matrix.database.replace({'MIMIC-III MetaVision': 'MIMIC-III'})
    matrix.stay_id = matrix.stay_id.astype(str)
    flow = pd.read_parquet(audit / 'private/02_cohort_exclusion_reasons.parquet')
    flow.stay_id = flow.stay_id.astype(str)
    initial_hashes = {str(p): sha256(p) for p in [frozenpath, matrixpath, out / '肺炎证据修订规则_拟合前固定.md']}
    qc = []

    def check(name, actual, expected=0, tolerance=0):
        error = abs(float(actual) - float(expected))
        passed = bool(np.isfinite(error) and error <= tolerance)
        qc.append(dict(check=name, actual=actual, expected=expected, difference=error, tolerance=tolerance, passed=passed))
        pd.DataFrame(qc).to_csv(tables / '01_资格门控独立核查.tsv', sep='\t', index=False)
        if not passed:
            raise RuntimeError('门控失败：' + name)

    dictionary_rows, code_rows, memberships = [], [], []
    specs = [('MIMIC-IV', 'mimiciv', raw / 'MIMIC-IV_3.1/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz',
              raw / 'MIMIC-IV_3.1/mimic-iv-3.1/hosp/d_icd_diagnoses.csv.gz', 'hadm_id', 'icd_code', 'icd_version'),
             ('MIMIC-III', 'mimiciii', raw / 'MIMIC-III_1.4/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv.gz',
              raw / 'MIMIC-III_1.4/mimic-iii-clinical-database-1.4/D_ICD_DIAGNOSES.csv.gz', 'HADM_ID', 'ICD9_CODE', None)]
    for db, prefix, filepath, dictpath, hadm, codecol, versioncol in specs:
        evidence = pd.read_parquet(audit / f'private/03_{prefix}_pneumonia_code_evidence.parquet')
        evidence.stay_id = evidence.stay_id.astype(str)
        evidence.hadm_id = pd.to_numeric(evidence.hadm_id).astype('int64')
        relevant = set(evidence.hadm_id)
        dictionary = pd.read_csv(dictpath, dtype='string')
        dictionary.columns = dictionary.columns.str.lower()
        dictionary['code'] = dictionary['icd_code' if db == 'MIMIC-IV' else 'icd9_code'].str.replace('.', '', regex=False).str.strip().str.upper()
        dictionary['version'] = pd.to_numeric(dictionary.icd_version) if db == 'MIMIC-IV' else 9
        categories = {}
        for _, row in dictionary.iterrows():
            c, version = row.code, int(row.version)
            pneumonia = (version == 9 and (c.startswith(tuple(str(i) for i in range(480, 487))) or c in {'4870', '48801', '48811', '99731'})) or (version == 10 and c.startswith(('J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J100', 'J110', 'J95851')))
            aspiration = (version == 9 and c == '5070') or (version == 10 and c == 'J690')
            legacy = pneumonia or aspiration or (version == 9 and c.startswith(('4880', '4881'))) or (version == 10 and c.startswith(('U071', 'J101')))
            if legacy:
                category = '肺炎特异编码' if pneumonia else ('吸入相关编码' if aspiration else '不得单独纳入')
                categories[(version, c)] = category
                dictionary_rows.append(dict(database=db, version=version, code=c, dictionary_title=row.long_title, category=category))
        parts = []
        usecols = [hadm, codecol] + ([versioncol] if versioncol else [])
        for chunk in pd.read_csv(filepath, usecols=usecols, dtype={codecol: 'string'}, chunksize=400000, low_memory=False):
            part = chunk.loc[chunk[hadm].isin(relevant)].copy()
            if not len(part):
                continue
            part['code'] = part[codecol].fillna('').str.replace('.', '', regex=False).str.strip().str.upper()
            part['version'] = pd.to_numeric(part[versioncol]) if versioncol else 9
            part['category'] = [categories.get((int(v), c), '') for v, c in zip(part.version, part.code)]
            part = part.loc[part.category.ne(''), [hadm, 'code', 'version', 'category']].rename(columns={hadm: 'hadm_id'})
            parts.append(part)
        rows = pd.concat(parts, ignore_index=True)
        explicit = set(rows.loc[rows.category.eq('肺炎特异编码'), 'hadm_id'])
        aspiration = set(rows.loc[rows.category.eq('吸入相关编码'), 'hadm_id'])
        evidence['explicit_pneumonia'] = evidence.hadm_id.isin(explicit)
        evidence['aspiration_code'] = evidence.hadm_id.isin(aspiration)
        evidence['specific_eligible'] = evidence.explicit_pneumonia | evidence.aspiration_code
        evidence['aspiration_only'] = evidence.aspiration_code & ~evidence.explicit_pneumonia
        evidence['early_specific'] = pd.NA
        evidence['earliest_specific_offset'] = np.nan
        check(db + '_肺炎规则资格覆盖', int((~evidence.qualifying_code).sum()), 0)
        membership = evidence[['database', 'stay_id', 'specific_eligible', 'explicit_pneumonia', 'aspiration_code', 'aspiration_only', 'early_specific', 'earliest_specific_offset']]
        memberships.append(membership)
        counts = rows.groupby(['version', 'code', 'category']).agg(diagnosis_rows=('hadm_id', 'size'), hospital_admissions=('hadm_id', 'nunique')).reset_index()
        counts['database'] = db
        code_rows.append(counts)
        rows['database'] = db
        rows.to_parquet(private / f'{prefix}_合格与被排除编码逐入院证据.parquet', index=False)
        print(db, '原始诊断资格核查完成', flush=True)

    old = pd.read_parquet(audit / 'private/04_eicu_patient_evidence.parquet')
    old.stay_id = old.stay_id.astype(str)
    selected = set(pd.to_numeric(old.stay_id))
    pattern = re.compile(r'\b(?:broncho)?pneumonia\b', re.IGNORECASE)
    concern = re.compile(r'\b(?:no|without|ruled\s+out|rule\s+out|history\s+of)\b|\br/o\b', re.IGNORECASE)
    evidence_rows = []
    eicu = raw / 'eICU-CRD_2.0/EICU 2.0数据'
    for filename, timecol, wanted in [('diagnosis.csv.gz', 'diagnosisoffset', ['diagnosisstring']),
                                       ('admissionDx.csv.gz', 'admitdxenteredoffset', ['admitdxpath', 'admitdxname', 'admitdxtext'])]:
        header = pd.read_csv(eicu / filename, nrows=0).columns.tolist()
        cols = [c for c in wanted if c in header]
        if not cols:
            raise RuntimeError('未找到指定诊断文本字段：' + filename)
        for chunk in pd.read_csv(eicu / filename, usecols=['patientunitstayid', timecol] + cols, chunksize=250000, low_memory=False):
            chunk = chunk.loc[chunk.patientunitstayid.isin(selected)]
            for col in cols:
                text = chunk[col].fillna('').astype(str)
                hit = text.str.contains(pattern, regex=True)
                if not hit.any():
                    continue
                part = chunk.loc[hit, ['patientunitstayid', timecol, col]].copy()
                part.columns = ['stay_id', 'offset', 'text']
                part['source'] = filename + ':' + col
                part['context_concern'] = text.loc[hit].str.contains(concern, regex=True).to_numpy()
                evidence_rows.append(part)
        print('eICU', filename, '明确词边界文本扫描完成', flush=True)
    texts = pd.concat(evidence_rows, ignore_index=True).drop_duplicates()
    texts.stay_id = texts.stay_id.astype(str)
    texts.offset = pd.to_numeric(texts.offset, errors='coerce')
    texts.to_parquet(private / 'eICU肺炎词边界逐条证据.parquet', index=False)
    summary = texts.groupby('stay_id').agg(earliest_specific_offset=('offset', 'min'), qualifying_text_rows=('text', 'size'), context_concern=('context_concern', 'any'))
    eu = old[['database', 'stay_id']].merge(summary, on='stay_id', how='left', validate='one_to_one')
    eu['specific_eligible'] = eu.qualifying_text_rows.notna()
    eu['explicit_pneumonia'] = eu.specific_eligible
    eu['aspiration_code'] = False
    eu['aspiration_only'] = False
    eu['early_specific'] = eu.specific_eligible & eu.earliest_specific_offset.le(720)
    check('eICU合格文本否定或病史提示', int(eu.context_concern.eq(True).sum()), 0)
    eu[['database', 'stay_id', 'specific_eligible', 'explicit_pneumonia', 'aspiration_code', 'aspiration_only', 'early_specific', 'earliest_specific_offset']].pipe(memberships.append)
    text_summary = texts.groupby(['source', 'text', 'context_concern']).agg(matching_rows=('stay_id', 'size'), distinct_stays=('stay_id', 'nunique')).reset_index()
    text_summary.to_csv(tables / '01_eICU合格诊断文本模式.tsv', sep='\t', index=False)
    pd.DataFrame(dictionary_rows).to_csv(tables / '01_修订ICD字典.tsv', sep='\t', index=False)
    pd.concat(code_rows, ignore_index=True).to_csv(tables / '01_实际纳入及排除编码计数.tsv', sep='\t', index=False)

    member = pd.concat(memberships, ignore_index=True)
    check('三库资格键唯一', int(member.duplicated(['database', 'stay_id']).sum()), 0)
    member = member.merge(flow[['database', 'stay_id', 'exclusion_reason', 'los_hours']], on=['database', 'stay_id'], how='left', validate='one_to_one')
    check('三库均有原流程记录', int(member.exclusion_reason.isna().sum()), 0)
    member['historical_landmark'] = member.exclusion_reason.eq('纳入12小时landmark')
    member['strict_boundary'] = member.historical_landmark & member.los_hours.gt(12)
    member['revised_landmark'] = member.strict_boundary & member.specific_eligible
    member = member.merge(frozen[['database', 'stay_id', 'phenotype']], on=['database', 'stay_id'], how='left', validate='one_to_one')
    member['revised_assigned'] = member.revised_landmark & member.phenotype.notna()
    check('历史可分配成员与冻结表同数', int(member.phenotype.notna().sum()), len(frozen))
    check('历史界标成员与冻结矩阵同数', int(member.historical_landmark.sum()), len(matrix))
    member.to_parquet(private / '三库修订肺炎资格与纳入标记.parquet', index=False)
    gate_rows, phenotype_rows = [], []
    for db, frame in member.groupby('database', sort=False):
        masks = [('原始首次肺炎相关ICU', pd.Series(True, index=frame.index)),
                 ('原始界标', frame.historical_landmark), ('严格界标', frame.strict_boundary),
                 ('仅资格修正', frame.specific_eligible), ('修订界标', frame.revised_landmark),
                 ('修订可分配', frame.revised_assigned),
                 ('修订界标且仅吸入相关编码', frame.revised_landmark & frame.aspiration_only)]
        for stage, mask in masks:
            gate_rows.append(dict(database=db, stage=stage, N=int(mask.sum()),
                                 assigned_N=int((mask & frame.phenotype.notna()).sum())))
        for version, mask in [('historical', frame.phenotype.notna()), ('revised', frame.revised_assigned)]:
            sub = frame.loc[mask]
            for phenotype in ['P1', 'P2', 'P3', 'P4']:
                phenotype_rows.append(dict(database=db, version=version, phenotype=phenotype,
                                           N=int(sub.phenotype.eq(phenotype).sum()), total_N=len(sub),
                                           percent=100 * sub.phenotype.eq(phenotype).mean()))
    pd.DataFrame(gate_rows).to_csv(tables / '01_结局盲分母与资格门控.tsv', sep='\t', index=False)
    pd.DataFrame(phenotype_rows).to_csv(tables / '01_冻结四型筛选后分布.tsv', sep='\t', index=False)
    pd.DataFrame([dict(path=p, sha256_before=h, sha256_after=sha256(Path(p)), passed=h == sha256(Path(p))) for p, h in initial_hashes.items()]).to_csv(tables / '01_输入保护.tsv', sep='\t', index=False)
    state = dict(stage='识别与分母门控', status='通过；尚未计算修订效应',
                 checks=len(qc), all_passed=all(r['passed'] for r in qc), no_K4_refitting=True)
    (out / '阶段检查点.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw', required=True)
    main(parser.parse_args())
