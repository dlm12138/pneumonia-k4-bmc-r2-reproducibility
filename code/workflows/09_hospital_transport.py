
import os, sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from release_runtime import load_actual, init_output
#!/usr/bin/env python3
"""修订队列医院层统计与逐院剔除；使用真实原公式并独立复核。"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


def load(path,name):
    return load_actual(path, name if "name" in locals() else "actual_functions")


def main(args):
    project,out=Path(args.project),Path(args.output);init_output(out);tables=out/'tables';private=out/'private'
    module=load(project/'revision_R1_R2_data/scripts/12_eicu_hospital_transport.py','hospital_methods')
    severity=load(project/'revision_R1_R2_data/scripts/05_harmonized_severity_and_apache.py','numerical_methods')
    source=pd.read_parquet(private/'修订三库全模块分析输入.parquet')
    source=source.loc[source.database.eq('eICU')].copy()
    frame=source.loc[source.phenotype.isin(['P2','P3'])].copy()
    frame['p3']=frame.phenotype.eq('P3').astype(int);frame['age10']=frame.age/10;frame['coverage10']=frame.observed_dimensions_48d/10
    frame.to_parquet(private/'修订医院P2P3输入.parquet',index=False)
    required=['death_24h','p3','age10','male','coverage10','hospital_id']
    complete=frame.dropna(subset=required).copy();qc=[]
    def check(name,actual,expected=0,tol=1e-7):
        error=abs(float(actual)-float(expected));passed=bool(np.isfinite(error) and error<=tol)
        qc.append(dict(check=name,actual=actual,expected=expected,difference=error,tolerance=tol,passed=passed))
        pd.DataFrame(qc).to_csv(tables/'19_医院层独立核查.tsv',sep='\t',index=False)
        if not passed:raise RuntimeError(name)
    site=module.site_effects(frame);sufficient=int(site.eligible_for_hospital_specific.sum())
    site['included_in_synthesis']=site.eligible_for_hospital_specific&(sufficient>=10)
    # 真实原研究是粗2x2比值比，不包装成医院内完全调整效应。
    for index,row in site.iterrows():
        d=frame.loc[frame.hospital_id.eq(row.hospital_id)]
        counts=np.array([[((d.p3==p)&(d.death_24h==e)).sum() for e in [1,0]] for p in [1,0]],float)
        check(f'院{index}_P2P3人数',counts.sum(),row.N,0)
        check(f'院{index}_P3事件',counts[0,0],row.P3_events,0);check(f'院{index}_P2事件',counts[1,0],row.P2_events,0)
        if np.any(counts==0):counts+=.5
        independent=np.log(counts[0,0])+np.log(counts[1,1])-np.log(counts[0,1])-np.log(counts[1,0])
        check(f'院{index}_独立粗logOR',independent,row.log_OR,1e-12)
        check(f'院{index}_独立粗SE',np.sqrt(np.sum(1/counts)),row.SE_log_OR,1e-12)
    site.to_parquet(private/'修订医院特异粗效应.parquet',index=False)
    loho=module.loho(frame)
    for index,row in loho.iterrows():
        d=complete.loc[complete.hospital_id.ne(row.excluded_hospital_id)]
        x=sm.add_constant(d[['p3','age10','male','coverage10']],has_constant='add').to_numpy(float)
        y=d.death_24h.to_numpy(float)
        beta,converged=severity.fit_logistic(x,y)
        check(f'逐院{index}_独立收敛',converged,1,0)
        covariance=severity.robust_covariance(x,y,beta,d.hospital_id.to_numpy())
        check(f'逐院{index}_N',len(d),row.N,0)
        check(f'逐院{index}_系数',beta[1],row.beta)
        check(f'逐院{index}_聚类SE',np.sqrt(covariance[1,1]),row.SE)
    loho.to_parquet(private/'修订逐院剔除效应.parquet',index=False)
    print('医院粗效应及全部逐院剔除独立核查通过',flush=True)
    member=pd.read_parquet(private/'三库修订肺炎资格与纳入标记.parquet')
    candidate=pd.read_parquet(project/'01_k5-7.29_history/k5-7.29/analysis_k4_evidence_deepening_20260801/01_candidate_outcomes_and_coverage.parquet')
    candidate.stay_id=candidate.stay_id.astype(str)
    keys=member.loc[member.database.eq('eICU')&member.revised_landmark,['database','stay_id']]
    candidate=candidate.merge(keys,on=['database','stay_id'],validate='one_to_one')
    transport=candidate.groupby('hospital_id').agg(candidate_n=('stay_id','size'),eligible_n=('k4_eligible','sum')).reset_index()
    posterior=source.groupby('hospital_id').agg(median_max_posterior=('max_posterior','median'),assigned_events=('death_24h','sum')).reset_index()
    transport=transport.merge(posterior,on='hospital_id',how='left',validate='one_to_one')
    transport['assignment_coverage']=transport.eligible_n/transport.candidate_n
    check('医院候选人数总和',transport.candidate_n.sum(),len(candidate),0)
    check('医院可分配人数总和',transport.eligible_n.sum(),len(source),0)
    counts=pd.crosstab(source.hospital_id,source.phenotype).reindex(columns=['P1','P2','P3','P4'],fill_value=0).reset_index()
    transport=transport.merge(counts,on='hospital_id',how='left',validate='one_to_one')
    transport.to_parquet(private/'修订医院覆盖率和后验汇总.parquet',index=False)
    eligible_hospitals=transport.loc[transport.candidate_n.ge(20)]
    low=eligible_hospitals.loc[eligible_hospitals.assignment_coverage.lt(.8)]
    posterior_available=eligible_hospitals.loc[eligible_hospitals.median_max_posterior.notna()]
    summary=dict(candidate_N=len(candidate),assigned_N=len(source),P2P3_N=len(frame),P2P3_complete_N=len(complete),
      P2P3_complete_events=int(complete.death_24h.sum()),candidate_hospitals=len(transport),assigned_hospitals=int(transport.eligible_n.gt(0).sum()),
      P2P3_hospitals=complete.hospital_id.nunique(),eligible_independent_hospitals=sufficient,
      synthesis_status='不足10院，未进行医院特异综合' if sufficient<10 else '达到数目门槛；需另行审定综合分析，未自动新增',
      hospitals_candidate_N_ge20=len(eligible_hospitals),hospitals_coverage_lt80_N=len(low),
      low_coverage_denominator=len(eligible_hospitals),low_coverage_fraction=len(low)/len(eligible_hospitals),
      posterior_assessable_hospitals=len(posterior_available),posterior_lt90_N=int(posterior_available.median_max_posterior.lt(.9).sum()),
      zero_assignable_hospitals_with20_N=int(eligible_hospitals.eligible_n.eq(0).sum()),
      LOHO_total=len(loho),LOHO_min_OR=loho.OR.min(),LOHO_max_OR=loho.OR.max(),LOHO_median_OR=loho.OR.median(),
      LOHO_OR_gt1=int(loho.OR.gt(1).sum()),independent_checks=len(qc))
    pd.DataFrame([summary]).to_csv(tables/'19_修订医院层汇总.tsv',sep='\t',index=False)
    # R将读取这个完全病例固定效应对照，不能将GLMM的条件OR与其强求相等。
    x=sm.add_constant(complete[['p3','age10','male','coverage10']],has_constant='add').to_numpy(float)
    beta,converged=severity.fit_logistic(x,complete.death_24h.to_numpy(float))
    cov=severity.robust_covariance(x,complete.death_24h.to_numpy(float),beta,complete.hospital_id.to_numpy())
    pd.DataFrame([dict(N=len(complete),events=int(complete.death_24h.sum()),beta=beta[1],SE=np.sqrt(cov[1,1]))]).to_csv(tables/'19_医院固定效应R核对输入.tsv',sep='\t',index=False)
    (out/'医院层Python检查点.json').write_text(json.dumps(dict(status='Python医院模块完成；R随机效应待运行',checks=len(qc),failed=0),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);main(p.parse_args())
