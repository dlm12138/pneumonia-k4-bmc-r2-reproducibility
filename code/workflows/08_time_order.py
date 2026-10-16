#!/usr/bin/env python3
"""修订肺炎队列：保持历史随机过程的时间顺序分析；逐模型审查分离。"""
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
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load(path, name):
    return load_actual(path, name if "name" in locals() else "actual_functions")


def independent_ari(a,b):
    cells=pd.crosstab(pd.Series(a),pd.Series(b)).to_numpy(np.int64)
    pairs=lambda x:np.sum(x*(x-1)/2)
    total=len(a)*(len(a)-1)/2
    observed=pairs(cells);left=pairs(cells.sum(axis=0));right=pairs(cells.sum(axis=1))
    expected=left*right/total;maximum=.5*(left+right)
    return (observed-expected)/(maximum-expected) if maximum!=expected else 1.


def main(args):
    project,out=Path(args.project),Path(args.output);init_output(out);tables=out/'tables';private=out/'private'
    helper=load(out/'scripts/09_domain_early_and_posterior.py','shared_assets')
    original=load(project/'revision_R1_R2_data/scripts/11_time_order_falsification.py','original_time_order')
    severity=load(project/'revision_R1_R2_data/scripts/05_harmonized_severity_and_apache.py','severity_functions')
    features,means,sds,model,mapping,assignment=helper.frozen_assets(project)
    if features!=original.FEATURES:raise RuntimeError('特征顺序不一致')
    inputpath=private/'修订三库全模块分析输入.parquet'
    candidatepath=project/'01_k5-7.29_history/k5-7.29/analysis_k4_evidence_deepening_20260801/01_candidate_outcomes_and_coverage.parquet'
    input_hash=hashlib.sha256(inputpath.read_bytes()).hexdigest()
    candidates=pd.read_parquet(candidatepath);candidates.stay_id=candidates.stay_id.astype(str)
    source=pd.read_parquet(inputpath)
    result_path=tables/'18_修订时间顺序逐次结果.tsv'
    qc_path=tables/'18_时间顺序独立核查.tsv'
    checkpoint=out/'时间顺序检查点.json'
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text(encoding='utf-8'))
        if previous['analysis_input_sha256']!=input_hash:raise RuntimeError('恢复输入与检查点不一致')
    rows=pd.read_csv(result_path,sep='\t').to_dict('records') if result_path.exists() else []
    checks=pd.read_csv(qc_path,sep='\t').to_dict('records') if qc_path.exists() else []
    completed={(r['database'],r['transformation'],int(r['replicate'])) for r in rows}
    def check(name,actual,expected=0,tol=1e-8):
        error=abs(float(actual)-float(expected));passed=bool(np.isfinite(error) and error<=tol)
        checks.append(dict(check=name,actual=actual,expected=expected,difference=error,tolerance=tol,passed=passed))
        if not passed:
            pd.DataFrame(checks).to_csv(qc_path,sep='\t',index=False)
            raise RuntimeError(name)
    def save(status):
        pd.DataFrame(rows).to_csv(result_path,sep='\t',index=False)
        pd.DataFrame(checks).to_csv(qc_path,sep='\t',index=False)
        checkpoint.write_text(json.dumps(dict(status=status,analysis_input_sha256=input_hash,completed=len(rows),
                  checks=len(checks),failed=sum(not q['passed'] for q in checks)),ensure_ascii=False,indent=2),encoding='utf-8')

    def effect(frame,labels,key):
        d=frame.copy();d['phenotype_test']=labels
        required=['death_24h','age','male','observed_dimensions_48d']
        if d.database.iloc[0]=='eICU':required+=['hospital_id']
        d=d.dropna(subset=required)
        x=np.column_stack([np.ones(len(d))]+[d.phenotype_test.eq(p).to_numpy(float) for p in ['P1','P3','P4']]+
                          [d.age.to_numpy(float)/10,d.male.to_numpy(float),d.observed_dimensions_48d.to_numpy(float)/10])
        y=d.death_24h.to_numpy(float)
        row=dict(model_N=len(d),events=int(y.sum()),P3_events=int(d.loc[d.phenotype_test.eq('P3'),'death_24h'].sum()),
             model_P3_N=int(d.phenotype_test.eq('P3').sum()),model_P2_N=int(d.phenotype_test.eq('P2').sum()),
             P2_events=int(d.loc[d.phenotype_test.eq('P2'),'death_24h'].sum()),design_rank=np.linalg.matrix_rank(x),parameters=x.shape[1])
        names=['p3_beta','P3_OR','OR_CI_low','OR_CI_high','p_value','P2_standardized_risk','P3_standardized_risk',
               'P3_RD_fraction','RD_CI_low','RD_CI_high']
        if row['design_rank']<x.shape[1]:
            row.update({n:np.nan for n in names});row['estimation_status']='设计矩阵不满秩，未报告常规估计';return row
        lp=linprog(-((2*y-1)[:,None]*x).sum(axis=0),A_ub=-(2*y-1)[:,None]*x,b_ub=np.zeros(len(y)),
                   bounds=[(-1,1)]*x.shape[1],method='highs')
        check(key+'_分离审查成功',lp.success,1,0);row['separation_objective']=lp.fun
        if lp.fun < -1e-7:
            row.update({n:np.nan for n in names});row['estimation_status']='准完全分离，常规全模型无有限MLE';return row
        beta,converged=severity.fit_logistic(x,y,max_iter=250)
        check(key+'_IRLS收敛',converged,1,0)
        clusters=d.hospital_id.to_numpy() if d.database.iloc[0]=='eICU' else None
        covariance=severity.robust_covariance(x,y,beta,clusters)
        fitted=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=250,tol=1e-12,
                 cov_type='HC0' if clusters is None else 'cluster',cov_kwds={} if clusters is None else {'groups':clusters})
        check(key+'_独立系数',np.max(np.abs(beta-fitted.params)),0,1e-6)
        check(key+'_独立方差矩阵',np.max(np.abs(covariance-fitted.cov_params())),0,1e-6)
        check(key+'_得分方程',np.max(np.abs(x.T@(y-expit(x@beta)))),0,1e-5)
        se=np.sqrt(covariance[2,2]);risks={};gradients={}
        for p in ['P2','P3']:
            current=x.copy();current[:,1:4]=0
            if p=='P3':current[:,2]=1
            probability=expit(current@beta);risks[p]=probability.mean()
            gradients[p]=np.mean(current*(probability*(1-probability))[:,None],axis=0)
        difference=risks['P3']-risks['P2'];gradient=gradients['P3']-gradients['P2']
        rdse=np.sqrt(gradient@covariance@gradient)
        row.update(p3_beta=beta[2],P3_OR=np.exp(beta[2]),OR_CI_low=np.exp(beta[2]-1.96*se),OR_CI_high=np.exp(beta[2]+1.96*se),
           p_value=2*norm.sf(abs(beta[2]/se)),P2_standardized_risk=risks['P2'],P3_standardized_risk=risks['P3'],
           P3_RD_fraction=difference,RD_CI_low=difference-1.96*rdse,RD_CI_high=difference+1.96*rdse,
           estimation_status='常规MLE及稳健方差独立通过',RD_interval_method='固定协变量分布的delta区间，非bootstrap')
        return row

    primary=pd.read_csv(tables/'02_修订主要效应与严重度.tsv',sep='\t')
    for db in original.DATABASES:
        old=candidates.loc[candidates.database.eq(db)&candidates.k4_eligible].copy()
        raw_all=old[features].to_numpy(float)
        subset=old.stay_id.isin(source.loc[source.database.eq(db),'stay_id']).to_numpy()
        frame=source.loc[source.database.eq(db)].set_index('stay_id').loc[old.loc[subset,'stay_id']].reset_index()
        check(db+'_分型样本对账',subset.sum(),len(frame),0)
        raw=raw_all[subset];scorer=original.FixedMaskScorer(raw,means,sds,model)
        posterior=scorer.posterior(raw);truth=np.asarray([mapping[int(i)] for i in posterior.argmax(axis=1)])
        clean_labels,clean_probability=assignment.assign_locked_k4(raw,means,sds,model,mapping)
        check(db+'_冻结独立概率',np.max(np.abs(posterior-clean_probability)),0,1e-10)
        check(db+'_冻结四型不变',np.sum(truth!=frame.phenotype.to_numpy()),0,0)
        for mode,rep in [('original',0),('reversed',0),('mean_flattened',0)]+[('permuted',i) for i in range(1,101)]:
            if (db,mode,rep) in completed:continue
            seed=original.SEED+rep*101+original.DATABASES.index(db) if rep else 0
            transformed=raw_all if mode=='original' else original.transform_raw(raw_all,mode,np.random.default_rng(seed) if rep else None)
            transformed=transformed[subset]
            key=f'{db}_{mode}_{rep}'
            check(key+'_缺失位置不变',np.sum(np.isfinite(raw)!=np.isfinite(transformed)),0,0)
            for i in range(4):
                before,after=raw[:,i*12:(i+1)*12],transformed[:,i*12:(i+1)*12]
                check(key+f'_变量{i}_逐人均值',np.max(np.abs(np.nanmean(before,axis=1)-np.nanmean(after,axis=1))),0,1e-9)
                if mode!='mean_flattened':
                    difference=np.nanmax(np.abs(np.sort(before,axis=1)-np.sort(after,axis=1)))
                    check(key+f'_变量{i}_逐人值多重集',difference,0,1e-9)
            probability=scorer.posterior(transformed)
            labels=np.asarray([mapping[int(i)] for i in probability.argmax(axis=1)])
            ari=adjusted_rand_score(truth,labels);check(key+'_独立组合数ARI',ari,independent_ari(truth,labels),1e-12)
            row=dict(database=db,transformation=mode,replicate=rep,seed=seed,assigned_N=len(frame),
                agreement=float(np.mean(truth==labels)),ARI=ari,NMI=normalized_mutual_info_score(truth,labels),
                median_max_posterior=np.median(probability.max(axis=1)),**original.distribution(labels),**effect(frame,labels,key))
            if mode=='original':
                expected=primary.loc[primary.database.eq(db)&primary.model.eq('Model_0_current')&primary.phenotype.eq('P3')].iloc[0]
                for name,field in [('model_N','N'),('events','events'),('P3_OR','OR')]:check(key+'_'+name,row[name],expected[field],1e-7)
            rows.append(row)
            if mode!='permuted' or rep%10==0:
                save('运行中，可恢复');print(db,mode,rep,'已保存',flush=True)
    result=pd.DataFrame(rows);summary=[]
    for (db,mode),d in result.groupby(['database','transformation'],sort=False):
        row=dict(database=db,transformation=mode,total_runs=len(d),estimable_runs=int(d.P3_OR.notna().sum()),
                 unestimable_runs=int(d.P3_OR.isna().sum()))
        for field in ['agreement','ARI','NMI','P3_OR','P3_RD_fraction','median_max_posterior']:
            row[field+'_median']=d[field].median();row[field+'_P2_5']=d[field].quantile(.025);row[field+'_P97_5']=d[field].quantile(.975)
        row['percentile_interpretation']='扰动重复分布百分位，不是效应置信区间；效应分位仅基于可估计重复'
        summary.append(row)
    pd.DataFrame(summary).to_csv(tables/'18_修订时间顺序汇总.tsv',sep='\t',index=False)
    check('全部103乘三库重复完备',len(rows),309,0)
    check('输入未修改',hashlib.sha256(inputpath.read_bytes()).hexdigest()==input_hash,1,0)
    save('完成；常规不可估重复单列，不作为有限估计')
    print(json.dumps(dict(rows=len(rows),checks=len(checks),unestimable=int(result.P3_OR.isna().sum())),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);main(p.parse_args())
