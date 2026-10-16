#!/usr/bin/env python3
"""2026-10-14：修订队列的冻结参考分布、早期分配及后验关联；不训练K4。"""
from __future__ import annotations

import os, sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from release_runtime import load_actual, init_output

import argparse
import ast
import hashlib
import importlib.util
import json
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from scipy.optimize import linprog
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load(path, name, no_plot=False):
    return load_actual(path, name if "name" in locals() else "actual_functions")


def frozen_assets(project):
    root=project/'revision_R1_R2_data/locked_K4_model'
    features=pd.read_csv(root/'feature_order.csv').sort_values('feature_position').feature.tolist()
    standard=pd.read_csv(root/'standardization_parameters.csv').set_index('feature').loc[features]
    weights=pd.read_csv(root/'component_weights.csv').sort_values('raw_component_zero_based')
    means=pd.read_csv(root/'component_means.csv')
    model=SimpleNamespace(n_components=4,covariance_type='full',weights_=weights.weight.to_numpy(float),
      means_=np.asarray([means.loc[means.raw_component_zero_based.eq(i)].set_index('feature').loc[features].component_mean_z for i in range(4)]),
      covariances_=np.asarray([pd.read_csv(root/('covariance_'+p+'.csv'),index_col=0).loc[features,features].to_numpy(float) for p in weights.phenotype]))
    mapping=dict(zip(weights.raw_component_zero_based.astype(int),weights.phenotype))
    assignment=load(root/'assignment_code_clean.py','locked_assignment')
    return features,standard.MIMIC_IV_observed_mean.to_numpy(float),standard.MIMIC_IV_observed_sd_ddof0.to_numpy(float),model,mapping,assignment


def main(args):
    project,out=Path(args.project),Path(args.output)
    init_output(out)
    table,private=out/'tables',out/'private'
    history=project/'01_k5-7.29_history/k5-7.29'
    deep=load(history/'analysis_k4_evidence_deepening_20260801/run_k4_evidence_deepening.py','historical_posterior')
    domain=load(project/'revision_R1_R2_data/scripts/09_domain_shift_diagnostics.py','real_domain',True)
    features,means,sds,model,mapping,assignment=frozen_assets(project)
    inputpath=private/'修订三库全模块分析输入.parquet'
    hash_before=hashlib.sha256(inputpath.read_bytes()).hexdigest()
    source=pd.read_parquet(inputpath)
    candidates=pd.read_parquet(history/'analysis_k4_evidence_deepening_20260801/01_candidate_outcomes_and_coverage.parquet')
    candidates.stay_id=candidates.stay_id.astype(str)
    old=pd.read_parquet(project/'revision_R1_R2_data/02_frozen_reproduced_assignments.parquet')
    old.database=old.database.replace({'MIMIC-III MetaVision':'MIMIC-III'});old.stay_id=old.stay_id.astype(str)
    reference=old.loc[old.database.eq('MIMIC-IV')].merge(candidates[['database','stay_id']+features],on=['database','stay_id'],validate='one_to_one')
    analysis=source.merge(candidates[['database','stay_id']+features],on=['database','stay_id'],validate='one_to_one')
    qc,regressions=[],[]

    def check(name,actual,expected=0,tolerance=1e-8):
        error=0. if pd.isna(actual) and pd.isna(expected) else abs(float(actual)-float(expected))
        passed=bool(np.isfinite(error) and error<=tolerance)
        qc.append(dict(check=name,actual=actual,expected=expected,difference=error,tolerance=tolerance,passed=passed))
        pd.DataFrame(qc).to_csv(table/'17_分布早期后验独立核查.tsv',sep='\t',index=False)
        if not passed: raise RuntimeError(name)

    def score(frame):
        raw=frame[features].to_numpy(float);observed=np.isfinite(raw).sum(axis=1)
        ll=domain.observed_component_log_likelihood(raw,means,sds,model)
        labels=np.array([mapping[int(i)] for i in ll.argmax(axis=1)])
        check('分布分配标签_'+frame.database.iloc[0],int(np.sum(labels!=frame.phenotype)),0,0)
        frame=frame.copy()
        frame['normalized_NLL']=-logsumexp(ll,axis=1)/observed
        frame['mahalanobis_per_observed_dimension']=domain.assigned_mahalanobis(raw,frame.selected_component.to_numpy(int)-1,means,sds,model)/observed
        check('马氏距离全部有限_'+frame.database.iloc[0],int((~np.isfinite(frame.mahalanobis_per_observed_dimension)).sum()),0,0)
        return frame

    reference=score(reference)
    analysis=pd.concat([score(g) for _,g in analysis.groupby('database',sort=False)],ignore_index=True)
    nll_threshold=reference.normalized_NLL.quantile(.95)
    mahal_threshold=reference.mahalanobis_per_observed_dimension.quantile(.95)
    check('冻结参考马氏阈值',mahal_threshold,pd.read_csv(project/'12_eicu_boundary_propagation_20261014/tables/12_三库正确马氏距离诊断.tsv',sep='\t').MIMIC_IV_mahalanobis_P95_threshold.iloc[0])
    oldnll=pd.read_csv(project/'revision_R1_R2_data/12_domain_shift_summary.csv').NLL_reference_threshold.iloc[0]
    check('冻结参考NLL阈值',nll_threshold,oldnll)
    # 将新IV临时改名，使真实统计函数的MIMIC-IV参考始终只有原推导样本。
    aliased=analysis.copy();aliased.loc[aliased.database.eq('MIMIC-IV'),'database']='修订MIMIC-IV'
    domain.DATABASES=['修订MIMIC-IV','MIMIC-III','eICU']
    feature_detail,feature_summary=domain.feature_shift(pd.concat([reference,aliased],ignore_index=True))
    feature_detail.database=feature_detail.database.replace({'修订MIMIC-IV':'MIMIC-IV'})
    feature_summary.database=feature_summary.database.replace({'修订MIMIC-IV':'MIMIC-IV'})
    feature_detail.to_csv(table/'17_修订48维特征漂移.tsv',sep='\t',index=False)
    reference_prevalence=reference.phenotype.value_counts(normalize=True).reindex(['P1','P2','P3','P4']).to_numpy(float)
    member=pd.read_parquet(private/'三库修订肺炎资格与纳入标记.parquet')
    summaries=[]
    for db,frame in analysis.groupby('database',sort=False):
        current=frame.phenotype.value_counts(normalize=True).reindex(['P1','P2','P3','P4'],fill_value=0).to_numpy(float)
        row=feature_summary.loc[feature_summary.database.eq(db)].iloc[0].to_dict()
        row.update(N=len(frame),reference_N=len(reference),reference_population='原冻结MIMIC-IV推导人群',
                   phenotype_prevalence_JSD=domain.jensen_shannon_divergence(reference_prevalence,current),
                   total_variation_distance=float(.5*np.abs(reference_prevalence-current).sum()),
                   NLL_median=frame.normalized_NLL.median(),NLL_reference_threshold=nll_threshold,
                   NLL_OOD_pct=100*frame.normalized_NLL.gt(nll_threshold).mean(),
                   Mahalanobis_median=frame.mahalanobis_per_observed_dimension.median(),Mahalanobis_reference_threshold=mahal_threshold,
                   Mahalanobis_OOD_pct=100*frame.mahalanobis_per_observed_dimension.gt(mahal_threshold).mean(),
                   max_posterior_median=frame.max_posterior.median(),entropy_median=frame.posterior_entropy_normalized.median(),
                   assignment_coverage_pct=100*len(frame)/int((member.database.eq(db)&member.revised_landmark).sum()))
        summaries.append(row)
        for _,r in frame.groupby('selected_component').head(8).iterrows():
            raw=r[features].to_numpy(float);mask=np.isfinite(raw);component=int(r.selected_component)-1
            diff=(raw[mask]-means[mask])/sds[mask]-model.means_[component,mask]
            independent=diff@np.linalg.solve(model.covariances_[component][np.ix_(mask,mask)],diff)/mask.sum()
            check(db+'_马氏独立线性求解_'+str(len(qc)),independent,r.mahalanobis_per_observed_dimension)
    pd.DataFrame(summaries).to_csv(table/'17_修订冻结参考分布漂移汇总.tsv',sep='\t',index=False)
    analysis[['database','stay_id','phenotype','normalized_NLL','mahalanobis_per_observed_dimension']].to_parquet(private/'修订逐人冻结参考分布核查.parquet',index=False)
    print('三库冻结参考分布漂移完成',flush=True)

    early_rows=[];early_frames=[]
    assign=load(history/'analysis_main12_clinical_actionability_20260730/02_common4_stability_transport.py','original_early_assignment')
    indices=np.array([i for i,f in enumerate(features) if int(f.rsplit('h',1)[1])<=8])
    for db,frame in analysis.groupby('database',sort=False):
        raw=frame[features].to_numpy(float);observed=np.isfinite(raw[:,indices])
        vital=np.column_stack([observed[:,[j for j,i in enumerate(indices) if features[i].startswith(v+'_')]].any(axis=1) for v in ['SBP','DBP','HR','RR']]).all(axis=1)
        gate=(observed.sum(axis=1)>=24)&vital
        components,posterior=assign.assign(raw[gate],means,sds,model,feature_indices=indices)
        obscured=raw[gate].copy();obscured[:,[i for i in range(48) if i not in indices]]=np.nan
        _,independent=assignment.assign_locked_k4(obscured,means,sds,model,mapping)
        check(db+'_早期独立遮维概率',np.max(np.abs(posterior-independent)),0,1e-10)
        truth=frame.loc[gate,'selected_component'].to_numpy(int)-1
        early_rows.append(dict(database=db,eligible_12h_N=len(frame),eligible_9h_N=int(gate.sum()),
            exact_agreement=float(np.mean(truth==components)),ARI=adjusted_rand_score(truth,components),NMI=normalized_mutual_info_score(truth,components),
            median_max_posterior=float(np.median(posterior.max(axis=1)))))
        early=frame.loc[gate,['database','stay_id','phenotype']].copy()
        early['early_phenotype']=[mapping[int(i)] for i in components];early['early_max_posterior']=posterior.max(axis=1)
        early_frames.append(early)
    pd.DataFrame(early_rows).to_csv(table/'17_修订早期分配稳定性.tsv',sep='\t',index=False)
    pd.concat(early_frames).to_parquet(private/'修订早期分配逐人核查.parquet',index=False)
    print('三库早期分配独立核查完成',flush=True)

    original_fit=deep.fit_glm
    quintile_intervals=[]
    def audited_fit(frame,formula,weight=None,cluster=None):
        result=original_fit(frame,formula,weight,cluster)
        m=deep.smf.glm(formula,frame,family=deep.sm.families.Binomial(),freq_weights=weight)
        independent=m.fit(maxiter=200,tol=1e-12)
        x,y=np.asarray(m.exog,float),np.asarray(m.endog,float)
        weights=np.ones(len(y)) if weight is None else np.asarray(weight,float)
        b=np.asarray(result.params,float);probability=expit(x@b)
        prefix=frame.database.iloc[0]+'_'+str(len(regressions))
        lp=linprog(-(((2*y-1)*weights)[:,None]*x).sum(axis=0),
                   A_ub=-(2*y-1)[:,None]*x,b_ub=np.zeros(len(y)),bounds=[(-1,1)]*x.shape[1],method='highs')
        check(prefix+'_分离审查成功',int(lp.success),1,0)
        separated=bool(lp.fun < -1e-7)
        scores=x*(weights*(y-probability))[:,None]
        covariance=None
        if not separated:
            check(prefix+'_独立系数',np.max(np.abs(b-independent.params)),0,1e-7)
            check(prefix+'_得分方程',np.max(np.abs(scores.sum(axis=0))),0,1e-5)
            bread=np.linalg.inv(x.T@(x*(weights*probability*(1-probability))[:,None]))
            if cluster is None:
                covariance=bread@(scores.T@scores)@bread
            else:
                codes,levels=pd.factorize(np.asarray(cluster),sort=True)
                grouped=np.vstack([scores[codes==i].sum(axis=0) for i in range(len(levels))])
                n,k,g=len(y),x.shape[1],len(levels)
                covariance=bread@(grouped.T@grouped)@bread*g/(g-1)*(n-1)/(n-k)
            check(prefix+'_独立稳健标准误',np.max(np.abs(np.sqrt(np.diag(covariance))-np.asarray(result.bse))),0,1e-7)
        regressions.append(dict(database=frame.database.iloc[0],formula=formula,rows=len(y),patients=frame.stay_id.nunique(),
              events=int(y.sum()) if weight is None else np.nan,weighted=weight is not None,rank=np.linalg.matrix_rank(x),parameters=x.shape[1],
              status='准完全分离，不正式报告该模型' if separated else '独立数值核查通过',separation_objective=lp.fun))
        if separated:
            return SimpleNamespace(params=result.params*np.nan,bse=result.bse*np.nan,predict=lambda scenario:np.full(len(scenario),np.nan))
        if 'p3_quintile' in formula:
            from patsy import build_design_matrices
            for q in ['Q1','Q2','Q3','Q4','Q5']:
                scenario=frame.copy();scenario['p3_quintile']=pd.Categorical([q]*len(frame),categories=['Q1','Q2','Q3','Q4','Q5'])
                xx=np.asarray(build_design_matrices([m.data.design_info],scenario)[0],float)
                probability_q=expit(xx@b);risk=float(probability_q.mean())
                gradient=np.mean(xx*(probability_q*(1-probability_q))[:,None],axis=0)
                se_logit=np.sqrt(gradient@covariance@gradient)/(risk*(1-risk))
                quintile_intervals.append(dict(database=frame.database.iloc[0],quintile=q,risk_CI_low=expit(logit(risk)-1.96*se_logit),
                       risk_CI_high=expit(logit(risk)+1.96*se_logit),risk_CI_method='固定协变量分布上logit-delta稳健区间；非重抽样区间'))
        return result
    deep.fit_glm=audited_fit;deep.write_csv=lambda frame,name:None
    results=deep.posterior_uncertainty_analysis(analysis)
    names=['17_修订连续P3后验关联.tsv','17_修订P3后验五分位风险.tsv','17_修订置信度与熵分层关联.tsv','17_修订后验加权伪类别关联.tsv']
    for i,(name,result) in enumerate(zip(names,results)):
        if i==1:
            result=result.merge(pd.DataFrame(quintile_intervals),on=['database','quintile'],how='left',validate='one_to_one')
            for db,g in result.groupby('database'):
                src=analysis.loc[analysis.database.eq(db)].dropna(subset=['death_24h','age','male','observed_dimensions_48d'])
                check(db+'_五分位人数完备',g.n.sum(),len(src),0);check(db+'_五分位事件完备',g.events.sum(),src.death_24h.sum(),0)
        result.to_csv(table/name,sep='\t',index=False)
    pd.DataFrame(regressions).to_csv(table/'17_后验回归诊断.tsv',sep='\t',index=False)
    check('本轮私有分析输入未修改',int(hashlib.sha256(inputpath.read_bytes()).hexdigest()==hash_before),1,0)
    state=dict(status='修订冻结参考分布、早期分配与后验模块完成',checks=len(qc),failed=0,
               unestimable_posterior_models=sum(r['status'].startswith('准完全分离') for r in regressions),
               scope='不含时间顺序、医院层或IPW；不修改稿件')
    (out/'分布早期后验检查点.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
