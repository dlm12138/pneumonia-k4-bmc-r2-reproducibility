#!/usr/bin/env python3
"""新肺炎队列的可分配性选择诊断；不作治疗因果推断或无支持层外推。"""
from __future__ import annotations

import os, sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from release_runtime import load_actual, init_output

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import linprog
from scipy.special import expit
from scipy.stats import norm


def load(path,name):
    return load_actual(path, name if "name" in locals() else "actual_functions")


def main(args):
    project,out,raw=Path(args.project),Path(args.output),Path(args.raw)
    init_output(out)
    tables,private=out/'tables',out/'private'
    old=load(project/'revision_R1_R2_data/scripts/04_selection_ipw_audit.py','selection_methods')
    severity=load(project/'revision_R1_R2_data/scripts/05_harmonized_severity_and_apache.py','independent_irls')
    member=pd.read_parquet(private/'三库修订肺炎资格与纳入标记.parquet')
    candidate=pd.read_parquet(project/'01_k5-7.29_history/k5-7.29/analysis_k4_evidence_deepening_20260801/01_candidate_outcomes_and_coverage.parquet')
    candidate.stay_id=candidate.stay_id.astype(str)
    candidate=candidate.merge(member.loc[member.revised_landmark,['database','stay_id']],on=['database','stay_id'],validate='one_to_one')
    analysis=pd.read_parquet(private/'修订三库全模块分析输入.parquet')
    metadata=pd.read_csv(raw/'MIMIC-III_1.4/mimic-iii-clinical-database-1.4/ICUSTAYS.csv.gz',
                         usecols=['ICUSTAY_ID','DBSOURCE','FIRST_CAREUNIT','INTIME'],low_memory=False)
    metadata['stay_id']=metadata.ICUSTAY_ID.astype(str)
    metadata['admission_year']=pd.to_datetime(metadata.INTIME).dt.year
    candidate=candidate.merge(metadata[['stay_id','DBSOURCE','FIRST_CAREUNIT','admission_year']].assign(database='MIMIC-III'),
                              on=['database','stay_id'],how='left',validate='one_to_one')
    qc,model_rows,weight_rows,balance_rows,effects,missing,matrices,diagnostics,private_weights=[],[],[],[],[],[],[],[],[]
    def check(name,actual,expected=0,tol=1e-7):
        difference=abs(float(actual)-float(expected));passed=bool(np.isfinite(difference) and difference<=tol)
        qc.append(dict(check=name,actual=actual,expected=expected,difference=difference,tolerance=tol,passed=passed))
        pd.DataFrame(qc).to_csv(tables/'20_选择加权独立核查.tsv',sep='\t',index=False)
        if not passed:raise RuntimeError(name)
    def separation(m):
        x,y=np.asarray(m.exog,float),np.asarray(m.endog,float)
        lp=linprog(-((2*y-1)[:,None]*x).sum(axis=0),A_ub=-(2*y-1)[:,None]*x,b_ub=np.zeros(len(y)),bounds=[(-1,1)]*x.shape[1],method='highs')
        if not lp.success:raise RuntimeError('选择或结局分离审查失败')
        return float(lp.fun)
    def outcome_fit(d,outcome,weight,trim):
        x=np.column_stack([np.ones(len(d))]+[d.phenotype.eq(p).to_numpy(float) for p in ['P1','P3','P4']]+
                        [d.age.to_numpy(float)/10,d.male.to_numpy(float),d.observed_dimensions_48d.to_numpy(float)/10])
        y=d[outcome].to_numpy(float);w=np.asarray(weight,float);db=d.database.iloc[0];key=db+'__'+outcome+'__'+trim
        beta,converged=severity.fit_logistic(x,y,w,max_iter=250)
        check(key+'_独立IRLS收敛',converged,1,0)
        smmodel=sm.GLM(y,x,family=sm.families.Binomial(),freq_weights=w)
        fitted=smmodel.fit(maxiter=250,tol=1e-12)
        objective=separation(smmodel);check(key+'_常规无分离',objective>=-1e-7,1,0)
        check(key+'_独立系数',np.max(np.abs(beta-fitted.params)),0,1e-7)
        probability=expit(x@beta);scores=x*(w*(y-probability))[:,None]
        check(key+'_得分方程',np.max(np.abs(scores.sum(axis=0))),0,1e-5)
        bread=np.linalg.inv(x.T@(x*(w*probability*(1-probability))[:,None]))
        cluster=d.hospital_id if db=='eICU' else None
        if cluster is None:
            covariance=bread@(scores.T@scores)@bread
        else:
            codes,levels=pd.factorize(cluster,sort=True);grouped=np.zeros((len(levels),x.shape[1]));np.add.at(grouped,codes,scores)
            covariance=bread@(grouped.T@grouped)@bread*len(levels)/(len(levels)-1)*(len(d)-1)/(len(d)-x.shape[1])
        # 另一真实实现：旧手算HC1，MIMIC取消明确的有限样本校正；不是标签替换。
        numeric=d.copy();numeric['p1']=numeric.phenotype.eq('P1').astype(int);numeric['p3']=numeric.phenotype.eq('P3').astype(int)
        numeric['p4']=numeric.phenotype.eq('P4').astype(int);numeric['age10']=numeric.age/10;numeric['coverage10']=numeric.observed_dimensions_48d/10
        oracle=old.fit_outcome_glm(numeric,outcome+' ~ p1 + p3 + p4 + age10 + male + coverage10',pd.Series(w,index=numeric.index),cluster)
        oracle_se=oracle.bse.to_numpy(float)
        if cluster is None:oracle_se/=np.sqrt(len(d)/(len(d)-x.shape[1]))
        check(key+'_独立稳健SE',np.max(np.abs(np.sqrt(np.diag(covariance))-oracle_se)),0,1e-7)
        se=np.sqrt(covariance[2,2])
        effects.append(dict(database=db,outcome=outcome,model='unweighted_complete_case' if trim=='none' else 'stabilized_selection_ipw',
           truncation=trim,N=len(d),events=int(y.sum()),OR=np.exp(beta[2]),CI_low=np.exp(beta[2]-1.96*se),CI_high=np.exp(beta[2]+1.96*se),
           beta=beta[2],se=se,p_value=2*norm.sf(abs(beta[2]/se)),variance='医院聚类稳健，原有限样本校正' if cluster is not None else 'HC0，已取消旧加权HC1校正',
           interpretation='观察关联；不能识别无可分配CareVue层的效应'))
        matrix=pd.DataFrame(x,columns=['x'+str(i) for i in range(x.shape[1])]);matrix['y']=y;matrix['weight']=w
        matrix['model_key']=key;matrix['cluster']=d.hospital_id.astype(str).to_numpy() if cluster is not None else ''
        matrices.append(matrix)

    source_rows=[]
    for db in old.DATABASES:
        pool=candidate.loc[candidate.database.eq(db)].copy()
        for outcome in old.OUTCOMES:
            needed=['age','male','source_tier',outcome]
            for name in needed:
                missing.append(dict(database=db,outcome=outcome,variable=name,N=len(pool),missing_N=int(pool[name].isna().sum())))
            complete=pool.dropna(subset=needed).copy();complete['selected']=complete.k4_eligible.astype(int)
            denominator_formula=f'selected ~ bs(age, df=4, include_intercept=False) + male + C(source_tier) + {outcome}'
            numerator_formula='selected ~ bs(age, df=4, include_intercept=False) + male'
            predictions={}
            for role,formula in [('denominator',denominator_formula),('numerator',numerator_formula)]:
                m=smf.glm(formula,data=complete,family=sm.families.Binomial());objective=separation(m)
                fitted=m.fit(maxiter=250,tol=1e-12)
                check(db+'_'+outcome+'_'+role+'_无分离',objective>=-1e-7,1,0)
                x,y=np.asarray(m.exog,float),np.asarray(m.endog,float)
                independent,converged=severity.fit_logistic(x,y,max_iter=250)
                check(db+'_'+outcome+'_'+role+'_独立收敛',converged,1,0)
                check(db+'_'+outcome+'_'+role+'_独立系数',np.max(np.abs(independent-fitted.params)),0,1e-7)
                check(db+'_'+outcome+'_'+role+'_得分残差',np.max(np.abs(x.T@(y-fitted.fittedvalues))),0,1e-5)
                predictions[role]=np.clip(fitted.predict(complete),1e-6,1-1e-6)
                diagnostics.append(dict(database=db,outcome=outcome,role=role,N=len(complete),selected_N=int(complete.selected.sum()),
                   converged=fitted.converged,rank=np.linalg.matrix_rank(x),parameters=x.shape[1],separation_objective=objective,
                   note='模型未含DBSOURCE；不能修复护理系统层面的结构性缺乏支持'))
                for term in fitted.params.index:
                    model_rows.append(dict(database=db,outcome=outcome,role=role,formula=formula,term=term,coefficient=fitted.params[term]))
            complete['denominator_probability']=predictions['denominator'];complete['numerator_probability']=predictions['numerator']
            selected=complete.loc[complete.selected.eq(1)].copy()
            selected['raw_weight']=selected.numerator_probability/selected.denominator_probability
            private_weights.append(complete[['database','stay_id','selected','denominator_probability','numerator_probability']].assign(outcome=outcome))
            for quantity,values in [('denominator_probability_all',complete.denominator_probability),('denominator_probability_selected',selected.denominator_probability)]:
                weight_rows.append(dict(database=db,outcome=outcome,quantity=quantity,truncation='not_applicable',
                    candidate_N=len(complete),selected_N=len(selected),**old.summarize_distribution(values)))
            weights={'none':np.ones(len(selected))}
            for trim,(lo,hi) in old.TRIMS.items():
                lower,upper=selected.raw_weight.quantile([lo,hi]);w=selected.raw_weight.clip(lower,upper);weights[trim]=w.to_numpy(float)
                ess=float(w.sum()**2/np.square(w).sum())
                check(db+'_'+outcome+'_'+trim+'_ESS独立',ess,1/np.sum(np.square(w/w.sum())),1e-8)
                weight_rows.append(dict(database=db,outcome=outcome,quantity='stabilized_weight',truncation=trim,
                   candidate_N=len(complete),selected_N=len(selected),selection_rate_fraction=len(selected)/len(complete),
                   **old.summarize_distribution(w),trim_low_value=lower,trim_high_value=upper,
                   trimmed_low_N=int(selected.raw_weight.lt(lower).sum()),trimmed_high_N=int(selected.raw_weight.gt(upper).sum()),
                   ESS=ess,ESS_fraction=ess/len(selected),weight_gt10_N=int(w.gt(10).sum())))
            variables={'age':(complete.age,selected.age),'male':(complete.male,selected.male),outcome:(complete[outcome],selected[outcome])}
            for level in sorted(complete.source_tier.unique()):
                variables['source_tier='+level]=(complete.source_tier.eq(level).astype(float),selected.source_tier.eq(level).astype(float))
            if db=='MIMIC-III':
                check('MIMICIII护理系统原始来源完整',complete.DBSOURCE.isna().sum(),0,0)
                variables['admission_year']=(complete.admission_year,selected.admission_year)
                for level in sorted(complete.DBSOURCE.unique()):
                    variables['DBSOURCE='+level]=(complete.DBSOURCE.eq(level).astype(float),selected.DBSOURCE.eq(level).astype(float))
                    target=complete.loc[complete.DBSOURCE.eq(level)]
                    source_rows.append(dict(outcome=outcome,DBSOURCE=level,N=len(target),assigned_N=int(target.selected.sum()),
                        unassigned_N=int(target.selected.eq(0).sum()),assignment_rate=target.selected.mean(),
                        selected_outcome_events=int(target.loc[target.selected.eq(1),outcome].sum()),
                        structural_no_support=bool(target.selected.sum()==0)))
                for vital in ['SBP','DBP','HR','RR']:
                    vs=[f'{vital}_h{h}' for h in range(12)]
                    variables[vital+'_observed_hours']=(complete[vs].notna().sum(axis=1),selected[vs].notna().sum(axis=1))
            for trim,w in weights.items():
                for name,(target,observed) in variables.items():
                    tm,sm0,smd0=old.standardized_difference(target,observed)
                    _,sm1,smd1=old.standardized_difference(target,observed,w)
                    check(db+'_'+outcome+'_'+trim+'_'+name+'_独立加权均值',sm1,float(np.sum(observed.to_numpy(float)*w)/np.sum(w)))
                    balance_rows.append(dict(database=db,outcome=outcome,truncation=trim,variable=name,target_mean=tm,
                       selected_mean=sm0,weighted_mean=sm1,SMD_before=smd0,SMD_after=smd1,
                       in_selection_model=not (name.startswith('DBSOURCE=') or name.endswith('observed_hours') or name=='admission_year')))
            columns=['database','stay_id',outcome]
            selected_keys=selected[columns].copy()
            for trim,w in weights.items():selected_keys['weight_'+trim]=w
            d=analysis.loc[analysis.database.eq(db)].drop(columns=[outcome]).merge(selected_keys,on=['database','stay_id'],validate='one_to_one')
            check(db+'_'+outcome+'_分配人数',len(d),len(selected),0)
            for trim in weights:outcome_fit(d,outcome,d['weight_'+trim].to_numpy(float),trim)
            print(db,outcome,'选择权重与独立核查完成',flush=True)
    for name,rows in [('20_修订选择模型系数.tsv',model_rows),('20_修订选择权重与ESS.tsv',weight_rows),
                      ('20_修订选择加权平衡.tsv',balance_rows),('20_修订选择加权效应.tsv',effects),
                      ('20_修订选择加权缺失.tsv',missing),('20_选择模型数值诊断.tsv',diagnostics),
                      ('20_MIMICIII护理系统结构性支持.tsv',source_rows)]:
        pd.DataFrame(rows).to_csv(tables/name,sep='\t',index=False)
    pd.concat(matrices,ignore_index=True).to_parquet(private/'修订选择加权独立核查矩阵.parquet',index=False)
    pd.concat(private_weights,ignore_index=True).to_parquet(private/'修订逐人选择概率.parquet',index=False)
    (out/'选择加权Python检查点.json').write_text(json.dumps(dict(status='Python选择加权完成；结构性缺乏支持未消除；R待核查',
          checks=len(qc),failed=0,models=len(effects)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=len(qc),outcome_models=len(effects)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);p.add_argument('--raw',required=True);main(p.parse_args())
