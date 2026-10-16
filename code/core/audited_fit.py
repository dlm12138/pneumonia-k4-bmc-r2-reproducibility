"""Actual current checked fitting functions; no substitute estimator."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy.special import expit
from core import association as s, prediction as pred
O=Path(os.environ['OUTPUT_ROOT'])
from release_runtime import init_output
init_output(O)
checks=[];diagnostics=[];missingrows=[]
qcp=O/'qc/association_checks.tsv'
def check(name,ok,detail=''):
    checks.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    pd.DataFrame(checks).to_csv(O/'qc/02_关联独立核查.tsv',sep='\t',index=False)
    if not ok:raise RuntimeError(name+': '+str(detail))

def effective(f,cov,key):
    result=[]
    for c in cov:
        if c=='SpO2_min_0_12h_missing' and f[c].nunique()==1:
            check(key+'_冗余缺失标记全零',f[c].eq(0).all())
            diagnostics.append(dict(database=f.database.iloc[0],model=key,issue='redundant_all_zero_missingness_indicator',action='omitted_only_SpO2_missing_indicator',N=len(f),detail='零列不携带信息；不删除RRT或其他协变量'))
        else:result.append(c)
    if 'active_RRT_at_12h' in cov:check(key+'_RRT保留','active_RRT_at_12h' in result)
    return result

def audit(f,outcome,key,cov,marker=False):
    cov=effective(f,cov,key)
    if marker:x=np.column_stack([np.ones(len(f)),f.fixed_HRRR_rule.to_numpy(float)]+[f[c].to_numpy(float) for c in cov]);names=['intercept','fixed_HRRR_rule']+cov
    else:x,names=s.design_matrix(f,cov)
    y=f[outcome].to_numpy(float)
    check(key+'_矩阵有限',np.isfinite(x).all())
    check(key+'_矩阵满秩',np.linalg.matrix_rank(x)==x.shape[1])
    signed=(2*y-1)[:,None]*x
    sep=linprog(-signed.sum(axis=0),A_ub=-signed,b_ub=np.zeros(len(y)),bounds=[(-1,1)]*x.shape[1],method='highs')
    check(key+'_分离优化成功',sep.success)
    issep=sep.fun < -1e-7
    diagnostics.append(dict(database=f.database.iloc[0],model=key,outcome=outcome,issue='quasi_complete_separation' if issep else 'regular_finite_MLE',action='not_estimable_no_substitute' if issep else 'original_unpenalized_fit',N=len(f),events=int(y.sum()),detail=';'.join(names),RRT_positive_N=int(f.active_RRT_at_12h.sum()) if 'active_RRT_at_12h' in cov else np.nan,RRT_positive_events=int(f.loc[f.active_RRT_at_12h.eq(1),outcome].sum()) if 'active_RRT_at_12h' in cov else np.nan,missing_SpO2_N=int(f.SpO2_min_0_12h_missing.sum()),missing_SpO2_events=int(f.loc[f.SpO2_min_0_12h_missing.eq(1),outcome].sum()),separating_terms=';'.join(names[i] for i in range(len(names)) if issep and abs(sep.x[i])>1e-7),separation_objective=float(sep.fun)))
    pd.DataFrame(diagnostics).to_csv(O/'tables/02_冗余列与分离诊断.tsv',sep='\t',index=False)
    if issep:
        check(key+'_分离项如实不输出有限MLE',True,'保留原MIMICIII两项及纠正后新增分离，不换算法')
        return cov,False
    b,conv=s.fit_logistic(x,y);check(key+'_原函数收敛',conv)
    groups=f.hospital_id.astype(str).to_numpy() if f.database.iloc[0]=='eICU' else None
    kw=dict(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True}) if groups is not None else dict(cov_type='HC0')
    independent=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-12,**kw)
    check(key+'_独立GLM系数',np.allclose(b,independent.params,atol=1e-7,rtol=0),np.max(np.abs(b-independent.params)))
    check(key+'_独立稳健协方差',np.allclose(s.robust_covariance(x,y,b,groups),independent.cov_params(),atol=1e-7,rtol=0))
    check(key+'_得分方程',np.max(np.abs(x.T@(y-expit(x@b))))<1e-5)
    return cov,True

def fit(f,outcome,name,cov,seed,marker=False):
    cached_name=('03_修订固定HRRR规则.tsv' if marker else '02_修订主要效应与严重度.tsv' if name.startswith('Model_') else '04_修订整个住院死亡.tsv' if name=='hospital_harmonized' else None)
    if cached_name and (O/'tables'/cached_name).exists():
        cached=pd.read_csv(O/'tables'/cached_name,sep='\t');saved=cached[cached.database.eq(f.database.iloc[0])&cached.model.eq(name)]
        if len(saved)==(1 if marker else 4):
            print('续用本20目录已核完成的血氧纠正结果',f.database.iloc[0],name,flush=True)
            return saved
    prepared,pars=s.prepare_model_data(f,outcome)
    key=f.database.iloc[0]+'_'+name
    missingrows.append(dict(database=f.database.iloc[0],model=name,outcome=outcome,N=len(prepared),events=int(prepared[outcome].sum()),SpO2_missing=int(prepared.SpO2_min_0_12h_missing.sum()),median_imputation=pars['SpO2_min_0_12h'],seed=s.SEED+seed))
    cov,ok=audit(prepared,outcome,key,cov,marker)
    if not ok:
        empty={c:np.nan for c in ['OR','CI_low','CI_high','p_value','standardized_risk_fraction','risk_CI_low','risk_CI_high','RD_vs_P2_fraction','RD_vs_P2_percentage_points','RD_CI_low','RD_CI_high','RD_fraction','RD_percentage_points']}
        return pd.DataFrame([dict(**empty,database=f.database.iloc[0],model=name,outcome=outcome,phenotype=p,reference='P2',N=len(prepared),events=int(prepared[outcome].sum()),bootstrap_successful=0,bootstrap_attempts=0,variance='HC0 sandwich robust' if f.database.iloc[0]!='eICU' else 'hospital-cluster robust',covariates_used=';'.join(cov),estimation_status='quasi_separation_no_finite_MLE') for p in (['marker'] if marker else s.PHENOTYPES)])
    if marker:r=pd.DataFrame([pred.marker_association(prepared,f.database.iloc[0],name,cov,s,seed)])
    else:r,_=s.run_model(prepared,outcome,name,cov,seed)
    r['estimation_status']='regular_MLE_numerically_verified';r['covariates_used']=';'.join(cov)
    r['variance']='hospital-cluster robust' if f.database.iloc[0]=='eICU' else 'HC0 sandwich robust'
    check(key+'_真实500次重抽样',r.bootstrap_successful.min()==500)
    print('完成受影响关联',key,'N',len(prepared),'events',int(prepared[outcome].sum()),flush=True)
    pd.DataFrame(missingrows).to_csv(O/'tables/02_各模型血氧缺失与插补.tsv',sep='\t',index=False)
    return r
