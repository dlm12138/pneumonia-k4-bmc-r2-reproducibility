"""复用真实关联函数、原始随机种子；仅拟合含血氧的既有分析。"""

import os, sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from release_runtime import load_actual, init_output, public_result_path

from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,shutil
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy.special import expit
P=Path(os.environ['DATA_ROOT']);O=Path(os.environ['OUTPUT_ROOT']);R=P/'current_inputs';V=R
init_output(O)
def load(path,name):
    return load_actual(path,name)
s=load(P/'revision_R1_R2_data/scripts/05_harmonized_severity_and_apache.py','original_severity')
pred=load(P/'revision_R1_R2_data/scripts/08_hrrr_and_incremental_performance.py','original_hrrr')
data=pd.read_parquet(O/'private/修订三库全模块分析输入.parquet')
base=['age10','male','coverage10'];oxy=base+['SpO2_min10','SpO2_min_0_12h_missing'];support=['active_invasive_ventilation_at_12h','active_vasopressor_at_12h','active_RRT_at_12h'];full=oxy+support
qcp=O/'qc/02_关联独立核查.tsv'
if qcp.exists() and not (O/'qc/02_初次发现新分离_检查记录.tsv').exists():shutil.copy2(qcp,O/'qc/02_初次发现新分离_检查记录.tsv')
checks=pd.read_csv(qcp,sep='\t').query('passed == True').to_dict('records') if qcp.exists() else []
diagnostics=pd.read_csv(O/'tables/02_冗余列与分离诊断.tsv',sep='\t').to_dict('records') if (O/'tables/02_冗余列与分离诊断.tsv').exists() else []
missingrows=pd.read_csv(O/'tables/02_各模型血氧缺失与插补.tsv',sep='\t').to_dict('records') if (O/'tables/02_各模型血氧缺失与插补.tsv').exists() else []
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
def replace_file(name,root,mask,new):
    dest=O/'tables'/name
    old=pd.read_csv(public_result_path(name),sep='\t')
    # 保持既有行号、未受影响行和列；只覆盖给定模型的结果。
    oldmask=mask(old);check(name+'_替换行数',int(oldmask.sum())==len(new),(oldmask.sum(),len(new)))
    for col in new:
        if col not in old:old[col]=pd.Series(np.nan,index=old.index,dtype='object' if new[col].dtype=='object' else 'float64')
    old.loc[oldmask,new.columns]=new.to_numpy();old.to_csv(dest,sep='\t',index=False)
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

primary=[];hrrr=[];hospital=[]
for i,db in enumerate(s.DATABASES):
    f=data[data.database.eq(db)].copy()
    for j,name,cov in [(1,'Model_1_harmonized_pre_landmark',oxy),(2,'Model_2_extended_landmark_support',full)]:
        primary.append(fit(f,'death_24h',name,cov,7000+i*10000+j*1000))
    hrrr.append(fit(f,'death_24h','harmonized_severity_adjusted',full,32000+i*1000+2*100,True).drop(columns=['phenotype','reference'],errors='ignore'))
    hospital.append(fit(f,'hospital_mortality','hospital_harmonized',full,60000+i*2000+1000))
replace_file('02_修订主要效应与严重度.tsv',R/'tables',lambda a:a.model.isin(['Model_1_harmonized_pre_landmark','Model_2_extended_landmark_support']),pd.concat(primary,ignore_index=True))
replace_file('03_修订固定HRRR规则.tsv',R/'tables',lambda a:a.model.eq('harmonized_severity_adjusted'),pd.concat(hrrr,ignore_index=True))
replace_file('04_修订整个住院死亡.tsv',R/'tables',lambda a:a.model.eq('hospital_harmonized'),pd.concat(hospital,ignore_index=True))
# 普通MIMIC复合保持原风险集/事件；不重做Fine–Gray或事件分类。
composite=[]
for db,seed in [('MIMIC-IV',24000),('MIMIC-III',28000)]:
    f=data[data.database.eq(db)&data.support_free_at_12h.eq(True)].copy()
    r=fit(f,'organ_support_or_death_24h','harmonized_severity_adjusted',oxy,seed)
    r['model_key']=db+'_harmonized_severity_adjusted';r['horizon_hours']=24;r['seed']=s.SEED+seed
    comp=f.dropna(subset=['organ_support_or_death_24h','phenotype','age','male','observed_dimensions_48d'])
    r['phenotype_N']=r.phenotype.map(comp.groupby('phenotype').size());r['phenotype_events']=r.phenotype.map(comp.groupby('phenotype').organ_support_or_death_24h.sum());r['raw_risk_fraction']=r.phenotype.map(comp.groupby('phenotype').organ_support_or_death_24h.mean())
    r['RD_CI_low_percentage_points']=100*r.RD_CI_low;r['RD_CI_high_percentage_points']=100*r.RD_CI_high
    composite.append(r)
replace_file('05_修订普通复合模型.tsv',V/'tables',lambda a:a.model.eq('harmonized_severity_adjusted'),pd.concat(composite,ignore_index=True))
# eICU两窗使用原保存状态，不重建结局或另换时钟。
f=pd.read_parquet(P/'current_inputs/private/修订eICU逐人原始结局证据.parquet');f.stay_id=f.stay_id.astype(str)
f=f.drop(columns=['SpO2_min_0_12h','acutephysiologyscore'],errors='ignore').merge(data.loc[data.database.eq('eICU'),['stay_id','SpO2_min_0_12h','acutephysiologyscore']],on='stay_id',validate='one_to_one')
events=pd.read_parquet(P/'current_inputs/private/修订eICU竞争风险逐人事件.parquet');aligned=[]
for horizon in [24,48]:
    e=events.loc[events.scenario.eq('FG_CI')&events.horizon_hours.eq(horizon)&events.status.isin([0,1,2])]
    g=f.merge(e[['stay_id','status','time_from_landmark']],on='stay_id',validate='one_to_one');g['aligned_composite']=g.status.eq(1).astype(float)
    r=fit(g,'aligned_composite','ICU范围复合_'+str(horizon)+'h_oxygenation_adjusted',oxy,110000+horizon*1000+100)
    r['horizon_hours']=horizon;r['target_semantics']='ICU范围内新器官支持或记录的ICU死亡；存活ICU出科为竞争事件';aligned.append(r)
replace_file('14_修订同范围复合普通回归.tsv',R/'tables',lambda a:a.model.str.endswith('oxygenation_adjusted'),pd.concat(aligned,ignore_index=True))

# 只更新临床描述中的两项血氧，不改变其他行。
clinical=load(P/'revision_R1_R2_data/scripts/07_clinical_characterization_and_stress.py','original_characterization')
tables,_=clinical.characterize(data)
for db in s.DATABASES:
    n='10_'+db+'_修订四型临床表征.tsv';fresh=tables[db][tables[db].variable.isin(['SpO2_mean','SpO2_minimum'])].copy()
    fresh['comorbidity_source_note']='MIMIC使用原始诊断表恢复的编码；有有效诊断且无该病编码记0，不等同临床确认无病；eICU沿用原可得描述'
    replace_file(n,R/'tables',lambda a:a.variable.isin(['SpO2_mean','SpO2_minimum']),fresh)
stress=clinical.stress_profile(data);replace_file('10_修订P3与P2临床描述.tsv',R/'tables',lambda a:a.variable.isin(['SpO2_mean','SpO2_minimum']),stress[stress.variable.isin(['SpO2_mean','SpO2_minimum'])])
pd.DataFrame(missingrows).to_csv(O/'tables/02_各模型血氧缺失与插补.tsv',sep='\t',index=False)
(O/'关联检查点.json').write_text(json.dumps(dict(status='当前32维人群全部血氧依赖关联完成',checks=len(checks),failed=0),ensure_ascii=False,indent=2))
