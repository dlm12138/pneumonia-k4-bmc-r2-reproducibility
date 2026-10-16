"""五个既有覆盖阈值只重算氧合列；冻结分配，不拟合混合模型。"""

import os, sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from release_runtime import load_actual, init_output, public_result_path

from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
P=Path(os.environ['DATA_ROOT']);O=Path(os.environ['OUTPUT_ROOT']);R=P/'current_inputs';V=R
init_output(O)
def load(path,name):
    return load_actual(path,name)
s=load(P/'revision_R1_R2_data/scripts/05_harmonized_severity_and_apache.py','severity_coverage')
mr=P/'revision_R1_R2_data/locked_K4_model';assignment=load(mr/'assignment_code_clean.py','frozen_assign')
order=pd.read_csv(mr/'feature_order.csv').sort_values('feature_position').feature.tolist();standard=pd.read_csv(mr/'standardization_parameters.csv').set_index('feature').loc[order];w=pd.read_csv(mr/'component_weights.csv').sort_values('raw_component_zero_based');m=pd.read_csv(mr/'component_means.csv')
model=SimpleNamespace(n_components=4,weights_=w.weight.to_numpy(float),means_=np.array([m.loc[m.raw_component_zero_based.eq(i)].set_index('feature').loc[order].component_mean_z for i in range(4)]),covariances_=np.array([pd.read_csv(mr/('covariance_'+p+'.csv'),index_col=0).loc[order,order] for p in w.phenotype]))
mapping=dict(zip(w.raw_component_zero_based.astype(int),w.phenotype))
c=pd.read_parquet(O/'private/24维以上严格肺炎候选_原特征结局不变.parquet');c=c.drop(columns=['SpO2_min_0_12h'],errors='ignore').merge(pd.read_parquet(O/'private/纠正外周血氧汇总_覆盖所有所需阈值.parquet')[['database','stay_id','SpO2_min_0_12h']],on=['database','stay_id'],how='left',validate='one_to_one')
for col in ['acutephysiologyscore','active_invasive_ventilation_at_12h','active_vasopressor_at_12h','active_RRT_at_12h']:c[col]=np.nan
oldsummary=pd.read_csv(public_result_path('15_修订覆盖阈值汇总.tsv'),sep='\t');olddetail=pd.read_csv(public_result_path('15_修订覆盖阈值完整效应.tsv'),sep='\t');summary=oldsummary.copy();detail=olddetail.copy()
current=pd.read_parquet(O/'private/修订三库全模块分析输入.parquet');primary=pd.read_csv(O/'tables/02_修订主要效应与严重度.tsv',sep='\t');checks=[];missing=[]
def check(name,ok,detail=''):
    checks.append(dict(check=name,passed=bool(ok),detail=str(detail)));pd.DataFrame(checks).to_csv(O/'qc/04_覆盖氧合独立核查.tsv',sep='\t',index=False)
    if not ok:raise RuntimeError(name)
for dbi,db in enumerate(s.DATABASES):
    for threshold in [24,28,32,36,40]:
        f=c.loc[c.database.eq(db)&c.observed_dimensions_48d.ge(threshold)].copy()
        labels,post=assignment.assign_locked_k4(f[order].to_numpy(float),standard.MIMIC_IV_observed_mean.to_numpy(float),standard.MIMIC_IV_observed_sd_ddof0.to_numpy(float),model,mapping)
        f['phenotype']=labels;f['max_posterior']=post.max(axis=1)
        z=f.loc[f.observed_dimensions_48d.ge(32)].merge(current[['database','stay_id','phenotype']],on=['database','stay_id'],suffixes=('','_old'),validate='one_to_one')
        check(db+str(threshold)+'_32维以上标签完全不变',z.phenotype.eq(z.phenotype_old).all())
        prepared,pars=s.prepare_model_data(f,'death_24h');oldrow=oldsummary[oldsummary.database.eq(db)&oldsummary.coverage_threshold.eq(threshold)].iloc[0]
        key=db+'_'+str(threshold);check(key+'_分母不变',len(f)==oldrow.assignable_N);check(key+'_主模型人数不变',len(prepared)==oldrow.primary_model_N);check(key+'_事件不变',prepared.death_24h.sum()==oldrow.primary_model_events)
        for p in s.PHENOTYPES:check(key+'_'+p+'_人数不变',int(f.phenotype.eq(p).sum())==oldrow[p+'_N'])
        cov=['age10','male','coverage10','SpO2_min10','SpO2_min_0_12h_missing']
        removed=prepared.SpO2_min_0_12h_missing.eq(0).all()
        if removed:cov.remove('SpO2_min_0_12h_missing')
        x,_=s.design_matrix(prepared,cov);y=prepared.death_24h.to_numpy(float);check(key+'_满秩',np.linalg.matrix_rank(x)==x.shape[1])
        signed=(2*y-1)[:,None]*x;sep=linprog(-signed.sum(axis=0),A_ub=-signed,b_ub=np.zeros(len(y)),bounds=[(-1,1)]*x.shape[1],method='highs');check(key+'_无分离',sep.success and sep.fun>=-1e-7)
        b,conv=s.fit_logistic(x,y);check(key+'_收敛',conv);groups=prepared.hospital_id.astype(str).to_numpy() if db=='eICU' else None
        kw=dict(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True}) if groups is not None else dict(cov_type='HC0')
        g=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-12,**kw);check(key+'_独立GLM',np.allclose(b,g.params,atol=1e-7,rtol=0));check(key+'_独立协方差',np.allclose(s.robust_covariance(x,y,b,groups),g.cov_params(),atol=1e-7,rtol=0))
        if threshold==32:r=primary[primary.database.eq(db)&primary.model.eq('Model_1_harmonized_pre_landmark')].copy()
        else:r,_=s.run_model(prepared,'death_24h','oxygenation_adjusted',cov,130000+dbi*10000+threshold*100+50)
        r['variance']='hospital-cluster robust' if db=='eICU' else 'HC0 sandwich robust';r['coverage_threshold']=threshold;r['coverage_model']='oxygenation_adjusted'
        p3=r[r.phenotype.eq('P3')].iloc[0];check(key+'_P3点估计一致',abs(p3.OR-np.exp(b[2]))<1e-8);check(key+'_重抽样次数',r.bootstrap_successful.min()==500)
        smask=summary.database.eq(db)&summary.coverage_threshold.eq(threshold)
        for col in ['OR','CI_low','CI_high']:summary.loc[smask,'oxygenation_adjusted_'+col]=p3[col]
        dmask=detail.database.eq(db)&detail.coverage_threshold.eq(threshold)&detail.coverage_model.eq('oxygenation_adjusted')
        for col in r:
            if col not in detail:detail[col]=np.nan
        detail.loc[dmask,r.columns]=r.to_numpy()
        missing.append(dict(database=db,coverage_threshold=threshold,N=len(prepared),events=int(y.sum()),SpO2_missing=int(prepared.SpO2_min_0_12h_missing.sum()),SpO2_median=pars['SpO2_min_0_12h'],redundant_missing_indicator_omitted=bool(removed)))
        summary.to_csv(O/'tables/15_修订覆盖阈值汇总.tsv',sep='\t',index=False);detail.to_csv(O/'tables/15_修订覆盖阈值完整效应.tsv',sep='\t',index=False);pd.DataFrame(missing).to_csv(O/'tables/04_覆盖阈值血氧缺失.tsv',sep='\t',index=False)
        print('只更新阈值氧合结果',key,'OR',p3.OR,flush=True)
(O/'覆盖氧合检查点.json').write_text(json.dumps(dict(status='五阈值仅氧合列更新完成',checks=len(checks),failed=0),ensure_ascii=False,indent=2))
