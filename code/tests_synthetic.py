"""Only synthetic fixtures. Never opens an actual study input or model asset.

Tests verify released calculations and SQL properties, not clinical truth,
privacy authorisation or full real-data reproduction.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import multivariate_normal
import statsmodels.api as sm

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
from core import association, locked_k4, oxygen, prediction, secondary_events, time_order, early_assignment, raw_vitals
from render_sql import render

def main(args):
    out=Path(args.output_root).resolve();repository=CODE.parent
    if out==repository or repository in out.parents: raise ValueError('Use a private output root outside the repository')
    out.mkdir(parents=True,exist_ok=True);os.environ['OUTPUT_ROOT']=str(out)
    checks=[]
    def check(name,condition,detail=''):
        checks.append(dict(check=name,passed=bool(condition),detail=str(detail)))
        if not condition:raise AssertionError(name+': '+str(detail))
    rng=np.random.default_rng(73421)
    n=360;x=np.column_stack([np.ones(n),rng.normal(size=(n,3))]);y=rng.binomial(1,expit(x @ np.array([-.4,.5,-.2,.1])))
    beta,converged=association.fit_logistic(x,y)
    oracle=sm.GLM(y,x,family=sm.families.Binomial()).fit(cov_type='HC0',tol=1e-12)
    check('unpenalised_IRLS_coefficients',converged and np.allclose(beta,oracle.params,atol=1e-7,rtol=0))
    check('HC0_covariance',np.allclose(association.robust_covariance(x,y,beta,None),oracle.cov_params(),atol=1e-7,rtol=0))
    clusters=np.repeat(np.arange(30),12)
    clustered=sm.GLM(y,x,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':clusters,'use_correction':True},tol=1e-12)
    check('cluster_covariance',np.allclose(association.robust_covariance(x,y,beta,clusters),clustered.cov_params(),atol=1e-7,rtol=0))
    # Random artificial means/weights are generated in memory, not fitted and not real model assets.
    means=rng.normal(size=(4,48));cov=np.array([np.eye(48)*(i+1) for i in range(4)])
    model=SimpleNamespace(n_components=4,covariance_type='full',weights_=np.array([.1,.2,.3,.4]),means_=means,covariances_=cov)
    raw=rng.normal(size=(9,48));raw[::2,::3]=np.nan
    mapping=dict(enumerate(['P1','P2','P3','P4']))
    labels,post=locked_k4.assign_locked_k4(raw,np.zeros(48),np.ones(48),model,mapping)
    ll=locked_k4.observed_dimension_log_likelihood(raw,model);independent=np.empty_like(ll)
    for i,row in enumerate(raw):
        observed=np.flatnonzero(np.isfinite(row))
        for k in range(4): independent[i,k]=np.log(model.weights_[k])+multivariate_normal.logpdf(row[observed],mean=means[k,observed],cov=cov[k][np.ix_(observed,observed)])
    check('observed_dimension_marginal_likelihood',np.allclose(ll,independent,atol=1e-10))
    check('posterior_normalisation',np.allclose(post.sum(axis=1),1))
    early_components,earlypost=early_assignment.assign(raw[:,:36],np.zeros(48),np.ones(48),model,feature_indices=np.arange(36))
    check('early_assignment_dependency_complete',len(early_components)==9 and np.allclose(earlypost.sum(axis=1),1))
    changed=time_order.transform_raw(raw,'reversed')
    check('time_transform_preserves_missing_mask',np.array_equal(np.isfinite(raw),np.isfinite(changed)))
    check('time_transform_preserves_multisets',np.allclose(np.sort(np.nan_to_num(raw,nan=-999),axis=1),np.sort(np.nan_to_num(changed,nan=-999),axis=1)))
    score=time_order.FixedMaskScorer(raw,np.zeros(48),np.ones(48),model).posterior(raw)
    check('fixed_mask_scoring_matches_assignment',np.allclose(score,post,atol=1e-12))
    frame=pd.DataFrame({'database':['SYNTHETIC']*2,'stay_id':['SYNTHETIC_A']*2,'SpO2':[80.,99.]})
    agg=oxygen.aggregate_hourly_spo2(frame)
    check('oxygen_minimum_is_hourly_mean_minimum',float(agg.SpO2_min_0_12h.iloc[0])==80 and float(agg.SpO2_mean_0_12h.iloc[0])==89.5)
    check('oxygen_count_means_hours',int(agg.SpO2_n_records_0_12h.iloc[0])==2)
    for v,lo,hi in [('SBP',40,300),('DBP',20,200),('HR',20,250),('RR',3,80)]:
        check('clean_range_'+v,raw_vitals.clean_value(v,str(lo),'')[0]==lo and raw_vitals.clean_value(v,str(hi),'')[0]==hi and raw_vitals.clean_value(v,str(lo-1),'')[0] is None)
    boundaries=[(12,np.nan,20,24,2,8),(13,np.nan,13,24,1,1),(14,np.nan,13,24,2,1),(np.nan,13,13,24,1,1),(36,np.nan,40,24,1,24),(36.00001,np.nan,40,24,0,24)]
    for i,(s,d,e,h,expected,t) in enumerate(boundaries):
        result=secondary_events.classify_scalar(s,d,e,h);check('competing_event_boundary_'+str(i),result==(expected,t))
    data=pd.DataFrame(rng.normal(size=(n,7)),columns=['age','male','observed_dimensions_48d','SpO2_min_0_12h','active_invasive_ventilation_at_12h','active_vasopressor_at_12h','active_RRT_at_12h'])
    data['phenotype']=np.resize(['P1','P2','P3','P4'],n);data['fixed_HRRR_rule']=np.resize([0,1],n)
    for v in ['SBP','DBP','HR','RR']:
        for s in ['mean','count','sd','slope']:data[v+'_'+s]=rng.normal(size=n)
    numeric=list(data.columns[:7]);conventional=[v+'_'+s for v in ['SBP','DBP','HR','RR'] for s in ['mean','count','sd','slope']]
    models={'A_base_severity':(numeric,[]),'B_base_plus_HRRR':(numeric+['fixed_HRRR_rule'],[]),
        'C_base_plus_K4':(numeric,['phenotype']),'D_base_plus_conventional':(numeric+conventional,[]),
        'E_base_conventional_plus_K4':(numeric+conventional,['phenotype'])}
    fold=np.repeat(np.arange(30)%5,12);probabilities={}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=FutureWarning)
        for name,(num,cat) in models.items():
            probabilities[name]=np.full(n,np.nan)
            for f in range(5):
                train=fold!=f;test=~train;pipe=prediction.prediction_pipeline(num,cat)
                pipe.fit(data.loc[train,num+cat],y[train]);est=pipe.steps[-1][1]
                check(name+'_fixed_settings_'+str(f),est.C==1 and est.penalty=='l2' and est.solver=='lbfgs' and est.tol==1e-4 and est.max_iter==2000 and est.n_iter_.max()<2000)
                check(name+'_group_disjoint_'+str(f),not set(clusters[train])&set(clusters[test]))
                probabilities[name][test]=pipe.predict_proba(data.loc[test,num+cat])[:,1]
            point=prediction.metric_values(y,probabilities[name])
            check(name+'_finite_metrics',np.isfinite([point['AUROC'],point['AUPRC'],point['Brier']]).all())
    # Full, unchanged 500-draw routine, on artificial observations only.
    metrics,deltas=prediction.bootstrap_prediction_metrics(y,probabilities,clusters.astype(str),'MIMIC-IV')
    check('paired_bootstrap_fixed_500',metrics.bootstrap_successful.eq(500).all() and deltas.bootstrap_successful.eq(500).all())
    check('paired_bootstrap_uses_five_and_three_models',len(metrics)==5 and len(deltas)==3)

    # Current quasi-separation implementation, retaining RRT and omitting only an all-zero missing flag.
    from core import audited_fit
    f=pd.DataFrame({'database':['MIMIC-III']*n,'death_24h':y,'phenotype':np.resize(['P1','P2','P3','P4'],n),
        'age10':x[:,1],'male':x[:,2],'coverage10':x[:,3], 'SpO2_min10':rng.normal(size=n),
        'SpO2_min_0_12h_missing':np.zeros(n),'active_RRT_at_12h':np.zeros(n)})
    zeros=np.flatnonzero(y==0)[:27];f.loc[zeros,'active_RRT_at_12h']=1
    used,estimable=audited_fit.audit(f,'death_24h','SYNTHETIC_separation',['age10','male','coverage10','SpO2_min10','SpO2_min_0_12h_missing','active_RRT_at_12h'])
    check('separation_not_silently_repaired',not estimable and 'active_RRT_at_12h' in used and 'SpO2_min_0_12h_missing' not in used)

    # All source SQL is run on tiny artificial CSV tables, never the raw databases.
    with tempfile.TemporaryDirectory(prefix='synthetic_sql_',dir=out) as tmp:
        tmp=Path(tmp);keys=tmp/'keys.parquet';pd.DataFrame({'stay_id':['1']}).to_parquet(keys,index=False)
        origin=pd.Timestamp('2020-01-01')
        directories={short:tmp/short for short in ['mimiciv','mimiciii','eicu']}
        for directory in directories.values():directory.mkdir()
        def csv(name,rows):pd.DataFrame(rows).to_csv(rawdir/name,index=False,compression='gzip')
        for upper in [False,True]:
            idc='icustay_id' if upper else 'stay_id';short='mimiciii' if upper else 'mimiciv'
            rawdir=directories[short]
            csv('ICUSTAYS.csv.gz' if upper else 'icustays.csv.gz',[{idc:1,'intime':str(origin)}])
            vals=[(0,646 if upper else 220277,50),(0,220277,100),(60,220277,99),(720,220277,60),(-1,220277,70),(120,220277,49),(180,220227,97)]
            csv('CHARTEVENTS.csv.gz' if upper else 'chartevents.csv.gz',[{idc:1,'charttime':str(origin+pd.Timedelta(minutes=t)),'itemid':item,'valuenum':v} for t,item,v in vals])
            rows=[{idc:1,'starttime':str(origin+pd.Timedelta(hours=10)),'endtime':str(origin+pd.Timedelta(hours=13)),'itemid':221906,'cancelreason':0}]
            csv('INPUTEVENTS_MV.csv.gz' if upper else 'inputevents.csv.gz',rows)
            csv('PROCEDUREEVENTS_MV.csv.gz' if upper else 'procedureevents.csv.gz',[dict(rows[0],itemid=225792),dict(rows[0],itemid=225441)])
        rawdir=directories['eicu']
        csv('vitalPeriodic.csv.gz',[{'patientunitstayid':1,'observationoffset':t,'sao2':v} for t,v in [(0,50),(0,100),(60,99),(720,60),(-1,70),(120,49)]])
        csv('infusionDrug.csv.gz',[{'patientunitstayid':1,'infusionoffset':720,'drugrate':1,'infusionrate':0,'drugname':'norepinephrine'}])
        csv('respiratoryCare.csv.gz',[{'patientunitstayid':1,'ventstartoffset':700,'ventendoffset':0}])
        csv('treatment.csv.gz',[{'patientunitstayid':1,'treatmentoffset':720,'treatmentstring':'mechanical ventilation'},{'patientunitstayid':1,'treatmentoffset':700,'treatmentstring':'crrt'}])
        for template in sorted((CODE/'sql').glob('*.sql')):
            query=render(template.read_text(),directories[template.stem.split('_')[0]],keys).rstrip().rstrip(';')
            result=tmp/(template.stem+'.parquet')
            escaped=str(result).replace("'","''")
            command=".bail on\nCOPY ("+query+") TO '"+escaped+"' (FORMAT PARQUET);"
            completed=subprocess.run([args.duckdb,'-batch'],input=command,text=True,capture_output=True)
            check('SQL_execution_'+template.stem,completed.returncode==0,completed.stderr)
            found=pd.read_parquet(result)
            check('SQL_selected_keys_'+template.stem,set(found.stay_id.astype(str)).issubset({'1'}))
            if 'pulse_oxygen' in template.stem:
                check('SQL_oxygen_window_and_source_'+template.stem,set(found.hour_index)=={0,1})
                grouped=found.groupby('hour_index')[['value_sum','n']].sum()
                check('SQL_oxygen_hour_mean_'+template.stem,np.isclose(grouped.loc[0,'value_sum']/grouped.loc[0,'n'],75))
    pd.DataFrame(checks).to_csv(out/'synthetic_checks.tsv',sep='\t',index=False)
    print('SYNTHETIC_ONLY_CHECKS',len(checks),'PASS',sum(r['passed'] for r in checks),'FAIL',sum(not r['passed'] for r in checks))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--duckdb',default='duckdb');main(p.parse_args())
