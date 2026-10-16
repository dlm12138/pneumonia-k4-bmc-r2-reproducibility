"""Packaging dispatcher only; actual checked fitting is in core/audited_fit.py.
Specifications and seed offsets are transcribed from the actual current workflow;
this is not a replacement estimator or a newly selected model specification.
"""
import os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from core import association as s
from core.audited_fit import fit, O

data=pd.read_parquet(Path(os.environ['ANALYSIS_INPUT']))
data['HR_mean10']=data.HR_mean/10
data['RR_mean10']=data.RR_mean/10
base=['age10','male','coverage10']
results={k:[] for k in ['primary','hrrr','hospital','apache','comorbidity','pneumonia']}
def save(kind, rows):
    results[kind].append(rows)
    pd.concat(results[kind],ignore_index=True).to_csv(O/'tables'/('public_'+kind+'.tsv'),sep='\t',index=False)
for dbi,db in enumerate(s.DATABASES):
    f=data.loc[data.database.eq(db)].copy()
    for name,cov,offset in [('Model_0_current',base,7000+dbi*10000),
        ('C_current_plus_conventional_HR_RR_means',base+['HR_mean10','RR_mean10'],10000+dbi*10000)]:
        save('primary',fit(f,'death_24h',name,cov,offset))
    for i,(name,cov) in enumerate([('unadjusted',[]),('current_adjusted',base)]):
        save('hrrr',fit(f,'death_24h',name,cov,32000+dbi*1000+i*100,True))
    save('hospital',fit(f,'hospital_mortality','hospital_current',base,60000+dbi*2000))
    if db=='eICU':
        g,_=s.prepare_model_data(f,'hospital_mortality',require_apache=True)
        for i,(name,cov) in enumerate([('APACHE_same_sample_baseline',base),
                ('APACHE_IVa_adjusted_existing_specification',base+['APS10'])]):
            save('apache',fit(g,'hospital_mortality',name,cov,16000+i*1000))
        save('pneumonia',fit(f.loc[f.early_specific.eq(True)],'death_24h','explicit_pneumonia_documented_by12',base,51000))
    else:
        recovered=['raw_'+c for c in s.COMORBIDITIES]
        g=f.dropna(subset=recovered).copy()
        if len(g)!=len(f): raise ValueError('Recovered comorbidity input is incomplete; do not treat missing as zero.')
        for i,(name,cov) in enumerate([('recovered_same_sample_baseline',base),('recovered_full_comorbidity',base+recovered)]):
            save('comorbidity',fit(g,'death_24h',name,cov,65000+dbi*2000+i*1000))
        save('pneumonia',fit(f.loc[~f.aspiration_only],'death_24h','exclude_aspiration_only',base,55000+dbi*1000))
