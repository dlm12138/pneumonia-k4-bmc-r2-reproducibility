"""Artificial boundary checks for recovered source logic; no real data/model access."""
from pathlib import Path
import argparse, ast, math, os, subprocess, sys, tempfile
import numpy as np
import pandas as pd

CODE=Path(__file__).resolve().parent
def functions(path,names,environment):
    tree=ast.parse(path.read_text())
    for n in tree.body:
        if isinstance(n,ast.Assign):
            try:value=ast.literal_eval(n.value)
            except (ValueError,TypeError):continue
            for target in n.targets:
                if isinstance(target,ast.Name):environment[target.id]=value
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise RuntimeError('Source function missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),environment)
    return environment

def main(args):
    out=Path(args.output_root).resolve();repo=CODE.parent
    if out==repo or repo in out.parents:raise ValueError('Use a private output directory outside repository')
    out.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    e=functions(CODE/'source_history/assembly/12_eicu_event_classification.py',
                ['classify','boundary_tests'],{'math':math,'np':np,'pd':pd,'LANDMARK':720.0})
    for r in e['boundary_tests']().itertuples(index=False):check('original_event_'+r.check,r.passed)
    h=functions(CODE/'source_history/raw/01_pneumonia_identification.py',
                ['icd9_match','icd10_match','keyword_match_pneumonia'],{'pd':pd})
    check('historical_ICD9_positive',h['icd9_match']('486'))
    check('historical_ICD9_negative',not h['icd9_match']('25000'))
    check('historical_U071_inclusion_visible',h['icd10_match']('U071'))
    check('historical_short_substring_problem_visible',h['keyword_match_pneumonia']('hypercapnia'))
    check('historical_clear_pneumonia_match',h['keyword_match_pneumonia']('pneumonia'))
    for path in (CODE/'source_history').rglob('*.py'):
        p=subprocess.run([sys.executable,str(path)],capture_output=True,text=True,env=os.environ.copy())
        check('direct_execution_blocked_'+path.name,p.returncode!=0 and 'Historical/source-reference program' in p.stderr)
    launch=subprocess.run([sys.executable,str(CODE/'run_source_reference.py'),'--execute-source-reference',
              'source_history/historical_model/19_outcome_free_grid.py'],capture_output=True,text=True)
    check('historical_fit_requires_opt_in',launch.returncode!=0 and 'Historical fitting is disabled' in launch.stderr)
    sys.path.insert(0,str(CODE))
    from core import clinical
    with tempfile.TemporaryDirectory(dir=out,prefix='artificial_features_') as tmp:
        rows={'database':['MIMIC-IV'],'stay_id':['SYNTHETIC_ONLY']}
        for vital in clinical.VITALS:
            for hour in range(12):rows[f'{vital}_h{hour}']=[float(hour+1)]
        f=Path(tmp)/'artificial.parquet';pd.DataFrame(rows).to_parquet(f,index=False)
        clinical.CANDIDATES=f
        result=clinical.append_vital_summaries(pd.DataFrame({'database':['MIMIC-IV'],'stay_id':['SYNTHETIC_ONLY']}))
        for vital in clinical.VITALS:
            check(vital+'_original_hourly_summary',result[f'{vital}_mean_0_12h'].iloc[0]==6.5 and
                    result[f'{vital}_min_0_12h'].iloc[0]==1 and result[f'{vital}_max_0_12h'].iloc[0]==12)
    pd.DataFrame(checks).to_csv(out/'source_synthetic_checks.tsv',sep='\t',index=False)
    print('SOURCE_SYNTHETIC_ONLY',len(checks),'PASS',sum(c['passed'] for c in checks))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);main(p.parse_args())
