"""Historical legacy eICU hourly source, not corrected oxygen.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'\nScript 04: Extract vital signs from eICU databases\nSBP/DBP from vitalAperiodic, HR/RR/TEMP from vitalPeriodic, TEMP supplement from nurseCharting\n'
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
RUN_DIR = Path(os.environ['SOURCE_1B251D2246'])
EICU_DIR = Path(os.environ['SOURCE_B41C1FC8CF'])
PROCESSED = RUN_DIR / '03_processed'
TABLES = RUN_DIR / '05_tables'
LOGS = RUN_DIR / '00_logs'
print('=' * 60)
print('eICU VITAL SIGN EXTRACTION')
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 60)
PHYS_RANGES = {'HR': (20, 250), 'RR': (5, 80), 'SBP': (50, 260), 'DBP': (20, 160), 'TEMP': (30, 43)}

def apply_phys_filter(df, col, vital):
    lo, hi = PHYS_RANGES[vital]
    before = len(df)
    df = df[(df[col] >= lo) & (df[col] <= hi)].copy()
    removed = before - len(df)
    if removed > 0:
        print(f'  Physiological filter {vital}: removed {removed:,} out-of-range values')
    return df
print('\nLoading eICU pneumonia stays...')
stays_raw = pd.read_csv(RUN_DIR / '02_raw_extract/eicu_pneumonia_stays_raw.csv', usecols=['patientunitstayid'])
pneumonia_ids = set(stays_raw['patientunitstayid'].tolist())
print(f'  Pneumonia patientunitstayids: {len(pneumonia_ids):,}')
print('\nLoading vitalPeriodic (HR, RR, TEMP)...')
vp = pd.read_csv(EICU_DIR / 'vitalPeriodic.csv.gz', usecols=['patientunitstayid', 'observationoffset', 'heartrate', 'respiration', 'temperature'])
print(f'  Total rows: {len(vp):,}')
vp = vp[vp['patientunitstayid'].isin(pneumonia_ids)].copy()
print(f'  Pneumonia rows: {len(vp):,}')
vp = vp[(vp['observationoffset'] >= 0) & (vp['observationoffset'] <= 720)].copy()
vp['hour'] = (vp['observationoffset'] / 60).apply(np.floor).astype(int)
print(f'  After 0-720 min filter: {len(vp):,}')
vp_long = vp.melt(id_vars=['patientunitstayid', 'observationoffset', 'hour'], value_vars=['heartrate', 'respiration', 'temperature'], var_name='raw_vital', value_name='value')
vital_rename = {'heartrate': 'HR', 'respiration': 'RR', 'temperature': 'TEMP'}
vp_long['vital'] = vp_long['raw_vital'].map(vital_rename)
vp_long = vp_long.dropna(subset=['value'])
vp_long = vp_long[vp_long['value'].notna()]
frames = []
for vital in ['HR', 'RR', 'TEMP']:
    sub = vp_long[vp_long['vital'] == vital].copy()
    sub = apply_phys_filter(sub, 'value', vital)
    frames.append(sub)
vp_long = pd.concat(frames, ignore_index=True)
print(f'  After physio filter: {len(vp_long):,} rows')
vp_hourly = vp_long.groupby(['patientunitstayid', 'hour', 'vital'])['value'].mean().reset_index()
print(f'  Hourly aggregated: {len(vp_hourly):,} stay-hour-vital rows')
print('\nLoading vitalAperiodic (SBP, DBP)...')
va = pd.read_csv(EICU_DIR / 'vitalAperiodic.csv.gz', usecols=['patientunitstayid', 'observationoffset', 'noninvasivesystolic', 'noninvasivediastolic'])
print(f'  Total rows: {len(va):,}')
va = va[va['patientunitstayid'].isin(pneumonia_ids)].copy()
va = va[(va['observationoffset'] >= 0) & (va['observationoffset'] <= 720)].copy()
va['hour'] = (va['observationoffset'] / 60).apply(np.floor).astype(int)
print(f'  Pneumonia rows (0-720 min): {len(va):,}')
va_long = va.melt(id_vars=['patientunitstayid', 'observationoffset', 'hour'], value_vars=['noninvasivesystolic', 'noninvasivediastolic'], var_name='raw_vital', value_name='value')
vital_rename2 = {'noninvasivesystolic': 'SBP', 'noninvasivediastolic': 'DBP'}
va_long['vital'] = va_long['raw_vital'].map(vital_rename2)
va_long = va_long.dropna(subset=['value'])
frames2 = []
for vital in ['SBP', 'DBP']:
    sub = va_long[va_long['vital'] == vital].copy()
    sub = apply_phys_filter(sub, 'value', vital)
    frames2.append(sub)
va_long = pd.concat(frames2, ignore_index=True)
print(f'  After physio filter: {len(va_long):,} rows')
va_hourly = va_long.groupby(['patientunitstayid', 'hour', 'vital'])['value'].mean().reset_index()
print(f'  Hourly aggregated: {len(va_hourly):,} stay-hour-vital rows')
print('\nLoading nurseCharting (TEMP supplement)...')
temp_source_mapping = []
try:
    nc_chunks = []
    for chunk in pd.read_csv(EICU_DIR / 'nurseCharting.csv.gz', usecols=['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue'], chunksize=500000):
        chunk = chunk[chunk['patientunitstayid'].isin(pneumonia_ids)].copy()
        if len(chunk) > 0:
            nc_chunks.append(chunk)
    if nc_chunks:
        nc = pd.concat(nc_chunks, ignore_index=True)
        print(f'  nurseCharting rows for pneumonia stays: {len(nc):,}')
        temp_labels = nc['nursingchartcelltypevalname'].dropna().unique()
        temp_matches = [l for l in temp_labels if 'temp' in str(l).lower() or 'temperature' in str(l).lower()]
        print(f'  Temperature-related labels found ({len(temp_matches)}):')
        for lbl in sorted(temp_matches):
            cnt = (nc['nursingchartcelltypevalname'] == lbl).sum()
            print(f"    '{lbl}': {cnt:,} rows")
            temp_source_mapping.append({'source': 'nurseCharting', 'label': lbl, 'n_rows': cnt})
        nc_temp = nc[nc['nursingchartcelltypevalname'].isin(temp_matches)].copy()
        nc_temp = nc_temp[(nc_temp['nursingchartoffset'] >= 0) & (nc_temp['nursingchartoffset'] <= 720)].copy()
        nc_temp['hour'] = (nc_temp['nursingchartoffset'] / 60).apply(np.floor).astype(int)
        nc_temp['value'] = pd.to_numeric(nc_temp['nursingchartvalue'], errors='coerce')
        nc_temp = nc_temp.dropna(subset=['value'])
        f_labels = [l for l in temp_matches if 'f' in str(l).lower() and 'fahrenheit' in str(l).lower()]
        if f_labels:
            f_mask = nc_temp['nursingchartcelltypevalname'].isin(f_labels)
            nc_temp.loc[f_mask, 'value'] = (nc_temp.loc[f_mask, 'value'] - 32) / 1.8
            print(f'  Converted Fahrenheit labels: {f_labels}')
        high_mask = nc_temp['value'] > 43
        nc_temp.loc[high_mask, 'value'] = (nc_temp.loc[high_mask, 'value'] - 32) / 1.8
        if high_mask.sum() > 0:
            print(f'  Auto-converted {high_mask.sum():,} values >43 from F to C')
        nc_temp['vital'] = 'TEMP'
        nc_temp = apply_phys_filter(nc_temp, 'value', 'TEMP')
        nc_temp_hourly = nc_temp.groupby(['patientunitstayid', 'hour', 'vital'])['value'].mean().reset_index()
        print(f'  nurseCharting TEMP hourly rows: {len(nc_temp_hourly):,}')
    else:
        nc_temp_hourly = pd.DataFrame(columns=['patientunitstayid', 'hour', 'vital', 'value'])
        print('  No nurseCharting data for pneumonia stays')
except Exception as e:
    print(f'  nurseCharting load error: {e}')
    nc_temp_hourly = pd.DataFrame(columns=['patientunitstayid', 'hour', 'vital', 'value'])
print('\nCombining all vital sources...')
vp_temp = vp_hourly[vp_hourly['vital'] == 'TEMP'].copy()
vp_hr_rr = vp_hourly[vp_hourly['vital'].isin(['HR', 'RR'])].copy()
if len(nc_temp_hourly) > 0:
    vp_temp_stack = vp_temp[['patientunitstayid', 'hour', 'vital', 'value']].copy()
    nc_temp_stack = nc_temp_hourly[['patientunitstayid', 'hour', 'vital', 'value']].copy()
    all_temp = pd.concat([vp_temp_stack, nc_temp_stack], ignore_index=True)
    temp_hourly = all_temp.groupby(['patientunitstayid', 'hour', 'vital'])['value'].mean().reset_index()
    print(f'  Combined TEMP rows: {len(temp_hourly):,}')
else:
    temp_hourly = vp_temp.copy()
    print(f'  Using vitalPeriodic TEMP only: {len(temp_hourly):,}')
all_hourly = pd.concat([vp_hr_rr, temp_hourly, va_hourly], ignore_index=True)
all_hourly = all_hourly.groupby(['patientunitstayid', 'hour', 'vital'])['value'].mean().reset_index()
print(f'  Total combined hourly rows: {len(all_hourly):,}')
df_wide = all_hourly.pivot_table(index=['patientunitstayid', 'hour'], columns='vital', values='value').reset_index()
df_wide.columns.name = None
for col in ['SBP', 'DBP', 'HR', 'RR', 'TEMP']:
    if col not in df_wide.columns:
        df_wide[col] = np.nan
        print(f'  WARNING: {col} missing entirely')
df_wide = df_wide[['patientunitstayid', 'hour', 'SBP', 'DBP', 'HR', 'RR', 'TEMP']]
print(f'\nWide hourly data: {len(df_wide):,} stay-hour rows')
for col in ['SBP', 'DBP', 'HR', 'RR', 'TEMP']:
    pct = df_wide[col].isna().mean() * 100
    print(f'  {col}: {pct:.1f}% missing')
out_all = PROCESSED / 'eicu_vital_12h_hourly_all.csv'
df_wide.to_csv(out_all, index=False)
print(f'\nSaved: {out_all}')
df_complete = df_wide.dropna(subset=['SBP', 'DBP', 'HR', 'RR', 'TEMP']).copy()
print(f'\nComplete hours: {len(df_complete):,}')
complete_counts = df_complete.groupby('patientunitstayid').size()
eligible = complete_counts[complete_counts >= 2].index
df_complete = df_complete[df_complete['patientunitstayid'].isin(eligible)].copy()
print(f'After ≥2 complete hours: {len(df_complete):,} rows, {len(eligible):,} stays')
out_complete = PROCESSED / 'eicu_vital_12h_complete_hour.csv'
df_complete.to_csv(out_complete, index=False)
print(f'Saved: {out_complete}')
temp_source_mapping.append({'source': 'vitalPeriodic', 'label': 'temperature', 'n_rows': len(vp_temp)})
temp_source_mapping.append({'source': 'vitalAperiodic', 'label': 'N/A (no temp)', 'n_rows': 0})
ts_df = pd.DataFrame(temp_source_mapping)
ts_df.to_csv(TABLES / 'Table_S3_temperature_source_mapping.csv', index=False)
print(f'\nSaved: Table_S3_temperature_source_mapping.csv')
with open(LOGS / 'CURRENT_STAGE.txt', 'w') as f:
    f.write(f'COMPLETED: 04_extract_vitals_eicu.py\n')
    f.write(f'NEXT: 05_extract_vitals_chunked_mimiciii.py\n')
    f.write(f'eICU complete-hour stays: {len(eligible):,}\n')
    f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
print('\n=== Script 04 COMPLETE ===')
