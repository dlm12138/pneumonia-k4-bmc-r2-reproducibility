"""Historical legacy MIMIC-III hourly source, not corrected oxygen.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'\nScript 05: Extract vital signs from MIMIC-III CHARTEVENTS (chunked streaming from ZIP)\nBoth CareVue and MetaVision itemids for SBP, DBP, HR, RR, TEMP\n'
import sys, os, zipfile, gzip, time
from pathlib import Path
import pandas as pd
import numpy as np
RUN_DIR = Path(os.environ['SOURCE_1B251D2246'])
MIMICIII_ZIP = os.environ['SOURCE_1D1D312607']
MIMICIII_RAW = RUN_DIR / '10_temp/mimiciii_raw'
PROCESSED = RUN_DIR / '03_processed'
LOGS = RUN_DIR / '00_logs'
print('=' * 60)
print('MIMIC-III VITAL SIGN EXTRACTION (CHUNKED)')
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 60)
HR_ITEMS = {211, 220045}
RR_ITEMS = {618, 220210, 224690}
SBP_ITEMS = {51, 442, 455, 220179, 220050, 225309}
SBP_ITEMS.add(3313)
DBP_ITEMS = {8368, 8440, 8441, 220180, 220051, 225310}
DBP_ITEMS.add(8502)
TEMP_C_ITEMS = {676, 677, 223762}
TEMP_F_ITEMS = {678, 679, 223761}
ALL_ITEMS = HR_ITEMS | RR_ITEMS | SBP_ITEMS | DBP_ITEMS | TEMP_C_ITEMS | TEMP_F_ITEMS
print(f'\nTargeted itemids: {len(ALL_ITEMS)} total')
print(f'  HR: {HR_ITEMS}')
print(f'  RR: {RR_ITEMS}')
print(f'  SBP: {SBP_ITEMS}')
print(f'  DBP: {DBP_ITEMS}')
print(f'  TEMP_C: {TEMP_C_ITEMS}')
print(f'  TEMP_F: {TEMP_F_ITEMS}')

def assign_vital_iii(itemid):
    if itemid in HR_ITEMS:
        return 'HR'
    if itemid in RR_ITEMS:
        return 'RR'
    if itemid in SBP_ITEMS:
        return 'SBP'
    if itemid in DBP_ITEMS:
        return 'DBP'
    if itemid in TEMP_C_ITEMS:
        return 'TEMP_C'
    if itemid in TEMP_F_ITEMS:
        return 'TEMP_F'
    return None
print('\nLoading MIMIC-III pneumonia ICU stays...')
stays_raw = pd.read_csv(RUN_DIR / '02_raw_extract/mimiciii_pneumonia_stays_raw.csv', usecols=['ICUSTAY_ID', 'INTIME'])
stays_raw['INTIME'] = pd.to_datetime(stays_raw['INTIME'])
pneumonia_icustay_ids = set(stays_raw['ICUSTAY_ID'].tolist())
intime_map = dict(zip(stays_raw['ICUSTAY_ID'], stays_raw['INTIME']))
print(f'  Pneumonia ICUSTAY_IDs: {len(pneumonia_icustay_ids):,}')
CHARTEVENTS_PATH = 'mimic-iii-clinical-database-1.4/CHARTEVENTS.csv.gz'
print(f'\nOpening MIMIC-III zip for CHARTEVENTS (4 GB compressed)...')
print('This will take 20-40 minutes...')
CHUNK_SIZE = 500000
collected = []
total_read = 0
matched_rows = 0
start_time = time.time()
with zipfile.ZipFile(MIMICIII_ZIP) as zf:
    with zf.open(CHARTEVENTS_PATH) as raw_gz:
        with gzip.open(raw_gz, 'rt', encoding='utf-8') as gz_file:
            reader = pd.read_csv(gz_file, chunksize=CHUNK_SIZE, usecols=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'ERROR'], dtype={'ICUSTAY_ID': 'Int64', 'ITEMID': 'Int64', 'VALUENUM': 'float64', 'ERROR': 'Int64'}, low_memory=False)
            for chunk_num, chunk in enumerate(reader):
                total_read += len(chunk)
                if 'ERROR' in chunk.columns:
                    chunk = chunk[chunk['ERROR'] != 1]
                mask = chunk['ICUSTAY_ID'].isin(pneumonia_icustay_ids) & chunk['ITEMID'].isin(ALL_ITEMS)
                matched = chunk[mask].copy()
                if len(matched) > 0:
                    collected.append(matched)
                    matched_rows += len(matched)
                if chunk_num % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = total_read / elapsed if elapsed > 0 else 0
                    print(f'  Chunk {chunk_num + 1}: {total_read / 1000000.0:.1f}M rows read, {matched_rows:,} matched, {rate / 1000000.0:.2f}M rows/sec, {elapsed:.0f}s elapsed')
print(f'\nTotal rows read: {total_read:,}')
print(f'Total matched rows: {matched_rows:,}')
if not collected:
    print('ERROR: No rows matched.')
    sys.exit(1)
df_chart = pd.concat(collected, ignore_index=True)
print(f'Combined DataFrame: {len(df_chart):,} rows')
df_chart['vital'] = df_chart['ITEMID'].apply(assign_vital_iii)
mask_f = df_chart['vital'] == 'TEMP_F'
df_chart.loc[mask_f, 'VALUENUM'] = (df_chart.loc[mask_f, 'VALUENUM'] - 32) / 1.8
df_chart.loc[mask_f, 'vital'] = 'TEMP'
df_chart.loc[df_chart['vital'] == 'TEMP_C', 'vital'] = 'TEMP'
df_chart['CHARTTIME'] = pd.to_datetime(df_chart['CHARTTIME'])
df_chart['intime'] = df_chart['ICUSTAY_ID'].map(intime_map)
df_chart['hours_after_icu'] = (df_chart['CHARTTIME'] - df_chart['intime']).dt.total_seconds() / 3600
df_chart = df_chart[(df_chart['hours_after_icu'] >= 0) & (df_chart['hours_after_icu'] < 12)].copy()
df_chart['hour'] = df_chart['hours_after_icu'].apply(np.floor).astype(int)
print(f'After 0-12h filter: {len(df_chart):,}')
PHYS_RANGES = {'HR': (20, 250), 'RR': (5, 80), 'SBP': (50, 260), 'DBP': (20, 160), 'TEMP': (30, 43)}
before = len(df_chart)
frames = []
for vital, (lo, hi) in PHYS_RANGES.items():
    sub = df_chart[df_chart['vital'] == vital].copy()
    sub = sub[(sub['VALUENUM'] >= lo) & (sub['VALUENUM'] <= hi)]
    frames.append(sub)
df_chart = pd.concat(frames, ignore_index=True)
print(f'After physio filter: {len(df_chart):,} (removed {before - len(df_chart):,})')
df_hourly = df_chart.groupby(['ICUSTAY_ID', 'hour', 'vital'])['VALUENUM'].mean().reset_index()
df_hourly.columns = ['stay_id', 'hour', 'vital', 'value_mean']
df_wide = df_hourly.pivot_table(index=['stay_id', 'hour'], columns='vital', values='value_mean').reset_index()
df_wide.columns.name = None
for col in ['SBP', 'DBP', 'HR', 'RR', 'TEMP']:
    if col not in df_wide.columns:
        df_wide[col] = np.nan
        print(f'  WARNING: {col} missing')
df_wide = df_wide[['stay_id', 'hour', 'SBP', 'DBP', 'HR', 'RR', 'TEMP']]
print(f'\nWide hourly data: {len(df_wide):,} stay-hour rows')
for col in ['SBP', 'DBP', 'HR', 'RR', 'TEMP']:
    pct = df_wide[col].isna().mean() * 100
    print(f'  {col}: {pct:.1f}% missing')
out_all = PROCESSED / 'mimiciii_vital_12h_hourly_all.csv'
df_wide.to_csv(out_all, index=False)
print(f'\nSaved: {out_all}')
df_complete = df_wide.dropna(subset=['SBP', 'DBP', 'HR', 'RR', 'TEMP']).copy()
complete_counts = df_complete.groupby('stay_id').size()
eligible = complete_counts[complete_counts >= 2].index
df_complete = df_complete[df_complete['stay_id'].isin(eligible)].copy()
print(f'\nComplete hours (≥2 per stay): {len(df_complete):,} rows, {len(eligible):,} stays')
out_complete = PROCESSED / 'mimiciii_vital_12h_complete_hour.csv'
df_complete.to_csv(out_complete, index=False)
print(f'Saved: {out_complete}')
with open(LOGS / 'CURRENT_STAGE.txt', 'w') as f:
    f.write(f'COMPLETED: 05_extract_vitals_chunked_mimiciii.py\n')
    f.write(f'NEXT: 06_build_complete_hour_vital_trajectory.py\n')
    f.write(f'MIMIC-III complete-hour stays: {len(eligible):,}\n')
    f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
print('\n=== Script 05 COMPLETE ===')
print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
