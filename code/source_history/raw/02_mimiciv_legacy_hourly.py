"""Historical legacy MIMIC-IV hourly source, not corrected oxygen.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'\nScript 03: Extract vital signs from MIMIC-IV chartevents (chunked streaming from ZIP)\nExtracts SBP, DBP, HR, RR, TEMP for pneumonia ICU stays, 0-12h after admission\n'
import sys, os, zipfile, io, gzip, time
from pathlib import Path
import pandas as pd
import numpy as np
RUN_DIR = Path(os.environ['SOURCE_1B251D2246'])
MIMICIV_ZIP = os.environ['SOURCE_E15CB3D6F9']
MIMICIV_RAW = RUN_DIR / '10_temp/mimiciv_raw'
PROCESSED = RUN_DIR / '03_processed'
PROCESSED.mkdir(exist_ok=True)
LOGS = RUN_DIR / '00_logs'
print('=' * 60)
print('MIMIC-IV VITAL SIGN EXTRACTION (CHUNKED)')
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 60)
HR_ITEMS = {220045}
RR_ITEMS = {220210, 224690}
SBP_ITEMS = {220050, 220179, 224167, 225309, 227243}
DBP_ITEMS = {220051, 220180, 224643, 225310, 227242}
TEMP_C_ITEMS = {223762}
TEMP_F_ITEMS = {223761}
ALL_ITEMS = HR_ITEMS | RR_ITEMS | SBP_ITEMS | DBP_ITEMS | TEMP_C_ITEMS | TEMP_F_ITEMS
print(f'\nTargeted itemids: {sorted(ALL_ITEMS)}')
print(f'  HR: {HR_ITEMS}')
print(f'  RR: {RR_ITEMS}')
print(f'  SBP: {SBP_ITEMS}')
print(f'  DBP: {DBP_ITEMS}')
print(f'  TEMP_C: {TEMP_C_ITEMS}')
print(f'  TEMP_F: {TEMP_F_ITEMS}')
print('\nLoading pneumonia ICU stays and intime...')
stays_raw = pd.read_csv(RUN_DIR / '02_raw_extract/mimiciv_pneumonia_stays_raw.csv', usecols=['stay_id', 'intime', 'outtime', 'los_hours'])
stays_raw['intime'] = pd.to_datetime(stays_raw['intime'])
stays_raw['outtime'] = pd.to_datetime(stays_raw['outtime'])
pneumonia_stay_ids = set(stays_raw['stay_id'].tolist())
intime_map = dict(zip(stays_raw['stay_id'], stays_raw['intime']))
print(f'  Pneumonia stay_ids: {len(pneumonia_stay_ids):,}')
print(f'\nOpening MIMIC-IV zip: {MIMICIV_ZIP}')
CHARTEVENTS_PATH = 'mimic-iv-3.1/icu/chartevents.csv.gz'
CHUNK_SIZE = 500000
collected_rows = []
total_read = 0
matched_rows = 0
start_time = time.time()
with zipfile.ZipFile(MIMICIV_ZIP) as zf:
    with zf.open(CHARTEVENTS_PATH) as raw_gz:
        with gzip.open(raw_gz, 'rt', encoding='utf-8') as gz_file:
            reader = pd.read_csv(gz_file, chunksize=CHUNK_SIZE, usecols=['stay_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'], dtype={'stay_id': 'Int64', 'itemid': 'Int64', 'valuenum': 'float64'}, low_memory=False)
            for chunk_num, chunk in enumerate(reader):
                total_read += len(chunk)
                mask = chunk['stay_id'].isin(pneumonia_stay_ids) & chunk['itemid'].isin(ALL_ITEMS)
                matched = chunk[mask].copy()
                if len(matched) > 0:
                    collected_rows.append(matched)
                    matched_rows += len(matched)
                if chunk_num % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = total_read / elapsed if elapsed > 0 else 0
                    print(f'  Chunk {chunk_num + 1}: {total_read / 1000000.0:.1f}M rows read, {matched_rows:,} matched, {rate / 1000000.0:.2f}M rows/sec, {elapsed:.0f}s elapsed')
print(f'\nTotal rows read: {total_read:,}')
print(f'Total matched rows: {matched_rows:,}')
if not collected_rows:
    print('ERROR: No rows matched. Cannot continue.')
    sys.exit(1)
df_chart = pd.concat(collected_rows, ignore_index=True)
print(f'Combined matched DataFrame: {len(df_chart):,} rows')

def assign_vital(itemid):
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
df_chart['vital'] = df_chart['itemid'].apply(assign_vital)
mask_f = df_chart['vital'] == 'TEMP_F'
df_chart.loc[mask_f, 'valuenum'] = (df_chart.loc[mask_f, 'valuenum'] - 32) / 1.8
df_chart.loc[mask_f, 'vital'] = 'TEMP'
df_chart.loc[df_chart['vital'] == 'TEMP_C', 'vital'] = 'TEMP'
df_chart['charttime'] = pd.to_datetime(df_chart['charttime'])
df_chart['intime'] = df_chart['stay_id'].map(intime_map)
df_chart['hours_after_icu'] = (df_chart['charttime'] - df_chart['intime']).dt.total_seconds() / 3600
df_chart = df_chart[(df_chart['hours_after_icu'] >= 0) & (df_chart['hours_after_icu'] < 12)].copy()
df_chart['hour'] = df_chart['hours_after_icu'].apply(np.floor).astype(int)
print(f'\nAfter 0-12h filter: {len(df_chart):,} rows')
PHYS_RANGES = {'HR': (20, 250), 'RR': (5, 80), 'SBP': (50, 260), 'DBP': (20, 160), 'TEMP': (30, 43)}
before = len(df_chart)
masks = []
for vital, (lo, hi) in PHYS_RANGES.items():
    vm = df_chart['vital'] == vital
    range_mask = (df_chart['valuenum'] >= lo) & (df_chart['valuenum'] <= hi)
    masks.append(~vm | range_mask)
combined_mask = masks[0]
for m in masks[1:]:
    combined_mask = combined_mask & m
df_chart = df_chart[combined_mask].copy()
print(f'After physiological range filter: {len(df_chart):,} rows (removed {before - len(df_chart):,})')
df_hourly = df_chart.groupby(['stay_id', 'hour', 'vital'])['valuenum'].mean().reset_index()
df_hourly.columns = ['stay_id', 'hour', 'vital', 'value_mean']
df_wide = df_hourly.pivot_table(index=['stay_id', 'hour'], columns='vital', values='value_mean').reset_index()
df_wide.columns.name = None
for col in ['SBP', 'DBP', 'HR', 'RR', 'TEMP']:
    if col not in df_wide.columns:
        df_wide[col] = np.nan
        print(f'  WARNING: {col} column missing from wide format (no data)')
df_wide = df_wide[['stay_id', 'hour', 'SBP', 'DBP', 'HR', 'RR', 'TEMP']]
print(f'\nWide-format hourly data: {len(df_wide):,} stay-hour rows')
print(f"Unique stays with any vital: {df_wide['stay_id'].nunique():,}")
print(f'\nMissingness per vital:')
for col in ['SBP', 'DBP', 'HR', 'RR', 'TEMP']:
    missing_pct = df_wide[col].isna().mean() * 100
    print(f'  {col}: {missing_pct:.1f}% missing')
out_all = PROCESSED / 'mimiciv_vital_12h_hourly_all.csv'
df_wide.to_csv(out_all, index=False)
print(f'\nSaved: {out_all} ({len(df_wide):,} rows)')
df_complete = df_wide.dropna(subset=['SBP', 'DBP', 'HR', 'RR', 'TEMP']).copy()
print(f'\nComplete hours (all 5 vitals): {len(df_complete):,} stay-hours')
print(f"Unique stays with ≥1 complete hour: {df_complete['stay_id'].nunique():,}")
complete_hour_counts = df_complete.groupby('stay_id').size()
eligible_stays = complete_hour_counts[complete_hour_counts >= 2].index
df_complete = df_complete[df_complete['stay_id'].isin(eligible_stays)].copy()
print(f'After ≥2 complete hour filter: {len(df_complete):,} rows, {len(eligible_stays):,} stays')
out_complete = PROCESSED / 'mimiciv_vital_12h_complete_hour.csv'
df_complete.to_csv(out_complete, index=False)
print(f'Saved: {out_complete} ({len(df_complete):,} rows)')
with open(LOGS / 'CURRENT_STAGE.txt', 'w') as f:
    f.write(f'COMPLETED: 03_extract_vitals_chunked_mimiciv.py\n')
    f.write(f'NEXT: 04_extract_vitals_eicu.py\n')
    f.write(f'MIMIC-IV complete-hour stays: {len(eligible_stays):,}\n')
    f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
print('\n=== Script 03 COMPLETE ===')
print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
