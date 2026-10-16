"""Actual pure pulse-oxygen reconstruction and historical-mapping comparison; outputs private only.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'Narrow, read-only oxygen source audit. No K4 or downstream model fitting.\n\nOnly saturation observations for the current assigned patients are scanned.\nPrivate hourly aggregates stay on the external drive. No patient identifiers\nare emitted in console output or aggregate tables.\n'
from pathlib import Path
import argparse
import importlib.util
import subprocess
import time
import numpy as np
import pandas as pd

def sqlstr(value):
    return "'" + str(value).replace("'", "''") + "'"

def main(project, output, databases):
    project, output = (Path(project), Path(output))
    private = output / 'private'
    private.mkdir(exist_ok=True)
    a = pd.read_parquet(project / '14_pneumonia_specificity_revision_20261014/private/修订三库全模块分析输入.parquet', columns=['database', 'stay_id', 'SpO2_min_0_12h', 'phenotype', 'death_24h'])
    a.stay_id = a.stay_id.astype(str)
    root = Path(os.environ['SOURCE_550D92AA9D'])
    locations = {'MIMIC-IV': root / 'MIMIC-IV_3.1/mimic-iv-3.1/icu', 'MIMIC-III': root / 'MIMIC-III_1.4/mimic-iii-clinical-database-1.4', 'eICU': root / 'eICU-CRD_2.0/EICU 2.0数据'}
    origin = project.parent / '重症数据库/STTT-6.3/2026-05-29/03_build_hourly_vital_trajectory_dataset.py'
    spec = importlib.util.spec_from_file_location('historical_source_rules', origin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping_rows = []
    for db in ['MIMIC-IV', 'MIMIC-III']:
        d = pd.read_csv(locations[db] / ('d_items.csv.gz' if db == 'MIMIC-IV' else 'D_ITEMS.csv.gz'))
        d.columns = d.columns.str.lower()
        pure = {220277} if db == 'MIMIC-IV' else {646, 220277}
        for row in d.itertuples(index=False):
            if module.label_to_var(row.label) == 'SpO2' and str(row.linksto).lower() == 'chartevents':
                mapping_rows.append(dict(database=db, itemid=int(row.itemid), label=row.label, historical_included=True, peripheral_pulse_oximetry=int(row.itemid) in pure))
    pd.DataFrame(mapping_rows).to_csv(output / 'tables/10_原小时血氧字段与外周血氧字段对照.tsv', sep='\t', index=False)
    for db in databases:
        start = time.monotonic()
        g = a.loc[a.database.eq(db)].copy()
        keys = private / (db + '_selected_keys.parquet')
        g[['stay_id']].to_parquet(keys, index=False)
        hourly = private / (db + '_raw_oxygen_by_hour_item.parquet')
        if not hourly.exists():
            settings = f"SET threads=4; SET memory_limit='3GB'; SET temp_directory={sqlstr(output / 'tmp')};"
            if db == 'eICU':
                raw = locations[db] / 'vitalPeriodic.csv.gz'
                query = f'\n                SELECT CAST(v.patientunitstayid AS VARCHAR) AS stay_id,\n                       CAST(floor(try_cast(v.observationoffset AS DOUBLE)/60.0) AS INTEGER) AS hour_index,\n                       0 AS itemid, sum(try_cast(v.sao2 AS DOUBLE)) AS value_sum, count(*) AS n\n                FROM read_csv_auto({sqlstr(raw)},all_varchar=true) v\n                JOIN read_parquet({sqlstr(keys)}) k\n                  ON CAST(v.patientunitstayid AS VARCHAR)=k.stay_id\n                WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<720\n                  AND try_cast(v.sao2 AS DOUBLE) BETWEEN 50 AND 100\n                GROUP BY 1,2,3\n                '
            else:
                upper = db == 'MIMIC-III'
                raw = locations[db] / ('CHARTEVENTS.csv.gz' if upper else 'chartevents.csv.gz')
                stays = locations[db] / ('ICUSTAYS.csv.gz' if upper else 'icustays.csv.gz')
                idcol = 'icustay_id' if upper else 'stay_id'
                ids = sorted((r['itemid'] for r in mapping_rows if r['database'] == db))
                query = f"\n                WITH selected AS (\n                  SELECT CAST(s.{idcol} AS BIGINT) AS stay_key, s.intime\n                  FROM read_csv_auto({sqlstr(stays)},sample_size=-1) s\n                  JOIN read_parquet({sqlstr(keys)}) k\n                    ON CAST(s.{idcol} AS VARCHAR)=k.stay_id\n                )\n                SELECT CAST(v.{idcol} AS VARCHAR) AS stay_id,\n                       CAST(floor(epoch(try_cast(v.charttime AS TIMESTAMP)-s.intime)/3600.0) AS INTEGER) AS hour_index,\n                       try_cast(v.itemid AS BIGINT) AS itemid,\n                       sum(try_cast(v.valuenum AS DOUBLE)) AS value_sum, count(*) AS n\n                FROM read_csv_auto({sqlstr(raw)},all_varchar=true) v\n                JOIN selected s ON try_cast(v.{idcol} AS BIGINT)=s.stay_key\n                WHERE try_cast(v.itemid AS BIGINT) IN ({','.join(map(str, ids))})\n                  AND try_cast(v.charttime AS TIMESTAMP)>=s.intime\n                  AND try_cast(v.charttime AS TIMESTAMP)<s.intime+INTERVAL '12 hours'\n                  AND try_cast(v.valuenum AS DOUBLE) BETWEEN 50 AND 100\n                GROUP BY 1,2,3\n                "
            pending = hourly.with_suffix('.inprogress.parquet')
            command = '.bail on\n' + settings + f'COPY ({query}) TO {sqlstr(pending)} (FORMAT PARQUET);'
            print('START_OXYGEN_ONLY', db, 'assigned_N', len(g), flush=True)
            result = subprocess.run([os.environ.get('DUCKDB', 'duckdb'), '-batch'], input=command, text=True, capture_output=True)
            (output / 'qc' / (db + '_oxygen_scan.log')).write_text(result.stdout + '\n' + result.stderr)
            if result.returncode or not pending.exists():
                raise RuntimeError(db + ' raw oxygen scan failed; see qc log')
            pd.read_parquet(pending, columns=['stay_id', 'hour_index']).head(0)
            pending.replace(hourly)
        h = pd.read_parquet(hourly)
        h.stay_id = h.stay_id.astype(str)
        pure_ids = [0] if db == 'eICU' else [220277] if db == 'MIMIC-IV' else [646, 220277]
        p = h.loc[h.itemid.isin(pure_ids)]

        def minima(data, name):
            byhour = data.groupby(['stay_id', 'hour_index']).agg(value_sum=('value_sum', 'sum'), n=('n', 'sum'))
            byhour[name] = byhour.value_sum / byhour.n
            return byhour.groupby('stay_id')[name].min()
        z = g.join(minima(h, 'raw_historical_mapping_min'), on='stay_id').join(minima(p, 'raw_peripheral_min'), on='stay_id')
        z['historical_nonperipheral_items_present'] = z.stay_id.isin(h.loc[~h.itemid.isin(pure_ids), 'stay_id'])
        z['desaturation_limit_present'] = z.stay_id.isin(h.loc[h.itemid.eq(226253), 'stay_id'])
        z['old_missing_raw_peripheral_present'] = z.SpO2_min_0_12h.isna() & z.raw_peripheral_min.notna()
        z['old_observed_peripheral_differs'] = z.SpO2_min_0_12h.notna() & ~np.isclose(z.SpO2_min_0_12h, z.raw_peripheral_min, atol=1e-09, rtol=0, equal_nan=True)
        z.to_parquet(private / (db + '_oxygen_source_comparison.parquet'), index=False)
        out = []
        for phenotype, k in [('ALL', z), *list(z.groupby('phenotype'))]:
            out.append(dict(database=db, phenotype=phenotype, assigned_N=len(k), old_observed_N=int(k.SpO2_min_0_12h.notna().sum()), raw_peripheral_observed_N=int(k.raw_peripheral_min.notna().sum()), recovered_from_old_missing_N=int(k.old_missing_raw_peripheral_present.sum()), recovered_deaths=int(k.loc[k.old_missing_raw_peripheral_present, 'death_24h'].sum()), observed_value_changed_or_absent_N=int(k.old_observed_peripheral_differs.sum()), historical_nonperipheral_items_N=int(k.historical_nonperipheral_items_present.sum()), desaturation_limit_items_N=int(k.desaturation_limit_present.sum()), old_observed_raw_same_mapping_mismatch_N=int((k.SpO2_min_0_12h.notna() & ~np.isclose(k.SpO2_min_0_12h, k.raw_historical_mapping_min, atol=1e-09, rtol=0, equal_nan=True)).sum())))
        pd.DataFrame(out).to_csv(output / 'tables' / ('11_' + db + '_血氧原始来源定向复核.tsv'), sep='\t', index=False)
        print('DONE_OXYGEN_ONLY', out[0], 'seconds', round(time.monotonic() - start, 1), flush=True)
if __name__ == '__source_reference_run__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--databases', nargs='+', default=['MIMIC-IV', 'MIMIC-III', 'eICU'])
    args = parser.parse_args()
    main(args.project, args.output, args.databases)
