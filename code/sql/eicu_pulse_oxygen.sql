-- Extracted from the actual corrected oxygen extraction query.
-- Render paths with code/render_sql.py; output contains restricted person-level data.
SELECT CAST(v.patientunitstayid AS VARCHAR) stay_id, CAST(floor(try_cast(v.observationoffset AS DOUBLE)/60.0) AS INTEGER) hour_index,0 itemid,sum(try_cast(v.sao2 AS DOUBLE)) value_sum,count(*) n
            FROM read_csv_auto('{RAW_DATABASE_DIR}/vitalPeriodic.csv.gz',all_varchar=true) v JOIN read_parquet('{SELECTED_KEYS}') k ON v.patientunitstayid=k.stay_id
            WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<720 AND try_cast(v.sao2 AS DOUBLE) BETWEEN 50 AND 100 GROUP BY 1,2,3;
