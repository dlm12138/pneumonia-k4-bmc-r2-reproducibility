-- Extracted from the actual corrected oxygen extraction query.
-- Render paths with code/render_sql.py; output contains restricted person-level data.
WITH selected AS (SELECT cast(s.icustay_id AS BIGINT) stay_key,s.intime FROM read_csv_auto('{RAW_DATABASE_DIR}/ICUSTAYS.csv.gz',sample_size=-1) s JOIN read_parquet('{SELECTED_KEYS}') k ON cast(s.icustay_id AS VARCHAR)=k.stay_id)
            SELECT cast(v.icustay_id AS VARCHAR) stay_id,cast(floor(epoch(try_cast(v.charttime AS TIMESTAMP)-s.intime)/3600.0) AS INTEGER) hour_index,try_cast(v.itemid AS BIGINT) itemid,sum(try_cast(v.valuenum AS DOUBLE)) value_sum,count(*) n
            FROM read_csv_auto('{RAW_DATABASE_DIR}/CHARTEVENTS.csv.gz',all_varchar=true) v JOIN selected s ON try_cast(v.icustay_id AS BIGINT)=s.stay_key
            WHERE try_cast(v.itemid AS BIGINT) IN (646,220277) AND try_cast(v.charttime AS TIMESTAMP)>=s.intime AND try_cast(v.charttime AS TIMESTAMP)<s.intime+INTERVAL '12 hours' AND try_cast(v.valuenum AS DOUBLE) BETWEEN 50 AND 100 GROUP BY 1,2,3;
