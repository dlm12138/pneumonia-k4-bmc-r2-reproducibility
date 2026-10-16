-- Actual support reconstruction SQL; path placeholders only.
WITH selected AS (
              SELECT cast(u.stay_id AS BIGINT) AS stay_key,u.intime
              FROM read_csv_auto('{RAW_DATABASE_DIR}/icustays.csv.gz',sample_size=-1) u
              JOIN read_parquet('{SELECTED_KEYS}') k ON cast(u.stay_id AS VARCHAR)=k.stay_id
            ) SELECT cast(u.stay_key AS VARCHAR) AS stay_id, true AS vaso FROM read_csv_auto('{RAW_DATABASE_DIR}/inputevents.csv.gz',all_varchar=true) v
             JOIN selected u ON try_cast(v.stay_id AS BIGINT)=u.stay_key
             WHERE try_cast(v.starttime AS TIMESTAMP)<=u.intime+INTERVAL '12 hours'
             AND (try_cast(v.endtime AS TIMESTAMP)>u.intime+INTERVAL '12 hours'
                  OR try_cast(v.endtime AS TIMESTAMP) IS NULL) AND try_cast(v.itemid AS BIGINT) IN (221289,221662,221749,221906,222315,229617,229630,229631,229632,229709,229764) GROUP BY 1;
