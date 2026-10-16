-- Actual support reconstruction SQL; path placeholders only.
WITH selected AS (
              SELECT cast(u.icustay_id AS BIGINT) AS stay_key,u.intime
              FROM read_csv_auto('{RAW_DATABASE_DIR}/ICUSTAYS.csv.gz',sample_size=-1) u
              JOIN read_parquet('{SELECTED_KEYS}') k ON cast(u.icustay_id AS VARCHAR)=k.stay_id
            ) SELECT cast(u.stay_key AS VARCHAR) AS stay_id,
             bool_or(try_cast(v.itemid AS BIGINT)=225792) AS vent,
             bool_or(try_cast(v.itemid AS BIGINT) IN (225441,225802,225803,225805,225809,225955)) AS rrt
             FROM read_csv_auto('{RAW_DATABASE_DIR}/PROCEDUREEVENTS_MV.csv.gz',all_varchar=true) v
             JOIN selected u ON try_cast(v.icustay_id AS BIGINT)=u.stay_key
             WHERE try_cast(v.starttime AS TIMESTAMP)<=u.intime+INTERVAL '12 hours'
             AND (try_cast(v.endtime AS TIMESTAMP)>u.intime+INTERVAL '12 hours'
                  OR try_cast(v.endtime AS TIMESTAMP) IS NULL) AND coalesce(try_cast(v.cancelreason AS DOUBLE),0)=0
              AND try_cast(v.itemid AS BIGINT) IN (225792,225441,225802,225803,225805,225809,225955) GROUP BY 1;
