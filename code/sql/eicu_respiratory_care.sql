-- Actual eICU record-proxy SQL; not exact clinical treatment intervals.
SELECT k.stay_id,true AS vent_resp
             FROM read_csv_auto('{RAW_DATABASE_DIR}/respiratoryCare.csv.gz',all_varchar=true) v
             JOIN read_parquet('{SELECTED_KEYS}') k ON v.patientunitstayid=k.stay_id
             WHERE try_cast(v.ventstartoffset AS DOUBLE)>=0 AND try_cast(v.ventstartoffset AS DOUBLE)<=720
              AND (try_cast(v.ventendoffset AS DOUBLE)=0 OR try_cast(v.ventendoffset AS DOUBLE)>720
                   OR try_cast(v.ventendoffset AS DOUBLE) IS NULL) GROUP BY 1;
