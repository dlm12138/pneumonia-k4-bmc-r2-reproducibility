-- Actual eICU record-proxy SQL; not exact clinical treatment intervals.
SELECT k.stay_id,
             bool_or(regexp_matches(v.treatmentstring,'(?i)mechanical ventilation|invasive ventilation|endotracheal tube\|insertion|reintubation') AND NOT regexp_matches(v.treatmentstring,'(?i)non[- ]?invasive|bipap|cpap|face mask|nasal mask')) AS vent_treatment,
             bool_or(regexp_matches(v.treatmentstring,'(?i)(^|\|)dialysis(\||$)|hemodialysis|haemodialysis|c v v h|c v v h d|c v v h d f|cvvh|cvvhd|cvvhdf|crrt|renal replacement|ultrafiltration|(^|\|)sled(\||$)') AND NOT regexp_matches(v.treatmentstring,'(?i)insertion of|catheter|access surgery|procedures/radiology|arteriovenous shunt|electrolyte correction')) AS rrt
             FROM read_csv_auto('{RAW_DATABASE_DIR}/treatment.csv.gz',all_varchar=true) v
             JOIN read_parquet('{SELECTED_KEYS}') k ON v.patientunitstayid=k.stay_id
             WHERE try_cast(v.treatmentoffset AS DOUBLE)<=720 GROUP BY 1;
