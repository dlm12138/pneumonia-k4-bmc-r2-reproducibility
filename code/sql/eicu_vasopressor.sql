-- Actual eICU record-proxy SQL; not exact clinical treatment intervals.
SELECT k.stay_id,true AS vaso
             FROM read_csv_auto('{RAW_DATABASE_DIR}/infusionDrug.csv.gz',all_varchar=true) v
             JOIN read_parquet('{SELECTED_KEYS}') k ON v.patientunitstayid=k.stay_id
             WHERE try_cast(v.infusionoffset AS DOUBLE)>360 AND try_cast(v.infusionoffset AS DOUBLE)<=720
              AND (try_cast(v.drugrate AS DOUBLE)>0 OR try_cast(v.infusionrate AS DOUBLE)>0)
              AND regexp_matches(v.drugname,'(?i)norepinephrine|noradrenaline|levophed|epinephrine|adrenaline|phenylephrine|neo[- ]?synephrine|neosynephrine|vasopressin|dopamine|angiotensin') AND NOT regexp_matches(v.drugname,'(?i)racemic|inhal|nebul|nasal') GROUP BY 1;
