# Private input contracts

All real inputs and generated outputs below remain **outside the public repository**. Identifiers in schema descriptions are column names, never shared values. No example real rows are supplied.

## Environment and directories

`DATA_ROOT` is an independently maintained private project/input root. `OUTPUT_ROOT` is a private output directory outside the repository. `ANALYSIS_INPUT` is the current licensed analysis-frame parquet path. `FEATURE_MATRIX` is the licensed 48-feature candidate-matrix path. Raw database directories are provided explicitly to SQL rendering or the gate's `--raw` option. There are no embedded machine-specific roots.

The retained drivers expect the original relative input names in several places. These are input contracts only, not downloadable files:

- `OUTPUT_ROOT/private/修订三库全模块分析输入.parquet`: current complete compiled application frame after the source-specific oxygen replacement.
- `OUTPUT_ROOT/private/24维以上严格肺炎候选_原特征结局不变.parquet`: strict eligible candidate frame with ≥24 observed dimensions and all four vitals represented.
- `OUTPUT_ROOT/private/纠正外周血氧汇总_覆盖所有所需阈值.parquet`: corrected pulse-oxygen summaries for every required coverage-threshold participant.
- `DATA_ROOT/current_inputs/private/修订eICU逐人原始结局证据.parquet` and `修订eICU竞争风险逐人事件.parquet`: event evidence and original classified competing-risk records, not included.
- `DATA_ROOT/revision_R1_R2_data/locked_K4_model/`: original restricted numerical CSV assets, not included.
- Historical gate/domain/selection/time-order drivers also read their named historical candidate, flow, assignment and reference files. See `workflow_inputs.tsv` for the exact relative paths extracted from the actual code. Their content must not be copied into the public release.

## Main analysis frame

Required fields include database, stay_id, patient_id, hospital_id (eICU clustering), phenotype (original P1–P4), age, male, observed_dimensions_48d, death_24h, hospital_mortality, fixed_HRRR_rule, corrected SpO2 summaries, three existing organ-support proxies, acutephysiologyscore, original saved fold, HR/RR means and all 16 conventional summaries.

MIMIC sensitivity fields include the eight recovered `raw_*` comorbidity indicators and `raw_valid_diagnosis_rows`; absent disease codes among valid diagnosis rows are distinct from absent diagnosis information. Additional flags include early_specific, aspiration_only and support_free_at_12h. Secondary-event analyses require audited support/death/discharge timestamps, event units, status, time_from_landmark and scenario. These fields are not to be reconstructed from outcome-driven guesses.

Original saved folds are mandatory for exact prediction comparison. Grouping is patient-based in MIMIC and hospital-based in eICU. Creating new folds may produce a different valid exercise but is not exact reproduction of the reported estimates.

## SQL contract

Each SQL template requires `RAW_DATABASE_DIR` and `SELECTED_KEYS`. The latter is a private parquet with a `stay_id` string column, restricted to independently prepared authorised candidates. Schema field names are intentionally retained for correct joins. The query output is patient-linked hourly/support data and must remain private.

For oxygen, pool `value_sum` and `n` over item IDs within each stay/hour before computing SpO2. Then call the actual `aggregate_hourly_spo2` function. Joining onto the full selected cohort preserves participants without oxygen as missing, rather than dropping them. Never combine the legacy oxygen-derived columns with the replacement as a missing-only fill.

## Runtime safety

Use fresh private output directories; do not mix cached estimates across source versions. Original cache/restart behaviours remain in some retained drivers. No generated `private/`, QC logs, predictor outputs, source text evidence, numerical assets or archives are eligible for automatic publication. `.gitignore` is an additional guard, not a privacy audit.
