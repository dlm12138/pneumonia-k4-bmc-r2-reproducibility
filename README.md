# Pneumonia K4: second-round revision reproducibility materials

This repository contains analysis implementations, aggregate results and documentation for **Early physiological phenotypes and mortality within 24 hours after a 12-hour landmark in pneumonia-associated critical illness: a three-database cohort study**.

The results correspond to the strict pneumonia-evidence application cohorts and corrected peripheral pulse-oxygen source analysed on 2026-10-15. This is research software, not a validated clinical decision tool. K4 did not demonstrate a stable incremental predictive advantage over simpler assessments.

## Contents

- `code/sql/`: ten SQL queries extracted from the pulse-oxygen and organ-support reconstruction programs for MIMIC-IV, MIMIC-III and eICU. Historical cohort extraction used the Python file-reading programs included under `code/source_history/raw/`, rather than a standalone SQL pipeline.
- `code/core/`: original scientific functions for vital cleaning, hourly oxygen aggregation, locked K4 assignment, association, prediction and sensitivity calculations.
- `code/workflows/`: study workflows with configurable input/output locations; the execution dispatcher calls the original regression functions and does not implement a separate estimator.
- `results/`: aggregate tables only; no patient rows, individual predictions, site identifiers, raw diagnostic text or numerical model assets.
- `docs/`: definitions, input contracts, execution order, provenance and reproducibility limitations.

## Current primary analysis

The endpoint is **recorded in-hospital death during (12,36] hours after ICU admission**. The eICU clock is a recorded proxy, not an independently adjudicated clinical death time. P2 remains the reference group.

| Database | Primary model N | Recorded deaths | P3 vs P2 primary OR (95% CI) |
| --- | ---: | ---: | --- |
| MIMIC-IV 3.1 | 14,636 | 281 | 1.7435 (1.2424–2.4468) |
| MIMIC-III 1.4 | 3,690 | 56 | 4.4050 (1.9225–10.0931) |
| eICU-CRD 2.0 | 14,779 | 341 | 3.5975 (2.5418–5.0915) |

eICU has 14,781 assigned participants; two lack recorded sex and are excluded from the complete-case primary model. The historical MIMIC-IV derivation population is distinct from the current strict application cohort.

## Reproduction is conditional, not end-to-end

Anyone can inspect aggregate tables and run the synthetic checks. Researchers must independently obtain authorised database access to run patient-level computations. With correctly prepared, legally held study inputs and the original saved labels/folds, the public code supports downstream statistical calculations. **Exact frozen K4 assignment, some coverage/domain/early/time-order analyses, and reconstruction of the complete original patient pipeline require restricted assets or inputs not distributed here.** Historical model-search, stability and finalisation source is included for provenance under `code/source_history/historical_model/`; direct execution is guarded. Current application analyses use the original frozen model. A new K=4 fit is not a reproduction of that model.

Start with [the reproduction guide](docs/reproduction_guide.md), [input contracts](docs/input_contracts.md) and [the scope matrix](docs/reproduction_scope.tsv). Do not run all workflow files blindly or treat archived intermediate files as available public inputs.

## Data access and model access

MIMIC-IV 3.1, MIMIC-III 1.4 and eICU-CRD 2.0 are credentialed-access databases. This package does not redistribute patient-level data. See [official access routes](docs/data_access.md). The locked standardisation values, component weights, means, covariance matrices and original numerical component mapping are excluded. **No operational parameter-distribution channel has been established; obtaining database credentials alone does not grant these assets.** See [model access status](docs/model_access.md).

## Synthetic verification

Install the core Python dependencies from `requirements.txt`. Install DuckDB CLI separately for SQL checks. Set `OUTPUT_ROOT` to a private directory **outside this repository**, then run:

```sh
python code/tests_synthetic.py --output-root "$OUTPUT_ROOT" --duckdb duckdb
```

The test generates explicitly synthetic inputs outside the repository; it never reads real patient records or loads a real K4 model. Passing synthetic checks does not establish clinical validity or full end-to-end reproducibility.

## Complete research process

Start with [the complete research process](docs/full_research_process.md) and [the 27-step source/input/output/check map](docs/full_process_steps.tsv). The included programs cover historical three-database cohort identification, both hourly-source tiers, landmark/source-tier assembly, support/outcome extraction, raw comorbidity recovery, strict frame assembly, eICU event evidence, the entire corrected-oxygen replacement and historical model derivation/freezing. See [the source inventory](docs/source_inventory.tsv), [source input/output contracts](docs/source_io_contracts.tsv) and [required resource settings](docs/source_resource_requirements.tsv). These source references have explicit private prerequisites and do not establish public-only end-to-end numerical reproduction.

See [the per-program execution manual](docs/program_execution_manual.md), [actual function catalogue](docs/function_catalog.tsv), [source argument index](docs/program_arguments.tsv) and [module prerequisites](docs/program_dependencies.tsv) for the detailed computational steps.

Variable names, units and time windows are listed in [the variable dictionary](docs/variable_dictionary.tsv). [Result-table notes](docs/results_notes.md) and [the result catalogue](docs/results_catalog.tsv) explain the aggregate outputs and their interpretation.
