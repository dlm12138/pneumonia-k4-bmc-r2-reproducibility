# Complete research process and source chain

This document describes the research process and its source programs, including steps that require inputs unavailable from public files alone. Process documentation does not imply that prerequisite patient inputs or numerical model assets are publicly distributed.

## 1. Access and source versions

Use MIMIC-IV 3.1, MIMIC-III 1.4 and eICU-CRD 2.0 under independently authorised access. See data_access.md. Keep raw sources read-only; all private intermediate/output files must remain outside this repository. No MySQL server is required. Historical source programs used Python file reads, ZIP/GZIP streams and later DuckDB SQL, rather than a standalone raw-to-paper SQL pipeline.

## 2. Historical candidate identification

The original three-database program is source_history/raw/01_pneumonia_identification.py. It reads MIMIC patients, admissions, ICU stays, diagnoses and code dictionaries, and eICU patient, diagnosis and admissionDx. It performs the historical exact/prefix ICD and substring text matching, links admissions to ICU stays and produces three pneumonia_stays_raw CSV files privately. The eICU historical program first reduces by patienthealthsystemstayid; the later landmark function also sorts/reduces by uniquepid. Reproduction must preserve this implemented sequence rather than substitute a different 'first admission' rule.

These historical candidate rules include problematic U07.1 and short substring matches. They are shown to explain the derivation and candidate provenance, NOT endorsed as the revised pneumonia definition. The strict gate in step 7 is mandatory for current application analyses. Historical 30-day mortality fields are not the current primary endpoint.

## 3. Two hourly vital-source tiers

The three legacy hourly programs raw/02–04 and corrected-source programs raw/05–06 explain where the original hourly records came from. The corrected-source subset was produced under an earlier feasibility filter; patients not in that subset can use the legacy hourly tier. Generic historical oxygen matching is NOT allowed for current pulse oxygen. The exact cleaning/aggregation functions used for the four phenotype vitals also appear in core/raw_vitals.py. Retain source-tier history and item mappings; do not silently replace all tiers with a newly chosen extraction.

## 4. Landmark candidates and feature matrix

assembly/07_landmark_source_functions.py retains normalize_id, load_hourly_sources, build_landmark_candidates and build_landmark_matrix. It sorts first records as actually implemented, applies adult age >=18, alive at 12 h, recorded ICU stay >=12 h and classifiable hospital outcome. For eICU, unitdischargeoffset—not the historical extracted hospital LOS—determines ICU landmark eligibility. It prefers corrected hourly sources at the patient level and otherwise uses the legacy tier, selects hours 0–11, applies recorded valid bounds, pivots hourly means and reports observed dimensions. This historical matrix included temperature; the actual common-four fit in step 5 excludes temperature and uses SBP/DBP/HR/RR only.

## 5. Original outcome-free model derivation and freezing

historical_model/19_outcome_free_grid.py contains the original model grid, observed-dimension likelihood and numerical/structural diagnostics. historical_model/20_original_stability_and_mapping.py contains original stability/resampling, finalist comparison and label mapping; 21_original_k4_finalization.py contains the finalisation logic. These programs document historical derivation; current application analyses retain the original frozen K4 without retraining. Model-selection criteria documented retrospectively are not a prospectively registered hierarchy.

Fit standardisation uses observed derivation means/SDs; missing standardised values are set to zero only for original fitting. External assignment uses the original standardisation and observed-dimension Gaussian marginal likelihood, without external re-standardisation or refitting. Full weights, means, covariances and original numeric mapping are withheld. Historical model code alone cannot reproduce the exact frozen numerical object. A new fit is not an acceptable substitute.

## 6. Outcome, support, candidate attributes and original folds

assembly/09_organ_support_endpoints.py retains actual support interval extraction and endpoint assembly for the three databases. assembly/08_candidate_attribute_source.py documents candidate attributes and source/system joins. assembly/11_eicu_evidence_collection.py and 12_eicu_event_classification.py contain the actual eICU evidence, status conflict and alternative event rules; these outputs are strictly private. The primary outcome remains recorded in-hospital death in (12,36] h after ICU admission. eICU uses recorded proxy timing, not adjudicated exact clinical timing. Original saved prediction folds must be retained; patient-level fold arrays are not released. Inspect the retained five-fold implementation; never regenerate folds just to improve a metric.

## 7. Revision of pneumonia eligibility, not K4

workflows/01_pneumonia_specificity.py applies the actual stricter gate to the historical candidate pool and links its original flow and frozen labels. MIMIC specific code evidence excludes U07.1-only inclusion; aspiration evidence is separately flagged. eICU uses explicit pneumonia/bronchopneumonia word boundaries rather than unrestricted CAP/HAP/VAP substring matching. The current strict boundary requires ICU exit later than 12 h. This revision occurred during review and is not presented as prespecified before outcomes were known. Pneumonia onset, CAP/HAP/VAP clinical adjudication and two-MIMIC patient overlap remain unresolved.

## 8. Recovery of comorbidity and strict frame assembly

assembly/10_diagnosis_recovery_functions.py contains the real flow and raw-diagnosis recovery functions; core/comorbidity_definitions.py and docs/comorbidity_code_dictionary.tsv retain the actual ICD prefixes. A valid diagnosis source with no matching condition code is distinct from an unavailable diagnosis source; do not fill every absent historical feature row with 'no comorbidity'. assembly/13_strict_frame_assembly_source.py shows the actual membership, posterior, trajectory summaries and original saved-fold joins; assembly/14_clinical_and_event_assembly.py shows recovered descriptors and strict event subsets. The original assembly can still contain pre-correction oxygen: it is an intermediate, not final analysis input. Step 9 must replace it.

## 9. Entire pulse-oxygen source replacement

assembly/17_original_pulse_oxygen_reconstruction.py documents the actual source reconstruction and comparison. Current sources are MIMIC-IV item 220277, MIMIC-III 646 and 220277, and eICU vitalPeriodic.sao2. SQL under code/sql is extracted from actual checked programs. Restrict measurement time to 0<=t<12 h and values to 50–100 inclusive. Within each hour, sum valid values and divide by record count across eligible items; then take the lowest observed hourly mean. This is not the minimum raw measurement.

assembly/18_current_oxygen_frame_replacement.py is the actual complete replacement program. It replaces all old oxygen summary columns, including old nonmissing values, and reads the same variable for newly included >=24/28-dimensional patients. Retain candidate labels, mortality, age, sex, coverage and saved folds. Current primary-model oxygen missing counts are 8, 0 and 114. MIMIC-III's all-zero oxygen missing indicator is redundant; removing it must not remove RRT. Current result files, not intermediate old oxygen outputs, are authoritative.

## 10. Association versus prediction

workflows/03_primary_and_fixed_sensitivities.py calls actual audited regression functions; workflow04 computes oxygen/full and related corrected models. MIMIC variance is HC0 sandwich; eICU is hospital-cluster sandwich with the implemented correction. Separation diagnostics prevent reporting a finite ordinary MLE when none exists. MIMIC-III full K4 and full HR/RR associations remain non-estimable; the additional separated ordinary composite is likewise not filled from an old result.

workflow05 uses the original fixed five folds and five prediction strategies, L2/C=1.0/lbfgs/tol=1e-4/max_iter=2000; it does not tune C. A penalised prediction fit is a different calculation from an unpenalised association MLE. Its existence does not resolve non-estimable association coefficients. Paired bootstrap uses existing out-of-fold predictions and original group resampling/settings.

## 11. Remaining sensitivities and secondary endpoints

workflow06 implements corrected-oxygen coverage thresholds 24/28/32/36/40. Workflows07–10 preserve domain/early/posterior, time-order, hospital and selection calculations. Workflows11–13 retain actual R Fine–Gray and hospital hierarchical code. assembly/15 and16 document eICU endpoint scenarios and MIMIC event assembly. Ordinary composite events include the specified recorded hospital death window, whereas competing-risk target events occur before/on ICU discharge, with discharge competing; therefore event counts can differ. Never label OR as SHR or permutation percentiles as CIs.

## 12. Results, tables and scientific interpretation

The 42 aggregate tables describe the current analysis; results_provenance.tsv identifies each numerical source. Clinical oxygen descriptions and adjusted effects use corrected step9 inputs. Primary model N/events are MIMIC-IV 14636/281, MIMIC-III 3690/56, eICU 14779/341; eICU assigned N=14781, with two missing-sex exclusions. Four-group actual deaths, 229 core conflict distribution and rules are aggregates only. No patient or site identifiers are published.

The findings include no stable incremental predictive benefit and no established clinical utility. Non-estimable models, treatment-modified physiology, CareVue 0/5173 and unresolved event/diagnosis validity remain limitations. Figure production is a separate rendering step using the documented sources; the historical layout functions do not by themselves establish identity with the submitted figure layout.

## 13. How to inspect or execute

Use full_process_steps.tsv for sequence, source entry points, input/output and stopping checks; source_io_contracts.tsv for exact source read/write expressions. source_inventory.tsv distinguishes whole-program sources from selected-function sources. Required environment resource keys are listed in source_resource_requirements.tsv. These keys point to privately prepared files and folders, not distributed patient data. Set DATA_ROOT and OUTPUT_ROOT outside the repository. Direct execution of source-history programs is guarded to prevent accidental legacy analysis/model fitting; inspect and configure prerequisites first. The guarded source runner configures execution rather than statistical estimation; it requires explicit arguments and legally held inputs.

The current workflows are the downstream entry points. Historical source programs document the preceding steps, but some rely on private manifests, source-tier inputs, original folds and frozen numerical assets. The documentation covers these stages; public-only end-to-end numerical replication is NOT established.
