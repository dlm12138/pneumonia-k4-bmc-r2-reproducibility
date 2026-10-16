# Analysis definitions and interpretation

The historical derivation population and the revised strict pneumonia application population are different. This revision changed eligibility after reviewer criticism; it was not prospectively registered or specified before all outcome inspection. The original K4, component labels and reference group remain frozen.

## Pneumonia evidence

The actual gate is in `code/workflows/01_pneumonia_specificity.py`. ICD code normalisation removes periods and uses uppercase. The stricter ICD-9 rule uses prefixes 480–486 and codes 4870, 48801, 48811, 99731; aspiration-related 5070 is tracked separately. ICD-10 pneumonia prefixes are J12–J18, J100, J110 and J95851, with J690 tracked as aspiration-related. Eligibility uses qualifying pneumonia or aspiration evidence; aspiration-only exclusion is a reported sensitivity. U07.1 alone is not qualifying evidence.

eICU uses a word-boundary expression for pneumonia/bronchopneumonia, not unrestricted cap/hap/vap substrings. Exact source fields and context checks are retained in the actual gate. Database documentation is not prospective clinical adjudication; onset and comprehensive CAP/HAP/VAP classification remain unconfirmed. The code refines the historical candidate pool; it does not independently regenerate every original candidate from raw tables.

## Vital signs and oxygen

Four phenotype vitals use the actual retained cleaning and hourly aggregation functions. Valid ranges are SBP 40–300 mmHg, DBP 20–200 mmHg, HR 20–250 beats/min and RR 3–80 breaths/min. Use only those four variables in the historical generic vital extractor; do not use its broad historical oxygen-label mapping. Pulse oxygen instead uses the source-specific SQL in `code/sql/`.

Pulse oxygen sources: MIMIC-IV item 220277; MIMIC-III items 646 and 220277; eICU `vitalPeriodic.sao2`. Keep values 50–100% during [0,12) hours. Combine sum/count within each hour across qualifying items, then calculate the minimum of the valid hourly means. It is not the minimum raw measurement. Mean, SD and low-oxygen burdens are summaries of hourly means, not raw-record weighted summaries. `SpO2_n_records_0_12h` is the historical column name but counts observed hourly means.

MIMIC organ support uses specified input/procedure intervals spanning 12 hours. eICU support variables retain legacy column names ending `at_12h` but represent recorded proxies: positive-rate vasopressor infusion during (6,12] h; a ventilation interval or qualifying treatment record by 12 h; RRT treatment records by 12 h. They do not establish exact clinical activity at the landmark. Treatment can itself affect measured vital signs.

## Endpoints

Primary `death_24h`: recorded **in-hospital** death during (12,36] hours after ICU admission. It is not restricted to ICU death, is not whole-stay hospital mortality, and does not claim complete post-hospital all-cause follow-up. eICU uses recorded time proxies; 229 core-conflict participants remain unresolved. Reliability B is not clinical truth certification.

`hospital_mortality` is a separate entire-hospital-stay sensitivity. Ordinary organ-support/death composite and ICU-scoped competing-risk outcomes have distinct observation scopes, event priorities and risk sets; consult `secondary_events.py`, `eicu_event_reconstruction.R`, the R workflows and aggregate event-count tables. Recorded hospital deaths after ICU discharge may count in an ordinary composite but not as target ICU-scoped events after a competing discharge. Do not equate OR, SHR and cumulative incidence.

## Association and prediction are different calculations

Unpenalised association models estimate conditional odds ratios. MIMIC uses HC0 sandwich uncertainty; eICU uses hospital-cluster robust uncertainty with the implemented finite-sample correction. Standardised risks/RDs use the original 500-replicate bootstrap implementation and unchanged seeds.

The all-zero MIMIC-III oxygen-missingness column is redundant and is omitted; RRT is retained. Two primary full-adjustment MIMIC-III associations (K4 and fixed HR/RR) remain non-estimable because of quasi-complete separation. An additional MIMIC-IV oxygen-adjusted ordinary composite is non-estimable. Empty estimates in these rows are deliberate, not zeros or missing successful fits.

Prediction uses fixed L2 regularisation, C=1.0, lbfgs, tolerance 1e-4, maximum 2000 iterations. The original five saved group folds are preserved; preprocessing is trained within each fold. The five models are severity baseline; baseline + fixed HR/RR; baseline + K4; baseline + 16 conventional vital summaries; conventional model + K4. Paired uncertainty resamples the same existing out-of-fold predictions 500 times at patient/hospital-group level; it is not full nested bootstrap training. Finite regularised prediction does not make an unpenalised separated OR estimable. No parameter search was performed to obtain significance.

## Limitations retained

No stable incremental predictive advantage, demonstrated clinical utility, treatment guidance or pneumonia-specific biological subtype is established. Time-order perturbations retain some prognostic associations but low ARI indicates unstable individual membership; this does not prove time order irrelevant. MIMIC-III is a same-center historical replication with unresolved patient overlap with MIMIC-IV. CareVue 0/5,173 assignability is a structural support failure that weighting cannot repair. Sparse hospital effects and interval-crossing secondary results remain reported. Only eight eICU hospitals meet the independent synthesis gate.
