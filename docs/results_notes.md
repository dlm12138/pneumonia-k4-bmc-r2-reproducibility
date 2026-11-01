# Reading aggregate results

The tables are projections of current actual source tables. Values and row order were retained; `results_provenance.tsv` identifies the source file name and SHA256 without private paths. Model/scenario keys are retained so endpoint or covariate differences remain visible.

- `primary_and_severity_effects.tsv`: four primary/mean-adjusted specifications × four phenotypes × three sources. Non-estimable rows have empty estimates and an explicit status.
- `hospital_mortality_effects.tsv`: MIMIC-IV has 3,250 recorded whole-hospital deaths among 14,636 patients. This includes three recorded hospital deaths without an exact death timestamp. Whole-hospital death is determined from hospital death status; missing death time must not be coded as survival. The two existing whole-hospital specifications and the MIMIC-IV hospital-mortality rows in the three selection-summary tables use this definition. Primary-window events, primary estimates, other databases and other endpoints are unchanged.
- `hrrr_comparison.tsv`: fixed HR mean >90 AND RR mean >20 rule, not a newly chosen cut point.
- `prediction_performance.tsv`: five point-performance models plus three paired comparisons per database; delta directions are newer minus reference. An AUROC delta is not an odds ratio.
- `phenotype_counts_and_deaths.tsv`: complete primary-model group sizes and actual recorded deaths. eICU assigned-clinical and complete-model denominators differ by two missing-sex records.
- `oxygen_availability.tsv` and `coverage_oxygen_missingness.tsv`: corrected current counts only, not old missingness.
- `eicu_conflict_distribution.tsv`, rule and rule-combination tables: aggregate conflict patterns only. Overlapping rules must not be summed; no 229-person list is provided.
- Composite and competing-risk files use different target/competing observation scopes. Do not equate the ordinary composite event total with Fine–Gray target-event totals.
- Time-order 2.5–97.5 percentiles summarise repeated perturbations; they are not ordinary confidence intervals for a fixed individual phenotype assignment. Low ARI is retained.
- Hospital transport tables provide overall summaries only, not individual hospital IDs or patient linkage.
- Selection tables include ESS and balance diagnostics; they do not restore representation of a source stratum with zero assignability.

Numerical units are preserved: risks and most RD bounds are fractions unless the column explicitly says percentage points/percent/pct. Standardised risks are not observed crude proportions. OR and SHR reference P2 where specified. Blank non-estimable values must not be filled with zero or an earlier version.

The historical MIMIC-IV reference population N=14,853 in domain-shift summaries is intentional and is not the current strict application denominator. Historical comparison/missingness columns were otherwise excluded. In particular, the old SpO2-missing column accompanying the unchanged secondary event-count reconciliation was omitted; use the corrected oxygen tables instead. The eICU support time-scope label corrections are descriptive and do not change participant classification or numerical estimates.

Small aggregate event counts are kept where needed to explain published statistical non-estimability and definitions. They are not real-patient example rows; no identifier, raw timestamp, diagnostic string or individual measurement accompanies them. No site-level granular table is released. Public disclosure remains subject to author/institution policy review; automated scanning is not a legal authorisation.

For reproduction of the corrected MIMIC-IV whole-hospital summaries, derive `hospital_mortality` from the recorded hospital death status in both the model input and the candidate selection input, independently of exact death-time availability. The selection numerator, denominator, truncation rules and HC0 outcome models are unchanged. An unknown death time does not establish an event within the primary (12,36]-hour window. The historical extraction sources are retained for provenance; the input outcome definition must follow this distinction.
