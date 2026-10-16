# Software and verification environment

The 2026-10-15 targeted analysis environment recorded numpy 2.4.3, pandas 2.3.3, scipy 1.17.1, statsmodels 0.14.6, scikit-learn 1.8.0 and pyarrow 26.0.0. `requirements.txt` pins these core dependencies plus the actually imported formula dependency patsy. Document-rendering packages are not analysis requirements and are omitted.

SQL verification uses DuckDB CLI 1.5.2. There is no MySQL dependency and no persistent database server is required. The official CLI must be available on PATH, or supplied to the synthetic test with `--duckdb`.

The R verification environment recorded on 2026-10-15 comprised R 4.5.3; arrow 24.0.0; survival 3.8.6; cmprsk 2.2.12; lme4 2.0.1; sandwich 3.1.1. These installed versions do not establish that every historical analysis used precisely this R environment. The retained R programs import the relevant subset.

Prediction uses the original fixed settings: L2 penalty, C=1.0, lbfgs, tol=1e-4, max_iter=2000, five saved group folds. The functions preserve the original seeds, study seed offsets and 500-replicate uncertainty settings; these are not retuned during reproduction.

Fitted numerical K4 assets, original saved fold arrays and patient-linked feature matrices are not software dependencies that can be replaced by installing a package. They remain separate restricted prerequisites.
