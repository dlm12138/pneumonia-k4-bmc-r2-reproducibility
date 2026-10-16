"""Historical final K4 selection source; no new fitting or relabelling.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'Complete early-assignment and model-card outputs from the locked K4 bundle.'
import hashlib
import importlib.util
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT = Path(os.environ['SOURCE_E14A791BB8'])
MATRIX = Path(os.environ['SOURCE_570F6B93E3'])
MODULE_PATH = ROOT / '02_common4_stability_transport.py'

def load_module():
    spec = importlib.util.spec_from_file_location('common4_transport', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(ROOT / filename, index=False, encoding='utf-8-sig')

def main() -> None:
    module = load_module()
    with (ROOT / '06_common4_selected_model.pkl').open('rb') as handle:
        bundle = pickle.load(handle)
    model = bundle['model']
    if int(model.n_components) != 4:
        raise RuntimeError('The final model must remain locked at K4')
    means = np.asarray(bundle['standardization_means'])
    sds = np.asarray(bundle['standardization_sds'])
    features = list(bundle['features'])
    matrix = pd.read_parquet(MATRIX)
    early_indices = np.array([index for index, feature in enumerate(features) if int(feature.rsplit('h', 1)[1]) <= 8])
    early_rows = []
    for database, frame in matrix.groupby('database', sort=False):
        eligible12 = frame.loc[module.eligible_mask(frame)].copy()
        raw = eligible12[features].to_numpy(float)
        full_components, _ = module.assign(raw, means, sds, model)
        early_raw = raw[:, early_indices]
        early_observed = np.isfinite(early_raw)
        vital_ok = np.column_stack([early_observed[:, [position for position, source_index in enumerate(early_indices) if features[source_index].startswith(f'{vital}_')]].any(axis=1) for vital in module.VITALS]).all(axis=1)
        early_eligible = (early_observed.sum(axis=1) >= 24) & vital_ok
        early_components, early_posterior = module.assign(raw[early_eligible], means, sds, model, feature_indices=early_indices)
        full_subset = full_components[early_eligible]
        early_rows.append({'database': database, 'comparison': '0-9h assignment vs 0-12h assignment', 'eligible_12h_n': int(len(eligible12)), 'eligible_9h_n': int(early_eligible.sum()), 'eligible_9h_percent': float(early_eligible.mean() * 100.0), 'exact_agreement': float(np.mean(early_components == full_subset)), 'adjusted_rand_index': float(adjusted_rand_score(full_subset, early_components)), 'normalized_mutual_information': float(normalized_mutual_info_score(full_subset, early_components)), 'median_9h_max_posterior': float(np.median(early_posterior.max(axis=1)))})
    write_csv(pd.DataFrame(early_rows), '09_common4_9h_assignment.csv')
    write_csv(pd.DataFrame({'feature_order': np.arange(1, len(features) + 1), 'feature': features, 'mimiciv_observed_mean': means, 'mimiciv_observed_sd': sds}), '10_common4_selected_standardization.csv')
    model_card = pd.DataFrame([{'primary_time_origin': '12 hours after ICU admission', 'primary_development_database': 'MIMIC-IV', 'historical_replication_database': 'MIMIC-III', 'primary_multicentre_external_validation_database': 'eICU', 'representation': 'SBP|DBP|HR|RR; hourly 0-12h; 48D', 'temperature_role': 'excluded from the final primary representation because cross-database recording coverage was not comparable', 'selected_covariance_type': model.covariance_type, 'selected_K': 4, 'K_status': 'locked final primary model', 'selection_outcome_used': False, 'assignment_operator': 'frozen GMM marginal likelihood over observed dimensions', 'external_refitting': False, 'other_K_role': 'model-selection audit only; not a final manuscript model'}])
    write_csv(model_card, '11_common4_selected_model_card.csv')
    manifest_rows = []
    for path in sorted(ROOT.glob('*')):
        if path.is_file() and path.name.startswith(('00_', '01_', '02_', '03_common4', '04_common4', '05_', '06_', '07_common4', '08_common4', '09_', '10_', '11_')):
            manifest_rows.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(pd.DataFrame(manifest_rows), '12_common4_transport_manifest.csv')
    print(pd.DataFrame(early_rows).to_string(index=False))
    print(model_card.to_string(index=False))
if __name__ == '__source_reference_run__':
    main()
