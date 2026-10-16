"""Historical selection/stability/mapping source; requires withheld fitted assets.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'Stability, variable-set, early-assignment, and transport gate.\n\nThis script decides between the outcome-free four-vital K4 and K5 finalists.\nNo mortality information is used in model selection or label alignment.\n'
import hashlib
import math
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT = Path(os.environ['SOURCE_E14A791BB8'])
INPUT = Path(os.environ['SOURCE_570F6B93E3'])
OLD_FIVE = Path(os.environ['SOURCE_BD75AD4448'])
SEED = 20260730
VITALS = ['SBP', 'DBP', 'HR', 'RR']
FEATURES = [f'{vital}_h{hour}' for vital in VITALS for hour in range(12)]
FINALISTS = [('full', 4), ('full', 5)]

def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(ROOT / filename, index=False, encoding='utf-8-sig')

def eligible_mask(frame: pd.DataFrame, minimum_dimensions: int=32) -> pd.Series:
    observed = frame[FEATURES].notna()
    enough = observed.sum(axis=1).ge(minimum_dimensions)
    each_vital = pd.concat({vital: observed[[f'{vital}_h{hour}' for hour in range(12)]].any(axis=1) for vital in VITALS}, axis=1).all(axis=1)
    return enough & each_vital

def standardization(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(raw, axis=0)
    sds = np.nanstd(raw, axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(sds).all() or (sds <= 0).any():
        raise RuntimeError('Invalid standardization')
    return (means, sds)

def observed_log_likelihood(raw: np.ndarray, standardization_means: np.ndarray, standardization_sds: np.ndarray, model: GaussianMixture, feature_indices: np.ndarray | None=None) -> np.ndarray:
    if feature_indices is None:
        feature_indices = np.arange(raw.shape[1])
    values_z = (raw[:, feature_indices] - standardization_means[feature_indices][None, :]) / standardization_sds[feature_indices][None, :]
    component_means = model.means_[:, feature_indices]
    d = len(feature_indices)
    mask = np.isfinite(values_z)
    packed = np.packbits(mask, axis=1)
    unique_masks, inverse = np.unique(packed, axis=0, return_inverse=True)
    result = np.full((len(raw), model.n_components), -np.inf)
    log_weights = np.log(np.clip(model.weights_, 1e-300, None))
    log_2pi = math.log(2.0 * math.pi)
    for mask_number, packed_mask in enumerate(unique_masks):
        rows = np.flatnonzero(inverse == mask_number)
        observed = np.unpackbits(packed_mask, count=d).astype(bool)
        columns = np.flatnonzero(observed)
        if len(columns) == 0:
            result[rows, :] = log_weights
            continue
        values = values_z[np.ix_(rows, columns)]
        source_feature_indices = feature_indices[columns]
        for component in range(model.n_components):
            mean = component_means[component, columns]
            diff = values - mean[None, :]
            if model.covariance_type == 'full':
                covariance = model.covariances_[component][np.ix_(source_feature_indices, source_feature_indices)]
            elif model.covariance_type == 'tied':
                covariance = model.covariances_[np.ix_(source_feature_indices, source_feature_indices)]
            elif model.covariance_type == 'diag':
                variance = model.covariances_[component, source_feature_indices]
                quadratic = np.sum(diff * diff / variance[None, :], axis=1)
                log_determinant = float(np.log(variance).sum())
                result[rows, component] = log_weights[component] - 0.5 * (len(columns) * log_2pi + log_determinant + quadratic)
                continue
            else:
                variance = float(model.covariances_[component])
                quadratic = np.sum(diff * diff / variance, axis=1)
                log_determinant = len(columns) * math.log(variance)
                result[rows, component] = log_weights[component] - 0.5 * (len(columns) * log_2pi + log_determinant + quadratic)
                continue
            jitter = 0.0
            for _ in range(7):
                try:
                    chol = np.linalg.cholesky(covariance + np.eye(len(columns)) * jitter)
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-08 if jitter == 0 else jitter * 10
            else:
                raise RuntimeError('Unable to factor covariance submatrix')
            solved = solve_triangular(chol, diff.T, lower=True, check_finite=False)
            quadratic = np.sum(solved * solved, axis=0)
            log_determinant = float(2.0 * np.log(np.diag(chol)).sum())
            result[rows, component] = log_weights[component] - 0.5 * (len(columns) * log_2pi + log_determinant + quadratic)
    return result

def assign(raw: np.ndarray, means: np.ndarray, sds: np.ndarray, model: GaussianMixture, feature_indices: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    ll = observed_log_likelihood(raw, means, sds, model, feature_indices)
    posterior = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    return (posterior.argmax(axis=1), posterior)

def align_labels(source: np.ndarray, target: np.ndarray, k: int) -> dict[int, int]:
    contingency = np.array([[np.sum((source == source_level) & (target == target_level)) for target_level in range(k)] for source_level in range(k)])
    rows, columns = linear_sum_assignment(-contingency)
    return {int(source_level): int(target_level) for source_level, target_level in zip(rows, columns)}

def fit_finalist(raw: np.ndarray, covariance_type: str, k: int, n_init: int, seed: int) -> tuple[np.ndarray, np.ndarray, GaussianMixture, np.ndarray, np.ndarray]:
    means, sds = standardization(raw)
    filled = np.nan_to_num((raw - means[None, :]) / sds[None, :], nan=0.0)
    model = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init, max_iter=500, tol=0.0001, reg_covar=1e-05, init_params='kmeans', random_state=seed).fit(filled)
    labels, posterior = assign(raw, means, sds, model)
    return (means, sds, model, labels, posterior)

def one_resample(raw: np.ndarray, full_labels: np.ndarray, covariance_type: str, k: int, iteration: int) -> dict[str, object]:
    rng = np.random.default_rng(SEED + 10000 * k + iteration)
    indices = rng.choice(len(raw), size=int(round(0.8 * len(raw))), replace=False)
    means, sds, model, _, _ = fit_finalist(raw[indices], covariance_type, k, n_init=3, seed=SEED + 10000 * k + iteration)
    sample_labels, sample_posterior = assign(raw, means, sds, model)
    mapping = align_labels(sample_labels, full_labels, k)
    aligned = np.array([mapping[int(label)] for label in sample_labels])
    counts = np.bincount(sample_labels, minlength=k)
    return {'covariance_type': covariance_type, 'K': k, 'iteration': iteration, 'subsample_n': int(len(indices)), 'converged': bool(model.converged_), 'adjusted_rand_index': float(adjusted_rand_score(full_labels, aligned)), 'normalized_mutual_information': float(normalized_mutual_info_score(full_labels, aligned)), 'exact_agreement': float(np.mean(full_labels == aligned)), 'minimum_class_percent': float(counts.min() / len(sample_labels) * 100.0), 'median_max_posterior': float(np.median(sample_posterior.max(axis=1)))}

def missingness_dependency(discovery: pd.DataFrame, labels: np.ndarray, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.DataFrame({vital: discovery[[f'{vital}_h{hour}' for hour in range(12)]].notna().sum(axis=1) for vital in VITALS})
    detail = counts.copy()
    detail['component'] = labels + 1
    detail = detail.groupby('component').agg(n=('SBP', 'size'), SBP_mean_hours=('SBP', 'mean'), DBP_mean_hours=('DBP', 'mean'), HR_mean_hours=('HR', 'mean'), RR_mean_hours=('RR', 'mean')).reset_index()
    detail.insert(0, 'model', model_name)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=None, random_state=SEED))
    predicted = cross_val_predict(classifier, counts.to_numpy(float), labels, cv=splitter, method='predict', n_jobs=1)
    majority_accuracy = float(np.bincount(labels).max() / len(labels))
    summary = pd.DataFrame([{'model': model_name, 'n': len(labels), 'missingness_only_cv_accuracy': float(np.mean(predicted == labels)), 'missingness_only_cv_balanced_accuracy': float(balanced_accuracy_score(labels, predicted)), 'majority_class_accuracy': majority_accuracy, 'accuracy_increment_over_majority': float(np.mean(predicted == labels) - majority_accuracy), 'interpretation': 'Observation counts only; outcome not used. Large increment would indicate recording-density dependence.'}])
    return (detail, summary)

def high_coverage_sensitivity(discovery: pd.DataFrame, full_labels: np.ndarray, covariance_type: str, k: int) -> dict[str, object]:
    mask = eligible_mask(discovery, minimum_dimensions=44)
    subset = discovery.loc[mask]
    raw = subset[FEATURES].to_numpy(float)
    means, sds, model, labels, posterior = fit_finalist(raw, covariance_type, k, n_init=20, seed=SEED + 50000 + k)
    full_subset_labels = full_labels[np.flatnonzero(mask.to_numpy())]
    mapping = align_labels(labels, full_subset_labels, k)
    aligned = np.array([mapping[int(label)] for label in labels])
    return {'covariance_type': covariance_type, 'K': k, 'high_coverage_definition': '>=44/48 dimensions and at least one observation per vital', 'high_coverage_n': int(len(subset)), 'adjusted_rand_index_vs_primary_on_overlap': float(adjusted_rand_score(full_subset_labels, aligned)), 'normalized_mutual_information_vs_primary_on_overlap': float(normalized_mutual_info_score(full_subset_labels, aligned)), 'exact_agreement_vs_primary_on_overlap': float(np.mean(full_subset_labels == aligned)), 'median_max_posterior': float(np.median(posterior.max(axis=1))), 'minimum_class_percent': float(np.bincount(labels, minlength=k).min() / len(labels) * 100.0)}

def map_to_old_labels(discovery_meta: pd.DataFrame, new_labels: np.ndarray, k: int) -> tuple[dict[int, str], pd.DataFrame]:
    old = pd.read_parquet(OLD_FIVE)
    old = old.loc[old['database'].eq('MIMIC-IV'), ['stay_id', 'refit_phenotype']].copy()
    crosswalk = discovery_meta[['stay_id']].copy()
    crosswalk['new_component'] = new_labels
    crosswalk = crosswalk.merge(old, on='stay_id', how='inner')
    if k != 4:
        mapping = {component: f'C{component + 1}' for component in range(k)}
        summary = pd.DataFrame([{'comparison': 'four-vital finalist vs five-vital K4', 'K': k, 'overlap_n': len(crosswalk), 'adjusted_rand_index': float(adjusted_rand_score(crosswalk['refit_phenotype'], crosswalk['new_component'])), 'normalized_mutual_information': float(normalized_mutual_info_score(crosswalk['refit_phenotype'], crosswalk['new_component'])), 'exact_agreement_after_alignment': np.nan, 'mapping': 'not one-to-one because K differs'}])
        return (mapping, summary)
    old_levels = ['P1', 'P2', 'P3', 'P4']
    contingency = np.array([[np.sum((crosswalk['new_component'].to_numpy() == component) & (crosswalk['refit_phenotype'].astype(str).to_numpy() == old_level)) for old_level in old_levels] for component in range(4)])
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {int(component): old_levels[int(old_index)] for component, old_index in zip(rows, columns)}
    aligned = crosswalk['new_component'].map(mapping)
    summary = pd.DataFrame([{'comparison': 'four-vital K4 vs five-vital K4', 'K': 4, 'overlap_n': len(crosswalk), 'adjusted_rand_index': float(adjusted_rand_score(crosswalk['refit_phenotype'], aligned)), 'normalized_mutual_information': float(normalized_mutual_info_score(crosswalk['refit_phenotype'], aligned)), 'exact_agreement_after_alignment': float(np.mean(crosswalk['refit_phenotype'].astype(str).to_numpy() == aligned.astype(str).to_numpy())), 'mapping': '|'.join((f'C{component + 1}->{mapping[component]}' for component in sorted(mapping)))}])
    return (mapping, summary)

def main() -> None:
    matrix = pd.read_parquet(INPUT)
    discovery = matrix.loc[matrix['database'].eq('MIMIC-IV') & eligible_mask(matrix)].copy()
    raw_discovery = discovery[FEATURES].to_numpy(float)
    grid = pd.read_csv(ROOT / '01_common4_model_grid.csv')
    full_models: dict[tuple[str, int], tuple] = {}
    for covariance_type, k in FINALISTS:
        log(f'Refitting finalist {covariance_type} K={k} with 50 starts')
        full_models[covariance_type, k] = fit_finalist(raw_discovery, covariance_type, k, n_init=50, seed=SEED + k)
    resampling_path = ROOT / '02_common4_resampling_stability.csv'
    if resampling_path.exists():
        resampling = pd.read_csv(resampling_path)
    else:
        log('Running 100 x 80% resampling for K4 and K5')
        jobs = [delayed(one_resample)(raw_discovery, full_models[covariance_type, k][3], covariance_type, k, iteration) for covariance_type, k in FINALISTS for iteration in range(1, 101)]
        resampling = pd.DataFrame(Parallel(n_jobs=4, verbose=5, max_nbytes='100M')(jobs))
        write_csv(resampling, '02_common4_resampling_stability.csv')
    stability_summary = resampling.groupby(['covariance_type', 'K']).agg(iterations=('iteration', 'size'), median_ARI=('adjusted_rand_index', 'median'), p05_ARI=('adjusted_rand_index', lambda x: np.quantile(x, 0.05)), median_exact_agreement=('exact_agreement', 'median'), p05_exact_agreement=('exact_agreement', lambda x: np.quantile(x, 0.05)), median_minimum_class_percent=('minimum_class_percent', 'median'), convergence_rate=('converged', 'mean')).reset_index()
    write_csv(stability_summary, '02_common4_resampling_stability_summary.csv')
    high_path = ROOT / '03_common4_high_coverage_sensitivity.csv'
    missing_path = ROOT / '04_common4_missingness_dependency_summary.csv'
    if high_path.exists() and missing_path.exists():
        high_coverage = pd.read_csv(high_path)
        missingness_summary = pd.read_csv(missing_path)
    else:
        high_rows = []
        detail_rows = []
        missing_rows = []
        for covariance_type, k in FINALISTS:
            labels = full_models[covariance_type, k][3]
            high_rows.append(high_coverage_sensitivity(discovery, labels, covariance_type, k))
            detail, missing = missingness_dependency(discovery, labels, f'{covariance_type}_K{k}')
            detail_rows.append(detail)
            missing_rows.append(missing)
        high_coverage = pd.DataFrame(high_rows)
        write_csv(high_coverage, '03_common4_high_coverage_sensitivity.csv')
        write_csv(pd.concat(detail_rows, ignore_index=True), '04_common4_missingness_by_component.csv')
        missingness_summary = pd.concat(missing_rows, ignore_index=True)
        write_csv(missingness_summary, '04_common4_missingness_dependency_summary.csv')
    decision = grid.loc[grid.apply(lambda row: (str(row['covariance_type']), int(row['K'])) in FINALISTS, axis=1)].merge(stability_summary, on=['covariance_type', 'K']).merge(high_coverage.rename(columns={'minimum_class_percent': 'high_coverage_minimum_class_percent'}), on=['covariance_type', 'K']).merge(missingness_summary.assign(covariance_type=lambda d: d['model'].str.split('_').str[0], K=lambda d: d['model'].str.extract('K(\\d+)')[0].astype(int)).drop(columns=['model']), on=['covariance_type', 'K'])
    decision['passes_stability_gate'] = decision['median_ARI'].ge(0.7) & decision['adjusted_rand_index_vs_primary_on_overlap'].ge(0.7) & decision['convergence_rate'].eq(1.0) & decision['minimum_class_percent'].ge(3.0) & decision['maximum_class_percent'].le(75.0)
    decision['missingness_dependency_flag'] = decision['accuracy_increment_over_majority'].gt(0.2)
    eligible_decision = decision.loc[decision['passes_stability_gate'] & ~decision['missingness_dependency_flag']].copy()
    if eligible_decision.empty:
        write_csv(decision, '05_common4_finalist_decision.csv')
        raise RuntimeError('No four-vital finalist passed the stability gate')
    selected = eligible_decision.sort_values(['approximate_ICL', 'K']).iloc[0]
    selected_key = (str(selected['covariance_type']), int(selected['K']))
    decision['selected_primary_model'] = decision.apply(lambda row: (str(row['covariance_type']), int(row['K'])) == selected_key, axis=1)
    decision['selection_rule'] = 'Among numerically admissible, structurally valid finalists passing resampling, high-coverage and missingness-dependence gates, select the minimum outcome-free approximate ICL; lower K breaks ties.'
    write_csv(decision, '05_common4_finalist_decision.csv')
    means, sds, selected_model, discovery_components, discovery_posterior = full_models[selected_key]
    phenotype_mapping, variable_set_summary = map_to_old_labels(discovery, discovery_components, selected_key[1])
    write_csv(variable_set_summary, '06_common4_vs_fivevital_patient_agreement.csv')
    with (ROOT / '06_common4_selected_model.pkl').open('wb') as handle:
        pickle.dump({'model': selected_model, 'standardization_means': means, 'standardization_sds': sds, 'features': FEATURES, 'vitals': VITALS, 'eligibility': 'alive/in ICU at 12h; >=32/48 observed dimensions; at least one observation per vital', 'component_to_phenotype': phenotype_mapping, 'selection_outcome_used': False}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    assignment_frames = []
    for database, frame in matrix.groupby('database', sort=False):
        eligible = frame.loc[eligible_mask(frame)].copy()
        raw = eligible[FEATURES].to_numpy(float)
        components, posterior = assign(raw, means, sds, selected_model)
        output_columns = ['database', 'stay_id', 'age', 'male', 'hospital_id', 'hospital_mortality', 'icu_los_hours']
        output = eligible[output_columns].copy()
        output['observed_dimensions_48d'] = np.isfinite(raw).sum(axis=1)
        output['selected_component'] = components + 1
        output['phenotype'] = [phenotype_mapping[int(component)] for component in components]
        output['max_posterior'] = posterior.max(axis=1)
        sorted_posterior = np.sort(posterior, axis=1)
        output['posterior_margin'] = sorted_posterior[:, -1] - sorted_posterior[:, -2]
        assignment_frames.append(output)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    assignments.to_parquet(ROOT / '07_common4_selected_assignments.parquet', index=False)
    distribution = assignments.groupby(['database', 'phenotype'], observed=True).agg(n=('stay_id', 'size'), deaths=('hospital_mortality', 'sum'), median_observed_dimensions=('observed_dimensions_48d', 'median'), median_max_posterior=('max_posterior', 'median'), median_posterior_margin=('posterior_margin', 'median')).reset_index()
    distribution['percent'] = distribution['n'] / distribution.groupby('database')['n'].transform('sum') * 100.0
    distribution['mortality_percent'] = distribution['deaths'] / distribution['n'] * 100.0
    write_csv(distribution, '07_common4_distribution_mortality.csv')
    profile_rows = []
    for component in range(selected_model.n_components):
        phenotype = phenotype_mapping[int(component)]
        for feature_index, feature in enumerate(FEATURES):
            profile_rows.append({'component': component + 1, 'phenotype': phenotype, 'feature_order': feature_index + 1, 'feature': feature, 'component_weight': selected_model.weights_[component], 'component_mean_z': selected_model.means_[component, feature_index], 'component_variance_z': selected_model.covariances_[component, feature_index, feature_index], 'mimiciv_observed_mean': means[feature_index], 'mimiciv_observed_sd': sds[feature_index], 'prototype_value_raw_scale': means[feature_index] + selected_model.means_[component, feature_index] * sds[feature_index]})
    write_csv(pd.DataFrame(profile_rows), '08_common4_component_profiles.csv')
    early_indices = np.array([index for index, feature in enumerate(FEATURES) if int(feature.rsplit('h', 1)[1]) <= 8])
    early_rows = []
    for database, frame in matrix.groupby('database', sort=False):
        eligible12 = frame.loc[eligible_mask(frame)].copy()
        raw = eligible12[FEATURES].to_numpy(float)
        full_components, _ = assign(raw, means, sds, selected_model)
        early_raw = raw[:, early_indices]
        early_observed = np.isfinite(early_raw)
        vital_ok = np.column_stack([early_observed[:, [position for position, source_index in enumerate(early_indices) if FEATURES[source_index].startswith(f'{vital}_')]].any(axis=1) for vital in VITALS]).all(axis=1)
        early_eligible = (early_observed.sum(axis=1) >= 24) & vital_ok
        early_components, early_posterior = assign(raw[early_eligible], means, sds, selected_model, feature_indices=early_indices)
        full_subset = full_components[early_eligible]
        early_rows.append({'database': database, 'comparison': '0-9h assignment vs 0-12h assignment', 'eligible_12h_n': int(len(eligible12)), 'eligible_9h_n': int(early_eligible.sum()), 'eligible_9h_percent': float(early_eligible.mean() * 100.0), 'exact_agreement': float(np.mean(early_components == full_subset)), 'adjusted_rand_index': float(adjusted_rand_score(full_subset, early_components)), 'normalized_mutual_information': float(normalized_mutual_info_score(full_subset, early_components)), 'median_9h_max_posterior': float(np.median(early_posterior.max(axis=1)))})
    write_csv(pd.DataFrame(early_rows), '09_common4_9h_assignment.csv')
    parameter_rows = []
    for feature_index, feature in enumerate(FEATURES):
        parameter_rows.append({'feature_order': feature_index + 1, 'feature': feature, 'mimiciv_observed_mean': means[feature_index], 'mimiciv_observed_sd': sds[feature_index]})
    write_csv(pd.DataFrame(parameter_rows), '10_common4_selected_standardization.csv')
    model_card = pd.DataFrame([{'primary_time_origin': '12 hours after ICU admission', 'primary_development_database': 'MIMIC-IV', 'historical_replication_database': 'MIMIC-III', 'primary_multicentre_external_validation_database': 'eICU', 'representation': 'SBP|DBP|HR|RR; hourly 0-12h; 48D', 'temperature_role': 'excluded from primary representation; five-vital model retained as temperature-enhanced sensitivity', 'selected_covariance_type': selected_key[0], 'selected_K': selected_key[1], 'selection_outcome_used': False, 'assignment_operator': 'frozen GMM marginal likelihood over observed dimensions', 'external_refitting': False, 'label_alignment': 'MIMIC-IV patient overlap with corrected five-vital K4; outcome not used' if selected_key[1] == 4 else 'component labels retained because K differs'}])
    write_csv(model_card, '11_common4_selected_model_card.csv')
    manifest_rows = []
    for path in sorted(ROOT.glob('*')):
        if path.is_file() and path.name.startswith(('02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_', '11_')):
            manifest_rows.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(pd.DataFrame(manifest_rows), '12_common4_transport_manifest.csv')
    log(f'Selected outcome-free primary model: {selected_key[0]} K={selected_key[1]}')
    print(decision.to_string(index=False))
    print(variable_set_summary.to_string(index=False))
    print(distribution.to_string(index=False))
if __name__ == '__source_reference_run__':
    main()
