"""Historical derivation source only; no fitting during packaging.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'Outcome-free 12-hour four-vital GMM model-selection grid.\n\nThe script uses the reconstructed 12-hour landmark matrix and restricts the\nprimary common-feature representation to SBP, DBP, HR, and RR. Temperature is\ndeliberately excluded from this gate because its recording coverage differs\nsubstantially across databases.\n'
import hashlib
import math
import pickle
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
ROOT = Path(os.environ['SOURCE_E14A791BB8'])
INPUT = Path(os.environ['SOURCE_570F6B93E3'])
SEED = 20260730
REG_COVAR = 1e-05
MIN_VARIANCE = 0.0001
MAX_CONDITION = 100000.0
VITALS = ['SBP', 'DBP', 'HR', 'RR']
FEATURES = [f'{vital}_h{hour}' for vital in VITALS for hour in range(12)]

def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(ROOT / filename, index=False, encoding='utf-8-sig')

def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    observed = frame[FEATURES].notna()
    enough_dimensions = observed.sum(axis=1).ge(32)
    each_vital = pd.concat({vital: observed[[f'{vital}_h{hour}' for hour in range(12)]].any(axis=1) for vital in VITALS}, axis=1).all(axis=1)
    return enough_dimensions & each_vital

def standardize(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(raw, axis=0)
    sds = np.nanstd(raw, axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(sds).all() or (sds <= 0).any():
        raise RuntimeError('Invalid MIMIC-IV standardization parameters')
    observed_z = (raw - means[None, :]) / sds[None, :]
    filled_z = np.nan_to_num(observed_z, nan=0.0)
    return (means, sds, filled_z)

def observed_log_likelihood(observed_z: np.ndarray, model: GaussianMixture) -> np.ndarray:
    """Component log likelihood using only dimensions observed in each row."""
    n, d = observed_z.shape
    if d != model.means_.shape[1]:
        raise ValueError('Feature dimension mismatch')
    mask = np.isfinite(observed_z)
    packed = np.packbits(mask, axis=1)
    unique_masks, inverse = np.unique(packed, axis=0, return_inverse=True)
    result = np.full((n, model.n_components), -np.inf, dtype=float)
    log_weights = np.log(np.clip(model.weights_, 1e-300, None))
    log_2pi = math.log(2.0 * math.pi)
    for mask_number, packed_mask in enumerate(unique_masks):
        rows = np.flatnonzero(inverse == mask_number)
        observed = np.unpackbits(packed_mask, count=d).astype(bool)
        columns = np.flatnonzero(observed)
        if len(columns) == 0:
            result[rows, :] = log_weights
            continue
        values = observed_z[np.ix_(rows, columns)]
        for component in range(model.n_components):
            mean = model.means_[component, columns]
            diff = values - mean[None, :]
            if model.covariance_type == 'spherical':
                variance = float(model.covariances_[component])
                quadratic = np.sum(diff * diff / variance, axis=1)
                log_determinant = len(columns) * math.log(variance)
            elif model.covariance_type == 'diag':
                variance = model.covariances_[component, columns]
                quadratic = np.sum(diff * diff / variance[None, :], axis=1)
                log_determinant = float(np.log(variance).sum())
            elif model.covariance_type in {'tied', 'full'}:
                source = model.covariances_ if model.covariance_type == 'tied' else model.covariances_[component]
                covariance = source[np.ix_(columns, columns)]
                jitter = 0.0
                for _ in range(7):
                    try:
                        chol = np.linalg.cholesky(covariance + np.eye(len(columns)) * jitter)
                        break
                    except np.linalg.LinAlgError:
                        jitter = 1e-08 if jitter == 0 else jitter * 10
                else:
                    raise RuntimeError('Unable to factor observed covariance submatrix')
                solved = solve_triangular(chol, diff.T, lower=True, check_finite=False)
                quadratic = np.sum(solved * solved, axis=0)
                log_determinant = float(2.0 * np.log(np.diag(chol)).sum())
            else:
                raise ValueError(model.covariance_type)
            result[rows, component] = log_weights[component] - 0.5 * (len(columns) * log_2pi + log_determinant + quadratic)
    return result

def numerical_diagnostics(model: GaussianMixture) -> dict[str, float | int | bool]:
    if model.covariance_type == 'spherical':
        values = np.asarray(model.covariances_, dtype=float)
        minimum = float(values.min())
        maximum = float(values.max())
        condition = float(maximum / minimum)
        floor_count = int(np.sum(values <= REG_COVAR * 1.1))
    elif model.covariance_type == 'diag':
        values = np.asarray(model.covariances_, dtype=float)
        minimum = float(values.min())
        maximum = float(values.max())
        condition = float(max((component.max() / component.min() for component in values)))
        floor_count = int(np.sum(values <= REG_COVAR * 1.1))
    elif model.covariance_type == 'tied':
        eigenvalues = np.linalg.eigvalsh(model.covariances_)
        minimum = float(eigenvalues.min())
        maximum = float(eigenvalues.max())
        condition = float(maximum / minimum)
        floor_count = int(np.sum(eigenvalues <= REG_COVAR * 1.1))
    elif model.covariance_type == 'full':
        spectra = [np.linalg.eigvalsh(covariance) for covariance in model.covariances_]
        minimum = float(min((values.min() for values in spectra)))
        maximum = float(max((values.max() for values in spectra)))
        condition = float(max((values.max() / values.min() for values in spectra)))
        floor_count = int(sum((np.sum(values <= REG_COVAR * 1.1) for values in spectra)))
    else:
        raise ValueError(model.covariance_type)
    admissible = bool(minimum >= MIN_VARIANCE and condition <= MAX_CONDITION and (floor_count == 0))
    return {'minimum_covariance_eigenvalue_or_variance': minimum, 'maximum_covariance_eigenvalue_or_variance': maximum, 'maximum_condition_number': condition, 'parameters_at_regularization_floor': floor_count, 'numerically_admissible': admissible}

def metric_row(model: GaussianMixture, observed_z: np.ndarray, filled_z: np.ndarray, fit_seconds: float) -> dict[str, object]:
    component_ll = observed_log_likelihood(observed_z, model)
    posterior = np.exp(component_ll - logsumexp(component_ll, axis=1, keepdims=True))
    labels = posterior.argmax(axis=1)
    counts = np.bincount(labels, minlength=model.n_components)
    entropy = float(-np.sum(posterior * np.log(np.clip(posterior, 1e-300, 1.0))))
    minimum_percent = float(counts.min() / len(labels) * 100.0)
    maximum_percent = float(counts.max() / len(labels) * 100.0)
    structure_valid = bool(counts.min() >= 300 and minimum_percent >= 3.0 and (maximum_percent <= 75.0))
    return {'representation': 'SBP|DBP|HR|RR; 0-12h; 48 dimensions', 'discovery_database': 'MIMIC-IV', 'eligibility': 'alive/in ICU at 12h; >=32/48 observed dimensions; at least one observation for each of four vital signs', 'covariance_type': model.covariance_type, 'K': int(model.n_components), 'n': int(len(filled_z)), 'converged': bool(model.converged_), 'n_iter': int(model.n_iter_), 'n_init': int(model.n_init), 'fit_seconds': fit_seconds, 'AIC_zero_filled_training_geometry': float(model.aic(filled_z)), 'BIC_zero_filled_training_geometry': float(model.bic(filled_z)), 'classification_entropy_observed_likelihood': entropy, 'approximate_ICL': float(model.bic(filled_z) + 2.0 * entropy), 'minimum_class_n': int(counts.min()), 'minimum_class_percent': minimum_percent, 'maximum_class_percent': maximum_percent, 'median_max_posterior': float(np.median(posterior.max(axis=1))), 'p10_max_posterior': float(np.quantile(posterior.max(axis=1), 0.1)), 'class_structure_valid': structure_valid, 'outcome_used_for_selection': False, **numerical_diagnostics(model)}

def audit_coverage(matrix: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for database, frame in matrix.groupby('database', sort=False):
        observed = frame[FEATURES].notna()
        primary = eligible_mask(frame)
        row: dict[str, object] = {'database': database, 'landmark_candidate_n': int(len(frame)), 'primary_common4_eligible_n': int(primary.sum()), 'primary_common4_eligible_percent': float(primary.mean() * 100.0), 'median_observed_dimensions_48d': float(observed.sum(axis=1).median())}
        for vital in VITALS + ['TEMP']:
            columns = [f'{vital}_h{hour}' for hour in range(12)]
            count = frame[columns].notna().sum(axis=1)
            row[f'{vital}_any_observation_n'] = int(count.gt(0).sum())
            row[f'{vital}_any_observation_percent'] = float(count.gt(0).mean() * 100.0)
            row[f'{vital}_median_observed_hours'] = float(count.median())
        rows.append(row)
    write_csv(pd.DataFrame(rows), '00_common4_landmark_coverage_audit.csv')

def save_models(models: dict[tuple[str, int], GaussianMixture]) -> None:
    with (ROOT / '01_common4_screening_models.pkl').open('wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

def write_manifest() -> None:
    rows = []
    for path in sorted(ROOT.glob('0[01]_*')):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': digest})
    write_csv(pd.DataFrame(rows), '01_common4_grid_manifest.csv')

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_parquet(INPUT)
    audit_coverage(matrix)
    discovery = matrix.loc[matrix['database'].eq('MIMIC-IV') & eligible_mask(matrix)].copy()
    raw = discovery[FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    means, sds, filled_z = standardize(raw)
    observed_z = (raw - means[None, :]) / sds[None, :]
    parameter_frame = pd.DataFrame({'feature_order': np.arange(1, len(FEATURES) + 1), 'feature': FEATURES, 'mimiciv_observed_mean': means, 'mimiciv_observed_sd': sds, 'temperature_in_primary_representation': False})
    write_csv(parameter_frame, '01_common4_standardization.csv')
    model_path = ROOT / '01_common4_screening_models.pkl'
    metrics_path = ROOT / '01_common4_model_grid.csv'
    if model_path.exists():
        with model_path.open('rb') as handle:
            models = pickle.load(handle)
    else:
        models = {}
    rows = pd.read_csv(metrics_path).to_dict('records') if metrics_path.exists() else []
    completed = {(str(row['covariance_type']), int(row['K'])) for row in rows}
    targets = [(covariance_type, k) for covariance_type in ['spherical', 'diag', 'tied', 'full'] for k in range(2, 11)]
    for covariance_type, k in targets:
        key = (covariance_type, k)
        if key in completed and key in models:
            log(f'Skipping completed {covariance_type} K={k}')
            continue
        n_init = 10
        log(f'Fitting outcome-free {covariance_type} K={k}')
        started = time.time()
        model = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init, max_iter=500, tol=0.0001, reg_covar=REG_COVAR, init_params='kmeans', random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            model.fit(filled_z)
        row = metric_row(model, observed_z, filled_z, time.time() - started)
        models[key] = model
        rows = [previous for previous in rows if (str(previous['covariance_type']), int(previous['K'])) != key]
        rows.append(row)
        save_models(models)
        pd.DataFrame(rows).sort_values(['covariance_type', 'K']).to_csv(metrics_path, index=False, encoding='utf-8-sig')
        log(f"{covariance_type} K={k}: BIC={row['BIC_zero_filled_training_geometry']:.1f}; ICL={row['approximate_ICL']:.1f}; min={row['minimum_class_percent']:.2f}%; admissible={row['numerically_admissible']}")
    grid = pd.DataFrame(rows).sort_values(['covariance_type', 'K']).reset_index(drop=True)
    valid = grid['converged'].astype(bool) & grid['numerically_admissible'].astype(bool) & grid['class_structure_valid'].astype(bool)
    grid['passes_initial_gate'] = valid
    grid['BIC_rank_among_initial_gate'] = np.nan
    grid['ICL_rank_among_initial_gate'] = np.nan
    grid.loc[valid, 'BIC_rank_among_initial_gate'] = grid.loc[valid, 'BIC_zero_filled_training_geometry'].rank(method='min').to_numpy()
    grid.loc[valid, 'ICL_rank_among_initial_gate'] = grid.loc[valid, 'approximate_ICL'].rank(method='min').to_numpy()
    grid.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    write_manifest()
    finalists = grid.loc[valid].sort_values(['ICL_rank_among_initial_gate', 'BIC_rank_among_initial_gate', 'K'])
    print(finalists[['covariance_type', 'K', 'BIC_zero_filled_training_geometry', 'approximate_ICL', 'minimum_class_percent', 'maximum_class_percent', 'median_max_posterior']].head(12).to_string(index=False))
if __name__ == '__source_reference_run__':
    main()
