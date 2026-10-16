#!/usr/bin/env python3
"""
Frozen K4 assignment helper.

This module performs assignment only. It contains no fitting, K selection,
standardisation estimation, or phenotype relabelling operation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp


def observed_dimension_log_likelihood(
    standardized_values: np.ndarray, model: Any
) -> np.ndarray:
    """Return component log likelihoods using only observed dimensions."""
    n_rows, n_features = standardized_values.shape
    mask = np.isfinite(standardized_values)
    packed = np.packbits(mask, axis=1)
    unique_masks, inverse = np.unique(packed, axis=0, return_inverse=True)
    result = np.full((n_rows, model.n_components), -np.inf, dtype=float)
    log_weights = np.log(np.clip(model.weights_, 1e-300, None))
    log_2pi = math.log(2.0 * math.pi)
    for mask_number, packed_mask in enumerate(unique_masks):
        rows = np.flatnonzero(inverse == mask_number)
        observed = np.unpackbits(packed_mask, count=n_features).astype(bool)
        columns = np.flatnonzero(observed)
        values = standardized_values[np.ix_(rows, columns)]
        for component in range(model.n_components):
            difference = values - model.means_[component, columns][None, :]
            covariance = model.covariances_[component][np.ix_(columns, columns)]
            cholesky = np.linalg.cholesky(covariance)
            solved = solve_triangular(
                cholesky, difference.T, lower=True, check_finite=False
            )
            quadratic = np.sum(solved * solved, axis=0)
            log_determinant = float(2.0 * np.log(np.diag(cholesky)).sum())
            result[rows, component] = log_weights[component] - 0.5 * (
                len(columns) * log_2pi + log_determinant + quadratic
            )
    return result


def assign_locked_k4(
    raw_values: np.ndarray,
    standardization_means: np.ndarray,
    standardization_sds: np.ndarray,
    model: Any,
    component_to_phenotype: dict[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign eligible rows and return phenotype labels and posteriors."""
    standardized = (
        np.asarray(raw_values, dtype=float)
        - np.asarray(standardization_means, dtype=float)[None, :]
    ) / np.asarray(standardization_sds, dtype=float)[None, :]
    component_ll = observed_dimension_log_likelihood(standardized, model)
    posterior = np.exp(
        component_ll - logsumexp(component_ll, axis=1, keepdims=True)
    )
    components = posterior.argmax(axis=1)
    phenotypes = np.asarray(
        [component_to_phenotype[int(component)] for component in components]
    )
    return phenotypes, posterior
