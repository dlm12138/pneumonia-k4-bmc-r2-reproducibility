"""Actual study functions extracted without changes to their calculations.
See docs/code_provenance.tsv and docs/reproduction_limits.md.
"""
from __future__ import annotations
import math
import os
import re
import csv
import gzip
import io
import sys
import zipfile
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any
from types import SimpleNamespace
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import solve_triangular
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



def observed_log_likelihood(
    raw: np.ndarray,
    standardization_means: np.ndarray,
    standardization_sds: np.ndarray,
    model: GaussianMixture,
    feature_indices: np.ndarray | None = None,
) -> np.ndarray:
    if feature_indices is None:
        feature_indices = np.arange(raw.shape[1])
    values_z = (
        raw[:, feature_indices]
        - standardization_means[feature_indices][None, :]
    ) / standardization_sds[feature_indices][None, :]
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
            if model.covariance_type == "full":
                covariance = model.covariances_[component][
                    np.ix_(source_feature_indices, source_feature_indices)
                ]
            elif model.covariance_type == "tied":
                covariance = model.covariances_[
                    np.ix_(source_feature_indices, source_feature_indices)
                ]
            elif model.covariance_type == "diag":
                variance = model.covariances_[component, source_feature_indices]
                quadratic = np.sum(diff * diff / variance[None, :], axis=1)
                log_determinant = float(np.log(variance).sum())
                result[rows, component] = (
                    log_weights[component]
                    - 0.5
                    * (
                        len(columns) * log_2pi
                        + log_determinant
                        + quadratic
                    )
                )
                continue
            else:
                variance = float(model.covariances_[component])
                quadratic = np.sum(diff * diff / variance, axis=1)
                log_determinant = len(columns) * math.log(variance)
                result[rows, component] = (
                    log_weights[component]
                    - 0.5
                    * (
                        len(columns) * log_2pi
                        + log_determinant
                        + quadratic
                    )
                )
                continue

            jitter = 0.0
            for _ in range(7):
                try:
                    chol = np.linalg.cholesky(
                        covariance + np.eye(len(columns)) * jitter
                    )
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-8 if jitter == 0 else jitter * 10
            else:
                raise RuntimeError("Unable to factor covariance submatrix")
            solved = solve_triangular(
                chol, diff.T, lower=True, check_finite=False
            )
            quadratic = np.sum(solved * solved, axis=0)
            log_determinant = float(2.0 * np.log(np.diag(chol)).sum())
            result[rows, component] = (
                log_weights[component]
                - 0.5
                * (
                    len(columns) * log_2pi
                    + log_determinant
                    + quadratic
                )
            )
    return result


def assign(
    raw: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    model: GaussianMixture,
    feature_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ll = observed_log_likelihood(raw, means, sds, model, feature_indices)
    posterior = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    return posterior.argmax(axis=1), posterior
