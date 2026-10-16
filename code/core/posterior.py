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

def write_csv(frame, name):
    root=Path(os.environ['OUTPUT_ROOT'])
    root.mkdir(parents=True,exist_ok=True)
    frame.to_csv(root/name,index=False)

PHENOTYPES = ["P1", "P2", "P3", "P4"]


DATABASES = ["MIMIC-IV", "MIMIC-III", "eICU"]


SEED = 20260801


def fit_glm(
    frame: pd.DataFrame,
    formula: str,
    weight: pd.Series | None = None,
    cluster: pd.Series | None = None,
):
    model = smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Binomial(),
        freq_weights=weight,
    )
    if weight is not None:
        if len(model.endog) != len(frame):
            raise ValueError("Weighted GLM input contains model-variable missingness; complete cases must be explicit")
        result = model.fit()
        exog = np.asarray(model.exog, float)
        endog = np.asarray(model.endog, float)
        fitted = np.asarray(result.fittedvalues, float)
        weights = np.asarray(weight, float)
        information = exog.T @ (exog * (weights * fitted * (1.0 - fitted))[:, None])
        bread = np.linalg.pinv(information)
        score = exog * (weights * (endog - fitted))[:, None]
        n, p = exog.shape
        if cluster is not None:
            groups = np.asarray(cluster)
            if len(groups) != n or pd.isna(groups).any():
                raise ValueError("Weighted clustered GLM requires a complete cluster ID for every model row")
            codes, levels = pd.factorize(groups, sort=False)
            group_score = np.zeros((len(levels), p), float)
            np.add.at(group_score, codes, score)
            meat = group_score.T @ group_score
            g = len(levels)
            correction = (g / (g - 1.0)) * ((n - 1.0) / (n - p)) if g > 1 and n > p else 1.0
        else:
            meat = score.T @ score
            correction = n / (n - p) if n > p else 1.0
        covariance = bread @ (meat * correction) @ bread
        return SimpleNamespace(
            params=result.params,
            bse=pd.Series(np.sqrt(np.maximum(np.diag(covariance), 0.0)), index=result.params.index),
        )
    if cluster is not None:
        row_labels = pd.Index(model.data.row_labels)
        groups = pd.Series(cluster, index=frame.index).reindex(row_labels)
        if groups.isna().any():
            raise ValueError("Clustered GLM requires a complete cluster ID for every model row")
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups.to_numpy()})
    return model.fit(cov_type="HC1")


def model_effect(result, term: str) -> dict[str, float]:
    beta = float(result.params[term])
    se = float(result.bse[term])
    return {
        "beta": beta,
        "se": se,
        "or": math.exp(beta),
        "ci_low": math.exp(beta - 1.96 * se),
        "ci_high": math.exp(beta + 1.96 * se),
        "p_value": float(2.0 * norm.sf(abs(beta / se))),
    }


def prepare_model_frame(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    result = frame.copy()
    result["age10"] = pd.to_numeric(result["age"], errors="coerce") / 10.0
    result["male"] = pd.to_numeric(result["male"], errors="coerce")
    result["coverage10"] = pd.to_numeric(result["observed_dimensions_48d"], errors="coerce") / 10.0
    result["p1"] = result["phenotype"].eq("P1").astype(int)
    result["p3"] = result["phenotype"].eq("P3").astype(int)
    result["p4"] = result["phenotype"].eq("P4").astype(int)
    return result.dropna(subset=[outcome, "age10", "male", "coverage10", "phenotype"])


def posterior_uncertainty_analysis(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuous_rows = []
    quintile_rows = []
    certainty_rows = []
    pseudo_rows = []
    for database, frame in analysis.groupby("database"):
        data = frame.copy()
        data["age10"] = pd.to_numeric(data["age"], errors="coerce") / 10.0
        data["coverage10"] = pd.to_numeric(data["observed_dimensions_48d"], errors="coerce") / 10.0
        data = data.dropna(subset=["death_24h", "age10", "male", "coverage10"])
        cluster = data["hospital_id"] if database == "eICU" else None
        fit = fit_glm(
            data,
            "death_24h ~ posterior_P1 + posterior_P3 + posterior_P4 + age10 + male + coverage10",
            cluster=cluster,
        )
        beta = float(fit.params["posterior_P3"]) * 0.1
        se = float(fit.bse["posterior_P3"]) * 0.1
        continuous_rows.append(
            {
                "database": database,
                "contrast": "per_0.1_higher_P3_posterior",
                "n": len(data), "events": int(data["death_24h"].sum()),
                "beta": beta, "se": se, "or": math.exp(beta),
                "ci_low": math.exp(beta - 1.96 * se), "ci_high": math.exp(beta + 1.96 * se),
                "p_value": float(2 * norm.sf(abs(beta / se))),
            }
        )

        data["p3_quintile"] = pd.qcut(
            data["posterior_P3"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
        )
        qfit = fit_glm(
            data,
            "death_24h ~ C(p3_quintile, Treatment(reference='Q1')) + age10 + male + coverage10",
            cluster=cluster,
        )
        for quintile in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            scenario = data.copy()
            scenario["p3_quintile"] = pd.Categorical(
                [quintile] * len(scenario), categories=["Q1", "Q2", "Q3", "Q4", "Q5"]
            )
            predicted = qfit.predict(scenario)
            observed = data.loc[data["p3_quintile"].eq(quintile), "death_24h"]
            quintile_rows.append(
                {
                    "database": database,
                    "quintile": quintile,
                    "n": int(data["p3_quintile"].eq(quintile).sum()),
                    "events": int(observed.sum()),
                    "observed_risk": observed.mean(),
                    "standardized_risk": float(predicted.mean()),
                    "posterior_min": data.loc[data["p3_quintile"].eq(quintile), "posterior_P3"].min(),
                    "posterior_max": data.loc[data["p3_quintile"].eq(quintile), "posterior_P3"].max(),
                }
            )

        for threshold in [0.70, 0.80, 0.90]:
            subset = data.loc[data["max_posterior"].ge(threshold)].copy()
            subset = prepare_model_frame(subset, "death_24h")
            cgroup = subset["hospital_id"] if database == "eICU" else None
            hfit = fit_glm(
                subset, "death_24h ~ p1 + p3 + p4 + age10 + male + coverage10", cluster=cgroup
            )
            certainty_rows.append(
                {
                    "database": database, "analysis": "hard_assignment_threshold",
                    "threshold_or_stratum": threshold, "n": len(subset),
                    "events": int(subset["death_24h"].sum()),
                    **model_effect(hfit, "p3"),
                }
            )
        data["entropy_tertile"] = pd.qcut(
            data["posterior_entropy_normalized"].rank(method="first"), 3,
            labels=["low_entropy", "mid_entropy", "high_entropy"],
        )
        for stratum in ["low_entropy", "mid_entropy", "high_entropy"]:
            subset = prepare_model_frame(data.loc[data["entropy_tertile"].eq(stratum)], "death_24h")
            cgroup = subset["hospital_id"] if database == "eICU" else None
            hfit = fit_glm(
                subset, "death_24h ~ p1 + p3 + p4 + age10 + male + coverage10", cluster=cgroup
            )
            certainty_rows.append(
                {
                    "database": database, "analysis": "entropy_tertile",
                    "threshold_or_stratum": stratum, "n": len(subset),
                    "events": int(subset["death_24h"].sum()),
                    **model_effect(hfit, "p3"),
                }
            )

        expanded = []
        for phenotype in PHENOTYPES:
            part = data.copy()
            part["pseudo_p1"] = int(phenotype == "P1")
            part["pseudo_p3"] = int(phenotype == "P3")
            part["pseudo_p4"] = int(phenotype == "P4")
            part["posterior_weight"] = part[f"posterior_{phenotype}"]
            expanded.append(part)
        expanded = pd.concat(expanded, ignore_index=True)
        pseudo_cluster = expanded["hospital_id"] if database == "eICU" else expanded["stay_id"]
        pfit = fit_glm(
            expanded,
            "death_24h ~ pseudo_p1 + pseudo_p3 + pseudo_p4 + age10 + male + coverage10",
            weight=expanded["posterior_weight"], cluster=pseudo_cluster,
        )
        pseudo_rows.append(
            {
                "database": database, "n_original": len(data),
                "events": int(data["death_24h"].sum()),
                **model_effect(pfit, "pseudo_p3"),
            }
        )
    continuous = pd.DataFrame(continuous_rows)
    quintiles = pd.DataFrame(quintile_rows)
    certainty = pd.DataFrame(certainty_rows)
    pseudo = pd.DataFrame(pseudo_rows)
    write_csv(continuous, "07_posterior_probability_models.csv")
    write_csv(quintiles, "08_p3_posterior_risk_gradient.csv")
    write_csv(certainty, "09_certainty_stratified_hard_assignment_models.csv")
    write_csv(pseudo, "09b_posterior_weighted_pseudoclass_models.csv")
    return continuous, quintiles, certainty, pseudo
