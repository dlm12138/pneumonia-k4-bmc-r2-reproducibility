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

def log(message):
    print(message, flush=True)

RANGES = {
    "SBP": (40, 300),
    "DBP": (20, 200),
    "MAP": (30, 250),
    "HR": (20, 250),
    "RR": (3, 80),
    "TEMP": (25, 45),
    "SpO2": (50, 100),
}


EICU_FIELD_MAP = {
    "vitalPeriodic": {
        "SBP": ["systemicsystolic"],
        "DBP": ["systemicdiastolic"],
        "MAP": ["systemicmean"],
        "HR": ["heartrate"],
        "RR": ["respiration"],
        "TEMP": ["temperature"],
        "SpO2": ["sao2"],
    },
    "vitalAperiodic": {
        "SBP": ["noninvasivesystolic"],
        "DBP": ["noninvasivediastolic"],
        "MAP": ["noninvasivemean"],
    },
}


def source_parts(path):
    if str(path).startswith("zip://"):
        rest = str(path)[len("zip://") :]
        z, member = rest.split("!", 1)
        return Path(z), member
    return Path(path), None


def open_source(path):
    p, member = source_parts(path)
    if member:
        zf = zipfile.ZipFile(p)
        raw = zf.open(member)
        if member.lower().endswith(".gz"):
            return gzip.open(raw, "rt", encoding="utf-8", errors="replace", newline=""), zf
        return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""), zf
    if str(p).lower().endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace", newline=""), None
    return open(p, "r", encoding="utf-8", errors="replace", newline=""), None


def available_columns(path):
    stream, holder = open_source(path)
    try:
        header = stream.readline().strip("\n\r")
        return next(csv.reader([header])) if header else []
    finally:
        stream.close()
        if holder:
            holder.close()


def actual_usecols(path, usecols):
    if not usecols:
        return None
    cols = available_columns(path)
    lower_map = {str(c).strip().lower(): c for c in cols}
    return [lower_map[c.lower()] for c in usecols if c.lower() in lower_map]


def read_csv_chunks(path, usecols=None, chunksize=600000):
    actual = actual_usecols(path, usecols)
    stream, holder = open_source(path)
    try:
        for chunk in pd.read_csv(stream, usecols=actual, dtype=str, chunksize=chunksize, low_memory=False):
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            yield chunk
    finally:
        stream.close()
        if holder:
            holder.close()


def read_csv_full(path, usecols=None):
    actual = actual_usecols(path, usecols)
    stream, holder = open_source(path)
    try:
        df = pd.read_csv(stream, usecols=actual, dtype=str, low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    finally:
        stream.close()
        if holder:
            holder.close()


def label_to_var(label):
    low = str(label).lower().strip()
    if not low:
        return None
    excluded_fragments = [
        "alarm",
        "set ",
        " setting",
        "resistance",
        "vital capacity",
        "temporary",
        "chromium",
        "ingr",
        "diet",
        "nutrition",
        "minute volume",
        "pulmonary artery",
        "central venous",
        "intra aortic",
    ]
    if any(x in low for x in excluded_fragments):
        return None
    if "o2 saturation" in low or "oxygen saturation" in low or "spo2" in low or "sao2" in low:
        return "SpO2"
    if "temperature" in low or low == "temp" or low.startswith("temp "):
        return "TEMP"
    if "respiratory rate" in low or low == "rr" or low == "respiration":
        return "RR"
    if "heart rate" in low or low == "hr" or low == "pulse":
        return "HR"
    if "mean arterial pressure" in low or "blood pressure mean" in low or "nibp mean" in low or low == "map":
        return "MAP"
    if "blood pressure systolic" in low or "systolic blood pressure" in low or "nibp systolic" in low or low == "sbp":
        return "SBP"
    if "blood pressure diastolic" in low or "diastolic blood pressure" in low or "nibp diastolic" in low or low == "dbp":
        return "DBP"
    return None


def clean_value(var, value, uom=""):
    raw = pd.to_numeric(value, errors="coerce")
    if pd.isna(raw):
        return None, False, False
    val = float(raw)
    converted = False
    unit = str(uom).lower()
    if var == "TEMP" and ("f" in unit or val > 60):
        val = (val - 32) * 5 / 9
        converted = True
    lo, hi = RANGES[var]
    if val < lo or val > hi:
        return None, True, converted
    return val, False, converted


def add_agg(agg, db, stay_id, hour, var, val, source_table, source_item):
    key = (db, str(stay_id), int(hour), var)
    rec = agg[key]
    rec["sum"] += val
    rec["n"] += 1
    rec["source_tables"][source_table] += 1
    rec["source_items"][str(source_item)] += 1


def scan_mimic_items(db, d_items_path):
    items = read_csv_full(d_items_path)
    unit_col = "unitname" if "unitname" in items.columns else None
    mapping = {}
    rows = []
    for _, row in items.iterrows():
        itemid = str(row.get("itemid", ""))
        label = str(row.get("label", ""))
        var = label_to_var(label)
        if var:
            mapping[itemid] = {
                "variable": var,
                "label": label,
                "unit": row.get(unit_col, "") if unit_col else "",
            }
        rows.append({
            "database": db,
            "table": "chartevents/d_items",
            "variable": var or "",
            "itemid_or_field": itemid,
            "label": label,
            "unit": row.get(unit_col, "") if unit_col else "",
            "included_in_step03": "yes" if var else "no",
        })
    return mapping, rows


def scan_mimic_vitals(db, paths, candidates, item_map):
    id_col = "stay_id" if db == "MIMIC-IV" else "icustay_id"
    cand = candidates[["subject_id", "hadm_id", id_col, "intime_dt"]].copy()
    cand[id_col] = cand[id_col].astype(str)
    intime_map = dict(zip(cand[id_col], cand["intime_dt"]))
    stay_set = set(intime_map)
    item_set = set(item_map)
    agg = defaultdict(lambda: {"sum": 0.0, "n": 0, "source_tables": Counter(), "source_items": Counter()})
    audit = {v: {"raw": 0, "valid": 0, "abnormal": 0, "converted": 0, "stays": set()} for v in VARS}
    item_counts = Counter()
    item_stays = defaultdict(set)
    usecols = [id_col, "itemid", "charttime", "valuenum", "valueuom"]
    log(f"{db}: 分块扫描 chartevents；候选 stays={len(stay_set)}，生命体征 itemids={len(item_set)}")
    for idx, chunk in enumerate(read_csv_chunks(paths["chartevents"]["path"], usecols=usecols, chunksize=600000), 1):
        if idx % 20 == 0:
            log(f"{db}: chartevents 已处理 {idx} 个分块")
        chunk[id_col] = chunk[id_col].astype(str)
        chunk["itemid"] = chunk["itemid"].astype(str)
        sub = chunk[chunk[id_col].isin(stay_set) & chunk["itemid"].isin(item_set)].copy()
        if sub.empty:
            continue
        sub["chart_dt"] = pd.to_datetime(sub["charttime"], errors="coerce")
        sub["intime_dt"] = sub[id_col].map(intime_map)
        sub["hours"] = (sub["chart_dt"] - sub["intime_dt"]).dt.total_seconds() / 3600
        sub = sub[(sub["hours"] >= 0) & (sub["hours"] < 12)]
        if sub.empty:
            continue
        sub["hour"] = sub["hours"].astype(int)
        for row in sub.itertuples(index=False):
            itemid = str(getattr(row, "itemid"))
            meta = item_map[itemid]
            var = meta["variable"]
            stay_id = str(getattr(row, id_col))
            audit[var]["raw"] += 1
            item_counts[(var, itemid)] += 1
            item_stays[(var, itemid)].add(stay_id)
            val, abnormal, converted = clean_value(var, getattr(row, "valuenum"), getattr(row, "valueuom", ""))
            if converted:
                audit[var]["converted"] += 1
            if abnormal:
                audit[var]["abnormal"] += 1
            if val is None:
                continue
            audit[var]["valid"] += 1
            audit[var]["stays"].add(stay_id)
            add_agg(agg, db, stay_id, getattr(row, "hour"), var, val, "chartevents", itemid)
    return agg, audit, item_counts, item_stays


def scan_eicu_vitals(paths, candidates):
    cand = candidates[["patientunitstayid"]].copy()
    cand["patientunitstayid"] = cand["patientunitstayid"].astype(str)
    stay_set = set(cand["patientunitstayid"])
    agg = defaultdict(lambda: {"sum": 0.0, "n": 0, "source_tables": Counter(), "source_items": Counter()})
    audit = {v: {"raw": 0, "valid": 0, "abnormal": 0, "converted": 0, "stays": set()} for v in VARS}
    field_counts = Counter()
    field_stays = defaultdict(set)
    for table, varmap in EICU_FIELD_MAP.items():
        path = paths[table]["path"]
        fields = ["patientunitstayid", "observationoffset"]
        for fs in varmap.values():
            fields.extend(fs)
        fields = sorted(set(fields))
        log(f"eICU: 分块扫描 {table}")
        for idx, chunk in enumerate(read_csv_chunks(path, usecols=fields, chunksize=600000), 1):
            if idx % 20 == 0:
                log(f"eICU: {table} 已处理 {idx} 个分块")
            chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(str)
            sub = chunk[chunk["patientunitstayid"].isin(stay_set)].copy()
            if sub.empty:
                continue
            sub["offset_min"] = pd.to_numeric(sub["observationoffset"], errors="coerce")
            sub = sub[(sub["offset_min"] >= 0) & (sub["offset_min"] < 720)]
            if sub.empty:
                continue
            sub["hour"] = (sub["offset_min"] // 60).astype(int)
            for var, fs in varmap.items():
                for field in fs:
                    if field not in sub.columns:
                        continue
                    nonmiss = sub[sub[field].notna()]
                    if nonmiss.empty:
                        continue
                    for row in nonmiss.itertuples(index=False):
                        stay_id = str(getattr(row, "patientunitstayid"))
                        audit[var]["raw"] += 1
                        field_counts[(var, field)] += 1
                        field_stays[(var, field)].add(stay_id)
                        val, abnormal, converted = clean_value(var, getattr(row, field), "C" if var == "TEMP" else "")
                        if converted:
                            audit[var]["converted"] += 1
                        if abnormal:
                            audit[var]["abnormal"] += 1
                        if val is None:
                            continue
                        audit[var]["valid"] += 1
                        audit[var]["stays"].add(stay_id)
                        add_agg(agg, "eICU", stay_id, getattr(row, "hour"), var, val, table, field)
    return agg, audit, field_counts, field_stays


def normalize_candidate_frame(db, final, id_col):
    frame = final.copy()
    if db == "eICU":
        out = pd.DataFrame({
            "database": db,
            "subject_id_or_patientunitstayid": frame["patientunitstayid"].astype(str),
            "hadm_id": "",
            "stay_id_or_icustay_id": frame["patientunitstayid"].astype(str),
            "intime": frame.get("unitadmittime24", ""),
            "outtime": frame.get("unitdischargetime24", ""),
            "age": frame.get("age_num", ""),
            "sex": frame.get("gender", ""),
            "hospital_mortality": frame.get("hospital_mortality", ""),
            "mortality_28d_available": False,
            "mortality_28d": "",
            "icu_los_hours": pd.to_numeric(frame.get("unitdischargeoffset", ""), errors="coerce") / 60,
        })
    else:
        out = pd.DataFrame({
            "database": db,
            "subject_id_or_patientunitstayid": frame["subject_id"].astype(str),
            "hadm_id": frame["hadm_id"].astype(str),
            "stay_id_or_icustay_id": frame[id_col].astype(str),
            "intime": frame.get("intime", ""),
            "outtime": frame.get("outtime", ""),
            "age": frame.get("age", ""),
            "sex": frame.get("gender", ""),
            "hospital_mortality": frame.get("hospital_mortality", ""),
            "mortality_28d_available": frame.get("mortality_28d_available", True),
            "mortality_28d": frame.get("mortality_28d", ""),
            "icu_los_hours": pd.to_numeric(frame.get("los_days", ""), errors="coerce") * 24,
        })
    return out


def build_hourly_tables(candidates_all, agg):
    meta = candidates_all.set_index(["database", "stay_id_or_icustay_id"]).to_dict("index")
    rows = []
    for (db, stay_id, hour, var), rec in agg.items():
        if rec["n"] <= 0:
            continue
        m = meta.get((db, stay_id), {})
        intime = m.get("intime", "")
        if db == "eICU":
            hour_start = f"offset_{hour * 60}_min"
            hour_end = f"offset_{(hour + 1) * 60}_min"
        else:
            dt = pd.to_datetime(intime, errors="coerce")
            hour_start_dt = dt + pd.Timedelta(hours=hour) if pd.notna(dt) else pd.NaT
            hour_end_dt = dt + pd.Timedelta(hours=hour + 1) if pd.notna(dt) else pd.NaT
            hour_start = hour_start_dt.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(hour_start_dt) else ""
            hour_end = hour_end_dt.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(hour_end_dt) else ""
        rows.append({
            "database": db,
            "subject_id_or_patientunitstayid": m.get("subject_id_or_patientunitstayid", ""),
            "hadm_id": m.get("hadm_id", ""),
            "stay_id_or_icustay_id": stay_id,
            "intime": intime,
            "hour_index": int(hour),
            "hour_start": hour_start,
            "hour_end": hour_end,
            "variable": var,
            "value": rec["sum"] / rec["n"],
            "n_raw_records": rec["n"],
            "source_table": "|".join(sorted(rec["source_tables"].keys())),
            "source_itemid_or_field": "|".join(sorted(rec["source_items"].keys())),
        })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("没有生成任何小时级生命体征记录")
    long_df = long_df.sort_values(["database", "stay_id_or_icustay_id", "hour_index", "variable"])

    value_wide = long_df.pivot_table(
        index=["database", "subject_id_or_patientunitstayid", "hadm_id", "stay_id_or_icustay_id", "intime", "hour_index"],
        columns="variable",
        values="value",
        aggfunc="first",
    ).reset_index()
    n_wide = long_df.pivot_table(
        index=["database", "subject_id_or_patientunitstayid", "hadm_id", "stay_id_or_icustay_id", "intime", "hour_index"],
        columns="variable",
        values="n_raw_records",
        aggfunc="first",
    ).reset_index()
    n_wide = n_wide.rename(columns={v: f"{v}_n" for v in VARS if v in n_wide.columns})
    wide_df = candidates_all[["database", "subject_id_or_patientunitstayid", "hadm_id", "stay_id_or_icustay_id", "intime"]].merge(
        pd.DataFrame({"hour_index": range(12)}), how="cross"
    )
    wide_df = wide_df.merge(value_wide, on=["database", "subject_id_or_patientunitstayid", "hadm_id", "stay_id_or_icustay_id", "intime", "hour_index"], how="left")
    wide_df = wide_df.merge(n_wide, on=["database", "subject_id_or_patientunitstayid", "hadm_id", "stay_id_or_icustay_id", "intime", "hour_index"], how="left")
    for var in VARS:
        if var not in wide_df.columns:
            wide_df[var] = pd.NA
        ncol = f"{var}_n"
        if ncol not in wide_df.columns:
            wide_df[ncol] = pd.NA
    return long_df, wide_df


def build_patient_matrix(candidates_all, wide_df):
    matrix = candidates_all.copy()
    base_cols = ["database", "subject_id_or_patientunitstayid", "hadm_id", "stay_id_or_icustay_id", "intime"]
    for var in VARS:
        p = wide_df.pivot_table(index=base_cols, columns="hour_index", values=var, aggfunc="first")
        p = p.rename(columns={h: f"{var}_h{int(h)}" for h in p.columns}).reset_index()
        matrix = matrix.merge(p, on=base_cols, how="left")
    ordered = [
        "database", "subject_id_or_patientunitstayid", "hadm_id", "stay_id_or_icustay_id",
        "intime", "outtime", "age", "sex", "hospital_mortality", "mortality_28d_available",
        "mortality_28d", "icu_los_hours",
    ]
    value_cols = [f"{var}_h{h}" for var in VARS for h in range(12)]
    for c in value_cols:
        if c not in matrix.columns:
            matrix[c] = pd.NA
    return matrix[ordered + value_cols]

# The published extractor is restricted to the four phenotype-defining vitals.
# Never use the historical generic oxygen label matcher for SpO2.
VARS = ['SBP', 'DBP', 'HR', 'RR']
EICU_FIELD_MAP = {table: {var: fields for var, fields in mapping.items() if var in VARS} for table, mapping in EICU_FIELD_MAP.items()}
