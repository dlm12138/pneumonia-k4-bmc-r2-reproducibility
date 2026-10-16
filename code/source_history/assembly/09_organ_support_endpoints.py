"""Actual three-database support and ordinary endpoint extraction.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'Reconstruct post-landmark organ-support escalation outcomes.\n\nPrimary clinical-actionability endpoint:\n  new invasive mechanical ventilation, vasopressor, RRT, or death during\n  (12 h, 36 h] after ICU admission among patients free of all three supports\n  at the 12-hour landmark.\n\nSecondary horizon:\n  the corresponding outcome during (12 h, 60 h].\n\nMIMIC-IV uses interval-timed input/procedure events. MIMIC-III is restricted\nto the MetaVision-era technical replication because comparable interval-timed\nsupport tables are not available for the full CareVue cohort. eICU uses\noffset-timed respiratoryCare, infusionDrug, and treatment records.\n'
import hashlib
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(os.environ['SOURCE_E14A791BB8'])
ASSIGNMENTS = ROOT / '07_common4_selected_assignments.parquet'
STAY_ROOT = Path(os.environ['SOURCE_31F1D1CB18'])
M4 = Path(os.environ['SOURCE_D31F440F3F'])
M3 = Path(os.environ['SOURCE_EB6E6A9325'])
EICU = Path(os.environ['SOURCE_9EC2BBF0E8'])
LANDMARK_HOURS = 12.0
PRIMARY_END_HOURS = 36.0
SECONDARY_END_HOURS = 60.0
LANDMARK_MINUTES = 720.0
PRIMARY_END_MINUTES = 2160.0
SECONDARY_END_MINUTES = 3600.0
M4_VASO = {221289: 'epinephrine', 221662: 'dopamine', 221749: 'phenylephrine', 221906: 'norepinephrine', 222315: 'vasopressin', 229617: 'epinephrine', 229630: 'phenylephrine', 229631: 'phenylephrine', 229632: 'phenylephrine', 229709: 'angiotensin_II', 229764: 'angiotensin_II'}
M4_VENT = {224385: 'intubation', 225792: 'invasive_ventilation'}
M4_RRT = {225441: 'hemodialysis', 225802: 'CRRT', 225803: 'CVVHD', 225805: 'peritoneal_dialysis', 225809: 'CVVHDF', 225955: 'SCUF'}
M3_VASO = {**M4_VASO, 30119: 'epinephrine', 30127: 'phenylephrine', 30043: 'dopamine', 30307: 'dopamine', 30309: 'epinephrine', 30044: 'epinephrine', 30047: 'norepinephrine', 30051: 'vasopressin', 30120: 'norepinephrine', 30128: 'phenylephrine', 42273: 'vasopressin', 42802: 'vasopressin'}
VASO_PATTERN = re.compile('norepinephrine|noradrenaline|levophed|epinephrine|adrenaline|phenylephrine|neo[- ]?synephrine|neosynephrine|vasopressin|dopamine|angiotensin', re.I)
VASO_EXCLUDE_PATTERN = re.compile('racemic|inhal|nebul|nasal', re.I)
INVASIVE_VENT_PATTERN = re.compile('mechanical ventilation|invasive ventilation|endotracheal tube\\|insertion|reintubation', re.I)
NONINVASIVE_PATTERN = re.compile('non[- ]?invasive|bipap|cpap|face mask|nasal mask', re.I)
RRT_PATTERN = re.compile('(^|\\|)dialysis(\\||$)|hemodialysis|haemodialysis|c v v h|c v v h d|c v v h d f|cvvh|cvvhd|cvvhdf|crrt|renal replacement|ultrafiltration|(^|\\|)sled(\\||$)', re.I)
RRT_EXCLUDE_PATTERN = re.compile('insertion of|catheter|access surgery|procedures/radiology|arteriovenous shunt|electrolyte correction', re.I)

def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(ROOT / filename, index=False, encoding='utf-8-sig')

def numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').astype('Int64')

def read_filtered_csv(path: Path, usecols: list[str], id_column: str, eligible_ids: set[int], item_column: str | None=None, item_ids: set[int] | None=None, chunksize: int=500000) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        ids = pd.to_numeric(chunk[id_column], errors='coerce')
        keep = ids.isin(eligible_ids)
        if item_column is not None and item_ids is not None:
            items = pd.to_numeric(chunk[item_column], errors='coerce')
            keep &= items.isin(item_ids)
        selected = chunk.loc[keep].copy()
        if not selected.empty:
            frames.append(selected)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=usecols)

def first_after(events: pd.DataFrame, id_column: str, time_column: str, lower: float) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=float)
    valid = events.loc[pd.to_numeric(events[time_column], errors='coerce').gt(lower)]
    return pd.to_numeric(valid[time_column], errors='coerce').groupby(valid[id_column]).min()

def interval_active(events: pd.DataFrame, id_column: str, start_column: str, end_column: str, landmark: float) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=bool)
    start = pd.to_numeric(events[start_column], errors='coerce')
    end = pd.to_numeric(events[end_column], errors='coerce')
    active = start.le(landmark) & (end.gt(landmark) | end.isna())
    return active.groupby(events[id_column]).any()

def assemble_endpoint(base: pd.DataFrame, id_column: str, active_vaso: pd.Series, active_mv: pd.Series, active_rrt: pd.Series, first_vaso: pd.Series, first_mv: pd.Series, first_rrt: pd.Series, death_time: pd.Series, database: str, support_source_scope: str, time_unit: str) -> pd.DataFrame:
    result = base.copy()
    identifier = result[id_column]
    result['database'] = database
    result['support_source_scope'] = support_source_scope
    result['active_vasopressor_at_12h'] = identifier.map(active_vaso).fillna(False).astype(bool)
    result['active_invasive_ventilation_at_12h'] = identifier.map(active_mv).fillna(False).astype(bool)
    result['active_RRT_at_12h'] = identifier.map(active_rrt).fillna(False).astype(bool)
    result['first_post12_vasopressor_time'] = identifier.map(first_vaso)
    result['first_post12_invasive_ventilation_time'] = identifier.map(first_mv)
    result['first_post12_RRT_time'] = identifier.map(first_rrt)
    result['death_time_from_icu_admission'] = identifier.map(death_time)
    result['event_time_unit'] = time_unit
    if time_unit == 'hours':
        primary_end = PRIMARY_END_HOURS
        secondary_end = SECONDARY_END_HOURS
    elif time_unit == 'minutes':
        primary_end = PRIMARY_END_MINUTES
        secondary_end = SECONDARY_END_MINUTES
    else:
        raise ValueError(time_unit)
    for suffix, horizon_end in [('24h', primary_end), ('48h', secondary_end)]:
        for support, active_column, time_column in [('vasopressor', 'active_vasopressor_at_12h', 'first_post12_vasopressor_time'), ('invasive_ventilation', 'active_invasive_ventilation_at_12h', 'first_post12_invasive_ventilation_time'), ('RRT', 'active_RRT_at_12h', 'first_post12_RRT_time')]:
            event = pd.to_numeric(result[time_column], errors='coerce').le(horizon_end).fillna(False)
            result[f'new_{support}_{suffix}'] = np.where(result[active_column], np.nan, event.astype(int))
        result[f'death_{suffix}'] = pd.to_numeric(result['death_time_from_icu_admission'], errors='coerce').le(horizon_end).fillna(False).astype(int)
        result[f'support_free_at_12h'] = ~(result['active_vasopressor_at_12h'] | result['active_invasive_ventilation_at_12h'] | result['active_RRT_at_12h'])
        new_any = result[f'new_vasopressor_{suffix}'].fillna(0).astype(int) | result[f'new_invasive_ventilation_{suffix}'].fillna(0).astype(int) | result[f'new_RRT_{suffix}'].fillna(0).astype(int)
        result[f'new_any_organ_support_{suffix}'] = np.where(result['support_free_at_12h'], new_any, np.nan)
        result[f'organ_support_or_death_{suffix}'] = np.where(result['support_free_at_12h'], (new_any | result[f'death_{suffix}'].astype(int)).astype(int), np.nan)
    return result

def mimiciv(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = assignments.loc[assignments['database'].eq('MIMIC-IV')].copy()
    selected['stay_id_num'] = numeric_id(selected['stay_id'])
    ids = set(selected['stay_id_num'].dropna().astype(int))
    stays = pd.read_csv(STAY_ROOT / 'mimiciv_pneumonia_stays_raw.csv', usecols=['stay_id', 'subject_id', 'hadm_id', 'intime', 'outtime', 'deathtime'], low_memory=False)
    stays['stay_id_num'] = numeric_id(stays['stay_id'])
    stays = stays.loc[stays['stay_id_num'].isin(ids)].copy()
    for column in ['intime', 'outtime', 'deathtime']:
        stays[column] = pd.to_datetime(stays[column], errors='coerce')
    base = selected.merge(stays[['stay_id_num', 'subject_id', 'hadm_id', 'intime', 'outtime', 'deathtime']], on='stay_id_num', how='left', validate='one_to_one')
    log('Reading MIMIC-IV vasopressor input events')
    vaso = read_filtered_csv(M4 / 'inputevents.csv.gz', ['stay_id', 'starttime', 'endtime', 'itemid', 'statusdescription'], 'stay_id', ids, 'itemid', set(M4_VASO))
    log('Reading MIMIC-IV ventilation and RRT procedure events')
    procedure = read_filtered_csv(M4 / 'procedureevents.csv.gz', ['stay_id', 'starttime', 'endtime', 'itemid', 'statusdescription'], 'stay_id', ids, 'itemid', set(M4_VENT) | set(M4_RRT))
    transformed = []
    for frame in [vaso, procedure]:
        frame['stay_id_num'] = numeric_id(frame['stay_id'])
        frame['starttime'] = pd.to_datetime(frame['starttime'], errors='coerce')
        frame['endtime'] = pd.to_datetime(frame['endtime'], errors='coerce')
        frame.drop(columns=['stay_id'], inplace=True)
        frame = frame.merge(stays[['stay_id_num', 'intime']], on='stay_id_num', how='left')
        frame['start_hour'] = (frame['starttime'] - frame['intime']).dt.total_seconds() / 3600.0
        frame['end_hour'] = (frame['endtime'] - frame['intime']).dt.total_seconds() / 3600.0
        transformed.append(frame)
    vaso, procedure = transformed
    active_vaso = interval_active(vaso, 'stay_id_num', 'start_hour', 'end_hour', LANDMARK_HOURS)
    first_vaso = first_after(vaso, 'stay_id_num', 'start_hour', LANDMARK_HOURS)
    invasive_interval = procedure.loc[pd.to_numeric(procedure['itemid'], errors='coerce').eq(225792)]
    active_mv = interval_active(invasive_interval, 'stay_id_num', 'start_hour', 'end_hour', LANDMARK_HOURS)
    vent_start = procedure.loc[pd.to_numeric(procedure['itemid'], errors='coerce').isin(M4_VENT)]
    first_mv = first_after(vent_start, 'stay_id_num', 'start_hour', LANDMARK_HOURS)
    rrt = procedure.loc[pd.to_numeric(procedure['itemid'], errors='coerce').isin(M4_RRT)]
    active_rrt = interval_active(rrt, 'stay_id_num', 'start_hour', 'end_hour', LANDMARK_HOURS)
    first_rrt = first_after(rrt, 'stay_id_num', 'start_hour', LANDMARK_HOURS)
    death = (base['deathtime'] - base['intime']).dt.total_seconds() / 3600.0
    death.index = base['stay_id_num']
    endpoint = assemble_endpoint(base, 'stay_id_num', active_vaso, active_mv, active_rrt, first_vaso, first_mv, first_rrt, death, 'MIMIC-IV', 'inputevents intervals for vasopressors; procedureevents intervals for invasive ventilation and RRT', 'hours')
    counts = pd.DataFrame([{'database': 'MIMIC-IV', 'source_table': 'inputevents', 'concept': 'vasopressor', 'rows': len(vaso), 'stays': vaso['stay_id_num'].nunique()}, {'database': 'MIMIC-IV', 'source_table': 'procedureevents', 'concept': 'invasive_ventilation', 'rows': len(vent_start), 'stays': vent_start['stay_id_num'].nunique()}, {'database': 'MIMIC-IV', 'source_table': 'procedureevents', 'concept': 'RRT', 'rows': len(rrt), 'stays': rrt['stay_id_num'].nunique()}])
    return (endpoint, counts)

def mimiciii_metavision(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = assignments.loc[assignments['database'].eq('MIMIC-III')].copy()
    selected['stay_id_num'] = numeric_id(selected['stay_id'])
    stays = pd.read_csv(STAY_ROOT / 'mimiciii_pneumonia_stays_raw.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DBSOURCE', 'INTIME', 'OUTTIME', 'DEATHTIME'], low_memory=False)
    stays['stay_id_num'] = numeric_id(stays['ICUSTAY_ID'])
    stays = stays.loc[stays['DBSOURCE'].astype(str).str.lower().isin(['metavision', 'both'])].copy()
    selected = selected.loc[selected['stay_id_num'].isin(stays['stay_id_num'])].copy()
    ids = set(selected['stay_id_num'].dropna().astype(int))
    stays = stays.loc[stays['stay_id_num'].isin(ids)].copy()
    for column in ['INTIME', 'OUTTIME', 'DEATHTIME']:
        stays[column] = pd.to_datetime(stays[column], errors='coerce')
    base = selected.merge(stays[['stay_id_num', 'SUBJECT_ID', 'HADM_ID', 'DBSOURCE', 'INTIME', 'OUTTIME', 'DEATHTIME']], on='stay_id_num', how='left', validate='one_to_one')
    log('Reading MIMIC-III MetaVision vasopressor events')
    vaso = read_filtered_csv(M3 / 'INPUTEVENTS_MV.csv.gz', ['ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'CANCELREASON', 'STATUSDESCRIPTION'], 'ICUSTAY_ID', ids, 'ITEMID', set(M4_VASO))
    log('Reading MIMIC-III MetaVision ventilation and RRT events')
    procedure = read_filtered_csv(M3 / 'PROCEDUREEVENTS_MV.csv.gz', ['ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'CANCELREASON', 'STATUSDESCRIPTION'], 'ICUSTAY_ID', ids, 'ITEMID', set(M4_VENT) | set(M4_RRT))
    transformed = []
    for frame in [vaso, procedure]:
        frame['stay_id_num'] = numeric_id(frame['ICUSTAY_ID'])
        frame['starttime'] = pd.to_datetime(frame['STARTTIME'], errors='coerce')
        frame['endtime'] = pd.to_datetime(frame['ENDTIME'], errors='coerce')
        frame = frame.loc[pd.to_numeric(frame['CANCELREASON'], errors='coerce').fillna(0).eq(0)].copy()
        frame = frame.merge(stays[['stay_id_num', 'INTIME']], on='stay_id_num', how='left')
        frame['start_hour'] = (frame['starttime'] - frame['INTIME']).dt.total_seconds() / 3600.0
        frame['end_hour'] = (frame['endtime'] - frame['INTIME']).dt.total_seconds() / 3600.0
        transformed.append(frame)
    vaso, procedure = transformed
    active_vaso = interval_active(vaso, 'stay_id_num', 'start_hour', 'end_hour', LANDMARK_HOURS)
    first_vaso = first_after(vaso, 'stay_id_num', 'start_hour', LANDMARK_HOURS)
    invasive_interval = procedure.loc[pd.to_numeric(procedure['ITEMID'], errors='coerce').eq(225792)]
    active_mv = interval_active(invasive_interval, 'stay_id_num', 'start_hour', 'end_hour', LANDMARK_HOURS)
    vent_start = procedure.loc[pd.to_numeric(procedure['ITEMID'], errors='coerce').isin(M4_VENT)]
    first_mv = first_after(vent_start, 'stay_id_num', 'start_hour', LANDMARK_HOURS)
    rrt = procedure.loc[pd.to_numeric(procedure['ITEMID'], errors='coerce').isin(M4_RRT)]
    active_rrt = interval_active(rrt, 'stay_id_num', 'start_hour', 'end_hour', LANDMARK_HOURS)
    first_rrt = first_after(rrt, 'stay_id_num', 'start_hour', LANDMARK_HOURS)
    death = (base['DEATHTIME'] - base['INTIME']).dt.total_seconds() / 3600.0
    death.index = base['stay_id_num']
    endpoint = assemble_endpoint(base, 'stay_id_num', active_vaso, active_mv, active_rrt, first_vaso, first_mv, first_rrt, death, 'MIMIC-III MetaVision', 'MetaVision-only interval-timed inputevents/procedureevents; technical historical replication', 'hours')
    counts = pd.DataFrame([{'database': 'MIMIC-III MetaVision', 'source_table': 'INPUTEVENTS_MV', 'concept': 'vasopressor', 'rows': len(vaso), 'stays': vaso['stay_id_num'].nunique()}, {'database': 'MIMIC-III MetaVision', 'source_table': 'PROCEDUREEVENTS_MV', 'concept': 'invasive_ventilation', 'rows': len(vent_start), 'stays': vent_start['stay_id_num'].nunique()}, {'database': 'MIMIC-III MetaVision', 'source_table': 'PROCEDUREEVENTS_MV', 'concept': 'RRT', 'rows': len(rrt), 'stays': rrt['stay_id_num'].nunique()}])
    return (endpoint, counts)

def eicu(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = assignments.loc[assignments['database'].eq('eICU')].copy()
    selected['stay_id_num'] = numeric_id(selected['stay_id'])
    ids = set(selected['stay_id_num'].dropna().astype(int))
    patient = pd.read_csv(EICU / 'patient.csv.gz', usecols=['patientunitstayid', 'hospitalid', 'unitdischargeoffset', 'unitdischargestatus', 'hospitaldischargeoffset', 'hospitaldischargestatus'], low_memory=False)
    patient['stay_id_num'] = numeric_id(patient['patientunitstayid'])
    patient = patient.loc[patient['stay_id_num'].isin(ids)].copy()
    base = selected.merge(patient[['stay_id_num', 'unitdischargeoffset', 'unitdischargestatus', 'hospitaldischargeoffset', 'hospitaldischargestatus']], on='stay_id_num', how='left', validate='one_to_one')
    log('Reading eICU vasopressor infusion records')
    infusion = read_filtered_csv(EICU / 'infusionDrug.csv.gz', ['patientunitstayid', 'infusionoffset', 'drugname', 'drugrate', 'infusionrate'], 'patientunitstayid', ids)
    name = infusion['drugname'].astype(str)
    positive_rate = pd.to_numeric(infusion['drugrate'], errors='coerce').gt(0) | pd.to_numeric(infusion['infusionrate'], errors='coerce').gt(0)
    infusion = infusion.loc[name.str.contains(VASO_PATTERN, na=False) & ~name.str.contains(VASO_EXCLUDE_PATTERN, na=False) & positive_rate].copy()
    infusion['stay_id_num'] = numeric_id(infusion['patientunitstayid'])
    infusion['event_minute'] = pd.to_numeric(infusion['infusionoffset'], errors='coerce')
    log('Reading eICU respiratoryCare records')
    respiratory = read_filtered_csv(EICU / 'respiratoryCare.csv.gz', ['patientunitstayid', 'respcarestatusoffset', 'airwaytype', 'ventstartoffset', 'ventendoffset', 'priorventstartoffset', 'priorventendoffset'], 'patientunitstayid', ids)
    respiratory['stay_id_num'] = numeric_id(respiratory['patientunitstayid'])
    respiratory['ventstartoffset'] = pd.to_numeric(respiratory['ventstartoffset'], errors='coerce')
    respiratory['ventendoffset'] = pd.to_numeric(respiratory['ventendoffset'], errors='coerce')
    respiratory_start = respiratory.loc[respiratory['ventstartoffset'].notna() & respiratory['ventstartoffset'].ge(0)].copy()
    respiratory_start['event_minute'] = respiratory_start['ventstartoffset']
    log('Reading eICU treatment records for invasive ventilation and RRT')
    treatment = read_filtered_csv(EICU / 'treatment.csv.gz', ['patientunitstayid', 'treatmentoffset', 'treatmentstring'], 'patientunitstayid', ids)
    text = treatment['treatmentstring'].astype(str)
    invasive = treatment.loc[text.str.contains(INVASIVE_VENT_PATTERN, na=False) & ~text.str.contains(NONINVASIVE_PATTERN, na=False)].copy()
    invasive['stay_id_num'] = numeric_id(invasive['patientunitstayid'])
    invasive['event_minute'] = pd.to_numeric(invasive['treatmentoffset'], errors='coerce')
    rrt = treatment.loc[text.str.contains(RRT_PATTERN, na=False) & ~text.str.contains(RRT_EXCLUDE_PATTERN, na=False)].copy()
    rrt['stay_id_num'] = numeric_id(rrt['patientunitstayid'])
    rrt['event_minute'] = pd.to_numeric(rrt['treatmentoffset'], errors='coerce')
    pre6 = infusion['event_minute'].gt(LANDMARK_MINUTES - 360) & infusion['event_minute'].le(LANDMARK_MINUTES)
    active_vaso = pre6.groupby(infusion['stay_id_num']).any()
    first_vaso = first_after(infusion, 'stay_id_num', 'event_minute', LANDMARK_MINUTES)
    respiratory_active = respiratory_start['event_minute'].le(LANDMARK_MINUTES) & (respiratory_start['ventendoffset'].eq(0) | respiratory_start['ventendoffset'].gt(LANDMARK_MINUTES) | respiratory_start['ventendoffset'].isna())
    active_mv_resp = respiratory_active.groupby(respiratory_start['stay_id_num']).any()
    active_mv_treatment = invasive['event_minute'].le(LANDMARK_MINUTES).groupby(invasive['stay_id_num']).any()
    active_mv = pd.concat([active_mv_resp.rename('resp'), active_mv_treatment.rename('treatment')], axis=1).fillna(False).any(axis=1)
    vent_events = pd.concat([respiratory_start[['stay_id_num', 'event_minute']], invasive[['stay_id_num', 'event_minute']]], ignore_index=True)
    first_mv = first_after(vent_events, 'stay_id_num', 'event_minute', LANDMARK_MINUTES)
    active_rrt = rrt['event_minute'].le(LANDMARK_MINUTES).groupby(rrt['stay_id_num']).any()
    first_rrt = first_after(rrt, 'stay_id_num', 'event_minute', LANDMARK_MINUTES)
    hospital_death = base['hospitaldischargestatus'].astype(str).str.contains('expired|death|dead', case=False, regex=True, na=False)
    death_values = np.where(hospital_death, pd.to_numeric(base['hospitaldischargeoffset'], errors='coerce'), np.nan)
    death = pd.Series(death_values, index=base['stay_id_num'])
    endpoint = assemble_endpoint(base, 'stay_id_num', active_vaso, active_mv, active_rrt, first_vaso, first_mv, first_rrt, death, 'eICU', 'infusionDrug positive-rate records; respiratoryCare vent starts plus invasive treatment strings; treatment-based RRT', 'minutes')
    counts = pd.DataFrame([{'database': 'eICU', 'source_table': 'infusionDrug', 'concept': 'vasopressor', 'rows': len(infusion), 'stays': infusion['stay_id_num'].nunique()}, {'database': 'eICU', 'source_table': 'respiratoryCare+treatment', 'concept': 'invasive_ventilation', 'rows': len(vent_events), 'stays': vent_events['stay_id_num'].nunique()}, {'database': 'eICU', 'source_table': 'treatment', 'concept': 'RRT', 'rows': len(rrt), 'stays': rrt['stay_id_num'].nunique()}])
    return (endpoint, counts)

def endpoint_summary(endpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for database, frame in endpoints.groupby('database', sort=False):
        row = {'database': database, 'assigned_landmark_n': int(len(frame)), 'active_vasopressor_at_12h_n': int(frame['active_vasopressor_at_12h'].sum()), 'active_invasive_ventilation_at_12h_n': int(frame['active_invasive_ventilation_at_12h'].sum()), 'active_RRT_at_12h_n': int(frame['active_RRT_at_12h'].sum()), 'support_free_at_12h_n': int(frame['support_free_at_12h'].sum())}
        for suffix in ['24h', '48h']:
            for column in [f'new_vasopressor_{suffix}', f'new_invasive_ventilation_{suffix}', f'new_RRT_{suffix}', f'death_{suffix}', f'organ_support_or_death_{suffix}']:
                available = frame[column].notna()
                events = pd.to_numeric(frame.loc[available, column], errors='coerce').sum()
                row[f'{column}_model_n'] = int(available.sum())
                row[f'{column}_events'] = int(events)
                row[f'{column}_percent'] = float(events / available.sum() * 100.0) if available.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def main() -> None:
    if not ASSIGNMENTS.exists():
        raise FileNotFoundError(f'Selected common-four-vital assignments not found: {ASSIGNMENTS}')
    assignments = pd.read_parquet(ASSIGNMENTS)
    m4_endpoint, m4_counts = mimiciv(assignments)
    m3_endpoint, m3_counts = mimiciii_metavision(assignments)
    e_endpoint, e_counts = eicu(assignments)
    endpoints = pd.concat([m4_endpoint, m3_endpoint, e_endpoint], ignore_index=True, sort=False)
    endpoints.to_parquet(ROOT / '13_organ_support_escalation_patient_level.parquet', index=False)
    write_csv(endpoints, '13_organ_support_escalation_patient_level.csv')
    write_csv(pd.concat([m4_counts, m3_counts, e_counts], ignore_index=True), '14_organ_support_event_source_counts.csv')
    summary = endpoint_summary(endpoints)
    write_csv(summary, '15_organ_support_endpoint_flow.csv')
    dictionary = pd.DataFrame([{'endpoint': 'primary clinical-actionability composite', 'time_zero': '12 hours after ICU admission', 'risk_window': '(12h, 36h]', 'risk_set': 'alive and in ICU at 12h; free of invasive ventilation, vasopressor and RRT at 12h', 'event': 'new invasive ventilation, new vasopressor, new RRT, or death', 'causal_interpretation': False}, {'endpoint': 'secondary clinical-actionability composite', 'time_zero': '12 hours after ICU admission', 'risk_window': '(12h, 60h]', 'risk_set': 'alive and in ICU at 12h; free of invasive ventilation, vasopressor and RRT at 12h', 'event': 'new invasive ventilation, new vasopressor, new RRT, or death', 'causal_interpretation': False}, {'endpoint': 'component-specific escalation', 'time_zero': '12 hours after ICU admission', 'risk_window': '(12h, 36h] and (12h, 60h]', 'risk_set': 'free of the corresponding support modality at 12h', 'event': 'first post-landmark record of that support modality', 'causal_interpretation': False}])
    write_csv(dictionary, '16_organ_support_endpoint_dictionary.csv')
    audit_rows = []
    for database, frame in endpoints.groupby('database', sort=False):
        for column in ['first_post12_vasopressor_time', 'first_post12_invasive_ventilation_time', 'first_post12_RRT_time']:
            values = pd.to_numeric(frame[column], errors='coerce').dropna()
            lower = LANDMARK_HOURS if frame['event_time_unit'].iloc[0] == 'hours' else LANDMARK_MINUTES
            audit_rows.append({'database': database, 'check': f'{column}_strictly_after_landmark', 'n_with_event_time': int(len(values)), 'minimum_event_time': float(values.min()) if len(values) else np.nan, 'threshold': lower, 'pass': bool(values.gt(lower).all())})
        audit_rows.append({'database': database, 'check': 'composite_defined_only_in_support_free_risk_set', 'n_with_event_time': int(frame['organ_support_or_death_24h'].notna().sum()), 'minimum_event_time': np.nan, 'threshold': int(frame['support_free_at_12h'].sum()), 'pass': bool(frame['organ_support_or_death_24h'].notna().sum() == frame['support_free_at_12h'].sum())})
    write_csv(pd.DataFrame(audit_rows), '17_organ_support_temporal_audit.csv')
    manifest_rows = []
    for path in sorted(ROOT.glob('1[3-7]_*')):
        if path.is_file():
            manifest_rows.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(pd.DataFrame(manifest_rows), '18_organ_support_manifest.csv')
    print(summary.to_string(index=False))
if __name__ == '__source_reference_run__':
    main()
