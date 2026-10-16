"""Original layout functions consumed by current figure producer; historical numerical anchors are superseded by that producer, not current results.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
from __future__ import annotations
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
from pathlib import Path
import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps
ROOT = Path(os.environ['SOURCE_2E845E495A'])
MAIN = ROOT / 'analysis_main12_clinical_actionability_20260730'
DEEP = ROOT / 'analysis_k4_evidence_deepening_20260801'
LANDMARK = ROOT / 'analysis_landmark_external_upgrade_20260730'
OUT = Path(os.environ['OUTPUT_ROOT'])
DPI = 600
ASSIGNMENTS = MAIN / '07_common4_selected_assignments.parquet'
MATRIX = LANDMARK / '14_landmark12_candidate_matrix.parquet'
OUTCOME_MODELS = MAIN / '19_k4_adjusted_outcome_models.csv'
ABSOLUTE_RISKS = MAIN / '20_k4_standardized_absolute_risks.csv'
RISK_GRADIENT = DEEP / '08_p3_posterior_risk_gradient.csv'
CERTAINTY_MODELS = DEEP / '09_certainty_stratified_hard_assignment_models.csv'
POSTERIOR_WEIGHTED = DEEP / '09b_posterior_weighted_pseudoclass_models.csv'
HOSPITAL_SUMMARY = DEEP / '10_eicu_hospital_transport_summary.csv'
HIERARCHICAL = DEEP / '11_eicu_hierarchical_models.csv'
LOHO = DEEP / '12_eicu_leave_one_hospital_out.csv'
SITE_FEASIBILITY = DEEP / '13_eicu_site_effect_feasibility.csv'
COLORS = {'P1': '#2767B1', 'P2': '#7446A8', 'P3': '#C9343A', 'P4': '#158064', 'navy': '#17365D', 'steel': '#607D9B', 'light_blue': '#EAF1F7', 'light_gray': '#F4F6F8', 'border': '#AAB8C5', 'grid': '#DDE3E8', 'text': '#17212B', 'muted': '#5D6975'}
DB_ORDER = ['MIMIC-IV', 'MIMIC-III', 'eICU']
DB_COLORS = {'MIMIC-IV': '#17365D', 'MIMIC-III': '#6A89A6', 'eICU': '#C9343A'}
DB_MARKERS = {'MIMIC-IV': 'o', 'MIMIC-III': 's', 'eICU': 'D'}
PHENO_ORDER = ['P1', 'P2', 'P3', 'P4']
EXPECTED_COUNTS = {'MIMIC-IV': {'P1': 4593, 'P2': 4420, 'P3': 2503, 'P4': 3337}, 'MIMIC-III': {'P1': 1083, 'P2': 1249, 'P3': 520, 'P4': 840}, 'eICU': {'P1': 3865, 'P2': 8179, 'P3': 839, 'P4': 2229}}

def configure_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 7.0, 'axes.titlesize': 7.8, 'axes.labelsize': 7.0, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'axes.linewidth': 0.65, 'legend.fontsize': 6.5, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding='utf-8-sig')

def stamp_png(path: Path) -> None:
    with Image.open(path) as image:
        if image.mode in {'RGBA', 'LA'} or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            rgb = Image.new('RGB', rgba.size, 'white')
            rgb.paste(rgba, mask=rgba.getchannel('A'))
        else:
            rgb = image.convert('RGB')
        rgb = ImageOps.expand(rgb, border=24, fill='white')
        rgb.save(path, 'PNG', dpi=(DPI, DPI), optimize=True)

def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor='white', bbox_inches=None, pad_inches=0)
    plt.close(fig)
    stamp_png(path)

def style_axis(ax: plt.Axes, grid_axis: str='y') -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.tick_params(length=2.5, width=0.6, color=COLORS['muted'])
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS['grid'], linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)

def draw_figure2(summary: pd.DataFrame, totals: dict[str, int]) -> Path:
    output = OUT / 'Figure2_K4_multitimepoint_vital_profiles.png'
    fig = plt.figure(figsize=(7.2, 6.45), dpi=DPI)
    fig.text(0.025, 0.985, '0-12 h multitimepoint vital-sign profiles', ha='left', va='top', fontsize=11.5, fontweight='bold', color=COLORS['text'])
    fig.text(0.025, 0.956, 'MIMIC-IV development and frozen K4 assignments in replication and transport cohorts', ha='left', va='top', fontsize=7.3, color=COLORS['muted'])
    legend_labels = {'P1': 'P1  lower-pressure', 'P2': 'P2  low-rate reference', 'P3': 'P3  tachycardic-tachypneic', 'P4': 'P4  higher-pressure'}
    handles = [Line2D([0], [0], color=COLORS[p], lw=2.0, marker='o', ms=3.4, label=legend_labels[p]) for p in PHENO_ORDER]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.6, 0.925), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.25)
    grid = fig.add_gridspec(3, 4, left=0.135, right=0.985, bottom=0.1, top=0.845, hspace=0.34, wspace=0.28)
    variables = ['SBP', 'DBP', 'HR', 'RR']
    titles = {'SBP': 'Systolic blood pressure\n(mmHg)', 'DBP': 'Diastolic blood pressure\n(mmHg)', 'HR': 'Heart rate\n(beats/min)', 'RR': 'Respiratory rate\n(breaths/min)'}
    row_labels = {'MIMIC-IV': 'MIMIC-IV\nDevelopment', 'MIMIC-III': 'MIMIC-III\nSame-center\nhistorical\nreplication', 'eICU': 'eICU\nExternal\ntransport'}
    y_limits: dict[str, tuple[float, float]] = {}
    for variable in variables:
        subset = summary[summary['variable'] == variable]
        low = float(subset['ci_low'].min())
        high = float(subset['ci_high'].max())
        padding = max((high - low) * 0.08, 0.5)
        y_limits[variable] = (low - padding, high + padding)
    for row, database in enumerate(DB_ORDER):
        center_y = 0.715 - row * 0.253
        fig.text(0.083, center_y, f'{row_labels[database]}\nn = {totals[database]:,}', ha='right', va='center', fontsize=6.6, fontweight='bold', color=COLORS['text'], linespacing=1.25)
        for column, variable in enumerate(variables):
            ax = fig.add_subplot(grid[row, column])
            panel = summary[(summary['database'] == database) & (summary['variable'] == variable)]
            for phenotype in PHENO_ORDER:
                values = panel[panel['phenotype'] == phenotype].sort_values('hour')
                x = values['hour'].to_numpy(dtype=float)
                mean = values['mean'].to_numpy(dtype=float)
                low = values['ci_low'].to_numpy(dtype=float)
                high = values['ci_high'].to_numpy(dtype=float)
                ax.fill_between(x, low, high, color=COLORS[phenotype], alpha=0.11, linewidth=0)
                ax.plot(x, mean, color=COLORS[phenotype], linewidth=1.55 if phenotype in {'P3', 'P4'} else 1.35, marker='o', markersize=1.8, markeredgewidth=0)
            ax.set_xlim(-0.2, 11.2)
            ax.set_ylim(*y_limits[variable])
            ax.set_xticks([0, 3, 6, 9, 11])
            if row == 2:
                ax.set_xlabel('Hours after ICU admission')
            else:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(titles[variable], pad=5, fontweight='bold', color=COLORS['text'])
            style_axis(ax, 'y')
    fig.text(0.135, 0.012, 'Lines show observed hourly means; shaded bands show 95% confidence intervals.\nMIMIC-III and eICU were assigned with the frozen MIMIC-IV K4 model; no refitting was performed.', ha='left', va='bottom', fontsize=5.6, color=COLORS['muted'], linespacing=1.2)
    save_figure(fig, output)
    return output

def model_row(models: pd.DataFrame, database: str, outcome: str, phenotype: str) -> pd.Series:
    selected = models[(models['database'] == database) & (models['outcome'] == outcome) & (models['phenotype'] == phenotype)]
    if len(selected) != 1:
        raise RuntimeError(f'Expected one model row: {database}, {outcome}, {phenotype}; got {len(selected)}')
    return selected.iloc[0]

def absolute_risk_row(risks: pd.DataFrame, database: str, outcome: str, phenotype: str='P3') -> pd.Series:
    selected = risks[(risks['database'] == database) & (risks['outcome'] == outcome) & (risks['phenotype'] == phenotype)]
    if len(selected) != 1:
        raise RuntimeError(f'Expected one risk row: {database}, {outcome}, {phenotype}; got {len(selected)}')
    return selected.iloc[0]

def draw_figure3() -> Path:
    output = OUT / 'Figure3_K4_cross_cohort_outcome_associations.png'
    models = read_csv(OUTCOME_MODELS)
    risks = read_csv(ABSOLUTE_RISKS)
    if set(models['K'].dropna().astype(int)) != {4}:
        raise RuntimeError('Outcome models are not K4-only')
    anchors = {('MIMIC-IV', 'death_24h'): 1.7337023630538637, ('MIMIC-III', 'death_24h'): 4.3928305453991126, ('eICU', 'death_24h'): 3.6389203741337703}
    for key, target in anchors.items():
        observed = float(model_row(models, key[0], key[1], 'P3')['odds_ratio'])
        if abs(observed - target) > 1e-12:
            raise RuntimeError(f'Figure 3 anchor mismatch for {key}: {observed}')
    fig = plt.figure(figsize=(7.2, 6.35), dpi=DPI)
    outer = fig.add_gridspec(2, 2, left=0.09, right=0.965, bottom=0.14, top=0.91, height_ratios=[1.22, 1.0], width_ratios=[0.95, 1.25], hspace=0.3, wspace=0.25)
    top = outer[0, :].subgridspec(1, 3, width_ratios=[1.35, 2.1, 1.65], wspace=0.02)
    ax_labels = fig.add_subplot(top[0, 0])
    ax_forest = fig.add_subplot(top[0, 1])
    ax_numbers = fig.add_subplot(top[0, 2])
    y_positions = {('death_24h', 'MIMIC-IV'): 7.5, ('death_24h', 'MIMIC-III'): 6.5, ('death_24h', 'eICU'): 5.5, ('hospital_mortality', 'MIMIC-IV'): 3.3, ('hospital_mortality', 'MIMIC-III'): 2.3, ('hospital_mortality', 'eICU'): 1.3}
    outcome_styles = {'death_24h': (COLORS['P3'], 'o'), 'hospital_mortality': (COLORS['navy'], 's')}
    for ax in [ax_labels, ax_forest, ax_numbers]:
        ax.set_ylim(0.5, 9.2)
    ax_labels.axis('off')
    ax_numbers.axis('off')
    fig.text(0.09, 0.965, 'a', fontsize=13, fontweight='bold', va='top', color=COLORS['text'])
    fig.text(0.121, 0.962, 'P3 vs P2: adjusted mortality associations', fontsize=9.5, fontweight='bold', va='top', color=COLORS['text'])
    ax_labels.text(0.0, 8.25, '12-36 h mortality', fontsize=7.3, fontweight='bold', color=COLORS['P3'])
    ax_labels.text(0.0, 4.05, 'In-hospital mortality', fontsize=7.3, fontweight='bold', color=COLORS['navy'])
    ax_numbers.text(0.0, 8.95, 'Adjusted OR (95% CI)', fontsize=6.7, fontweight='bold', va='top')
    ax_numbers.text(0.73, 8.95, 'Events / n', fontsize=6.7, fontweight='bold', va='top')
    for outcome in ['death_24h', 'hospital_mortality']:
        color, marker = outcome_styles[outcome]
        for database in DB_ORDER:
            row = model_row(models, database, outcome, 'P3')
            y = y_positions[outcome, database]
            ax_labels.text(0.08, y, database, va='center', fontsize=7.0, color=COLORS['text'])
            ax_forest.errorbar(float(row['odds_ratio']), y, xerr=[[float(row['odds_ratio'] - row['ci95_low'])], [float(row['ci95_high'] - row['odds_ratio'])]], fmt=marker, color=color, ecolor=color, elinewidth=1.25, capsize=2.4, markersize=5.0, markeredgecolor='white', markeredgewidth=0.6, zorder=3)
            ax_numbers.text(0.0, y, f"{row['odds_ratio']:.2f} ({row['ci95_low']:.2f}-{row['ci95_high']:.2f})", va='center', fontsize=6.5)
            ax_numbers.text(0.75, y, f"{int(row['events']):,} / {int(row['model_n']):,}", va='center', fontsize=6.5)
    ax_forest.axvline(1, color=COLORS['muted'], linestyle='--', linewidth=0.85)
    ax_forest.set_xscale('log')
    ax_forest.set_xlim(0.5, 12.0)
    ax_forest.set_xticks([0.5, 1, 2, 4, 8])
    ax_forest.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_forest.set_yticks([])
    ax_forest.set_xlabel('Odds ratio, logarithmic scale')
    style_axis(ax_forest, 'x')
    for y in [4.7]:
        ax_forest.axhline(y, color=COLORS['grid'], linewidth=0.8)
    ax_rd = fig.add_subplot(outer[1, 0])
    ax_rd.text(-0.15, 1.08, 'b', transform=ax_rd.transAxes, fontsize=13, fontweight='bold', va='top')
    ax_rd.set_title('P3 vs P2 standardized risk difference', loc='left', pad=10, fontweight='bold')
    outcomes = ['death_24h', 'hospital_mortality', 'organ_support_or_death_24h']
    outcome_labels = {'death_24h': 'Recent\nmortality', 'hospital_mortality': 'In-hospital\nmortality', 'organ_support_or_death_24h': 'Deterioration\nby 36 h'}
    rd_rows = []
    y = 8.5
    for outcome in outcomes:
        for database in DB_ORDER:
            row = absolute_risk_row(risks, database, outcome)
            rd_rows.append((y, outcome, database, row))
            y -= 0.72
        y -= 0.34
    for y, outcome, database, row in rd_rows:
        value = float(row['risk_difference_vs_P2']) * 100
        low = float(row['risk_difference_ci95_low']) * 100
        high = float(row['risk_difference_ci95_high']) * 100
        ax_rd.errorbar(value, y, xerr=[[value - low], [high - value]], fmt=DB_MARKERS[database], color=DB_COLORS[database], ecolor=DB_COLORS[database], elinewidth=1.0, capsize=1.8, markersize=3.8)
    ax_rd.axvline(0, color=COLORS['muted'], linestyle='--', linewidth=0.8)
    ax_rd.set_xlim(-7, 16)
    ax_rd.set_ylim(0.15, 9.2)
    group_centers = [7.78, 5.28, 2.78]
    ax_rd.set_yticks(group_centers)
    ax_rd.set_yticklabels([outcome_labels[o] for o in outcomes])
    ax_rd.tick_params(axis='y', labelsize=5.9, pad=3)
    ax_rd.set_xlabel('Risk difference (percentage points)')
    style_axis(ax_rd, 'x')
    handles = [Line2D([0], [0], marker=DB_MARKERS[d], color='none', markerfacecolor=DB_COLORS[d], markeredgecolor=DB_COLORS[d], label=d, markersize=4) for d in DB_ORDER]
    ax_rd.legend(handles=handles, loc='lower right', frameon=False, ncol=1, handletextpad=0.3)
    ax_heat = fig.add_subplot(outer[1, 1])
    ax_heat.text(-0.1, 1.08, 'c', transform=ax_heat.transAxes, fontsize=13, fontweight='bold', va='top')
    ax_heat.set_title('Exploratory P1 and P4 contrasts vs P2', loc='left', pad=10, fontweight='bold')
    heat_rows = []
    row_labels = []
    for outcome in outcomes:
        for phenotype in ['P1', 'P4']:
            values = []
            pvalues = []
            for database in DB_ORDER:
                row = model_row(models, database, outcome, phenotype)
                values.append(float(row['odds_ratio']))
                pvalues.append(float(row['p_value']))
            heat_rows.append(values)
            row_labels.append(f'{outcome_labels[outcome]} | {phenotype}')
    heat = np.log2(np.asarray(heat_rows, dtype=float))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.3)
    image = ax_heat.imshow(heat, cmap='RdBu_r', norm=norm, aspect='auto')
    for row_index, (values, outcome) in enumerate(zip(heat_rows, [x for x in outcomes for _ in (0, 1)])):
        phenotype = 'P1' if row_index % 2 == 0 else 'P4'
        for column_index, (database, value) in enumerate(zip(DB_ORDER, values)):
            pvalue = float(model_row(models, database, outcome, phenotype)['p_value'])
            star = '*' if pvalue < 0.05 else ''
            text_color = 'white' if abs(heat[row_index, column_index]) > 0.75 else COLORS['text']
            ax_heat.text(column_index, row_index, f'{value:.2f}{star}', ha='center', va='center', fontsize=6.2, color=text_color)
    ax_heat.set_xticks(range(3))
    ax_heat.set_xticklabels(DB_ORDER)
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_color(COLORS['border'])
        spine.set_linewidth(0.65)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.03, pad=0.02)
    colorbar.set_ticks([-1, 0, 1, 2])
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    colorbar.ax.set_title('log2\nOR', fontsize=5.3, pad=3)
    fig.text(0.09, 0.05, '* P < 0.05. Deterioration risk sets: MIMIC-IV 926/8,656; MIMIC-III 291/2,092; eICU 522/7,978 events/patients.', fontsize=5.5, color=COLORS['muted'])
    fig.text(0.09, 0.027, 'Models adjust for age, sex and observed 48-dimensional coverage; eICU uses hospital-clustered robust standard errors. P2 is the reference group.', fontsize=5.7, color=COLORS['muted'])
    save_figure(fig, output)
    return output

def draw_figure4() -> Path:
    output = OUT / 'Figure4_K4_uncertainty_hospital_transport.png'
    gradient = read_csv(RISK_GRADIENT)
    certainty = read_csv(CERTAINTY_MODELS)
    weighted = read_csv(POSTERIOR_WEIGHTED)
    models = read_csv(OUTCOME_MODELS)
    hospitals = read_csv(HOSPITAL_SUMMARY)
    hierarchical = read_csv(HIERARCHICAL)
    loho = read_csv(LOHO)
    feasibility = read_csv(SITE_FEASIBILITY)
    if gradient.groupby('database').size().to_dict() != {'MIMIC-III': 5, 'MIMIC-IV': 5, 'eICU': 5}:
        raise RuntimeError('Posterior gradient is incomplete')
    high_certainty = certainty[(certainty['analysis'] == 'hard_assignment_threshold') & (certainty['threshold_or_stratum'].astype(str) == '0.8')]
    if len(high_certainty) != 3 or len(weighted) != 3:
        raise RuntimeError('Uncertainty sensitivity inputs are incomplete')
    eligible_hospitals = hospitals[(hospitals['candidate_n'] >= 20) & hospitals['coverage_rate'].notna()].copy()
    if len(eligible_hospitals) != 153:
        raise RuntimeError(f'Expected 153 hospitals with >=20 candidates, got {len(eligible_hospitals)}')
    coverage_median = float(eligible_hospitals['coverage_rate'].median())
    posterior_fraction = float((eligible_hospitals['median_max_posterior'] >= 0.8).mean())
    if abs(coverage_median - 0.952) > 0.005 or abs(posterior_fraction - 0.987) > 0.005:
        raise RuntimeError('Hospital transport anchors changed')
    random_intercept = hierarchical[hierarchical['model'] == 'random_intercept'].iloc[0]
    loho_min = float(loho['or'].min())
    loho_max = float(loho['or'].max())
    if not (3.42 < loho_min < 3.45 and 3.86 < loho_max < 3.89):
        raise RuntimeError('Leave-one-hospital-out range changed')
    fig = plt.figure(figsize=(7.2, 6.35), dpi=DPI)
    grid = fig.add_gridspec(2, 2, left=0.075, right=0.98, bottom=0.13, top=0.95, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.92], hspace=0.53, wspace=0.3)
    ax_gradient = fig.add_subplot(grid[0, 0])
    ax_gradient.text(-0.18, 1.08, 'a', transform=ax_gradient.transAxes, fontsize=13, fontweight='bold', va='top')
    ax_gradient.set_title('P3 posterior probability and 12-36 h mortality', loc='left', pad=10, fontweight='bold')
    for database in DB_ORDER:
        data = gradient[gradient['database'] == database].copy()
        data['q'] = data['quintile'].str.replace('Q', '', regex=False).astype(int)
        data = data.sort_values('q')
        ax_gradient.plot(data['q'], data['standardized_risk'] * 100, marker=DB_MARKERS[database], color=DB_COLORS[database], linewidth=1.7, markersize=4.0, label=database)
        last = data.iloc[-1]
        endpoint_offset = {'MIMIC-IV': 0.11, 'MIMIC-III': -0.11, 'eICU': 0.0}[database]
        ax_gradient.text(5.08, float(last['standardized_risk'] * 100) + endpoint_offset, database, color=DB_COLORS[database], fontsize=6.2, va='center')
    ax_gradient.set_xlim(0.8, 5.75)
    ax_gradient.set_xticks(range(1, 6))
    ax_gradient.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax_gradient.set_xlabel('Database-specific quintile of P3 posterior probability')
    ax_gradient.set_ylabel('Standardized mortality risk (%)')
    style_axis(ax_gradient, 'y')
    ax_gradient.text(0.0, -0.23, 'Q5 contains the highest P3 posterior probabilities; no causal interpretation is implied.', transform=ax_gradient.transAxes, fontsize=5.5, color=COLORS['muted'])
    ax_robust = fig.add_subplot(grid[0, 1])
    ax_robust.text(-0.18, 1.08, 'b', transform=ax_robust.transAxes, fontsize=13, fontweight='bold', va='top')
    ax_robust.set_title('P3 vs P2 under uncertainty analyses', loc='left', pad=10, fontweight='bold')
    method_specs = [('Hard', 'o', COLORS['navy']), ('Posterior >=0.80', 's', COLORS['P3']), ('Posterior-weighted', 'D', COLORS['P4'])]
    y_base = {'MIMIC-IV': 3, 'MIMIC-III': 2, 'eICU': 1}
    offsets = {method_specs[0][0]: 0.18, method_specs[1][0]: 0.0, method_specs[2][0]: -0.18}
    for database in DB_ORDER:
        baseline = model_row(models, database, 'death_24h', 'P3')
        certainty_row = high_certainty[high_certainty['database'] == database].iloc[0]
        weighted_row = weighted[weighted['database'] == database].iloc[0]
        rows = [(method_specs[0], float(baseline['odds_ratio']), float(baseline['ci95_low']), float(baseline['ci95_high'])), (method_specs[1], float(certainty_row['or']), float(certainty_row['ci_low']), float(certainty_row['ci_high'])), (method_specs[2], float(weighted_row['or']), float(weighted_row['ci_low']), float(weighted_row['ci_high']))]
        for (label, marker, color), estimate, low, high in rows:
            y = y_base[database] + offsets[label]
            ax_robust.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt=marker, color=color, ecolor=color, elinewidth=1.05, capsize=2.0, markersize=4.0)
    ax_robust.axvline(1, color=COLORS['muted'], linestyle='--', linewidth=0.8)
    ax_robust.set_xscale('log')
    ax_robust.set_xlim(0.7, 30)
    ax_robust.set_xticks([1, 2, 4, 8, 16])
    ax_robust.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_robust.set_yticks([1, 2, 3])
    ax_robust.set_yticklabels(['eICU', 'MIMIC-III', 'MIMIC-IV'])
    ax_robust.set_xlabel('Odds ratio, logarithmic scale')
    style_axis(ax_robust, 'x')
    handles = [Line2D([0], [0], marker=marker, color=color, markerfacecolor=color, lw=0, label=label, markersize=4) for label, marker, color in method_specs]
    ax_robust.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=5.6, handletextpad=0.25, columnspacing=0.8)
    bottom = grid[1, :].subgridspec(1, 3, wspace=0.34)
    ax_cov = fig.add_subplot(bottom[0, 0])
    ax_cert = fig.add_subplot(bottom[0, 1])
    ax_loho = fig.add_subplot(bottom[0, 2])
    ax_cov.text(-0.3, 1.13, 'c', transform=ax_cov.transAxes, fontsize=13, fontweight='bold', va='top')
    rng = np.random.default_rng(20260801)
    coverage = eligible_hospitals['coverage_rate'].to_numpy(dtype=float)
    parts = ax_cov.violinplot([coverage], positions=[0], vert=False, widths=0.56, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(COLORS['light_blue'])
        body.set_edgecolor(COLORS['steel'])
        body.set_alpha(1.0)
    ax_cov.scatter(coverage, rng.normal(0, 0.055, len(coverage)), s=6, color=COLORS['steel'], alpha=0.42, linewidth=0)
    ax_cov.scatter([np.median(coverage)], [0], marker='D', s=32, color=COLORS['navy'], edgecolor='white', linewidth=0.6, zorder=4)
    ax_cov.axvline(0.8, color=COLORS['P3'], linestyle='--', linewidth=0.8)
    ax_cov.set_xlim(0, 1.02)
    ax_cov.set_yticks([])
    ax_cov.set_xlabel('Assignable proportion')
    ax_cov.set_title('eICU coverage across 153 hospitals', loc='left', fontsize=7.3, fontweight='bold')
    ax_cov.text(0.02, 0.08, f'Median {coverage_median * 100:.1f}%', transform=ax_cov.transAxes, fontsize=6.1, fontweight='bold')
    style_axis(ax_cov, 'x')
    posterior = eligible_hospitals['median_max_posterior'].dropna().to_numpy(dtype=float)
    parts = ax_cert.violinplot([posterior], positions=[0], vert=False, widths=0.56, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(COLORS['light_blue'])
        body.set_edgecolor(COLORS['steel'])
        body.set_alpha(1.0)
    ax_cert.scatter(posterior, rng.normal(0, 0.055, len(posterior)), s=6, color=COLORS['steel'], alpha=0.42, linewidth=0)
    ax_cert.scatter([np.median(posterior)], [0], marker='D', s=32, color=COLORS['navy'], edgecolor='white', linewidth=0.6, zorder=4)
    ax_cert.axvline(0.8, color=COLORS['P3'], linestyle='--', linewidth=0.8)
    ax_cert.set_xlim(0.65, 1.005)
    ax_cert.set_yticks([])
    ax_cert.set_xlabel('Hospital median maximum posterior')
    ax_cert.set_title('Assignment certainty', loc='left', fontsize=7.3, fontweight='bold')
    ax_cert.text(0.02, 0.08, f'{posterior_fraction * 100:.1f}% >= 0.80', transform=ax_cert.transAxes, fontsize=6.1, fontweight='bold')
    style_axis(ax_cert, 'x')
    loho_sorted = loho.sort_values('or').reset_index(drop=True)
    x = np.arange(1, len(loho_sorted) + 1)
    ax_loho.axhspan(float(random_intercept['ci_low']), float(random_intercept['ci_high']), color=COLORS['light_blue'], alpha=0.75)
    ax_loho.axhline(float(random_intercept['or_p3']), color=COLORS['navy'], linewidth=1.25)
    ax_loho.scatter(x, loho_sorted['or'], s=7, color=COLORS['P3'], alpha=0.55, linewidth=0)
    ax_loho.set_xlim(0, len(loho_sorted) + 1)
    ax_loho.set_ylim(2.3, 5.4)
    ax_loho.set_xticks([1, len(loho_sorted) // 2, len(loho_sorted)])
    ax_loho.set_xticklabels(['1', f'{len(loho_sorted) // 2}', f'{len(loho_sorted)}'])
    ax_loho.set_xlabel('Hospital omitted, ordered by estimate')
    ax_loho.set_ylabel('P3 vs P2 odds ratio')
    ax_loho.set_title('Leave-one-hospital-out analysis', loc='left', fontsize=7.3, fontweight='bold')
    style_axis(ax_loho, 'y')
    eligible_meta = int(feasibility['eligible_for_site_meta'].fillna(False).astype(bool).sum())
    ax_loho.text(0.02, 0.08, f"Pooled OR {float(random_intercept['or_p3']):.2f}; LOHO {loho_min:.2f}-{loho_max:.2f}\nOnly {eligible_meta} hospitals met the site-effect threshold", transform=ax_loho.transAxes, fontsize=5.8, color=COLORS['text'], linespacing=1.35)
    fig.text(0.075, 0.018, 'Posterior probabilities quantify assignment certainty, not biological membership.\nHospital-level results support aggregate transportability; sparse hospitals do not constitute independent site-level validation.', fontsize=5.4, color=COLORS['muted'], linespacing=1.25)
    save_figure(fig, output)
    return output
