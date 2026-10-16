"""Actual strict-cohort figure production retained unchanged in the latest oxygen repair; private hospital/distribution inputs and original layout source required.
Actual source retained with path-only adaptations; see source_inventory.tsv.
"""
import os
from pathlib import Path as _ReleasePath
if __name__ == '__main__':
    raise SystemExit('Historical/source-reference program. Read docs/full_research_process.md and use the guarded run_source_reference.py launcher; not a current-results batch runner.')
'仅由已审定结果重绘；复用历史图2–4绘图布局，不调用拟合程序。'
from pathlib import Path
import sys
import ast, copy, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageOps
P = Path(os.environ['SOURCE_7E9AB01024'])
O = P / '18_author_review_revision_20261014'
OUT = O / 'figures'
T = P / '14_pneumonia_specificity_revision_20261014/tables'
V = P / '16_MIMIC_secondary_completion_20261014/tables'

def tab(n, root=T):
    return pd.read_csv(root / n, sep='\t')
old = P / '08_revision_submission_20260903/scripts/build_figures2_4_revised.py'
tree = ast.parse(old.read_text().replace('(eligible_hospitals["median_max_posterior"] >= 0.80).mean()', '(eligible_hospitals["median_max_posterior"].dropna() >= 0.80).mean()'))
keep = {'configure_style', 'stamp_png', 'save_figure', 'style_axis', 'draw_figure2', 'model_row', 'absolute_risk_row', 'draw_figure3', 'draw_figure4'}
constant_names = {'COLORS', 'DB_ORDER', 'DB_COLORS', 'DB_MARKERS', 'PHENO_ORDER', 'DPI'}

class Adapt(ast.NodeTransformer):

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'ax_gradient') and (node.func.attr == 'set_title'):
            for kw in node.keywords:
                if kw.arg == 'pad':
                    kw.value = ast.Constant(3)
        return node

    def visit_If(self, node):
        s = ast.unparse(node)
        if any((x in s for x in ['Figure 3 anchor mismatch', 'Expected 153 hospitals', 'Hospital transport anchors changed', 'Leave-one-hospital-out range changed'])):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            replacements = {'MIMIC-IV\nDevelopment': 'MIMIC-IV\nRevised\napplication', 'MIMIC-IV development and frozen K4 assignments in replication and transport cohorts': 'Revised pneumonia-evidence application cohorts; original K4 parameters unchanged', '12-36 h mortality': 'Recorded hospital death\n(12,36] h', 'In-hospital mortality': 'Whole-hospital mortality', 'Recent\nmortality': '12-36 h\nhospital death', 'In-hospital\nmortality': 'Whole-hospital\nmortality', 'Deterioration\nby 36 h': 'Support/death\nby 36 h*', 'eICU coverage across 153 hospitals': 'eICU coverage across 152 hospitals', '* P < 0.05. Deterioration risk sets: MIMIC-IV 926/8,656; MIMIC-III 291/2,092; eICU 522/7,978 events/patients.': '* P < 0.05. Support/death: MIMIC-IV 922/8,468; MIMIC-III 291/2,091; eICU 481/7,693. eICU uses an ICU-scoped endpoint.', 'P3 posterior probability and Recorded hospital death\n(12,36] h': 'P3 probability and hospital death\n(12,36] h after ICU admission', 'Pooled OR ': 'Conditional OR '}
            for a, b in replacements.items():
                node.value = node.value.replace(a, b)
        return node
nodes = []
for n in tree.body:
    if isinstance(n, ast.Assign) and any((isinstance(x, ast.Name) and x.id in constant_names for x in n.targets)):
        nodes.append(n)
    if isinstance(n, ast.FunctionDef) and n.name in keep:
        nodes.append(Adapt().visit(n))
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(old), 'exec'))
configure_style()
models = tab('02_修订主要效应与严重度.tsv')
primary = models[models.model.eq('Model_0_current')].copy()
hospital = tab('04_修订整个住院死亡.tsv')
hospital = hospital[hospital.model.eq('hospital_current')].copy()
composite = tab('05_修订普通复合模型.tsv', V)
composite = composite[composite.model.eq('current_adjusted')].copy()
eu = tab('14_修订同范围复合普通回归.tsv')
eu = eu[eu.model.eq('ICU范围复合_24h_current_adjusted')].copy()
eu['outcome'] = 'organ_support_or_death_24h'
allmodels = pd.concat([primary, hospital, composite, eu], ignore_index=True)
for key in ['OUTCOME_MODELS', 'ABSOLUTE_RISKS', 'RISK_GRADIENT', 'CERTAINTY_MODELS', 'POSTERIOR_WEIGHTED', 'HOSPITAL_SUMMARY', 'HIERARCHICAL', 'LOHO', 'SITE_FEASIBILITY']:
    globals()[key] = key
alias = allmodels.rename(columns={'OR': 'odds_ratio', 'CI_low': 'ci95_low', 'CI_high': 'ci95_high', 'N': 'model_n', 'RD_vs_P2_fraction': 'risk_difference_vs_P2', 'RD_CI_low': 'risk_difference_ci95_low', 'RD_CI_high': 'risk_difference_ci95_high'})
alias['K'] = 4
hos = pd.read_parquet(P / '14_pneumonia_specificity_revision_20261014/private/修订医院覆盖率和后验汇总.parquet').rename(columns={'assignment_coverage': 'coverage_rate'})
if hos.coverage_rate.max() > 1:
    hos['coverage_rate'] /= 100
hier = tab('19_修订医院随机效应模型.tsv').rename(columns={'OR': 'or_p3', 'CI_low': 'ci_low', 'CI_high': 'ci_high'})
loho = pd.read_parquet(P / '14_pneumonia_specificity_revision_20261014/private/修订逐院剔除效应.parquet').rename(columns={'OR': 'or'})
mapping = {'OUTCOME_MODELS': alias, 'ABSOLUTE_RISKS': alias, 'RISK_GRADIENT': tab('17_修订P3后验五分位风险.tsv'), 'CERTAINTY_MODELS': tab('17_修订置信度与熵分层关联.tsv'), 'POSTERIOR_WEIGHTED': tab('17_修订后验加权伪类别关联.tsv'), 'HOSPITAL_SUMMARY': hos, 'HIERARCHICAL': hier, 'LOHO': loho, 'SITE_FEASIBILITY': pd.DataFrame({'eligible_for_site_meta': [True] * 8})}

def read_csv(path):
    return mapping[path].copy()
assert primary.groupby('database').N.first().to_dict() == {'MIMIC-IV': 14636, 'MIMIC-III': 3690, 'eICU': 14779}
assert len(hos[hos.candidate_n.ge(20)]) == 152 and len(loho) == 188
if '--uncertainty-only' in sys.argv:
    draw_figure4()
    print('UNCERTAINTY_FIGURE_UPDATED_ONLY')
    raise SystemExit(0)
flow = tab('21_全流程逐步排除完整对账.tsv')
fig, ax = plt.subplots(figsize=(7.2, 7.2 * 4.8 / 6.55), dpi=DPI)
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.995, 'Revised cohort selection and locked K4 application', fontsize=10, weight='bold', va='top')
roles = ['Revised MIMIC-IV application', 'Same-center historical replication', 'Multicenter external transport']
for k, db in enumerate(DB_ORDER):
    f = flow[flow.database.eq(db)]
    x = 0.018 + k * 0.329
    ax.text(x + 0.15, 0.95, db + '\n' + roles[k], ha='center', va='top', fontsize=6.7, weight='bold', linespacing=1.0)
    initial = int(f.iloc[0].remaining_N)
    excl = f[f.step.gt(0) & f.step.lt(10)]
    history = int(f[f.step.lt(10)].iloc[-1].remaining_N)
    boundary = int(f[f.step.eq(10)].excluded_N.iloc[0])
    pneu = int(f[f.step.eq(11)].excluded_N.iloc[0])
    land = int(f[f.step.eq(11)].remaining_N.iloc[0])
    ass = int(f[f.step.eq(12)].remaining_N.iloc[0])
    n = int(f.iloc[-1].remaining_N)
    reason = {'MIMIC-IV': 'ICU stay <12 h: 425\nDeath by 12 h: 40', 'MIMIC-III': 'Age <18 y: 54\nICU stay <12 h: 88\nDeath by 12 h: 24', 'eICU': 'Age <18 y: 21; age missing: 1\nICU stay <12 h: 1,027\nDeath by 12 h: 4\nHospital outcome unavailable: 202'}[db]
    boxes = [(f'Original first pneumonia-associated\nICU records: {initial:,}', 0.83, 0.065), ('Sequential exclusions\n' + reason, 0.695, 0.12), (f'Historical landmark records: {history:,}\nICU exit exactly 12 h removed: {boundary}\nInsufficient revised pneumonia\nevidence: {pneu}', 0.56, 0.115), (f'Revised landmark cohort: {land:,}\nBelow coverage rule: {land - ass:,}', 0.44, 0.08), (f'Assignable: {ass:,} ({100 * ass / land:.2f}%)\nP1–P4 mapping unchanged', 0.33, 0.08), (f'Main regression: {n:,}\nMissing sex excluded: {ass - n}', 0.22, 0.08)]
    prev = None
    for text, y, h in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.302, h, boxstyle='round,pad=0.005', facecolor=COLORS['light_blue'], edgecolor=COLORS['border'], linewidth=0.8))
        ax.text(x + 0.151, y + h / 2, text, ha='center', va='center', fontsize=6.2, linespacing=1.1)
        if prev is not None:
            ax.annotate('', xy=(x + 0.151, y + h + 0.006), xytext=(x + 0.151, prev - 0.006), arrowprops={'arrowstyle': '->', 'lw': 0.8, 'color': COLORS['navy']})
        prev = y
ax.text(0.5, 0.125, 'Original MIMIC-IV derivation: 14,853 patients; 48 features; K=4\nFixed standardization, weights, means, covariances, feature order and labels.\nRevised eligibility changes application populations, not the fitted model.', ha='center', va='center', fontsize=7, bbox={'boxstyle': 'round,pad=.7', 'fc': '#F4F6F8', 'ec': COLORS['border']})
ax.text(0.02, 0.025, 'Landmark: alive and still in ICU strictly after 12 h. Primary endpoint: recorded in-hospital death during (12,36] h.\nExclusions are sequential, not counts to subtract independently from every intermediate denominator.', fontsize=5.8)
save_figure(fig, OUT / 'Figure1_revised_cohort_flow.png')
if '--flow-only' in sys.argv:
    print('FLOW_FIGURE_UPDATED_ONLY')
    raise SystemExit(0)
data = pd.read_parquet(P / '14_pneumonia_specificity_revision_20261014/private/修订三库全模块分析输入.parquet', columns=['database', 'stay_id', 'phenotype'])
matrix = P / '01_k5-7.29_history/k5-7.29/analysis_landmark_external_upgrade_20260730/14_landmark12_candidate_matrix.parquet'
features = [f'{v}_h{h}' for v in ['SBP', 'DBP', 'HR', 'RR'] for h in range(12)]
m = pd.read_parquet(matrix, columns=['database', 'stay_id'] + features)
m['database'] = m.database.replace({'MIMIC-III MetaVision': 'MIMIC-III'})
m['stay_id'] = m.stay_id.astype(str)
data['stay_id'] = data.stay_id.astype(str)
d = data.merge(m, on=['database', 'stay_id'], validate='one_to_one')
assert len(d) == 33107
recs = []
for v in ['SBP', 'DBP', 'HR', 'RR']:
    for h in range(12):
        g = d.groupby(['database', 'phenotype'])[f'{v}_h{h}'].agg(['count', 'mean', 'std']).reset_index()
        g['variable'] = v
        g['hour'] = h
        g['se'] = g['std'] / np.sqrt(g['count'])
        g['ci_low'] = g['mean'] - 1.96 * g.se
        g['ci_high'] = g['mean'] + 1.96 * g.se
        recs.extend(g.to_dict('records'))
summary = pd.DataFrame(recs)
summary.to_csv(O / 'qc/图2逐小时聚合来源.tsv', sep='\t', index=False)
draw_figure2(summary, d.groupby('database').size().to_dict())
draw_figure3()
draw_figure4()
cov = tab('15_修订覆盖阈值汇总.tsv')
for name, col, ylabel in [('S2', 'assignment_coverage_pct', 'Assignable patients (%)'), ('S3', 'current_adjusted_OR', 'P3 vs P2 adjusted odds ratio')]:
    fig, ax = plt.subplots(figsize=(6.5, 3.7), dpi=DPI)
    for db in DB_ORDER:
        r = cov[cov.database.eq(db)].sort_values('coverage_threshold')
        x = r.coverage_threshold.to_numpy()
        if name == 'S3':
            ax.errorbar(x, r[col], yerr=[r[col] - r.current_adjusted_CI_low, r.current_adjusted_CI_high - r[col]], marker=DB_MARKERS[db], label=db, color=DB_COLORS[db], capsize=3)
        else:
            ax.plot(x, r[col], marker=DB_MARKERS[db], label=db, color=DB_COLORS[db])
    ax.axvline(32, ls='--', color=COLORS['muted'])
    ax.set_xticks([24, 28, 32, 36, 40])
    ax.set_xlabel('Minimum observed features (of 48)')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    style_axis(ax)
    if name == 'S3':
        ax.set_yscale('log')
        ax.axhline(1, ls=':', color=COLORS['muted'])
    fig.tight_layout()
    save_figure(fig, OUT / f'Figure_{name}_revised.png')
dist = pd.read_parquet(P / '14_pneumonia_specificity_revision_20261014/private/修订逐人冻结参考分布核查.parquet')
for name, col, threshold, label in [('S4', 'mahalanobis_per_observed_dimension', 1.6550214007230482, 'Assigned-component squared Mahalanobis distance / observed dimension'), ('S5', 'normalized_NLL', 1.3155081267235247, 'Negative log likelihood / observed dimension')]:
    fig, ax = plt.subplots(figsize=(6.5, 3.7), dpi=DPI)
    for db in DB_ORDER:
        vals = np.sort(dist.loc[dist.database.eq(db), col].dropna().to_numpy())
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), color=DB_COLORS[db], label=db, lw=1.5)
    ax.axvline(threshold, color=COLORS['muted'], ls='--', label=f'Original derivation 95th percentile: {threshold:.4f}')
    ax.set_xlabel(label, fontsize=7)
    ax.set_ylabel('Empirical cumulative distribution')
    ax.legend(frameon=False, fontsize=7)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT / f'Figure_{name}_revised.png')
(O / 'qc/图件来源与边界.md').write_text('图1：当前21全流程表。图2：当前严格成员与原小时矩阵，仅聚合；图3：当前主效应、整个住院死亡和普通复合结果；eICU复合明确为ICU范围。图4：当前后验及医院结果。S2/S3：当前覆盖结果。S4/S5：当前逐人冻结参考诊断仅聚合绘图，冻结阈值不变。S1原模型选择图原样保留并注明历史推导。图2–4复用08目录绘图函数布局，去掉旧数值断言并以当前断言取代。未重拟合任何统计模型。\n')
print('FIGURES_COMPLETE', len(list(OUT.glob('*.png'))))
