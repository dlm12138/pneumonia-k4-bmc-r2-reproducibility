"""Packaging-only input wiring. Scientific functions are in core/."""
from pathlib import Path
import importlib.util
import os
_CODE=Path(__file__).resolve().parent
_ROOT=_CODE.parent
MODULES = {'05_harmonized_severity_and_apache.py': 'core/association.py', '08_hrrr_and_incremental_performance.py': 'core/prediction.py', '07_clinical_characterization_and_stress.py': 'core/clinical.py', '09_domain_shift_diagnostics.py': 'core/domain_shift.py', '11_time_order_falsification.py': 'core/time_order.py', '12_eicu_hospital_transport.py': 'core/hospital.py', '04_selection_ipw_audit.py': 'core/selection.py', 'assignment_code_clean.py': 'core/locked_k4.py', 'run_k4_evidence_deepening.py': 'core/posterior.py', '02_common4_stability_transport.py': 'core/early_assignment.py', '09_domain_early_and_posterior.py': 'workflows/07_domain_early_posterior.py', '03_preserved_fold_prediction.py': 'workflows/05_fixed_fold_prediction.py'}
RESULTS = {'02_修订主要效应与严重度.tsv': 'primary_and_severity_effects.tsv', '03_修订固定HRRR规则.tsv': 'hrrr_comparison.tsv', '04_修订整个住院死亡.tsv': 'hospital_mortality_effects.tsv', '05_修订普通复合模型.tsv': 'ordinary_composite_effects.tsv', '08_修订预测性能与配对增量.tsv': 'prediction_performance.tsv', '08_修订各折人数与事件.tsv': 'prediction_fold_counts.tsv', '10_MIMIC-IV_修订四型临床表征.tsv': 'clinical_characteristics_mimiciv.tsv', '10_MIMIC-III_修订四型临床表征.tsv': 'clinical_characteristics_mimiciii.tsv', '10_eICU_修订四型临床表征.tsv': 'clinical_characteristics_eicu.tsv', '10_修订P3与P2临床描述.tsv': 'p3_p2_clinical_descriptors.tsv', '14_修订同范围复合普通回归.tsv': 'aligned_icu_composite_effects.tsv', '15_修订覆盖阈值完整效应.tsv': 'coverage_effects.tsv', '15_修订覆盖阈值汇总.tsv': 'coverage_summary.tsv', '19_02_三库四型主模型人数与实际死亡.tsv': 'phenotype_counts_and_deaths.tsv', '19_03_eICU核心冲突与死亡按四型分布.tsv': 'eicu_conflict_distribution.tsv', '19_04_eICU各冲突规则分组汇总_规则可重叠.tsv': 'eicu_conflict_rules.tsv', '19_05_eICU冲突规则互斥组合.tsv': 'eicu_conflict_rule_combinations.tsv', '01_血氧可用性新旧对照.tsv': 'oxygen_availability.tsv', '04_覆盖阈值血氧缺失.tsv': 'coverage_oxygen_missingness.tsv', '21_全流程逐步排除完整对账.tsv': 'cohort_flow.tsv', '05_修订APACHE同样本敏感性.tsv': 'apache_same_sample.tsv', '06_修订共病同样本敏感性.tsv': 'comorbidity_same_sample.tsv', '07_修订肺炎时点与吸入编码敏感性.tsv': 'pneumonia_definition_sensitivities.tsv', '11_修订明确肺炎记录时点.tsv': 'pneumonia_documentation_timing.tsv', '17_修订冻结参考分布漂移汇总.tsv': 'domain_shift_summary.tsv', '17_修订早期分配稳定性.tsv': 'early_assignment_stability.tsv', '17_修订置信度与熵分层关联.tsv': 'confidence_entropy_effects.tsv', '17_修订P3后验五分位风险.tsv': 'posterior_quintile_risks.tsv', '17_修订后验加权伪类别关联.tsv': 'posterior_pseudoclass_effects.tsv', '17_修订连续P3后验关联.tsv': 'continuous_posterior_effects.tsv', '18_修订时间顺序汇总.tsv': 'time_order_summary.tsv', '19_修订医院层汇总.tsv': 'hospital_transportability_summary.tsv', '19_修订医院随机效应模型.tsv': 'hospital_random_effects.tsv', '20_修订选择权重与ESS.tsv': 'selection_weight_summary.tsv', '20_修订选择加权效应.tsv': 'selection_weighted_effects.tsv', '21_选择加权平衡分层汇总.tsv': 'selection_balance_summary.tsv', '20_MIMICIII护理系统结构性支持.tsv': 'mimiciii_system_support.tsv', '13_修订FineGray实际重拟合.tsv': 'eicu_competing_risks.tsv', '21_eICU不同死亡终点同样本比较.tsv': 'eicu_endpoint_sensitivities.tsv', '10_修订MIMIC_FineGray.tsv': 'mimic_competing_risks.tsv', '12_各表型竞争风险累积发生率.tsv': 'mimic_cumulative_incidence.tsv', '03_新旧样本及结局定义差异.tsv': 'composite_risk_set_counts.tsv'}
def load_actual(path, name='actual_study_functions'):
    basename=Path(path).name
    if basename not in MODULES:
        raise FileNotFoundError('Scientific source not included: '+basename+'; see reproduction limits.')
    selected=_CODE/MODULES[basename]
    spec=importlib.util.spec_from_file_location(name,selected)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def init_output(root):
    root=Path(root).resolve()
    if root==_ROOT or _ROOT in root.parents:
        raise ValueError('OUTPUT_ROOT must be outside the public repository: generated inputs/outputs are private.')
    for sub in ['tables','private','qc','tmp']: (root/sub).mkdir(parents=True,exist_ok=True)
def public_result_path(name):
    return _ROOT/'results'/RESULTS[name]
