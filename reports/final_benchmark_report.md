# Final pharma commercial-review controlled benchmark report

Executed source: `pharma-data.csv` (167,760 × 18); modeling rows 167,756.
Coverage: 2017-01-01 to 2019-04-01. The brief's approximate 254,082 × 18 shape matched: **False**.
Currency: source currency unavailable; generated monetary tables use `UNK` and perform no conversion.

## Finalized architecture

The frozen primary review model is **PCA Reconstruction** at rep × month. It retained 1 components with cumulative explained variance 0.9987. Raw score is mean squared reconstruction error after clean-training median imputation, signed-log compression and robust scaling. Contributions are non-causal reconstruction evidence, not SHAP values.

Selected raw threshold: 169.75496, chosen from the unlabeled validation review budget before test evaluation. Manager review capacity: 5.0%.

## Test metrics (controlled synthetic labels)

- ROC-AUC: 0.6307
- PR-AUC / average precision: 0.4337
- Precision / recall / F1 at frozen threshold: 0.1000 / 0.0556 / 0.0714
- Precision@1/5/10%: 1.0000 / 0.3333 / 0.1667
- Recall@1/5/10%: 0.0556 / 0.0556 / 0.0556
- Lift@1/5/10%: 2.8889 / 0.9630 / 0.4815

Confusion matrix at the frozen threshold (final test):

```
                 Predicted normal  Predicted review
Actual normal                  25                 9
Actual injected                17                 1
```

Detection by controlled type/category/severity (`no_final_test_support` is reported as N/A, never zero):

```
        grouping                                     value  positive_support  overall_truth_support  detection_rate_at_threshold  detection_rate_at_top5pct             support_status                                                   evaluation_scope
    anomaly_type              customer_sales_concentration                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                   duplicate_expense_claim                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type            duplicate_incentive_adjustment                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                 end_of_period_sales_spike                 3                      4                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                    extremely_short_visits                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type              high_activity_low_engagement                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type            incentivized_product_mix_shift                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                incorrect_accelerator_tier                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                  inflated_travel_distance                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type             late_repeated_target_revision                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                 low_volume_customer_spike                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type       multi_signal_sales_discount_returns                 1                      1                        1.000                      1.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                 overlap_impossible_travel                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                    peer_incentive_outlier                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                       post_payout_returns                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type         sales_without_supporting_activity                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type territory_potential_explained_performance                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type               threshold_crossing_discount                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type               unsupported_manual_override                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type                       unusual_return_rate                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                  activity                 4                      4                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                  customer                 2                      2                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                  discount                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                   expense                 2                      2                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                 incentive                 3                      3                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                              multi_signal                 1                      1                        1.000                      1.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                              order_timing                 3                      4                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                   product                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                     quota                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                   returns                 2                      2                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
anomaly_category                                 territory                 1                      1                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
        severity                                      high                 5                      5                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
        severity                                       low                 7                      7                        0.000                      0.000    evaluated_on_final_test                                       commercial PCA final holdout
        severity                                    medium                 8                      9                        0.125                      0.125    evaluated_on_final_test                                       commercial PCA final holdout
    anomaly_type       territory_workload_exceeds_capacity                40                     40                        1.000                        NaN evaluated_by_capacity_rule                               deterministic capacity overload rule
    anomaly_type         persistent_priority_undercoverage                 3                      1                        1.000                        NaN evaluated_by_capacity_rule priority coverage gap above frozen clean 95th percentile (6.48125)
```

Detection by manager, team and territory (final holdout):

```
    grouping             value  observations  positive_support  precision_at_threshold  recall_at_threshold
  manager_id  MGR_31981a685b7e            16                 6                0.000000                  0.0
  manager_id  MGR_4b97498897f6            12                 5                0.333333                  0.2
  manager_id  MGR_5e1c70ff0ce6            12                 4                0.000000                  0.0
  manager_id  MGR_eb86f856200b            12                 3                0.000000                  0.0
     team_id TEAM_4f4a9410ffcd            16                 6                0.000000                  0.0
     team_id TEAM_a405eba78bf2            12                 4                0.000000                  0.0
     team_id TEAM_b9dd960c1753            12                 5                0.333333                  0.2
     team_id TEAM_f144a6907dc4            12                 3                0.000000                  0.0
territory_id TERR_07a5ea4d7ed4            12                 3                0.000000                  0.0
territory_id TERR_49c0203db75f            16                 6                0.000000                  0.0
territory_id TERR_938541fb7311            12                 5                0.333333                  0.2
territory_id TERR_b31415d90eb9            12                 4                0.000000                  0.0
```

Score stability across the most recent periods:

```
population     period  observations  mean_raw_score  median_raw_score  std_raw_score  mean_anomaly_score  p95_anomaly_score  review_rate  selected_entity_overlap_previous_period  ground_truth_prevalence  precision_at_threshold  recall_at_threshold  mean_score_change_from_previous_period
  injected 2017-09-01            13        1.307488          1.190288       0.272850            0.344517           0.712766     0.000000                                     0.00                 0.000000                   0.000                  NaN                                0.090344
  injected 2017-10-01            13        1.117669          1.105245       0.415328            0.222095           0.581702     0.000000                                     0.00                 0.000000                   0.000                  NaN                               -0.122422
  injected 2017-11-01            13       12.414743          1.263578      38.510171            0.371358           0.847234     0.000000                                     0.00                 0.000000                   0.000                  NaN                                0.149264
  injected 2017-12-01            13       69.372914          3.278650      80.097708            0.694763           0.982553     0.153846                                     0.00                 0.000000                   0.000                  NaN                                0.323404
  injected 2018-01-01            13        1.640643          1.536836       0.679528            0.576923           0.820851     0.000000                                     0.00                 0.000000                   0.000                  NaN                               -0.117840
  injected 2018-02-01            13        1.565935          1.561106       0.202718            0.585106           0.757872     0.000000                                     0.00                 0.000000                   0.000                  NaN                                0.008183
  injected 2018-03-01            13        1.781471          1.802042       0.296073            0.672177           0.804681     0.000000                                     0.00                 0.000000                   0.000                  NaN                                0.087070
  injected 2018-04-01            13       23.745291          1.505828      52.205386            0.564812           0.957872     0.000000                                     0.00                 0.000000                   0.000                  NaN                               -0.107365
  injected 2018-05-01            13       34.984648          1.753644      60.859671            0.702291           0.958723     0.000000                                     0.00                 0.000000                   0.000                  NaN                                0.137480
  injected 2018-06-01            13        1.672323          1.605702       0.302314            0.620786           0.805532     0.000000                                     0.00                 0.000000                   0.000                  NaN                               -0.081506
  injected 2018-07-01            13        1.688422          1.687287       0.182787            0.657784           0.764043     0.000000                                     0.00                 0.000000                   0.000                  NaN                                0.036998
  injected 2018-08-01            13       12.519880          1.656605      38.581920            0.496146           0.860213     0.000000                                     0.00                 0.000000                   0.000                  NaN                               -0.161638
  injected 2018-09-01            13       24.115656          1.750588      52.046889            0.691326           0.957292     0.000000                                     0.00                 0.076923                   0.000                 0.00                                0.195180
  injected 2018-10-01            13        1.736322          1.742711       0.074544            0.692849           0.742435     0.000000                                     0.00                 0.000000                   0.000                  NaN                                0.001523
  injected 2018-11-01            13       12.718911          1.632433      38.551210            0.637791           0.831392     0.000000                                     0.00                 0.000000                   0.000                  NaN                               -0.055058
  injected 2018-12-01            13       72.860869          2.189852      84.060779            0.632320           0.984354     0.307692                                     0.00                 0.000000                   0.000                  NaN                               -0.005471
  injected 2019-01-01            13      181.270619        198.654789      39.211786            0.975201           0.989763     0.615385                                     0.25                 0.307692                   0.125                 0.25                                0.342880
  injected 2019-02-01            13        8.992648          3.023779      17.962220            0.827340           0.901305     0.000000                                     0.00                 0.461538                   0.000                 0.00                               -0.147860
  injected 2019-03-01            13        3.798685          2.361672       3.093069            0.809300           0.883671     0.000000                                     0.00                 0.461538                   0.000                 0.00                               -0.018041
  injected 2019-04-01            13       47.957660          3.089091      75.136017            0.800930           0.984667     0.153846                                     0.00                 0.153846                   0.000                 0.00                               -0.008369
```

False-positive review by leading PCA reconstruction driver:

```
       driver_feature           driver_name  false_positive_count  mean_raw_score  median_anomaly_score                                                                                       example_observation_ids                                                                               review_interpretation
incentive_peer_median Incentive peer median                     9      198.791151              0.985136 RPER_8afd0f88595a10ae|RPER_4c8ca2c81587e694|RPER_2830872bf4889d63|RPER_2ddbd68d4476b0b2|RPER_e45c04d0d84e35a9 Clean-labeled review candidate; validate the driver and source context, not evidence of misconduct.
```

The complete executed tables are persisted under `artifacts/commercial_review/model/`.

## Generated clean datasets

{
  "rep_master": 13,
  "manager_master": 4,
  "team_master": 4,
  "customer_master": 751,
  "product_master": 240,
  "territory_master": 8,
  "rep_targets_quotas": 364,
  "incentive_policy_rules": 12,
  "orders": 167756,
  "discount_detail": 167756,
  "returns_cancellations": 6452,
  "field_visits": 30268,
  "crm_interactions": 70925,
  "travel_expenses": 21925,
  "incentive_calculations": 364,
  "capacity_calendar": 364,
  "capacity_customer_drilldown": 18627,
  "capacity_territory_allocation": 2022,
  "capacity_territory_summary": 176
}

Feature-store shape: 364 × 244 including identifiers/context; model feature count 205.

## Controlled injections

Commercial rep-period injections and capacity overload truth remain in a separate benchmark table. Counts by scenario:

anomaly_type
territory_workload_exceeds_capacity          40
end_of_period_sales_spike                     4
duplicate_expense_claim                       1
duplicate_incentive_adjustment                1
extremely_short_visits                        1
customer_sales_concentration                  1
high_activity_low_engagement                  1
incentivized_product_mix_shift                1
inflated_travel_distance                      1
incorrect_accelerator_tier                    1
low_volume_customer_spike                     1
multi_signal_sales_discount_returns           1
overlap_impossible_travel                     1
late_repeated_target_revision                 1
peer_incentive_outlier                        1
persistent_priority_undercoverage             1
sales_without_supporting_activity             1
post_payout_returns                           1
territory_potential_explained_performance     1
threshold_crossing_discount                   1
unsupported_manual_override                   1
unusual_return_rate                           1

## Capacity

The new deterministic hours calendar preserves the existing normalized workload index as an audit feature while adding working/leave/training/admin/meeting availability, visit/travel/required-coverage hours, utilization, required/available FTE, FTE gap and coverage gaps. Clean high/critical capacity rows: 65. Capacity output supports workload review, territory redesign, sharing or further hiring analysis; it is not an automated hiring decision.

Capacity evaluation:

 risk_medium_threshold_pct  overload_threshold_pct  risk_critical_threshold_pct  precision  recall  overload_precision  overload_recall  true_positive  false_positive  false_negative  true_negative  ground_truth_overload_count  capacity_ground_truth_row_count  undercoverage_ground_truth_row_count  unmatched_overload_truth_row_count  predicted_overload_count  above_medium_threshold_count  above_high_threshold_count  above_critical_threshold_count  overloaded_rep_period_count  reps_above_medium_threshold  reps_above_high_threshold  reps_above_critical_threshold  overloaded_territory_count  territories_above_medium_threshold  territories_above_critical_threshold  overloaded_territory_period_count  territory_period_count  clean_row_count  injected_row_count  territory_count  mae_required_total_hours  workload_mae  workload_mae_observations                         workload_mae_basis  mae_utilization_pct  utilization_mae  utilization_mae_observations                 utilization_mae_basis  numeric_truth_independent_flag                                                                                  numeric_mae_interpretation  clean_injected_required_hours_mae  clean_injected_utilization_pct_mae  territory_ranking_agreement  territory_rank_spearman                                         territory_ranking_basis  territory_allocation_sensitivity_spearman                                                                                                                    territory_allocation_sensitivity_basis  territory_truth_independent_flag                                                 territory_capacity_basis  low_risk_count  medium_risk_count  high_risk_count  critical_risk_count  total_required_fte  total_available_fte  total_fte_gap  positive_fte_gap_fte_months                                                fte_aggregate_basis      data_lineage
                      85.0                   100.0                        120.0        0.4     1.0                 0.4              1.0             40              60               0            264                           40                               41                                     1                                   0                       100                           186                         100                              21                          100                           13                         13                              7                           8                                   8                                     8                                 79                     176              364                 364                8                       0.0           0.0                         39 ground_truth.injected_required_total_hours                  0.0              0.0                            39 ground_truth.injected_utilization_pct                           False Deterministic reconciliation to controlled injected values; not an independent predictive-accuracy estimate                           4.832482                            4.120495                          NaN                      NaN not_available: no independently targeted territory-period truth                                   0.021044 allocation-sensitivity diagnostic at territory-period grain: utilization versus fractionally propagated rep-period truth; not independent territory truth                             False conserved rep-territory-period allocation aggregated to territory-period             178                 86               79                   21          224.333068           250.525278      -26.19221                    10.056156 sum across rep-periods (FTE-months), not a point-in-time headcount synthetic_derived

## Dashboard datasets and visualizations

Compact dashboard inventory:

```
                          dashboard_dataset  rows
             dashboard_anomaly_evidence.csv   364
         dashboard_anomaly_type_metrics.csv    48
  dashboard_capacity_customer_drilldown.csv 18627
             dashboard_capacity_summary.csv   364
dashboard_capacity_territory_allocation.csv  2022
   dashboard_capacity_territory_summary.csv   176
             dashboard_confusion_matrix.csv     4
                 dashboard_data_quality.csv   248
        dashboard_feature_contributions.csv  1092
                  dashboard_kpi_summary.csv     1
         dashboard_manager_review_queue.csv    19
                  dashboard_model_curve.csv   120
                dashboard_model_metrics.csv    38
                 dashboard_pca_variance.csv     1
              dashboard_peer_comparison.csv  1456
             dashboard_period_stability.csv    56
           dashboard_rep_period_summary.csv   364
                 dashboard_run_manifest.csv     1
           dashboard_score_distribution.csv   220
```

The existing Streamlit application was extended with seven commercial-review pages: Executive Overview, Manager Review Queue, Rep Anomaly Drill-down, Team and Manager View, Capacity Overview, Model Benchmark View, and Data and Model Health. Generated figures are `clean_vs_injected_score_distribution.png`, `roc_precision_recall_curves.png`, `pca_cumulative_explained_variance.png`, and `capacity_utilization_review.png` under `reports/figures/`.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m field_rep_anomaly.commercial_review.pipeline --config configs/synthetic_data.yaml --input data/raw/pharma-data.csv
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/reports/commercial_review_tests.xml
.\.venv\Scripts\streamlit.exe run app.py
```

## Automated-test record

Latest retained pytest JUnit result: **passed**; 111 tests, 0 failures, 0 errors, 0 skipped, 63.15s. Artifact: `artifacts/reports/commercial_review_tests.xml`.

## Implementation and artifact inventory

The extension adds `configs/synthetic_data.yaml`, the `src/field_rep_anomaly/commercial_review/` package, four focused `tests/test_commercial_review_*.py` suites, the synthetic methodology/model card/dashboard-layer documentation, and the compact dashboard/model/report artifacts listed above. It updates `app.py`, `README.md`, packaging/dependency metadata, and ignore rules while preserving the legacy benchmark. The exhaustive generated-file names, hashes and row counts are recorded in `artifacts/commercial_review/output_manifest.csv` and `run_manifest.json`.

## Responsible use and limitations

- The benchmark does **not** prove fraud. A high score is a review candidate or unusual observation.
- All targets, policies, incentives, discounts, visits, CRM, expenses, capacity inputs and injected labels are controlled synthetic or derived data.
- Results require human validation against governed source systems and must not support punitive action without further investigation.
- Capacity outputs support, but never automate, hiring or employment decisions.
- Synthetic relationships simplify real compensation plans, approvals, territories, travel and customer engagement.
- Actual source coverage changes between Poland and Germany remain material context.
