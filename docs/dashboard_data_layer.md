# Dashboard Data Layer — schema and populated previews

Generated directly from the exported manager CSVs. Each schema table includes the first three actual rows; `null` means unavailable, not zero. Displayed decimals are shortened here only; CSVs preserve their numeric precision.

The layer is read-only with respect to all existing raw, processed, model, metric and report artifacts. Only manager-facing exports and this documentation are produced.

Percentiles are 0–1. All incentive amounts retain the simulated_ prefix. Review priority is a deterministic presentation policy, not a probability. Month columns are ISO first-of-month dates.

## dashboard_anomaly_review.csv

Rows: **2,184**. Columns: **82**. Grain: **representative × product_class × month**.

| Field | Export dtype | Null rows | Row 1 | Row 2 | Row 3 |
|---|---|---:|---|---|---|
| observation_id | object | 0 | obs_6503b2dd1b66b41d | obs_a1e4ca82ef0309af | obs_438e87ccdfc7619e |
| representative | object | 0 | Jessica Smith | Jessica Smith | Jessica Smith |
| manager | object | 0 | Britanny Bold | Britanny Bold | Britanny Bold |
| team | object | 0 | Delta | Delta | Delta |
| country | object | 0 | Germany | Germany | Germany / Poland |
| product_class | object | 0 | Analgesics | Analgesics | Analgesics |
| month | object | 0 | 2019-02-01 | 2019-01-01 | 2018-12-01 |
| source_partition | object | 0 | test | test | validation |
| sales | float64 | 0 | 2.14202e+06 | 941037 | 3.39602e+06 |
| quantity | float64 | 0 | 5665 | 2272 | 14342 |
| transaction_count | int64 | 0 | 54 | 59 | 96 |
| transaction_value | float64 | 0 | 39667.1 | 15949.8 | 35375.2 |
| unique_customers | int64 | 0 | 51 | 55 | 87 |
| new_customers | int64 | 0 | 6 | 3 | 8 |
| product_breadth | int64 | 0 | 31 | 31 | 39 |
| city_coverage | int64 | 0 | 51 | 55 | 87 |
| geographic_spread | float64 | 0 | 2.6164 | 2.39962 | 5.78787 |
| distributor_count | int64 | 0 | 4 | 3 | 15 |
| distributor_concentration | float64 | 0 | 0.379468 | 0.453412 | 0.217821 |
| distributor_mix_change | float64 | 0 | 0.184259 | 0.748755 | 0.657564 |
| channel_mix | float64 | 0 | 0.471359 | 0.494042 | 0.400724 |
| subchannel_mix | float64 | 0 | 0.679594 | 0.680716 | 0.688068 |
| simulated_target | float64 | 0 | 1.18404e+06 | 3.4979e+06 | 3.24182e+06 |
| simulated_expected_incentive | float64 | 0 | 2011.34 | 451.772 | 1059.46 |
| simulated_actual_payout | float64 | 0 | 2011.34 | 451.772 | 1059.46 |
| simulated_adjustment | float64 | 0 | 0 | 0 | 0 |
| simulated_payout_delta | float64 | 0 | 0 | 0 | 0 |
| simulated_attainment | float64 | 0 | 180.908 | 26.9029 | 104.757 |
| repeat_customers | Int64 | 0 | 45 | 52 | 79 |
| lost_customers | Int64 | 78 | 52 | 83 | 88 |
| distributor_mix_summary | object | 0 | 4 distributors; concentration index 0.379; mix change 0.184 | 3 distributors; concentration index 0.453; mix change 0.749 | 15 distributors; concentration index 0.218; mix change 0.658 |
| channel_summary | object | 0 | Channel diversity index 0.471; subchannel diversity index 0.680 | Channel diversity index 0.494; subchannel diversity index 0.681 | Channel diversity index 0.401; subchannel diversity index 0.688 |
| pca_raw_score | float64 | 0 | 4.23645 | 3.95039 | 3.75196 |
| pca_score_percentile | float64 | 0 | 0.999976 | 0.999974 | 0.999972 |
| pca_review_flag | boolean | 0 | True | True | True |
| pca_threshold_exceedance | boolean | 0 | True | True | True |
| pca_raw_threshold | float64 | 0 | 0.0837369 | 0.0837369 | 0.0837369 |
| ewma_score | float64 | 0 | 0.662847 | 0.885985 | 0.511998 |
| robust_peer_percentile | float64 | 0 | 0.460794 | 0.564589 | 0.599285 |
| kmeans_percentile | float64 | 0 | 0.99565 | 0.995512 | 0.995353 |
| autoencoder_percentile | float64 | 0 | 0.99792 | 0.997866 | 0.997834 |
| isolation_forest_percentile | float64 | 0 | 0.980251 | 0.989662 | 0.948735 |
| temporal_review_flag | boolean | 0 | False | False | False |
| robust_peer_flag | boolean | 0 | False | False | False |
| peer_flag | boolean | 0 | False | False | False |
| kmeans_distance | float64 | 0 | 30.9872 | 28.9688 | 26.6656 |
| kmeans_cluster | Int64 | 0 | 3 | 0 | 2 |
| dbscan_cluster | Int64 | 0 | -1 | -1 | -1 |
| dbscan_noise | boolean | 0 | True | True | True |
| top_driver_1_feature | object | 0 | total_quantity_lag_3 | total_quantity_lag_2 | total_quantity_lag_1 |
| top_driver_1 | object | 0 | Quantity: lag 3 | Quantity: lag 2 | Quantity: lag 1 |
| top_driver_1_contribution | float64 | 0 | 335.654 | 319.282 | 364.362 |
| top_driver_2_feature | object | 0 | total_sales_trend | distinct_customers_lag_2 | simulated_actual_payout_acceleration |
| top_driver_2 | object | 0 | Sales: prior trend | Customer coverage: lag 2 | DEMO payout: change in growth |
| top_driver_2_contribution | float64 | 0 | 26.7743 | 13.0901 | 8.03822 |
| top_driver_3_feature | object | 0 | simulated_actual_payout_trend | total_sales_acceleration | total_sales_acceleration |
| top_driver_3 | object | 0 | DEMO payout: prior trend | Sales: change in growth | Sales: change in growth |
| top_driver_3_contribution | float64 | 0 | 18.8372 | 11.0877 | 7.20077 |
| strongest_peer_deviation_metric | object | 0 | Customer coverage | Customer coverage | DEMO payout |
| strongest_peer_deviation_value | float64 | 0 | -1.14663 | 1.34021 | 1.43564 |
| peer_comparison_cohort | object | 0 | product_class / peer_country / team / date | product_class / peer_country / team / date | product_class / peer_country / team / date |
| strongest_history_deviation_metric | object | 234 | Customer coverage: personal history deviation | Sales: personal history deviation | Quantity: personal history deviation |
| strongest_history_deviation_value | float64 | 234 | -4.66955 | -7.27823 | 1.53914 |
| temporal_metric_feature | object | 0 | distinct_customers | total_sales | total_quantity |
| temporal_observed | float64 | 0 | 51 | 941037 | 14342 |
| temporal_expected | float64 | 78 | 84.2473 | 3.78554e+06 | 7394.89 |
| temporal_direction | object | 234 | down | down | up |
| temporal_history_length | Int64 | 0 | 25 | 24 | 23 |
| temporal_available | bool | 0 | True | True | True |
| temporal_metric | object | 0 | Customer coverage | Sales | Quantity |
| ewma_sales_observed | float64 | 0 | 2.14202e+06 | 941037 | 3.39602e+06 |
| ewma_sales_expected | float64 | 78 | 2.93219e+06 | 3.78554e+06 | 3.95248e+06 |
| ewma_sales_raw_score | float64 | 0 | 0.449887 | 8.2272 | 0.35112 |
| ewma_sales_history_length | int64 | 0 | 25 | 24 | 23 |
| ewma_sales_available | bool | 0 | True | True | True |
| business_rule_flag | boolean | 0 | False | False | True |
| business_rule_summary | object | 1532 | null | null | Sales growth is concentrated in few customers |
| number_of_supporting_signals | int64 | 0 | 0 | 0 | 1 |
| review_priority | object | 0 | High | High | High |
| model_agreement_summary | object | 0 | PCA only | PCA only | PCA + Business Rule |
| pca_rank | int64 | 0 | 1 | 2 | 3 |
| review_rank | int64 | 0 | 1 | 2 | 3 |

## dashboard_rep_summary.csv

Rows: **13**. Columns: **24**. Grain: **representative**.

| Field | Export dtype | Null rows | Row 1 | Row 2 | Row 3 |
|---|---|---:|---|---|---|
| representative | object | 0 | Abigail Thompson | Alan Ray | Anne Wu |
| manager | object | 0 | Tracy Banks | James Goodwill | Britanny Bold |
| team | object | 0 | Bravo | Alfa | Delta |
| primary_country | object | 0 | Germany | Germany | Germany |
| latest_month_available | object | 0 | 2019-04-01 | 2019-04-01 | 2019-04-01 |
| total_observations | int64 | 0 | 168 | 168 | 168 |
| high_priority_review_count | int64 | 0 | 9 | 12 | 16 |
| medium_priority_review_count | int64 | 0 | 44 | 39 | 44 |
| top_5_percent_review_count | int64 | 0 | 5 | 10 | 12 |
| maximum_pca_percentile | float64 | 0 | 0.997468 | 0.997609 | 0.997017 |
| mean_pca_percentile | float64 | 0 | 0.639206 | 0.642729 | 0.720949 |
| latest_pca_percentile | float64 | 0 | 0.997468 | 0.997003 | 0.997017 |
| latest_review_priority | object | 0 | High | Medium | Medium |
| strongest_recent_driver | object | 0 | New customers | New customers | New customers |
| temporal_flag_count | int64 | 0 | 8 | 8 | 10 |
| business_rule_flag_count | int64 | 0 | 40 | 54 | 56 |
| peer_flag_count | int64 | 0 | 9 | 5 | 10 |
| model_agreement_high_count | int64 | 0 | 11 | 6 | 8 |
| total_sales | float64 | 0 | 6.09748e+08 | 5.14322e+08 | 5.53541e+08 |
| recent_3m_sales | float64 | 0 | 6.90164e+07 | 4.99911e+07 | 4.77069e+07 |
| prior_3m_sales | float64 | 0 | 4.99545e+07 | 5.22549e+07 | 3.83024e+07 |
| sales_growth_3m | float64 | 0 | 0.381584 | -0.0433218 | 0.245534 |
| unique_customers_latest | int64 | 0 | 232 | 216 | 218 |
| customer_change_3m | int64 | 0 | 24 | -10 | -17 |

## dashboard_capacity_base.csv

Rows: **48**. Columns: **36**. Grain: **team × country × product_class**.

| Field | Export dtype | Null rows | Row 1 | Row 2 | Row 3 |
|---|---|---:|---|---|---|
| team | object | 0 | Alfa | Alfa | Alfa |
| country | object | 0 | Germany | Germany | Germany |
| product_class | object | 0 | Analgesics | Antibiotics | Antimalarial |
| forecast_horizon | object | 0 | 2019-05-01 | 2019-05-01 | 2019-05-01 |
| eligible_for_capacity_recommendation | bool | 0 | True | True | True |
| selected_forecast_method | object | 24 | exponential_smoothing | exponential_smoothing | exponential_smoothing |
| forecast_workload | float64 | 24 | 3.00959 | 2.68911 | 2.61435 |
| forecast_error_metric_used_for_selection | object | 24 | WAPE | WAPE | WAPE |
| sustainable_workload_per_rep | float64 | 24 | 7.48169 | 7.48169 | 7.48169 |
| allocated_fte | float64 | 24 | 0.51703 | 0.457649 | 0.400035 |
| required_fte | float64 | 24 | 0.402261 | 0.359425 | 0.349433 |
| fte_gap | float64 | 24 | -0.114769 | -0.0982233 | -0.0506015 |
| required_fte_lower | float64 | 24 | 0.177744 | 0.173832 | 0.182882 |
| required_fte_upper | float64 | 24 | 0.59114 | 0.5053 | 0.470087 |
| fte_gap_lower | float64 | 24 | -0.339285 | -0.283817 | -0.217152 |
| fte_gap_upper | float64 | 24 | 0.0741108 | 0.0476515 | 0.0700523 |
| forecast_lower_scenario | float64 | 24 | 1.89196 | 1.85032 | 1.94665 |
| forecast_upper_scenario | float64 | 24 | 4.12722 | 3.5279 | 3.28205 |
| forecast_scenario_basis | object | 0 | Recovered from saved FTE scenario bounds and TRAIN capacity quantiles; validation absolute-error sensitivity, not a statistical confidence interval. | Recovered from saved FTE scenario bounds and TRAIN capacity quantiles; validation absolute-error sensitivity, not a statistical confidence interval. | Recovered from saved FTE scenario bounds and TRAIN capacity quantiles; validation absolute-error sensitivity, not a statistical confidence interval. |
| workload_score_raw | float64 | 24 | 3.00959 | 2.68911 | 2.61435 |
| workload_score_winsorized | float64 | 24 | 3.13415 | 2.85359 | 2.71912 |
| customer_load | float64 | 24 | 143.689 | 132.346 | 127.879 |
| transaction_load | float64 | 24 | 193 | 158 | 156 |
| geography_load | float64 | 24 | 143.637 | 132.313 | 127.529 |
| product_load | float64 | 24 | 41.5554 | 33.6559 | 34.3289 |
| distributor_load | float64 | 24 | 5 | 5 | 5.33333 |
| workload_components_basis | object | 0 | Independently forecast counts of customers, transactions, cities, products, and distributors; not additive weighted-workload contributions. | Independently forecast counts of customers, transactions, cities, products, and distributors; not additive weighted-workload contributions. | Independently forecast counts of customers, transactions, cities, products, and distributors; not additive weighted-workload contributions. |
| last_observed_month | object | 0 | 2019-04-01 | 2019-04-01 | 2019-04-01 |
| coverage_note | object | 0 | Observed latest-month rep allocation | Observed latest-month rep allocation | Observed latest-month rep allocation |
| validation_wape | float64 | 24 | 0.130214 | 0.130214 | 0.130214 |
| test_wape | float64 | 24 | 0.156886 | 0.156886 | 0.156886 |
| forecast_metric_scope | object | 0 | Pooled workload WAPE across available business-unit observations in each split; not unit-specific. Validation and test coverage may differ. | Pooled workload WAPE across available business-unit observations in each split; not unit-specific. Validation and test coverage may differ. | Pooled workload WAPE across available business-unit observations in each split; not unit-specific. Validation and test coverage may differ. |
| latest_observed_workload | float64 | 0 | 2.57157 | 2.27394 | 1.99127 |
| recent_workload_growth | float64 | 0 | 0.137199 | 0.0996808 | 0.0262668 |
| recent_workload_growth_basis | object | 0 | Mean workload in the latest 3 calendar months / mean in the preceding 3 months - 1; as of last_observed_month, requiring all 6 saved actuals. | Mean workload in the latest 3 calendar months / mean in the preceding 3 months - 1; as of last_observed_month, requiring all 6 saved actuals. | Mean workload in the latest 3 calendar months / mean in the preceding 3 months - 1; as of last_observed_month, requiring all 6 saved actuals. |
| capacity_priority | object | 0 | Potential Spare Capacity | Potential Spare Capacity | Potential Spare Capacity |

## dashboard_capacity_scenarios.csv

Rows: **170**. Columns: **20**. Grain: **team × country × product_class × scenario_name**.

| Field | Export dtype | Null rows | Row 1 | Row 2 | Row 3 |
|---|---|---:|---|---|---|
| team | object | 0 | Alfa | Alfa | Alfa |
| country | object | 0 | Germany | Germany | Germany |
| product_class | object | 0 | Analgesics | Analgesics | Analgesics |
| forecast_horizon | object | 0 | 2019-05-01 | 2019-05-01 | 2019-05-01 |
| scenario_name | object | 0 | Add 1 FTE | Add 2 FTE | Base |
| scenario_description | object | 0 | One FTE is added to this unit independently; workload and per-rep capacity are unchanged. | Two FTE are added to this unit independently; workload and per-rep capacity are unchanged. | Persisted base workload forecast, fractional FTE allocation, and sustainable per-rep capacity. |
| forecast_workload | float64 | 0 | 3.00959 | 3.00959 | 3.00959 |
| allocated_fte | float64 | 0 | 1.51703 | 2.51703 | 0.51703 |
| sustainable_capacity_per_rep | float64 | 0 | 7.48169 | 7.48169 | 7.48169 |
| required_fte | float64 | 0 | 0.402261 | 0.402261 | 0.402261 |
| fte_gap | float64 | 0 | -1.11477 | -2.11477 | -0.114769 |
| capacity_priority | object | 0 | Potential Spare Capacity | Potential Spare Capacity | Potential Spare Capacity |
| eligible_for_capacity_recommendation | bool | 0 | True | True | True |
| source_unit | object | 168 | null | null | null |
| target_unit | object | 168 | null | null | null |
| fte_reallocated | float64 | 168 | null | null | null |
| reallocation_role | object | 168 | null | null | null |
| allocated_fte_change | float64 | 0 | 1 | 2 | 1.11022e-16 |
| scenario_source_name | object | 0 | Add 1 representative | Add 2 representatives | Base case |
| scenario_source_description | object | 0 | demand=+0%; FTE=+1; capacity=+0% | demand=+0%; FTE=+2; capacity=+0% | demand=+0%; FTE=+0; capacity=+0% |

## dashboard_model_summary.csv

Rows: **12**. Columns: **13**. Grain: **model**.

| Field | Export dtype | Null rows | Row 1 | Row 2 | Row 3 |
|---|---|---:|---|---|---|
| model | object | 0 | PCA Reconstruction | Autoencoder | Best Ensemble |
| role | object | 0 | Primary anomaly ranking | Reconstruction comparator | Benchmark comparator only |
| manager_facing_label | object | 0 | Pattern deviation | Nonlinear pattern deviation | Combined comparison signal |
| selected_for_primary_use | bool | 0 | True | False | False |
| recall_at_5pct | float64 | 0 | 0.421053 | 0.421053 | 0.473684 |
| lift_at_5pct | float64 | 0 | 8.21053 | 8.21053 | 9.23684 |
| precision_at_5pct | float64 | 0 | 0.5 | 0.5 | 0.5625 |
| pr_auc | float64 | 0 | 0.394625 | 0.382063 | 0.349644 |
| f1 | float64 | 0 | 0.457143 | 0.457143 | 0.514286 |
| f2 | float64 | 0 | 0.434783 | 0.434783 | 0.48913 |
| stability | float64 | 0 | 1 | 0.916667 | 1 |
| runtime_seconds | float64 | 0 | 0.0612263 | 4.72209 | 12.8953 |
| business_interpretation | object | 0 | Linear reconstruction deviation from training patterns; review signal, not a probability. | Nonlinear reconstruction comparator; consult the recorded convergence limitations. | Combined signals evaluated on the same controlled benchmark; selection remains validation-based. |

## Provenance and definitions

Source SHA-256: `f77609eaa7700c964eee2f87a6bb1c75ca6263099969f7bc981fe6822f104cd3`.

Source rows: 167,760; analytical rows: 2,184; seed: 42.

Poland source coverage ends December 2018. Its May 2019 units remain ineligible; missing records are not zero demand or zero staffing.

- **review_priority:** High: persisted PCA exact-5% flag, or TRAIN percentile >= very_high with >=1 support. Medium: percentile >= elevated, or >= multiple_support_count supporting families. Otherwise Low. Thresholds are presentation rules, not test-optimized.
- **review_rank:** Snapshot-wide PCA raw-score descending, ties observation_id ascending. Retrospective display rank; never a historical model input.
- **pca_review_flag:** Exact budget flag already executed within each train/validation/test partition, not a new global top-5% selection.
- **pca_threshold_exceedance:** Persisted boolean exceedance of the raw-score TRAIN-reference threshold.
- **supporting_signals:** Three supporting families: EWMA exact-budget flag, robust-peer exact-budget flag, and any executed business-rule binary flag. Count excludes PCA and correlated model comparators. Not statistical independence or certainty.
- **transaction_value:** Mean transaction value, copied from average_transaction_value; sales is the monthly total.
- **repeat_customers:** unique_customers × persisted repeat_customer_ratio, only when it identifies an integer count.
- **lost_customers:** Previous calendar-month same rep/class customer count × current loss rate; null without the previous month or integer-identifying arithmetic.
- **top_driver_contribution:** Unmodified per-feature squared reconstruction error in transformed model space, NOT a percentage. Top three exported PCA features; technical feature ids retained.
- **strongest_deviation:** Largest absolute signed peer robust-z or personal-history deviation; original sign retained.
- **temporal_context:** Strongest EWMA metric by persisted raw signal, with separate SALES-only observed/expected fields for charts; unavailable direction remains null.
- **channel_summary:** Persisted diversity indices only; channel/distributor names were not retained and are not invented.
- **latest_rep_percentile:** Maximum across product-class observations in that representative’s latest month; latest priority is the highest latest-month category.
- **strongest_recent_driver:** Top PCA contributor of the highest-percentile product-class row in the representative’s latest month.
- **model_agreement_high_count:** Representative-level count of observations with at least two of the three supporting signal families; not a count of confirmed outcomes.
- **recent_3m_sales:** Latest available month plus two preceding calendar months; prior_3m_sales is the preceding three. Incomplete windows remain null. Growth=(recent-prior)/abs(prior); zero prior denominator is null.
- **unique_customers_latest:** Direct rep-month unique count, never summed across product classes.
- **customer_change_3m:** Latest direct rep-month unique count minus the exact three-calendar-month-earlier count; null if either is unavailable.
- **primary_country:** Latest-month singleton country only; multi-country ambiguity remains null.
- **fte_availability:** Base case assumes 13 observed representatives are available to the observed Germany scope; missing Poland records do not verify their actual cross-country availability.
- **capacity_priority:** Signed modeled FTE gap with numerical tolerance, not a hiring or employment decision.
- **forecast_scenario_bounds:** Scenario bounds reconstructed only from persisted forecast uncertainty/capacity bounds; not confidence intervals.

Capacity workload components are independently forecast counts, not additive shares of the composite index. Workload forecast bounds invert persisted FTE bounds using saved training capacity quantiles. Validation/test WAPE is pooled by selected method; country coverage differs by split. Net-zero reallocation uses signed FTE changes and conserved donor/receiver pairing.

The metadata git commit identifies HEAD at export time; git_worktree_dirty discloses uncommitted implementation changes. It does not claim to identify the later commit that includes the generated files.

## Executed verification

Preserved original source/technical files: 141; unchanged: True.

Full benchmark executed in an isolated directory: True; runtime: 127.233 seconds.

Benchmark metrics unchanged: True; model selection unchanged: True; capacity outputs unchanged: True.

Tests: 90; failures: 0; errors: 0; skipped: 0.
