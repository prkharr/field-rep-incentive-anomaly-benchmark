# Commercial review generated-data dictionary

All fields below are generated from the executed schema. `source_observed` rows retain source facts; `synthetic_derived`, `synthetic_normal`, and `synthetic_injected` identify controlled additions. `UNK` currency means no reliable source currency was available, so no conversion was attempted.

## `anomaly_ground_truth`

Executed rows: 64.

| Field | Type | Definition |
|---|---|---|
| `injection_id` | `object` | Stable identifier for injection. |
| `entity_type` | `object` | Entity type. |
| `entity_id` | `object` | Stable identifier for entity. |
| `rep_id` | `object` | Stable identifier for rep. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `anomaly_type` | `object` | Anomaly type. |
| `anomaly_category` | `object` | Anomaly category. |
| `severity` | `object` | Severity. |
| `affected_dataset` | `object` | Affected dataset. |
| `affected_record_ids` | `object` | Affected record ids. |
| `injection_description` | `object` | Injection description. |
| `original_value` | `object` | Original value. |
| `injected_value` | `object` | Injected value. |
| `expected_detection_signals` | `object` | Expected detection signals. |
| `ground_truth_label` | `int64` | Controlled synthetic evaluation label; excluded from all model features. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `correlated_case_flag` | `bool` | Boolean indicator for correlated case. |

## `capacity_calendar`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `capacity_record_id` | `object` | Stable identifier for capacity record. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `active_territory_count` | `int64` | Active territory count. |
| `dominant_territory_activity_share` | `float64` | Ratio for dominant territory activity share; zero-denominator handling is explicit. |
| `fractional_territory_allocation` | `object` | Fractional territory allocation. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `working_days` | `int64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `standard_field_hours_per_day` | `float64` | Standard field hours per day. |
| `standard_working_days_per_month` | `int64` | Standard working days per month. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `customer_coverage_pct` | `float64` | Percentage value for customer coverage. |
| `average_travel_hours` | `float64` | Average travel hours. |
| `workload_per_active_customer` | `float64` | Workload per active customer. |
| `legacy_normalized_workload_index` | `float64` | Legacy normalized workload index. |
| `workload_risk_band` | `object` | Workload risk band. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `distinct_customers` | `int64` | Distinct customers. |
| `transaction_count` | `int64` | Transaction count. |
| `distinct_cities` | `int64` | Distinct cities. |
| `distinct_products` | `int64` | Distinct products. |
| `distributor_count` | `int64` | Distributor count. |
| `legacy_distinct_customers_training_median` | `float64` | Legacy distinct customers training median. |
| `legacy_transaction_count_training_median` | `float64` | Legacy transaction count training median. |
| `legacy_distinct_cities_training_median` | `float64` | Legacy distinct cities training median. |
| `legacy_distinct_products_training_median` | `float64` | Legacy distinct products training median. |
| `legacy_distributor_count_training_median` | `float64` | Legacy distributor count training median. |
| `legacy_workload_training_end` | `datetime64[ns]` | Legacy workload training end. |
| `legacy_workload_formula` | `object` | Legacy workload formula. |
| `hire_date` | `datetime64[ns]` | Hire date. |
| `employment_status` | `object` | Employment status. |
| `role_grade` | `object` | Role grade. |
| `active_customer_count` | `int64` | Active customer count. |
| `priority_customer_count` | `int64` | Priority customer count. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `observed_visit_count` | `int64` | Observed visit count. |
| `observed_completed_visit_count` | `int64` | Observed completed visit count. |
| `priority_customer_coverage_gap_pct` | `float64` | Percentage value for priority customer coverage gap. |
| `core_required_hours` | `float64` | Core required hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `capacity_zero_denominator_flag` | `bool` | Boolean indicator for capacity zero denominator. |
| `required_hours` | `float64` | Required hours. |
| `available_hours` | `float64` | Available hours. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `float64` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `required_hours_formula` | `object` | Required hours formula. |
| `required_workload_scope` | `object` | Required workload scope. |
| `numeric_visit_frequency_period_divisor` | `float64` | Numeric visit frequency period divisor. |
| `capacity_methodology` | `object` | Capacity methodology. |

## `capacity_customer_drilldown`

Executed rows: 18,627.

| Field | Type | Definition |
|---|---|---|
| `rep_id` | `object` | Stable identifier for rep. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `customer_id` | `object` | Stable identifier for customer. |
| `customer_name` | `object` | Customer name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `geography` | `object` | Geography. |
| `channel` | `object` | Channel. |
| `customer_type` | `object` | Customer type. |
| `customer_segment` | `object` | Customer segment. |
| `customer_priority` | `object` | Customer priority. |
| `potential_score` | `float64` | Potential score. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `servicing_rep_count` | `float64` | Servicing rep count. |
| `observed_visit_count` | `float64` | Observed visit count. |
| `observed_completed_visit_count` | `float64` | Observed completed visit count. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `planned_coverage_gap_count` | `float64` | Planned coverage gap count. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `estimated_travel_hours_per_visit` | `float64` | Estimated travel hours per visit. |
| `required_coverage_visit_hours` | `float64` | Required coverage visit hours. |
| `required_coverage_travel_hours` | `float64` | Required coverage travel hours. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `customer_coverage_pct` | `float64` | Percentage value for customer coverage. |
| `coverage_met_flag` | `bool` | Boolean indicator for coverage met. |
| `priority_customer_flag` | `bool` | Boolean indicator for priority customer. |
| `coverage_visit_scope` | `object` | Coverage visit scope. |
| `coverage_status` | `object` | Coverage status. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `synthetic_seed` | `int64` | Synthetic seed. |

## `capacity_territory_allocation`

Executed rows: 2,022.

| Field | Type | Definition |
|---|---|---|
| `capacity_territory_allocation_id` | `object` | Stable identifier for capacity territory allocation. |
| `capacity_record_id` | `object` | Stable identifier for capacity record. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `transaction_count` | `float64` | Transaction count. |
| `territory_allocation_share` | `float64` | Ratio for territory allocation share; zero-denominator handling is explicit. |
| `dominant_territory_id` | `object` | Stable identifier for dominant territory. |
| `dominant_territory_name` | `object` | Dominant territory name. |
| `dominant_territory_flag` | `boolean` | Boolean indicator for dominant territory. |
| `allocation_basis` | `object` | Allocation basis. |
| `geographic_workload_attribution_flag` | `bool` | Boolean indicator for geographic workload attribution. |
| `residual_allocation_flag` | `bool` | Boolean indicator for residual allocation. |
| `injected_residual_allocation_flag` | `bool` | Boolean indicator for injected residual allocation. |
| `unrepresented_workload_residual_hours` | `float64` | Unrepresented workload residual hours. |
| `residual_allocation_columns` | `object` | Residual allocation columns. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `observed_visit_count` | `float64` | Observed visit count. |
| `observed_completed_visit_count` | `float64` | Observed completed visit count. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `float64` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `source_rep_period_utilization_pct` | `float64` | Percentage value for source rep period utilization. |
| `source_rep_period_risk_band` | `object` | Source rep period risk band. |
| `source_rep_period_overload_flag` | `bool` | Boolean indicator for source rep period overload. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `core_required_hours` | `float64` | Core required hours. |
| `territory_activity_count` | `float64` | Territory activity count. |
| `working_days` | `float64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `available_hours` | `float64` | Available hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `required_hours` | `float64` | Required hours. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `workload_risk_band` | `object` | Workload risk band. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `core_workload_geographic_attribution_flag` | `bool` | Boolean indicator for core workload geographic attribution. |
| `availability_geographic_attribution_flag` | `bool` | Boolean indicator for availability geographic attribution. |
| `allocation_scope` | `object` | Allocation scope. |

## `capacity_territory_summary`

Executed rows: 176.

| Field | Type | Definition |
|---|---|---|
| `capacity_territory_record_id` | `object` | Stable identifier for capacity territory record. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `rep_count` | `int64` | Rep count. |
| `fractional_rep_equivalent` | `float64` | Fractional rep equivalent. |
| `allocation_row_count` | `int64` | Allocation row count. |
| `required_total_hours` | `float64` | Required total hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `positive_fte_gap` | `float64` | Positive fte gap. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `transaction_count` | `float64` | Transaction count. |
| `working_days` | `float64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `observed_visit_count` | `float64` | Observed visit count. |
| `observed_completed_visit_count` | `float64` | Observed completed visit count. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `core_required_hours` | `float64` | Core required hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `required_hours` | `float64` | Required hours. |
| `available_hours` | `float64` | Available hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `rep_ids` | `object` | Rep ids. |
| `allocation_basis` | `object` | Allocation basis. |
| `allocation_scope` | `object` | Allocation scope. |
| `geographic_workload_attribution_flag` | `bool` | Boolean indicator for geographic workload attribution. |
| `core_workload_geographic_attribution_flag` | `bool` | Boolean indicator for core workload geographic attribution. |
| `availability_geographic_attribution_flag` | `bool` | Boolean indicator for availability geographic attribution. |
| `residual_allocation_flag` | `bool` | Boolean indicator for residual allocation. |
| `injected_residual_allocation_flag` | `bool` | Boolean indicator for injected residual allocation. |
| `unrepresented_workload_residual_hours` | `float64` | Unrepresented workload residual hours. |
| `manager_ids` | `object` | Manager ids. |
| `manager_names` | `object` | Manager names. |
| `team_ids` | `object` | Team ids. |
| `team_names` | `object` | Team names. |
| `risk_medium_threshold_pct` | `object` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `object` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `object` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `object` | Percentage value for overload threshold. |
| `synthetic_seed` | `object` | Synthetic seed. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `workload_risk_band` | `object` | Workload risk band. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `capacity_methodology` | `object` | Capacity methodology. |

## `crm_interactions`

Executed rows: 70,925.

| Field | Type | Definition |
|---|---|---|
| `interaction_id` | `object` | Stable identifier for interaction. |
| `visit_id` | `object` | Stable identifier for visit. |
| `rep_id` | `object` | Stable identifier for rep. |
| `customer_id` | `object` | Stable identifier for customer. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `interaction_datetime` | `datetime64[ns]` | Interaction datetime. |
| `interaction_type` | `object` | Interaction type. |
| `product_focus` | `object` | Product focus. |
| `interaction_outcome` | `object` | Interaction outcome. |
| `follow_up_required_flag` | `bool` | Boolean indicator for follow up required. |
| `next_action_date` | `datetime64[ns]` | Next action date. |
| `sentiment_or_interest_score` | `float64` | Sentiment or interest score. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `customer_master`

Executed rows: 751.

| Field | Type | Definition |
|---|---|---|
| `customer_id` | `object` | Stable identifier for customer. |
| `customer_name` | `object` | Customer name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `primary_rep_id` | `object` | Stable identifier for primary rep. |
| `geography` | `object` | Geography. |
| `country` | `object` | Country. |
| `channel` | `object` | Channel. |
| `customer_type` | `object` | Customer type. |
| `source_sales` | `float64` | Source sales. |
| `expected_monthly_volume` | `float64` | Expected monthly volume. |
| `source_return_rows` | `int64` | Source return rows. |
| `source_rows` | `int64` | Source rows. |
| `usual_product_classes` | `object` | Usual product classes. |
| `customer_segment` | `object` | Customer segment. |
| `potential_score` | `float64` | Potential score. |
| `expected_sales_band` | `object` | Expected sales band. |
| `required_visit_frequency` | `int64` | Required visit frequency. |
| `customer_priority` | `object` | Customer priority. |
| `historical_return_rate` | `float64` | Ratio for historical return rate; zero-denominator handling is explicit. |
| `synthetic_latitude` | `float64` | Synthetic latitude. |
| `synthetic_longitude` | `float64` | Synthetic longitude. |
| `coordinate_lineage` | `object` | Coordinate lineage. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `attribute_reference_end_date` | `datetime64[s]` | Attribute reference end date. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `dashboard_anomaly_evidence`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `observation_id` | `object` | Stable identifier for observation. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `split` | `object` | Split. |
| `anomaly_score` | `float64` | Training-reference percentile of PCA reconstruction error; higher is more unusual. |
| `raw_score` | `float64` | Raw score. |
| `threshold_flag` | `bool` | Boolean indicator for threshold. |
| `manager_review_flag` | `bool` | Boolean indicator for manager review. |
| `primary_reason` | `object` | Primary reason. |
| `recommended_review_action` | `object` | Recommended review action. |
| `ground_truth_label` | `int64` | Controlled synthetic evaluation label; excluded from all model features. |
| `anomaly_type` | `object` | Anomaly type. |
| `anomaly_category` | `object` | Anomaly category. |
| `severity` | `object` | Severity. |
| `injection_count` | `int64` | Injection count. |
| `benchmark_mode_disclosure` | `object` | Benchmark mode disclosure. |

## `dashboard_anomaly_type_metrics`

Executed rows: 48.

| Field | Type | Definition |
|---|---|---|
| `group_kind` | `object` | Group kind. |
| `grouping` | `object` | Grouping. |
| `value` | `object` | Value. |
| `observations` | `int64` | Observations. |
| `positive_support` | `int64` | Positive support. |
| `overall_truth_support` | `int64` | Overall truth support. |
| `selected_at_threshold` | `int64` | Selected at threshold. |
| `captured_at_threshold` | `int64` | Captured at threshold. |
| `false_positives_at_threshold` | `float64` | False positives at threshold. |
| `precision_at_threshold` | `float64` | Precision at threshold. |
| `recall_at_threshold` | `float64` | Recall at threshold. |
| `detection_rate_at_threshold` | `float64` | Detection rate at threshold. |
| `captured_at_top5pct` | `float64` | Captured at top5pct. |
| `recall_at_top5pct` | `float64` | Recall at top5pct. |
| `detection_rate_at_top5pct` | `float64` | Detection rate at top5pct. |
| `support_status` | `object` | Support status. |
| `evaluation_scope` | `object` | Evaluation scope. |

## `dashboard_capacity_customer_drilldown`

Executed rows: 18,627.

| Field | Type | Definition |
|---|---|---|
| `rep_id` | `object` | Stable identifier for rep. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `customer_id` | `object` | Stable identifier for customer. |
| `customer_name` | `object` | Customer name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `geography` | `object` | Geography. |
| `channel` | `object` | Channel. |
| `customer_type` | `object` | Customer type. |
| `customer_segment` | `object` | Customer segment. |
| `customer_priority` | `object` | Customer priority. |
| `potential_score` | `float64` | Potential score. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `servicing_rep_count` | `float64` | Servicing rep count. |
| `observed_visit_count` | `float64` | Observed visit count. |
| `observed_completed_visit_count` | `float64` | Observed completed visit count. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `planned_coverage_gap_count` | `float64` | Planned coverage gap count. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `estimated_travel_hours_per_visit` | `float64` | Estimated travel hours per visit. |
| `required_coverage_visit_hours` | `float64` | Required coverage visit hours. |
| `required_coverage_travel_hours` | `float64` | Required coverage travel hours. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `customer_coverage_pct` | `float64` | Percentage value for customer coverage. |
| `coverage_met_flag` | `bool` | Boolean indicator for coverage met. |
| `priority_customer_flag` | `bool` | Boolean indicator for priority customer. |
| `coverage_visit_scope` | `object` | Coverage visit scope. |
| `coverage_status` | `object` | Coverage status. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `rep_name` | `object` | Rep name. |

## `dashboard_capacity_summary`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `capacity_record_id` | `object` | Stable identifier for capacity record. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `active_territory_count` | `int64` | Active territory count. |
| `dominant_territory_activity_share` | `float64` | Ratio for dominant territory activity share; zero-denominator handling is explicit. |
| `fractional_territory_allocation` | `object` | Fractional territory allocation. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `working_days` | `int64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `standard_field_hours_per_day` | `float64` | Standard field hours per day. |
| `standard_working_days_per_month` | `int64` | Standard working days per month. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `customer_coverage_pct` | `float64` | Percentage value for customer coverage. |
| `average_travel_hours` | `float64` | Average travel hours. |
| `workload_per_active_customer` | `float64` | Workload per active customer. |
| `legacy_normalized_workload_index` | `float64` | Legacy normalized workload index. |
| `workload_risk_band` | `object` | Workload risk band. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `distinct_customers` | `int64` | Distinct customers. |
| `transaction_count` | `int64` | Transaction count. |
| `distinct_cities` | `int64` | Distinct cities. |
| `distinct_products` | `int64` | Distinct products. |
| `distributor_count` | `int64` | Distributor count. |
| `legacy_distinct_customers_training_median` | `float64` | Legacy distinct customers training median. |
| `legacy_transaction_count_training_median` | `float64` | Legacy transaction count training median. |
| `legacy_distinct_cities_training_median` | `float64` | Legacy distinct cities training median. |
| `legacy_distinct_products_training_median` | `float64` | Legacy distinct products training median. |
| `legacy_distributor_count_training_median` | `float64` | Legacy distributor count training median. |
| `legacy_workload_training_end` | `datetime64[ns]` | Legacy workload training end. |
| `legacy_workload_formula` | `object` | Legacy workload formula. |
| `hire_date` | `datetime64[ns]` | Hire date. |
| `employment_status` | `object` | Employment status. |
| `role_grade` | `object` | Role grade. |
| `active_customer_count` | `int64` | Active customer count. |
| `priority_customer_count` | `int64` | Priority customer count. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `observed_visit_count` | `int64` | Observed visit count. |
| `observed_completed_visit_count` | `int64` | Observed completed visit count. |
| `priority_customer_coverage_gap_pct` | `float64` | Percentage value for priority customer coverage gap. |
| `core_required_hours` | `float64` | Core required hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `capacity_zero_denominator_flag` | `bool` | Boolean indicator for capacity zero denominator. |
| `required_hours` | `float64` | Required hours. |
| `available_hours` | `float64` | Available hours. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `float64` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `required_hours_formula` | `object` | Required hours formula. |
| `required_workload_scope` | `object` | Required workload scope. |
| `numeric_visit_frequency_period_divisor` | `float64` | Numeric visit frequency period divisor. |
| `capacity_methodology` | `object` | Capacity methodology. |

## `dashboard_capacity_territory_allocation`

Executed rows: 2,022.

| Field | Type | Definition |
|---|---|---|
| `capacity_territory_allocation_id` | `object` | Stable identifier for capacity territory allocation. |
| `capacity_record_id` | `object` | Stable identifier for capacity record. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `transaction_count` | `float64` | Transaction count. |
| `territory_allocation_share` | `float64` | Ratio for territory allocation share; zero-denominator handling is explicit. |
| `dominant_territory_id` | `object` | Stable identifier for dominant territory. |
| `dominant_territory_name` | `object` | Dominant territory name. |
| `dominant_territory_flag` | `boolean` | Boolean indicator for dominant territory. |
| `allocation_basis` | `object` | Allocation basis. |
| `geographic_workload_attribution_flag` | `bool` | Boolean indicator for geographic workload attribution. |
| `residual_allocation_flag` | `bool` | Boolean indicator for residual allocation. |
| `injected_residual_allocation_flag` | `bool` | Boolean indicator for injected residual allocation. |
| `unrepresented_workload_residual_hours` | `float64` | Unrepresented workload residual hours. |
| `residual_allocation_columns` | `object` | Residual allocation columns. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `observed_visit_count` | `float64` | Observed visit count. |
| `observed_completed_visit_count` | `float64` | Observed completed visit count. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `float64` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `source_rep_period_utilization_pct` | `float64` | Percentage value for source rep period utilization. |
| `source_rep_period_risk_band` | `object` | Source rep period risk band. |
| `source_rep_period_overload_flag` | `bool` | Boolean indicator for source rep period overload. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `core_required_hours` | `float64` | Core required hours. |
| `territory_activity_count` | `float64` | Territory activity count. |
| `working_days` | `float64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `available_hours` | `float64` | Available hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `required_hours` | `float64` | Required hours. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `workload_risk_band` | `object` | Workload risk band. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `core_workload_geographic_attribution_flag` | `bool` | Boolean indicator for core workload geographic attribution. |
| `availability_geographic_attribution_flag` | `bool` | Boolean indicator for availability geographic attribution. |
| `allocation_scope` | `object` | Allocation scope. |

## `dashboard_capacity_territory_summary`

Executed rows: 176.

| Field | Type | Definition |
|---|---|---|
| `capacity_territory_record_id` | `object` | Stable identifier for capacity territory record. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `rep_count` | `int64` | Rep count. |
| `fractional_rep_equivalent` | `float64` | Fractional rep equivalent. |
| `allocation_row_count` | `int64` | Allocation row count. |
| `required_total_hours` | `float64` | Required total hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `positive_fte_gap` | `float64` | Positive fte gap. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `transaction_count` | `float64` | Transaction count. |
| `working_days` | `float64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `completed_visit_count` | `float64` | Completed visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `observed_visit_count` | `float64` | Observed visit count. |
| `observed_completed_visit_count` | `float64` | Observed completed visit count. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `core_required_hours` | `float64` | Core required hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `required_hours` | `float64` | Required hours. |
| `available_hours` | `float64` | Available hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `rep_ids` | `object` | Rep ids. |
| `allocation_basis` | `object` | Allocation basis. |
| `allocation_scope` | `object` | Allocation scope. |
| `geographic_workload_attribution_flag` | `bool` | Boolean indicator for geographic workload attribution. |
| `core_workload_geographic_attribution_flag` | `bool` | Boolean indicator for core workload geographic attribution. |
| `availability_geographic_attribution_flag` | `bool` | Boolean indicator for availability geographic attribution. |
| `residual_allocation_flag` | `bool` | Boolean indicator for residual allocation. |
| `injected_residual_allocation_flag` | `bool` | Boolean indicator for injected residual allocation. |
| `unrepresented_workload_residual_hours` | `float64` | Unrepresented workload residual hours. |
| `manager_ids` | `object` | Manager ids. |
| `manager_names` | `object` | Manager names. |
| `team_ids` | `object` | Team ids. |
| `team_names` | `object` | Team names. |
| `risk_medium_threshold_pct` | `object` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `object` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `object` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `object` | Percentage value for overload threshold. |
| `synthetic_seed` | `object` | Synthetic seed. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `workload_risk_band` | `object` | Workload risk band. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `capacity_methodology` | `object` | Capacity methodology. |

## `dashboard_confusion_matrix`

Executed rows: 4.

| Field | Type | Definition |
|---|---|---|
| `actual` | `object` | Actual. |
| `predicted` | `object` | Predicted. |
| `count` | `int64` | Count. |

## `dashboard_data_quality`

Executed rows: 248.

| Field | Type | Definition |
|---|---|---|
| `check_name` | `object` | Check name. |
| `status` | `object` | Status. |
| `value` | `object` | Value. |
| `detail` | `object` | Detail. |
| `exists` | `object` | Exists. |

## `dashboard_feature_contributions`

Executed rows: 1,092.

| Field | Type | Definition |
|---|---|---|
| `rep_id` | `object` | Stable identifier for rep. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `manager_id` | `object` | Stable identifier for manager. |
| `team_id` | `object` | Stable identifier for team. |
| `territory_id` | `object` | Stable identifier for territory. |
| `observation_id` | `object` | Stable identifier for observation. |
| `population` | `object` | Population. |
| `anomaly_score` | `float64` | Training-reference percentile of PCA reconstruction error; higher is more unusual. |
| `primary_reason_code` | `object` | Primary reason code. |
| `primary_reason` | `object` | Primary reason. |
| `secondary_reason` | `object` | Secondary reason. |
| `recommended_review_action` | `object` | Recommended review action. |
| `driver_rank` | `int64` | Driver rank. |
| `feature` | `object` | Feature. |
| `name` | `object` | Name. |
| `value` | `float64` | Value. |
| `peer_value` | `float64` | Peer value. |
| `percentile` | `float64` | Percentile. |
| `contribution` | `float64` | Contribution. |

## `dashboard_kpi_summary`

Executed rows: 1.

| Field | Type | Definition |
|---|---|---|
| `total_gross_sales` | `float64` | Total gross sales. |
| `total_net_sales` | `float64` | Total net sales. |
| `total_incentive_paid` | `float64` | Total incentive paid. |
| `review_candidate_count` | `int64` | Review candidate count. |
| `high_priority_review_candidate_count` | `int64` | High priority review candidate count. |
| `review_rate` | `float64` | Ratio for review rate; zero-denominator handling is explicit. |
| `overloaded_representative_count` | `int64` | Overloaded representative count. |
| `overloaded_territory_count` | `int64` | Overloaded territory count. |
| `total_positive_fte_gap` | `float64` | Total positive fte gap. |
| `selected_threshold` | `float64` | Selected threshold. |
| `test_precision_at_selected_threshold` | `float64` | Test precision at selected threshold. |
| `test_recall_at_selected_threshold` | `float64` | Test recall at selected threshold. |
| `benchmark_mode_label_disclosure` | `object` | Benchmark mode label disclosure. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |

## `dashboard_manager_review_queue`

Executed rows: 19.

| Field | Type | Definition |
|---|---|---|
| `review_rank` | `int64` | Review rank. |
| `observation_id` | `object` | Stable identifier for observation. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `split` | `object` | Split. |
| `anomaly_score` | `float64` | Training-reference percentile of PCA reconstruction error; higher is more unusual. |
| `anomaly_percentile` | `float64` | Anomaly percentile. |
| `raw_score` | `float64` | Raw score. |
| `raw_threshold` | `float64` | Raw threshold. |
| `threshold_flag` | `bool` | Boolean indicator for threshold. |
| `manager_review_flag` | `bool` | Boolean indicator for manager review. |
| `review_priority` | `object` | Review priority. |
| `risk_band` | `object` | Risk band. |
| `primary_reason_code` | `object` | Primary reason code. |
| `primary_reason` | `object` | Primary reason. |
| `secondary_reason` | `object` | Secondary reason. |
| `driver_1_name` | `object` | Driver 1 name. |
| `driver_1_value` | `float64` | Driver 1 value. |
| `driver_1_peer_value` | `float64` | Driver 1 peer value. |
| `driver_1_percentile` | `float64` | Driver 1 percentile. |
| `driver_2_name` | `object` | Driver 2 name. |
| `driver_2_value` | `float64` | Driver 2 value. |
| `driver_2_peer_value` | `float64` | Driver 2 peer value. |
| `driver_3_name` | `object` | Driver 3 name. |
| `driver_3_value` | `float64` | Driver 3 value. |
| `recommended_review_action` | `object` | Recommended review action. |
| `gross_sales` | `float64` | Gross sales. |
| `net_sales` | `float64` | Net sales. |
| `target_sales` | `float64` | Target sales. |
| `attainment_pct` | `float64` | Percentage value for attainment. |
| `final_incentive_paid` | `float64` | Final incentive paid. |
| `expected_incentive` | `float64` | Expected incentive. |
| `incentive_residual` | `float64` | Incentive residual. |
| `payout_to_peer_median_ratio` | `float64` | Ratio for payout to peer median ratio; zero-denominator handling is explicit. |
| `average_discount_pct` | `float64` | Percentage value for average discount. |
| `post_incentive_return_rate` | `float64` | Ratio for post incentive return rate; zero-denominator handling is explicit. |
| `impossible_travel_count` | `int64` | Impossible travel count. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `dashboard_model_curve`

Executed rows: 120.

| Field | Type | Definition |
|---|---|---|
| `false_positive_rate` | `float64` | Ratio for false positive rate; zero-denominator handling is explicit. |
| `true_positive_rate` | `float64` | Ratio for true positive rate; zero-denominator handling is explicit. |
| `threshold` | `float64` | Threshold. |
| `curve_type` | `object` | Curve type. |
| `precision` | `float64` | Precision. |
| `recall` | `float64` | Recall. |
| `review_rank` | `float64` | Review rank. |
| `review_fraction` | `float64` | Review fraction. |
| `captured_count` | `float64` | Captured count. |
| `lift` | `float64` | Lift. |

## `dashboard_model_metrics`

Executed rows: 38.

| Field | Type | Definition |
|---|---|---|
| `model` | `object` | Model. |
| `split` | `object` | Split. |
| `threshold` | `float64` | Threshold. |
| `manager_review_fraction` | `float64` | Manager review fraction. |
| `metric_name` | `object` | Metric name. |
| `metric_value` | `float64` | Metric value. |
| `review_fraction` | `float64` | Review fraction. |
| `review_count` | `float64` | Review count. |
| `positive_count` | `float64` | Positive count. |
| `captured_count` | `float64` | Captured count. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `precision` | `float64` | Precision. |
| `recall` | `float64` | Recall. |
| `overload_precision` | `float64` | Overload precision. |
| `overload_recall` | `float64` | Overload recall. |
| `true_positive` | `float64` | True positive. |
| `false_positive` | `float64` | False positive. |
| `false_negative` | `float64` | False negative. |
| `true_negative` | `float64` | True negative. |
| `ground_truth_overload_count` | `float64` | Ground truth overload count. |
| `capacity_ground_truth_row_count` | `float64` | Capacity ground truth row count. |
| `undercoverage_ground_truth_row_count` | `float64` | Undercoverage ground truth row count. |
| `unmatched_overload_truth_row_count` | `float64` | Unmatched overload truth row count. |
| `predicted_overload_count` | `float64` | Predicted overload count. |
| `above_medium_threshold_count` | `float64` | Above medium threshold count. |
| `above_high_threshold_count` | `float64` | Above high threshold count. |
| `above_critical_threshold_count` | `float64` | Above critical threshold count. |
| `overloaded_rep_period_count` | `float64` | Overloaded rep period count. |
| `reps_above_medium_threshold` | `float64` | Reps above medium threshold. |
| `reps_above_high_threshold` | `float64` | Reps above high threshold. |
| `reps_above_critical_threshold` | `float64` | Reps above critical threshold. |
| `overloaded_territory_count` | `float64` | Overloaded territory count. |
| `territories_above_medium_threshold` | `float64` | Territories above medium threshold. |
| `territories_above_critical_threshold` | `float64` | Territories above critical threshold. |
| `overloaded_territory_period_count` | `float64` | Overloaded territory period count. |
| `territory_period_count` | `float64` | Territory period count. |
| `clean_row_count` | `float64` | Clean row count. |
| `injected_row_count` | `float64` | Injected row count. |
| `territory_count` | `float64` | Territory count. |
| `mae_required_total_hours` | `float64` | Mae required total hours. |
| `workload_mae` | `float64` | Workload mae. |
| `workload_mae_observations` | `float64` | Workload mae observations. |
| `workload_mae_basis` | `object` | Workload mae basis. |
| `mae_utilization_pct` | `float64` | Percentage value for mae utilization. |
| `utilization_mae` | `float64` | Utilization mae. |
| `utilization_mae_observations` | `float64` | Utilization mae observations. |
| `utilization_mae_basis` | `object` | Utilization mae basis. |
| `numeric_truth_independent_flag` | `object` | Boolean indicator for numeric truth independent. |
| `numeric_mae_interpretation` | `object` | Numeric mae interpretation. |
| `clean_injected_required_hours_mae` | `float64` | Clean injected required hours mae. |
| `clean_injected_utilization_pct_mae` | `float64` | Clean injected utilization pct mae. |
| `territory_ranking_agreement` | `float64` | Territory ranking agreement. |
| `territory_rank_spearman` | `float64` | Territory rank spearman. |
| `territory_ranking_basis` | `object` | Territory ranking basis. |
| `territory_allocation_sensitivity_spearman` | `float64` | Territory allocation sensitivity spearman. |
| `territory_allocation_sensitivity_basis` | `object` | Territory allocation sensitivity basis. |
| `territory_truth_independent_flag` | `object` | Boolean indicator for territory truth independent. |
| `territory_capacity_basis` | `object` | Territory capacity basis. |
| `low_risk_count` | `float64` | Low risk count. |
| `medium_risk_count` | `float64` | Medium risk count. |
| `high_risk_count` | `float64` | High risk count. |
| `critical_risk_count` | `float64` | Critical risk count. |
| `total_required_fte` | `float64` | Total required fte. |
| `total_available_fte` | `float64` | Total available fte. |
| `total_fte_gap` | `float64` | Total fte gap. |
| `positive_fte_gap_fte_months` | `float64` | Positive fte gap fte months. |
| `fte_aggregate_basis` | `object` | Fte aggregate basis. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `dashboard_pca_variance`

Executed rows: 1.

| Field | Type | Definition |
|---|---|---|
| `component` | `int64` | Component. |
| `explained_variance_ratio` | `float64` | Ratio for explained variance ratio; zero-denominator handling is explicit. |
| `cumulative_explained_variance` | `float64` | Cumulative explained variance. |

## `dashboard_peer_comparison`

Executed rows: 1,456.

| Field | Type | Definition |
|---|---|---|
| `observation_id` | `object` | Stable identifier for observation. |
| `metric_name` | `object` | Metric name. |
| `actual_value` | `float64` | Actual value. |
| `peer_median_value` | `float64` | Peer median value. |
| `peer_z_score` | `float64` | Peer z score. |
| `peer_percentile` | `float64` | Peer percentile. |
| `peer_group_basis` | `object` | Peer group basis. |
| `rep_id` | `object` | Stable identifier for rep. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `rep_name` | `object` | Rep name. |
| `manager_name` | `object` | Manager name. |
| `team_name` | `object` | Team name. |
| `territory_name` | `object` | Territory name. |

## `dashboard_period_stability`

Executed rows: 56.

| Field | Type | Definition |
|---|---|---|
| `population` | `object` | Population. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `observations` | `int64` | Observations. |
| `mean_raw_score` | `float64` | Mean raw score. |
| `median_raw_score` | `float64` | Median raw score. |
| `std_raw_score` | `float64` | Std raw score. |
| `mean_anomaly_score` | `float64` | Mean anomaly score. |
| `p95_anomaly_score` | `float64` | P95 anomaly score. |
| `review_rate` | `float64` | Ratio for review rate; zero-denominator handling is explicit. |
| `selected_entity_overlap_previous_period` | `float64` | Selected entity overlap previous period. |
| `ground_truth_prevalence` | `float64` | Ground truth prevalence. |
| `precision_at_threshold` | `float64` | Precision at threshold. |
| `recall_at_threshold` | `float64` | Recall at threshold. |
| `mean_score_change_from_previous_period` | `float64` | Mean score change from previous period. |

## `dashboard_rep_period_summary`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `observation_id` | `object` | Stable identifier for observation. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `payout_date` | `datetime64[ns]` | Payout date. |
| `split` | `object` | Split. |
| `anomaly_score` | `float64` | Training-reference percentile of PCA reconstruction error; higher is more unusual. |
| `anomaly_percentile` | `float64` | Anomaly percentile. |
| `raw_score` | `float64` | Raw score. |
| `threshold_flag` | `bool` | Boolean indicator for threshold. |
| `manager_review_flag` | `bool` | Boolean indicator for manager review. |
| `review_priority` | `object` | Review priority. |
| `risk_band` | `object` | Risk band. |
| `primary_reason` | `object` | Primary reason. |
| `gross_sales` | `float64` | Gross sales. |
| `net_sales` | `float64` | Net sales. |
| `sales_growth` | `float64` | Sales growth. |
| `rolling_sales_mean` | `float64` | Rolling sales mean. |
| `target_sales` | `float64` | Target sales. |
| `attainment_pct` | `float64` | Percentage value for attainment. |
| `final_incentive_paid` | `float64` | Final incentive paid. |
| `expected_incentive` | `float64` | Expected incentive. |
| `incentive_calculation_residual` | `float64` | Incentive calculation residual. |
| `average_discount_pct` | `float64` | Percentage value for average discount. |
| `return_rate` | `float64` | Ratio for return rate; zero-denominator handling is explicit. |
| `post_incentive_return_rate` | `float64` | Ratio for post incentive return rate; zero-denominator handling is explicit. |
| `end_of_period_sales_share` | `float64` | Ratio for end of period sales share; zero-denominator handling is explicit. |
| `completed_visit_count` | `int64` | Completed visit count. |
| `average_visit_duration` | `float64` | Average visit duration. |
| `crm_interaction_count` | `int64` | Crm interaction count. |
| `claimed_expense_amount` | `float64` | Claimed expense amount. |
| `distance_claim_ratio` | `float64` | Ratio for distance claim ratio; zero-denominator handling is explicit. |
| `available_field_hours` | `float64` | Available field hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `dashboard_run_manifest`

Executed rows: 1.

| Field | Type | Definition |
|---|---|---|
| `execution_timestamp` | `object` | Execution timestamp. |
| `random_seed` | `int64` | Random seed. |
| `input_file_name` | `object` | Input file name. |
| `input_file_hash` | `object` | Input file hash. |
| `input_rows` | `int64` | Input rows. |
| `input_columns` | `int64` | Input columns. |
| `modeling_row_count` | `int64` | Modeling row count. |
| `source_date_min` | `object` | Source date min. |
| `source_date_max` | `object` | Source date max. |
| `primary_analytical_grain` | `object` | Primary analytical grain. |
| `model_feature_count` | `int64` | Model feature count. |
| `capacity_methodology` | `object` | Capacity methodology. |
| `key_software_versions` | `object` | Key software versions. |
| `configuration_hash` | `object` | Configuration hash. |
| `configuration_file` | `object` | Configuration file. |
| `finalized_model_name` | `object` | Finalized model name. |
| `model_parameters` | `object` | Model parameters. |
| `scoring_threshold` | `float64` | Scoring threshold. |
| `manager_review_fraction` | `float64` | Manager review fraction. |
| `evaluation_metrics` | `object` | Evaluation metrics. |
| `git_commit_hash` | `object` | Git commit hash. |
| `git_worktree_dirty` | `bool` | Git worktree dirty. |
| `implementation_sha256` | `object` | Implementation sha256. |
| `source_unchanged` | `bool` | Source unchanged. |
| `manifest_scope` | `object` | Manifest scope. |
| `output_file_names` | `object` | Output file names. |
| `output_row_counts` | `object` | Output row counts. |

## `dashboard_score_distribution`

Executed rows: 220.

| Field | Type | Definition |
|---|---|---|
| `population` | `object` | Population. |
| `split` | `object` | Split. |
| `bin` | `int64` | Bin. |
| `score_lower` | `float64` | Score lower. |
| `score_upper` | `float64` | Score upper. |
| `count` | `int64` | Count. |
| `share` | `float64` | Share. |
| `mean_raw_score` | `float64` | Mean raw score. |
| `mean_anomaly_score` | `float64` | Mean anomaly score. |

## `discount_detail`

Executed rows: 167,756.

| Field | Type | Definition |
|---|---|---|
| `discount_id` | `object` | Stable identifier for discount. |
| `order_line_id` | `object` | Stable identifier for order line. |
| `rep_id` | `object` | Stable identifier for rep. |
| `discount_pct` | `float64` | Percentage value for discount. |
| `discount_amount` | `float64` | Discount amount. |
| `discount_reason` | `object` | Discount reason. |
| `approval_required_flag` | `bool` | Boolean indicator for approval required. |
| `approved_flag` | `bool` | Boolean indicator for approved. |
| `approver_id` | `object` | Stable identifier for approver. |
| `approval_date` | `datetime64[ns]` | Approval date. |
| `exception_flag` | `bool` | Boolean indicator for exception. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `feature_store_rep_period`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `observation_id` | `object` | Stable identifier for observation. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `peer_group_id` | `object` | Stable identifier for peer group. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `policy_id` | `object` | Stable identifier for policy. |
| `policy_version` | `object` | Policy version. |
| `eligible_gross_sales` | `float64` | Eligible gross sales. |
| `eligible_net_sales` | `float64` | Eligible net sales. |
| `target_sales` | `float64` | Target sales. |
| `attainment_pct` | `float64` | Percentage value for attainment. |
| `base_incentive` | `float64` | Base incentive. |
| `accelerator_amount` | `float64` | Accelerator amount. |
| `product_mix_bonus` | `float64` | Product mix bonus. |
| `new_customer_bonus` | `float64` | New customer bonus. |
| `discount_penalty` | `float64` | Discount penalty. |
| `return_clawback` | `float64` | Return clawback. |
| `manual_adjustment` | `float64` | Manual adjustment. |
| `calculated_incentive` | `float64` | Calculated incentive. |
| `final_incentive_paid` | `float64` | Final incentive paid. |
| `payout_date` | `datetime64[ns]` | Payout date. |
| `payout_status` | `object` | Payout status. |
| `payout_to_sales_ratio` | `float64` | Ratio for payout to sales ratio; zero-denominator handling is explicit. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `reconciliation_tolerance` | `float64` | Reconciliation tolerance. |
| `incentive_record_id` | `object` | Stable identifier for incentive record. |
| `target_units` | `float64` | Target units. |
| `quota_difficulty_index` | `float64` | Quota difficulty index. |
| `target_revision_flag` | `bool` | Boolean indicator for target revision. |
| `target_effective_date` | `datetime64[ns]` | Target effective date. |
| `target_version` | `int64` | Target version. |
| `target_visit_count` | `int64` | Target visit count. |
| `gross_sales` | `float64` | Gross sales. |
| `net_sales` | `float64` | Net sales. |
| `total_quantity` | `float64` | Total quantity. |
| `order_count` | `int64` | Order count. |
| `average_order_value` | `float64` | Average order value. |
| `maximum_order_value` | `float64` | Maximum order value. |
| `end_period_order_count` | `int64` | End period order count. |
| `end_period_sales` | `float64` | End period sales. |
| `end_of_period_sales_share` | `float64` | Ratio for end of period sales share; zero-denominator handling is explicit. |
| `average_selling_price` | `float64` | Average selling price. |
| `order_frequency` | `float64` | Order frequency. |
| `unusual_order_timing_score` | `float64` | Unusual order timing score. |
| `duplicate_order_signal` | `float64` | Duplicate order signal. |
| `active_customer_count` | `int64` | Active customer count. |
| `total_customer_sales` | `float64` | Total customer sales. |
| `new_customer_count` | `float64` | New customer count. |
| `high_priority_customer_share` | `float64` | Ratio for high priority customer share; zero-denominator handling is explicit. |
| `low_potential_customer_sales_share` | `float64` | Ratio for low potential customer sales share; zero-denominator handling is explicit. |
| `customer_concentration_hhi` | `float64` | Customer concentration hhi. |
| `top_customer_concentration_share` | `float64` | Ratio for top customer concentration share; zero-denominator handling is explicit. |
| `customer_sales_concentration` | `float64` | Customer sales concentration. |
| `top_customer_sales_share` | `float64` | Ratio for top customer sales share; zero-denominator handling is explicit. |
| `customer_mix_shift` | `float64` | Customer mix shift. |
| `active_product_count` | `int64` | Active product count. |
| `product_price_deviation` | `float64` | Product price deviation. |
| `highly_incentivized_product_sales` | `float64` | Highly incentivized product sales. |
| `highly_incentivized_product_share` | `float64` | Ratio for highly incentivized product share; zero-denominator handling is explicit. |
| `product_concentration_hhi` | `float64` | Product concentration hhi. |
| `top_product_concentration_share` | `float64` | Ratio for top product concentration share; zero-denominator handling is explicit. |
| `product_mix_entropy` | `float64` | Product mix entropy. |
| `product_mix_shift` | `float64` | Product mix shift. |
| `priority_product_share` | `float64` | Ratio for priority product share; zero-denominator handling is explicit. |
| `low_volume_product_spike` | `float64` | Low volume product spike. |
| `average_discount_pct` | `float64` | Percentage value for average discount. |
| `maximum_discount_pct` | `float64` | Percentage value for maximum discount. |
| `unapproved_discount_rate` | `float64` | Ratio for unapproved discount rate; zero-denominator handling is explicit. |
| `discount_spike` | `float64` | Discount spike. |
| `return_amount` | `float64` | Return amount. |
| `return_count` | `float64` | Return count. |
| `cancellation_count` | `float64` | Cancellation count. |
| `post_payout_return_amount` | `float64` | Post payout return amount. |
| `completed_visit_count` | `int64` | Completed visit count. |
| `average_visit_duration` | `float64` | Average visit duration. |
| `extremely_short_visit_rate` | `float64` | Ratio for extremely short visit rate; zero-denominator handling is explicit. |
| `impossible_travel_count` | `int64` | Impossible travel count. |
| `overlapping_visit_count` | `int64` | Overlapping visit count. |
| `estimated_visit_travel_km` | `float64` | Estimated visit travel km. |
| `visit_converted_customer_count` | `int64` | Visit converted customer count. |
| `crm_interaction_count` | `int64` | Crm interaction count. |
| `average_interest_score` | `float64` | Average interest score. |
| `claimed_distance_km` | `float64` | Claimed distance km. |
| `estimated_distance_km` | `float64` | Estimated distance km. |
| `claimed_expense_amount` | `float64` | Claimed expense amount. |
| `expected_expense_amount` | `float64` | Expected expense amount. |
| `missing_receipt_rate` | `float64` | Ratio for missing receipt rate; zero-denominator handling is explicit. |
| `duplicate_expense_signal` | `float64` | Duplicate expense signal. |
| `capacity_record_id` | `object` | Stable identifier for capacity record. |
| `active_territory_count` | `int64` | Active territory count. |
| `dominant_territory_activity_share` | `float64` | Ratio for dominant territory activity share; zero-denominator handling is explicit. |
| `fractional_territory_allocation` | `object` | Fractional territory allocation. |
| `working_days` | `int64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `standard_field_hours_per_day` | `float64` | Standard field hours per day. |
| `standard_working_days_per_month` | `int64` | Standard working days per month. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `customer_coverage_pct` | `float64` | Percentage value for customer coverage. |
| `average_travel_hours` | `float64` | Average travel hours. |
| `workload_per_active_customer` | `float64` | Workload per active customer. |
| `legacy_normalized_workload_index` | `float64` | Legacy normalized workload index. |
| `workload_risk_band` | `object` | Workload risk band. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `distinct_customers` | `int64` | Distinct customers. |
| `transaction_count` | `int64` | Transaction count. |
| `distinct_cities` | `int64` | Distinct cities. |
| `distinct_products` | `int64` | Distinct products. |
| `distributor_count` | `int64` | Distributor count. |
| `legacy_distinct_customers_training_median` | `float64` | Legacy distinct customers training median. |
| `legacy_transaction_count_training_median` | `float64` | Legacy transaction count training median. |
| `legacy_distinct_cities_training_median` | `float64` | Legacy distinct cities training median. |
| `legacy_distinct_products_training_median` | `float64` | Legacy distinct products training median. |
| `legacy_distributor_count_training_median` | `float64` | Legacy distributor count training median. |
| `legacy_workload_training_end` | `datetime64[ns]` | Legacy workload training end. |
| `legacy_workload_formula` | `object` | Legacy workload formula. |
| `employment_status` | `object` | Employment status. |
| `role_grade` | `object` | Role grade. |
| `priority_customer_count` | `int64` | Priority customer count. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `observed_visit_count` | `int64` | Observed visit count. |
| `observed_completed_visit_count` | `int64` | Observed completed visit count. |
| `priority_customer_coverage_gap_pct` | `float64` | Percentage value for priority customer coverage gap. |
| `core_required_hours` | `float64` | Core required hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `capacity_zero_denominator_flag` | `bool` | Boolean indicator for capacity zero denominator. |
| `required_hours` | `float64` | Required hours. |
| `available_hours` | `float64` | Available hours. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `float64` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `required_hours_formula` | `object` | Required hours formula. |
| `required_workload_scope` | `object` | Required workload scope. |
| `numeric_visit_frequency_period_divisor` | `float64` | Numeric visit frequency period divisor. |
| `capacity_methodology` | `object` | Capacity methodology. |
| `hire_date` | `datetime64[ns]` | Hire date. |
| `travel_complexity_index` | `float64` | Travel complexity index. |
| `territory_potential` | `float64` | Territory potential. |
| `tenure_months` | `int32` | Tenure months. |
| `tenure_band` | `object` | Tenure band. |
| `potential_band` | `object` | Potential band. |
| `travel_band` | `object` | Travel band. |
| `rolling_sales_mean` | `float64` | Rolling sales mean. |
| `rolling_sales_std` | `float64` | Rolling sales std. |
| `sales_growth` | `float64` | Sales growth. |
| `quantity_growth` | `float64` | Quantity growth. |
| `sales_volatility` | `float64` | Sales volatility. |
| `price_deviation_from_product_norm` | `float64` | Price deviation from product norm. |
| `sales_vs_territory_potential` | `float64` | Sales vs territory potential. |
| `target_attainment_pct` | `float64` | Percentage value for target attainment. |
| `expected_incentive` | `float64` | Expected incentive. |
| `incentive_calculation_residual` | `float64` | Incentive calculation residual. |
| `manual_adjustment_ratio` | `float64` | Ratio for manual adjustment ratio; zero-denominator handling is explicit. |
| `accelerator_cliff_distance` | `float64` | Accelerator cliff distance. |
| `return_rate` | `float64` | Ratio for return rate; zero-denominator handling is explicit. |
| `post_incentive_return_rate` | `float64` | Ratio for post incentive return rate; zero-denominator handling is explicit. |
| `return_clawback_ratio` | `float64` | Ratio for return clawback ratio; zero-denominator handling is explicit. |
| `incentive_growth` | `float64` | Incentive growth. |
| `incentive_volatility` | `float64` | Incentive volatility. |
| `cancelled_order_rate` | `float64` | Ratio for cancelled order rate; zero-denominator handling is explicit. |
| `visits_per_customer` | `float64` | Visits per customer. |
| `sales_per_visit` | `float64` | Sales per visit. |
| `visit_to_sales_conversion` | `float64` | Visit to sales conversion. |
| `missed_priority_visit_count` | `float64` | Missed priority visit count. |
| `crm_interactions_per_customer` | `float64` | Crm interactions per customer. |
| `distance_claim_ratio` | `float64` | Ratio for distance claim ratio; zero-denominator handling is explicit. |
| `expense_per_visit` | `float64` | Expense per visit. |
| `capacity_risk_code` | `int64` | Capacity risk code. |
| `sales_peer_median` | `float64` | Sales peer median. |
| `sales_peer_z` | `float64` | Sales peer z. |
| `sales_peer_percentile` | `float64` | Sales peer percentile. |
| `sales_peer_cohort` | `object` | Sales peer cohort. |
| `incentive_peer_median` | `float64` | Incentive peer median. |
| `incentive_peer_z` | `float64` | Incentive peer z. |
| `incentive_peer_percentile` | `float64` | Incentive peer percentile. |
| `incentive_peer_cohort` | `object` | Incentive peer cohort. |
| `discount_peer_median` | `float64` | Discount peer median. |
| `discount_peer_z` | `float64` | Discount peer z. |
| `discount_peer_percentile` | `float64` | Discount peer percentile. |
| `discount_peer_cohort` | `object` | Discount peer cohort. |
| `expense_peer_median` | `float64` | Expense peer median. |
| `expense_peer_z` | `float64` | Expense peer z. |
| `expense_peer_percentile` | `float64` | Expense peer percentile. |
| `expense_peer_cohort` | `object` | Expense peer cohort. |
| `payout_to_peer_median_ratio` | `float64` | Ratio for payout to peer median ratio; zero-denominator handling is explicit. |
| `discount_vs_peer` | `float64` | Discount vs peer. |
| `expense_vs_peer` | `float64` | Expense vs peer. |
| `peer_group_z_score` | `float64` | Peer group z score. |
| `peer_percentile` | `float64` | Peer percentile. |
| `rep_historical_mean` | `float64` | Rep historical mean. |
| `rep_historical_std` | `float64` | Rep historical std. |
| `rep_historical_z_score` | `float64` | Rep historical z score. |
| `territory_adjusted_sales_residual` | `float64` | Territory adjusted sales residual. |
| `territory_adjusted_incentive_residual` | `float64` | Territory adjusted incentive residual. |
| `tenure_adjusted_performance` | `float64` | Tenure adjusted performance. |
| `product_mix_adjusted_performance` | `float64` | Product mix adjusted performance. |
| `channel_hhi` | `float64` | Channel hhi. |
| `channel_adjusted_performance` | `float64` | Channel adjusted performance. |
| `month_over_month_behavior_change` | `float64` | Month over month behavior change. |
| `rolling_anomaly_score` | `float64` | Rolling anomaly score. |
| `threshold_crossing_discount_signal` | `int64` | Threshold crossing discount signal. |
| `rolling_sales_mean_missing` | `int64` | Rolling sales mean missing. |
| `rolling_sales_std_missing` | `int64` | Rolling sales std missing. |
| `rep_historical_mean_missing` | `int64` | Rep historical mean missing. |
| `rep_historical_std_missing` | `int64` | Rep historical std missing. |

## `field_visits`

Executed rows: 30,268.

| Field | Type | Definition |
|---|---|---|
| `visit_id` | `object` | Stable identifier for visit. |
| `rep_id` | `object` | Stable identifier for rep. |
| `customer_id` | `object` | Stable identifier for customer. |
| `territory_id` | `object` | Stable identifier for territory. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `visit_date` | `datetime64[ns]` | Visit date. |
| `scheduled_start_time` | `datetime64[ns]` | Scheduled start time. |
| `actual_start_time` | `datetime64[ns]` | Actual start time. |
| `actual_end_time` | `datetime64[ns]` | Actual end time. |
| `visit_duration_minutes` | `int64` | Visit duration minutes. |
| `visit_type` | `object` | Visit type. |
| `product_discussed` | `object` | Product discussed. |
| `visit_outcome` | `object` | Visit outcome. |
| `check_in_latitude` | `float64` | Check in latitude. |
| `check_in_longitude` | `float64` | Check in longitude. |
| `check_out_latitude` | `float64` | Check out latitude. |
| `check_out_longitude` | `float64` | Check out longitude. |
| `estimated_travel_km` | `float64` | Estimated travel km. |
| `visit_completed_flag` | `bool` | Boolean indicator for visit completed. |
| `impossible_travel_flag` | `bool` | Boolean indicator for impossible travel. |
| `overlapping_visit_flag` | `bool` | Boolean indicator for overlapping visit. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `incentive_calculations`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `rep_id` | `object` | Stable identifier for rep. |
| `manager_id` | `object` | Stable identifier for manager. |
| `team_id` | `object` | Stable identifier for team. |
| `territory_id` | `object` | Stable identifier for territory. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `policy_id` | `object` | Stable identifier for policy. |
| `policy_version` | `object` | Policy version. |
| `eligible_gross_sales` | `float64` | Eligible gross sales. |
| `eligible_net_sales` | `float64` | Eligible net sales. |
| `target_sales` | `float64` | Target sales. |
| `attainment_pct` | `float64` | Percentage value for attainment. |
| `base_incentive` | `float64` | Base incentive. |
| `accelerator_amount` | `float64` | Accelerator amount. |
| `product_mix_bonus` | `float64` | Product mix bonus. |
| `new_customer_bonus` | `float64` | New customer bonus. |
| `discount_penalty` | `float64` | Discount penalty. |
| `return_clawback` | `float64` | Return clawback. |
| `manual_adjustment` | `float64` | Manual adjustment. |
| `calculated_incentive` | `float64` | Calculated incentive. |
| `final_incentive_paid` | `float64` | Final incentive paid. |
| `payout_date` | `datetime64[ns]` | Payout date. |
| `payout_status` | `object` | Payout status. |
| `payout_to_sales_ratio` | `float64` | Ratio for payout to sales ratio; zero-denominator handling is explicit. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `reconciliation_tolerance` | `float64` | Reconciliation tolerance. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `incentive_record_id` | `object` | Stable identifier for incentive record. |

## `incentive_policy_rules`

Executed rows: 12.

| Field | Type | Definition |
|---|---|---|
| `policy_id` | `object` | Stable identifier for policy. |
| `policy_version` | `object` | Policy version. |
| `effective_start_date` | `datetime64[ns]` | Effective start date. |
| `effective_end_date` | `datetime64[ns]` | Effective end date. |
| `metric_name` | `object` | Metric name. |
| `lower_attainment_pct` | `float64` | Percentage value for lower attainment. |
| `upper_attainment_pct` | `float64` | Percentage value for upper attainment. |
| `payout_rate` | `float64` | Ratio for payout rate; zero-denominator handling is explicit. |
| `accelerator_multiplier` | `float64` | Accelerator multiplier. |
| `decelerator_multiplier` | `float64` | Decelerator multiplier. |
| `maximum_payout` | `float64` | Maximum payout. |
| `minimum_eligibility` | `float64` | Minimum eligibility. |
| `product_weight` | `float64` | Product weight. |
| `new_customer_bonus` | `float64` | New customer bonus. |
| `discount_penalty_threshold_pct` | `float64` | Percentage value for discount penalty threshold. |
| `discount_penalty_rule` | `object` | Discount penalty rule. |
| `return_clawback_rule` | `object` | Return clawback rule. |
| `payout_delay_days` | `int64` | Payout delay days. |
| `policy_description` | `object` | Policy description. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `manager_master`

Executed rows: 4.

| Field | Type | Definition |
|---|---|---|
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `region` | `object` | Region. |
| `active_rep_count` | `int64` | Active rep count. |
| `management_span` | `int64` | Management span. |
| `attribute_reference_end_date` | `datetime64[s]` | Attribute reference end date. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `model_clean_scores`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `observation_id` | `object` | Stable identifier for observation. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `peer_group_id` | `object` | Stable identifier for peer group. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `policy_id` | `object` | Stable identifier for policy. |
| `policy_version` | `object` | Policy version. |
| `eligible_gross_sales` | `float64` | Eligible gross sales. |
| `eligible_net_sales` | `float64` | Eligible net sales. |
| `target_sales` | `float64` | Target sales. |
| `attainment_pct` | `float64` | Percentage value for attainment. |
| `base_incentive` | `float64` | Base incentive. |
| `accelerator_amount` | `float64` | Accelerator amount. |
| `product_mix_bonus` | `float64` | Product mix bonus. |
| `new_customer_bonus` | `float64` | New customer bonus. |
| `discount_penalty` | `float64` | Discount penalty. |
| `return_clawback` | `float64` | Return clawback. |
| `manual_adjustment` | `float64` | Manual adjustment. |
| `calculated_incentive` | `float64` | Calculated incentive. |
| `final_incentive_paid` | `float64` | Final incentive paid. |
| `payout_date` | `datetime64[ns]` | Payout date. |
| `payout_status` | `object` | Payout status. |
| `payout_to_sales_ratio` | `float64` | Ratio for payout to sales ratio; zero-denominator handling is explicit. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `reconciliation_tolerance` | `float64` | Reconciliation tolerance. |
| `incentive_record_id` | `object` | Stable identifier for incentive record. |
| `target_units` | `float64` | Target units. |
| `quota_difficulty_index` | `float64` | Quota difficulty index. |
| `target_revision_flag` | `bool` | Boolean indicator for target revision. |
| `target_effective_date` | `datetime64[ns]` | Target effective date. |
| `target_version` | `int64` | Target version. |
| `target_visit_count` | `int64` | Target visit count. |
| `gross_sales` | `float64` | Gross sales. |
| `net_sales` | `float64` | Net sales. |
| `total_quantity` | `float64` | Total quantity. |
| `order_count` | `int64` | Order count. |
| `average_order_value` | `float64` | Average order value. |
| `maximum_order_value` | `float64` | Maximum order value. |
| `end_period_order_count` | `int64` | End period order count. |
| `end_period_sales` | `float64` | End period sales. |
| `end_of_period_sales_share` | `float64` | Ratio for end of period sales share; zero-denominator handling is explicit. |
| `average_selling_price` | `float64` | Average selling price. |
| `order_frequency` | `float64` | Order frequency. |
| `unusual_order_timing_score` | `float64` | Unusual order timing score. |
| `duplicate_order_signal` | `float64` | Duplicate order signal. |
| `active_customer_count` | `int64` | Active customer count. |
| `total_customer_sales` | `float64` | Total customer sales. |
| `new_customer_count` | `float64` | New customer count. |
| `high_priority_customer_share` | `float64` | Ratio for high priority customer share; zero-denominator handling is explicit. |
| `low_potential_customer_sales_share` | `float64` | Ratio for low potential customer sales share; zero-denominator handling is explicit. |
| `customer_concentration_hhi` | `float64` | Customer concentration hhi. |
| `top_customer_concentration_share` | `float64` | Ratio for top customer concentration share; zero-denominator handling is explicit. |
| `customer_sales_concentration` | `float64` | Customer sales concentration. |
| `top_customer_sales_share` | `float64` | Ratio for top customer sales share; zero-denominator handling is explicit. |
| `customer_mix_shift` | `float64` | Customer mix shift. |
| `active_product_count` | `int64` | Active product count. |
| `product_price_deviation` | `float64` | Product price deviation. |
| `highly_incentivized_product_sales` | `float64` | Highly incentivized product sales. |
| `highly_incentivized_product_share` | `float64` | Ratio for highly incentivized product share; zero-denominator handling is explicit. |
| `product_concentration_hhi` | `float64` | Product concentration hhi. |
| `top_product_concentration_share` | `float64` | Ratio for top product concentration share; zero-denominator handling is explicit. |
| `product_mix_entropy` | `float64` | Product mix entropy. |
| `product_mix_shift` | `float64` | Product mix shift. |
| `priority_product_share` | `float64` | Ratio for priority product share; zero-denominator handling is explicit. |
| `low_volume_product_spike` | `float64` | Low volume product spike. |
| `average_discount_pct` | `float64` | Percentage value for average discount. |
| `maximum_discount_pct` | `float64` | Percentage value for maximum discount. |
| `unapproved_discount_rate` | `float64` | Ratio for unapproved discount rate; zero-denominator handling is explicit. |
| `discount_spike` | `float64` | Discount spike. |
| `return_amount` | `float64` | Return amount. |
| `return_count` | `float64` | Return count. |
| `cancellation_count` | `float64` | Cancellation count. |
| `post_payout_return_amount` | `float64` | Post payout return amount. |
| `completed_visit_count` | `int64` | Completed visit count. |
| `average_visit_duration` | `float64` | Average visit duration. |
| `extremely_short_visit_rate` | `float64` | Ratio for extremely short visit rate; zero-denominator handling is explicit. |
| `impossible_travel_count` | `int64` | Impossible travel count. |
| `overlapping_visit_count` | `int64` | Overlapping visit count. |
| `estimated_visit_travel_km` | `float64` | Estimated visit travel km. |
| `visit_converted_customer_count` | `int64` | Visit converted customer count. |
| `crm_interaction_count` | `int64` | Crm interaction count. |
| `average_interest_score` | `float64` | Average interest score. |
| `claimed_distance_km` | `float64` | Claimed distance km. |
| `estimated_distance_km` | `float64` | Estimated distance km. |
| `claimed_expense_amount` | `float64` | Claimed expense amount. |
| `expected_expense_amount` | `float64` | Expected expense amount. |
| `missing_receipt_rate` | `float64` | Ratio for missing receipt rate; zero-denominator handling is explicit. |
| `duplicate_expense_signal` | `float64` | Duplicate expense signal. |
| `capacity_record_id` | `object` | Stable identifier for capacity record. |
| `active_territory_count` | `int64` | Active territory count. |
| `dominant_territory_activity_share` | `float64` | Ratio for dominant territory activity share; zero-denominator handling is explicit. |
| `fractional_territory_allocation` | `object` | Fractional territory allocation. |
| `working_days` | `int64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `standard_field_hours_per_day` | `float64` | Standard field hours per day. |
| `standard_working_days_per_month` | `int64` | Standard working days per month. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `customer_coverage_pct` | `float64` | Percentage value for customer coverage. |
| `average_travel_hours` | `float64` | Average travel hours. |
| `workload_per_active_customer` | `float64` | Workload per active customer. |
| `legacy_normalized_workload_index` | `float64` | Legacy normalized workload index. |
| `workload_risk_band` | `object` | Workload risk band. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `distinct_customers` | `int64` | Distinct customers. |
| `transaction_count` | `int64` | Transaction count. |
| `distinct_cities` | `int64` | Distinct cities. |
| `distinct_products` | `int64` | Distinct products. |
| `distributor_count` | `int64` | Distributor count. |
| `legacy_distinct_customers_training_median` | `float64` | Legacy distinct customers training median. |
| `legacy_transaction_count_training_median` | `float64` | Legacy transaction count training median. |
| `legacy_distinct_cities_training_median` | `float64` | Legacy distinct cities training median. |
| `legacy_distinct_products_training_median` | `float64` | Legacy distinct products training median. |
| `legacy_distributor_count_training_median` | `float64` | Legacy distributor count training median. |
| `legacy_workload_training_end` | `datetime64[ns]` | Legacy workload training end. |
| `legacy_workload_formula` | `object` | Legacy workload formula. |
| `employment_status` | `object` | Employment status. |
| `role_grade` | `object` | Role grade. |
| `priority_customer_count` | `int64` | Priority customer count. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `observed_visit_count` | `int64` | Observed visit count. |
| `observed_completed_visit_count` | `int64` | Observed completed visit count. |
| `priority_customer_coverage_gap_pct` | `float64` | Percentage value for priority customer coverage gap. |
| `core_required_hours` | `float64` | Core required hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `capacity_zero_denominator_flag` | `bool` | Boolean indicator for capacity zero denominator. |
| `required_hours` | `float64` | Required hours. |
| `available_hours` | `float64` | Available hours. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `float64` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `required_hours_formula` | `object` | Required hours formula. |
| `required_workload_scope` | `object` | Required workload scope. |
| `numeric_visit_frequency_period_divisor` | `float64` | Numeric visit frequency period divisor. |
| `capacity_methodology` | `object` | Capacity methodology. |
| `hire_date` | `datetime64[ns]` | Hire date. |
| `travel_complexity_index` | `float64` | Travel complexity index. |
| `territory_potential` | `float64` | Territory potential. |
| `tenure_months` | `int32` | Tenure months. |
| `tenure_band` | `object` | Tenure band. |
| `potential_band` | `object` | Potential band. |
| `travel_band` | `object` | Travel band. |
| `rolling_sales_mean` | `float64` | Rolling sales mean. |
| `rolling_sales_std` | `float64` | Rolling sales std. |
| `sales_growth` | `float64` | Sales growth. |
| `quantity_growth` | `float64` | Quantity growth. |
| `sales_volatility` | `float64` | Sales volatility. |
| `price_deviation_from_product_norm` | `float64` | Price deviation from product norm. |
| `sales_vs_territory_potential` | `float64` | Sales vs territory potential. |
| `target_attainment_pct` | `float64` | Percentage value for target attainment. |
| `expected_incentive` | `float64` | Expected incentive. |
| `incentive_calculation_residual` | `float64` | Incentive calculation residual. |
| `manual_adjustment_ratio` | `float64` | Ratio for manual adjustment ratio; zero-denominator handling is explicit. |
| `accelerator_cliff_distance` | `float64` | Accelerator cliff distance. |
| `return_rate` | `float64` | Ratio for return rate; zero-denominator handling is explicit. |
| `post_incentive_return_rate` | `float64` | Ratio for post incentive return rate; zero-denominator handling is explicit. |
| `return_clawback_ratio` | `float64` | Ratio for return clawback ratio; zero-denominator handling is explicit. |
| `incentive_growth` | `float64` | Incentive growth. |
| `incentive_volatility` | `float64` | Incentive volatility. |
| `cancelled_order_rate` | `float64` | Ratio for cancelled order rate; zero-denominator handling is explicit. |
| `visits_per_customer` | `float64` | Visits per customer. |
| `sales_per_visit` | `float64` | Sales per visit. |
| `visit_to_sales_conversion` | `float64` | Visit to sales conversion. |
| `missed_priority_visit_count` | `float64` | Missed priority visit count. |
| `crm_interactions_per_customer` | `float64` | Crm interactions per customer. |
| `distance_claim_ratio` | `float64` | Ratio for distance claim ratio; zero-denominator handling is explicit. |
| `expense_per_visit` | `float64` | Expense per visit. |
| `capacity_risk_code` | `int64` | Capacity risk code. |
| `sales_peer_median` | `float64` | Sales peer median. |
| `sales_peer_z` | `float64` | Sales peer z. |
| `sales_peer_percentile` | `float64` | Sales peer percentile. |
| `sales_peer_cohort` | `object` | Sales peer cohort. |
| `incentive_peer_median` | `float64` | Incentive peer median. |
| `incentive_peer_z` | `float64` | Incentive peer z. |
| `incentive_peer_percentile` | `float64` | Incentive peer percentile. |
| `incentive_peer_cohort` | `object` | Incentive peer cohort. |
| `discount_peer_median` | `float64` | Discount peer median. |
| `discount_peer_z` | `float64` | Discount peer z. |
| `discount_peer_percentile` | `float64` | Discount peer percentile. |
| `discount_peer_cohort` | `object` | Discount peer cohort. |
| `expense_peer_median` | `float64` | Expense peer median. |
| `expense_peer_z` | `float64` | Expense peer z. |
| `expense_peer_percentile` | `float64` | Expense peer percentile. |
| `expense_peer_cohort` | `object` | Expense peer cohort. |
| `payout_to_peer_median_ratio` | `float64` | Ratio for payout to peer median ratio; zero-denominator handling is explicit. |
| `discount_vs_peer` | `float64` | Discount vs peer. |
| `expense_vs_peer` | `float64` | Expense vs peer. |
| `peer_group_z_score` | `float64` | Peer group z score. |
| `peer_percentile` | `float64` | Peer percentile. |
| `rep_historical_mean` | `float64` | Rep historical mean. |
| `rep_historical_std` | `float64` | Rep historical std. |
| `rep_historical_z_score` | `float64` | Rep historical z score. |
| `territory_adjusted_sales_residual` | `float64` | Territory adjusted sales residual. |
| `territory_adjusted_incentive_residual` | `float64` | Territory adjusted incentive residual. |
| `tenure_adjusted_performance` | `float64` | Tenure adjusted performance. |
| `product_mix_adjusted_performance` | `float64` | Product mix adjusted performance. |
| `channel_hhi` | `float64` | Channel hhi. |
| `channel_adjusted_performance` | `float64` | Channel adjusted performance. |
| `month_over_month_behavior_change` | `float64` | Month over month behavior change. |
| `rolling_anomaly_score` | `float64` | Rolling anomaly score. |
| `threshold_crossing_discount_signal` | `int64` | Threshold crossing discount signal. |
| `rolling_sales_mean_missing` | `int64` | Rolling sales mean missing. |
| `rolling_sales_std_missing` | `int64` | Rolling sales std missing. |
| `rep_historical_mean_missing` | `int64` | Rep historical mean missing. |
| `rep_historical_std_missing` | `int64` | Rep historical std missing. |
| `split` | `object` | Split. |
| `model_name` | `object` | Model name. |
| `raw_score` | `float64` | Raw score. |
| `anomaly_score` | `float64` | Training-reference percentile of PCA reconstruction error; higher is more unusual. |
| `anomaly_percentile` | `float64` | Anomaly percentile. |
| `raw_threshold` | `float64` | Raw threshold. |
| `threshold_flag` | `bool` | Boolean indicator for threshold. |
| `manager_review_flag` | `bool` | Boolean indicator for manager review. |
| `review_budget_flag` | `bool` | Boolean indicator for review budget. |
| `primary_reason_code` | `object` | Primary reason code. |
| `primary_reason` | `object` | Primary reason. |
| `secondary_reason` | `object` | Secondary reason. |
| `recommended_review_action` | `object` | Recommended review action. |
| `driver_1_feature` | `object` | Driver 1 feature. |
| `driver_1_name` | `object` | Driver 1 name. |
| `driver_1_value` | `float64` | Driver 1 value. |
| `driver_1_peer_value` | `float64` | Driver 1 peer value. |
| `driver_1_percentile` | `float64` | Driver 1 percentile. |
| `driver_1_contribution` | `float64` | Driver 1 contribution. |
| `driver_2_feature` | `object` | Driver 2 feature. |
| `driver_2_name` | `object` | Driver 2 name. |
| `driver_2_value` | `float64` | Driver 2 value. |
| `driver_2_peer_value` | `float64` | Driver 2 peer value. |
| `driver_2_percentile` | `float64` | Driver 2 percentile. |
| `driver_2_contribution` | `float64` | Driver 2 contribution. |
| `driver_3_feature` | `object` | Driver 3 feature. |
| `driver_3_name` | `object` | Driver 3 name. |
| `driver_3_value` | `float64` | Driver 3 value. |
| `driver_3_peer_value` | `float64` | Driver 3 peer value. |
| `driver_3_percentile` | `float64` | Driver 3 percentile. |
| `driver_3_contribution` | `float64` | Driver 3 contribution. |

## `model_false_positive_review`

Executed rows: 1.

| Field | Type | Definition |
|---|---|---|
| `driver_feature` | `object` | Driver feature. |
| `driver_name` | `object` | Driver name. |
| `false_positive_count` | `int64` | False positive count. |
| `mean_raw_score` | `float64` | Mean raw score. |
| `median_anomaly_score` | `float64` | Median anomaly score. |
| `example_observation_ids` | `object` | Example observation ids. |
| `review_interpretation` | `object` | Review interpretation. |

## `model_feature_contributions`

Executed rows: 2,184.

| Field | Type | Definition |
|---|---|---|
| `rep_id` | `object` | Stable identifier for rep. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `manager_id` | `object` | Stable identifier for manager. |
| `team_id` | `object` | Stable identifier for team. |
| `territory_id` | `object` | Stable identifier for territory. |
| `observation_id` | `object` | Stable identifier for observation. |
| `population` | `object` | Population. |
| `anomaly_score` | `float64` | Training-reference percentile of PCA reconstruction error; higher is more unusual. |
| `primary_reason_code` | `object` | Primary reason code. |
| `primary_reason` | `object` | Primary reason. |
| `secondary_reason` | `object` | Secondary reason. |
| `recommended_review_action` | `object` | Recommended review action. |
| `driver_rank` | `int64` | Driver rank. |
| `feature` | `object` | Feature. |
| `name` | `object` | Name. |
| `value` | `float64` | Value. |
| `peer_value` | `float64` | Peer value. |
| `percentile` | `float64` | Percentile. |
| `contribution` | `float64` | Contribution. |

## `model_group_metrics`

Executed rows: 52.

| Field | Type | Definition |
|---|---|---|
| `group_kind` | `object` | Group kind. |
| `grouping` | `object` | Grouping. |
| `value` | `object` | Value. |
| `observations` | `int64` | Observations. |
| `positive_support` | `int64` | Positive support. |
| `overall_truth_support` | `int64` | Overall truth support. |
| `selected_at_threshold` | `int64` | Selected at threshold. |
| `captured_at_threshold` | `int64` | Captured at threshold. |
| `false_positives_at_threshold` | `float64` | False positives at threshold. |
| `precision_at_threshold` | `float64` | Precision at threshold. |
| `recall_at_threshold` | `float64` | Recall at threshold. |
| `detection_rate_at_threshold` | `float64` | Detection rate at threshold. |
| `captured_at_top5pct` | `float64` | Captured at top5pct. |
| `recall_at_top5pct` | `float64` | Recall at top5pct. |
| `detection_rate_at_top5pct` | `float64` | Detection rate at top5pct. |
| `support_status` | `object` | Support status. |
| `evaluation_scope` | `object` | Evaluation scope. |

## `model_injected_scores`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `observation_id` | `object` | Stable identifier for observation. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `peer_group_id` | `object` | Stable identifier for peer group. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
| `policy_id` | `object` | Stable identifier for policy. |
| `policy_version` | `object` | Policy version. |
| `eligible_gross_sales` | `float64` | Eligible gross sales. |
| `eligible_net_sales` | `float64` | Eligible net sales. |
| `target_sales` | `float64` | Target sales. |
| `attainment_pct` | `float64` | Percentage value for attainment. |
| `base_incentive` | `float64` | Base incentive. |
| `accelerator_amount` | `float64` | Accelerator amount. |
| `product_mix_bonus` | `float64` | Product mix bonus. |
| `new_customer_bonus` | `float64` | New customer bonus. |
| `discount_penalty` | `float64` | Discount penalty. |
| `return_clawback` | `float64` | Return clawback. |
| `manual_adjustment` | `float64` | Manual adjustment. |
| `calculated_incentive` | `float64` | Calculated incentive. |
| `final_incentive_paid` | `float64` | Final incentive paid. |
| `payout_date` | `datetime64[ns]` | Payout date. |
| `payout_status` | `object` | Payout status. |
| `payout_to_sales_ratio` | `float64` | Ratio for payout to sales ratio; zero-denominator handling is explicit. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `reconciliation_tolerance` | `float64` | Reconciliation tolerance. |
| `incentive_record_id` | `object` | Stable identifier for incentive record. |
| `target_units` | `float64` | Target units. |
| `quota_difficulty_index` | `float64` | Quota difficulty index. |
| `target_revision_flag` | `bool` | Boolean indicator for target revision. |
| `target_effective_date` | `datetime64[ns]` | Target effective date. |
| `target_version` | `int64` | Target version. |
| `target_visit_count` | `int64` | Target visit count. |
| `gross_sales` | `float64` | Gross sales. |
| `net_sales` | `float64` | Net sales. |
| `total_quantity` | `float64` | Total quantity. |
| `order_count` | `int64` | Order count. |
| `average_order_value` | `float64` | Average order value. |
| `maximum_order_value` | `float64` | Maximum order value. |
| `end_period_order_count` | `int64` | End period order count. |
| `end_period_sales` | `float64` | End period sales. |
| `end_of_period_sales_share` | `float64` | Ratio for end of period sales share; zero-denominator handling is explicit. |
| `average_selling_price` | `float64` | Average selling price. |
| `order_frequency` | `float64` | Order frequency. |
| `unusual_order_timing_score` | `float64` | Unusual order timing score. |
| `duplicate_order_signal` | `float64` | Duplicate order signal. |
| `active_customer_count` | `int64` | Active customer count. |
| `total_customer_sales` | `float64` | Total customer sales. |
| `new_customer_count` | `float64` | New customer count. |
| `high_priority_customer_share` | `float64` | Ratio for high priority customer share; zero-denominator handling is explicit. |
| `low_potential_customer_sales_share` | `float64` | Ratio for low potential customer sales share; zero-denominator handling is explicit. |
| `customer_concentration_hhi` | `float64` | Customer concentration hhi. |
| `top_customer_concentration_share` | `float64` | Ratio for top customer concentration share; zero-denominator handling is explicit. |
| `customer_sales_concentration` | `float64` | Customer sales concentration. |
| `top_customer_sales_share` | `float64` | Ratio for top customer sales share; zero-denominator handling is explicit. |
| `customer_mix_shift` | `float64` | Customer mix shift. |
| `active_product_count` | `int64` | Active product count. |
| `product_price_deviation` | `float64` | Product price deviation. |
| `highly_incentivized_product_sales` | `float64` | Highly incentivized product sales. |
| `highly_incentivized_product_share` | `float64` | Ratio for highly incentivized product share; zero-denominator handling is explicit. |
| `product_concentration_hhi` | `float64` | Product concentration hhi. |
| `top_product_concentration_share` | `float64` | Ratio for top product concentration share; zero-denominator handling is explicit. |
| `product_mix_entropy` | `float64` | Product mix entropy. |
| `product_mix_shift` | `float64` | Product mix shift. |
| `priority_product_share` | `float64` | Ratio for priority product share; zero-denominator handling is explicit. |
| `low_volume_product_spike` | `float64` | Low volume product spike. |
| `average_discount_pct` | `float64` | Percentage value for average discount. |
| `maximum_discount_pct` | `float64` | Percentage value for maximum discount. |
| `unapproved_discount_rate` | `float64` | Ratio for unapproved discount rate; zero-denominator handling is explicit. |
| `discount_spike` | `float64` | Discount spike. |
| `return_amount` | `float64` | Return amount. |
| `return_count` | `float64` | Return count. |
| `cancellation_count` | `float64` | Cancellation count. |
| `post_payout_return_amount` | `float64` | Post payout return amount. |
| `completed_visit_count` | `int64` | Completed visit count. |
| `average_visit_duration` | `float64` | Average visit duration. |
| `extremely_short_visit_rate` | `float64` | Ratio for extremely short visit rate; zero-denominator handling is explicit. |
| `impossible_travel_count` | `int64` | Impossible travel count. |
| `overlapping_visit_count` | `int64` | Overlapping visit count. |
| `estimated_visit_travel_km` | `float64` | Estimated visit travel km. |
| `visit_converted_customer_count` | `int64` | Visit converted customer count. |
| `crm_interaction_count` | `int64` | Crm interaction count. |
| `average_interest_score` | `float64` | Average interest score. |
| `claimed_distance_km` | `float64` | Claimed distance km. |
| `estimated_distance_km` | `float64` | Estimated distance km. |
| `claimed_expense_amount` | `float64` | Claimed expense amount. |
| `expected_expense_amount` | `float64` | Expected expense amount. |
| `missing_receipt_rate` | `float64` | Ratio for missing receipt rate; zero-denominator handling is explicit. |
| `duplicate_expense_signal` | `float64` | Duplicate expense signal. |
| `capacity_record_id` | `object` | Stable identifier for capacity record. |
| `active_territory_count` | `int64` | Active territory count. |
| `dominant_territory_activity_share` | `float64` | Ratio for dominant territory activity share; zero-denominator handling is explicit. |
| `fractional_territory_allocation` | `object` | Fractional territory allocation. |
| `working_days` | `int64` | Working days. |
| `leave_days` | `float64` | Leave days. |
| `holiday_days` | `float64` | Holiday days. |
| `training_hours` | `float64` | Training hours. |
| `administrative_hours` | `float64` | Administrative hours. |
| `meeting_hours` | `float64` | Meeting hours. |
| `standard_field_hours_per_day` | `float64` | Standard field hours per day. |
| `standard_working_days_per_month` | `int64` | Standard working days per month. |
| `gross_rostered_field_hours` | `float64` | Gross rostered field hours. |
| `non_field_hours` | `float64` | Non field hours. |
| `available_field_hours` | `float64` | Available field hours. |
| `planned_visit_hours` | `float64` | Planned visit hours. |
| `planned_travel_hours` | `float64` | Planned travel hours. |
| `observed_visit_hours` | `float64` | Observed visit hours. |
| `observed_travel_hours` | `float64` | Observed travel hours. |
| `excess_service_visit_hours` | `float64` | Excess service visit hours. |
| `excess_service_travel_hours` | `float64` | Excess service travel hours. |
| `excess_service_hours` | `float64` | Excess service hours. |
| `required_customer_coverage_hours` | `float64` | Required customer coverage hours. |
| `required_priority_customer_coverage_hours` | `float64` | Required priority customer coverage hours. |
| `required_total_hours` | `float64` | Required total hours. |
| `utilization_pct` | `float64` | Required workload hours divided by available field hours, expressed as percent. |
| `capacity_utilization_pct` | `float64` | Percentage value for capacity utilization. |
| `required_fte` | `float64` | Required fte. |
| `available_fte` | `float64` | Available fte. |
| `fte_gap` | `float64` | Required FTE minus available FTE; evidence for workload review, not an employment decision. |
| `customer_coverage_gap` | `float64` | Customer coverage gap. |
| `priority_customer_coverage_gap` | `float64` | Priority customer coverage gap. |
| `customer_coverage_pct` | `float64` | Percentage value for customer coverage. |
| `average_travel_hours` | `float64` | Average travel hours. |
| `workload_per_active_customer` | `float64` | Workload per active customer. |
| `legacy_normalized_workload_index` | `float64` | Legacy normalized workload index. |
| `workload_risk_band` | `object` | Workload risk band. |
| `capacity_risk_band` | `object` | Capacity risk band. |
| `overload_flag` | `bool` | Boolean indicator for overload. |
| `capacity_overload_flag` | `bool` | Boolean indicator for capacity overload. |
| `synthetic_seed` | `int64` | Synthetic seed. |
| `distinct_customers` | `int64` | Distinct customers. |
| `transaction_count` | `int64` | Transaction count. |
| `distinct_cities` | `int64` | Distinct cities. |
| `distinct_products` | `int64` | Distinct products. |
| `distributor_count` | `int64` | Distributor count. |
| `legacy_distinct_customers_training_median` | `float64` | Legacy distinct customers training median. |
| `legacy_transaction_count_training_median` | `float64` | Legacy transaction count training median. |
| `legacy_distinct_cities_training_median` | `float64` | Legacy distinct cities training median. |
| `legacy_distinct_products_training_median` | `float64` | Legacy distinct products training median. |
| `legacy_distributor_count_training_median` | `float64` | Legacy distributor count training median. |
| `legacy_workload_training_end` | `datetime64[ns]` | Legacy workload training end. |
| `legacy_workload_formula` | `object` | Legacy workload formula. |
| `employment_status` | `object` | Employment status. |
| `role_grade` | `object` | Role grade. |
| `priority_customer_count` | `int64` | Priority customer count. |
| `required_visit_count` | `float64` | Required visit count. |
| `priority_required_visit_count` | `float64` | Priority required visit count. |
| `credited_planned_visit_count` | `float64` | Credited planned visit count. |
| `credited_completed_visit_count` | `float64` | Credited completed visit count. |
| `planned_visit_count` | `float64` | Planned visit count. |
| `excess_service_visit_count` | `float64` | Excess service visit count. |
| `observed_visit_count` | `int64` | Observed visit count. |
| `observed_completed_visit_count` | `int64` | Observed completed visit count. |
| `priority_customer_coverage_gap_pct` | `float64` | Percentage value for priority customer coverage gap. |
| `core_required_hours` | `float64` | Core required hours. |
| `workload_buffer_hours` | `float64` | Workload buffer hours. |
| `nominal_full_time_hours` | `float64` | Nominal full time hours. |
| `capacity_zero_denominator_flag` | `bool` | Boolean indicator for capacity zero denominator. |
| `required_hours` | `float64` | Required hours. |
| `available_hours` | `float64` | Available hours. |
| `risk_medium_threshold_pct` | `float64` | Percentage value for risk medium threshold. |
| `risk_high_threshold_pct` | `float64` | Percentage value for risk high threshold. |
| `risk_critical_threshold_pct` | `float64` | Percentage value for risk critical threshold. |
| `overload_threshold_pct` | `float64` | Percentage value for overload threshold. |
| `required_hours_formula` | `object` | Required hours formula. |
| `required_workload_scope` | `object` | Required workload scope. |
| `numeric_visit_frequency_period_divisor` | `float64` | Numeric visit frequency period divisor. |
| `capacity_methodology` | `object` | Capacity methodology. |
| `hire_date` | `datetime64[ns]` | Hire date. |
| `travel_complexity_index` | `float64` | Travel complexity index. |
| `territory_potential` | `float64` | Territory potential. |
| `tenure_months` | `int32` | Tenure months. |
| `tenure_band` | `object` | Tenure band. |
| `potential_band` | `object` | Potential band. |
| `travel_band` | `object` | Travel band. |
| `rolling_sales_mean` | `float64` | Rolling sales mean. |
| `rolling_sales_std` | `float64` | Rolling sales std. |
| `sales_growth` | `float64` | Sales growth. |
| `quantity_growth` | `float64` | Quantity growth. |
| `sales_volatility` | `float64` | Sales volatility. |
| `price_deviation_from_product_norm` | `float64` | Price deviation from product norm. |
| `sales_vs_territory_potential` | `float64` | Sales vs territory potential. |
| `target_attainment_pct` | `float64` | Percentage value for target attainment. |
| `expected_incentive` | `float64` | Expected incentive. |
| `incentive_calculation_residual` | `float64` | Incentive calculation residual. |
| `manual_adjustment_ratio` | `float64` | Ratio for manual adjustment ratio; zero-denominator handling is explicit. |
| `accelerator_cliff_distance` | `float64` | Accelerator cliff distance. |
| `return_rate` | `float64` | Ratio for return rate; zero-denominator handling is explicit. |
| `post_incentive_return_rate` | `float64` | Ratio for post incentive return rate; zero-denominator handling is explicit. |
| `return_clawback_ratio` | `float64` | Ratio for return clawback ratio; zero-denominator handling is explicit. |
| `incentive_growth` | `float64` | Incentive growth. |
| `incentive_volatility` | `float64` | Incentive volatility. |
| `cancelled_order_rate` | `float64` | Ratio for cancelled order rate; zero-denominator handling is explicit. |
| `visits_per_customer` | `float64` | Visits per customer. |
| `sales_per_visit` | `float64` | Sales per visit. |
| `visit_to_sales_conversion` | `float64` | Visit to sales conversion. |
| `missed_priority_visit_count` | `float64` | Missed priority visit count. |
| `crm_interactions_per_customer` | `float64` | Crm interactions per customer. |
| `distance_claim_ratio` | `float64` | Ratio for distance claim ratio; zero-denominator handling is explicit. |
| `expense_per_visit` | `float64` | Expense per visit. |
| `capacity_risk_code` | `int64` | Capacity risk code. |
| `sales_peer_median` | `float64` | Sales peer median. |
| `sales_peer_z` | `float64` | Sales peer z. |
| `sales_peer_percentile` | `float64` | Sales peer percentile. |
| `sales_peer_cohort` | `object` | Sales peer cohort. |
| `incentive_peer_median` | `float64` | Incentive peer median. |
| `incentive_peer_z` | `float64` | Incentive peer z. |
| `incentive_peer_percentile` | `float64` | Incentive peer percentile. |
| `incentive_peer_cohort` | `object` | Incentive peer cohort. |
| `discount_peer_median` | `float64` | Discount peer median. |
| `discount_peer_z` | `float64` | Discount peer z. |
| `discount_peer_percentile` | `float64` | Discount peer percentile. |
| `discount_peer_cohort` | `object` | Discount peer cohort. |
| `expense_peer_median` | `float64` | Expense peer median. |
| `expense_peer_z` | `float64` | Expense peer z. |
| `expense_peer_percentile` | `float64` | Expense peer percentile. |
| `expense_peer_cohort` | `object` | Expense peer cohort. |
| `payout_to_peer_median_ratio` | `float64` | Ratio for payout to peer median ratio; zero-denominator handling is explicit. |
| `discount_vs_peer` | `float64` | Discount vs peer. |
| `expense_vs_peer` | `float64` | Expense vs peer. |
| `peer_group_z_score` | `float64` | Peer group z score. |
| `peer_percentile` | `float64` | Peer percentile. |
| `rep_historical_mean` | `float64` | Rep historical mean. |
| `rep_historical_std` | `float64` | Rep historical std. |
| `rep_historical_z_score` | `float64` | Rep historical z score. |
| `territory_adjusted_sales_residual` | `float64` | Territory adjusted sales residual. |
| `territory_adjusted_incentive_residual` | `float64` | Territory adjusted incentive residual. |
| `tenure_adjusted_performance` | `float64` | Tenure adjusted performance. |
| `product_mix_adjusted_performance` | `float64` | Product mix adjusted performance. |
| `channel_hhi` | `float64` | Channel hhi. |
| `channel_adjusted_performance` | `float64` | Channel adjusted performance. |
| `month_over_month_behavior_change` | `float64` | Month over month behavior change. |
| `rolling_anomaly_score` | `float64` | Rolling anomaly score. |
| `threshold_crossing_discount_signal` | `int64` | Threshold crossing discount signal. |
| `rolling_sales_mean_missing` | `int64` | Rolling sales mean missing. |
| `rolling_sales_std_missing` | `int64` | Rolling sales std missing. |
| `rep_historical_mean_missing` | `int64` | Rep historical mean missing. |
| `rep_historical_std_missing` | `int64` | Rep historical std missing. |
| `split` | `object` | Split. |
| `model_name` | `object` | Model name. |
| `raw_score` | `float64` | Raw score. |
| `anomaly_score` | `float64` | Training-reference percentile of PCA reconstruction error; higher is more unusual. |
| `anomaly_percentile` | `float64` | Anomaly percentile. |
| `raw_threshold` | `float64` | Raw threshold. |
| `threshold_flag` | `bool` | Boolean indicator for threshold. |
| `manager_review_flag` | `bool` | Boolean indicator for manager review. |
| `review_budget_flag` | `bool` | Boolean indicator for review budget. |
| `ground_truth_label` | `bool` | Controlled synthetic evaluation label; excluded from all model features. |
| `anomaly_type` | `object` | Anomaly type. |
| `anomaly_category` | `object` | Anomaly category. |
| `severity` | `object` | Severity. |
| `primary_reason_code` | `object` | Primary reason code. |
| `primary_reason` | `object` | Primary reason. |
| `secondary_reason` | `object` | Secondary reason. |
| `recommended_review_action` | `object` | Recommended review action. |
| `driver_1_feature` | `object` | Driver 1 feature. |
| `driver_1_name` | `object` | Driver 1 name. |
| `driver_1_value` | `float64` | Driver 1 value. |
| `driver_1_peer_value` | `float64` | Driver 1 peer value. |
| `driver_1_percentile` | `float64` | Driver 1 percentile. |
| `driver_1_contribution` | `float64` | Driver 1 contribution. |
| `driver_2_feature` | `object` | Driver 2 feature. |
| `driver_2_name` | `object` | Driver 2 name. |
| `driver_2_value` | `float64` | Driver 2 value. |
| `driver_2_peer_value` | `float64` | Driver 2 peer value. |
| `driver_2_percentile` | `float64` | Driver 2 percentile. |
| `driver_2_contribution` | `float64` | Driver 2 contribution. |
| `driver_3_feature` | `object` | Driver 3 feature. |
| `driver_3_name` | `object` | Driver 3 name. |
| `driver_3_value` | `float64` | Driver 3 value. |
| `driver_3_peer_value` | `float64` | Driver 3 peer value. |
| `driver_3_percentile` | `float64` | Driver 3 percentile. |
| `driver_3_contribution` | `float64` | Driver 3 contribution. |

## `model_lift_curve`

Executed rows: 52.

| Field | Type | Definition |
|---|---|---|
| `review_rank` | `int64` | Review rank. |
| `review_fraction` | `float64` | Review fraction. |
| `threshold` | `float64` | Threshold. |
| `captured_count` | `int64` | Captured count. |
| `precision` | `float64` | Precision. |
| `recall` | `float64` | Recall. |
| `lift` | `float64` | Lift. |

## `model_metrics_summary`

Executed rows: 1.

| Field | Type | Definition |
|---|---|---|
| `model` | `object` | Model. |
| `split` | `object` | Split. |
| `threshold` | `float64` | Threshold. |
| `threshold_basis` | `object` | Threshold basis. |
| `manager_review_fraction` | `float64` | Manager review fraction. |
| `validation_rows` | `int64` | Validation rows. |
| `validation_target_review_count` | `int64` | Validation target review count. |
| `test_rows` | `int64` | Test rows. |
| `test_prevalence` | `float64` | Test prevalence. |
| `precision` | `float64` | Precision. |
| `recall` | `float64` | Recall. |
| `f1` | `float64` | F1. |
| `f2` | `float64` | F2. |
| `specificity` | `float64` | Specificity. |
| `balanced_accuracy` | `float64` | Balanced accuracy. |
| `roc_auc` | `float64` | Roc auc. |
| `pr_auc` | `float64` | Pr auc. |
| `true_positives` | `int64` | True positives. |
| `false_positives` | `int64` | False positives. |
| `false_negatives` | `int64` | False negatives. |
| `true_negatives` | `int64` | True negatives. |
| `predicted_anomalies` | `int64` | Predicted anomalies. |
| `predicted_anomaly_pct` | `float64` | Percentage value for predicted anomaly. |
| `precision_at_1pct` | `float64` | Precision at 1pct. |
| `recall_at_1pct` | `float64` | Recall at 1pct. |
| `lift_at_1pct` | `float64` | Lift at 1pct. |
| `precision_at_5pct` | `float64` | Precision at 5pct. |
| `recall_at_5pct` | `float64` | Recall at 5pct. |
| `lift_at_5pct` | `float64` | Lift at 5pct. |
| `precision_at_10pct` | `float64` | Precision at 10pct. |
| `recall_at_10pct` | `float64` | Recall at 10pct. |
| `lift_at_10pct` | `float64` | Lift at 10pct. |
| `top_decile_capture` | `float64` | Top decile capture. |

## `model_period_stability`

Executed rows: 56.

| Field | Type | Definition |
|---|---|---|
| `population` | `object` | Population. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `observations` | `int64` | Observations. |
| `mean_raw_score` | `float64` | Mean raw score. |
| `median_raw_score` | `float64` | Median raw score. |
| `std_raw_score` | `float64` | Std raw score. |
| `mean_anomaly_score` | `float64` | Mean anomaly score. |
| `p95_anomaly_score` | `float64` | P95 anomaly score. |
| `review_rate` | `float64` | Ratio for review rate; zero-denominator handling is explicit. |
| `selected_entity_overlap_previous_period` | `float64` | Selected entity overlap previous period. |
| `ground_truth_prevalence` | `float64` | Ground truth prevalence. |
| `precision_at_threshold` | `float64` | Precision at threshold. |
| `recall_at_threshold` | `float64` | Recall at threshold. |
| `mean_score_change_from_previous_period` | `float64` | Mean score change from previous period. |

## `model_pr_curve`

Executed rows: 53.

| Field | Type | Definition |
|---|---|---|
| `precision` | `float64` | Precision. |
| `recall` | `float64` | Recall. |
| `threshold` | `float64` | Threshold. |

## `model_roc_curve`

Executed rows: 15.

| Field | Type | Definition |
|---|---|---|
| `false_positive_rate` | `float64` | Ratio for false positive rate; zero-denominator handling is explicit. |
| `true_positive_rate` | `float64` | Ratio for true positive rate; zero-denominator handling is explicit. |
| `threshold` | `float64` | Threshold. |

## `model_score_distributions`

Executed rows: 220.

| Field | Type | Definition |
|---|---|---|
| `population` | `object` | Population. |
| `split` | `object` | Split. |
| `bin` | `int64` | Bin. |
| `score_lower` | `float64` | Score lower. |
| `score_upper` | `float64` | Score upper. |
| `count` | `int64` | Count. |
| `share` | `float64` | Share. |
| `mean_raw_score` | `float64` | Mean raw score. |
| `mean_anomaly_score` | `float64` | Mean anomaly score. |

## `model_top_k_metrics`

Executed rows: 3.

| Field | Type | Definition |
|---|---|---|
| `model` | `object` | Model. |
| `split` | `object` | Split. |
| `review_fraction` | `float64` | Review fraction. |
| `review_count` | `int64` | Review count. |
| `positive_count` | `int64` | Positive count. |
| `captured_count` | `int64` | Captured count. |
| `precision` | `float64` | Precision. |
| `recall` | `float64` | Recall. |
| `lift` | `float64` | Lift. |

## `normalized_source_transactions`

Executed rows: 167,756.

| Field | Type | Definition |
|---|---|---|
| `source_row_id` | `object` | Stable identifier for the unchanged source CSV row. |
| `source_file_row_number` | `int64` | Source file row number. |
| `transaction_id` | `object` | Stable identifier for transaction. |
| `transaction_date` | `datetime64[ns]` | Transaction date. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `manager_name` | `object` | Manager name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `customer_id` | `object` | Stable identifier for customer. |
| `customer_name` | `object` | Customer name. |
| `product_id` | `object` | Stable identifier for product. |
| `product_name` | `object` | Product name. |
| `product_class` | `object` | Product class. |
| `distributor` | `object` | Distributor. |
| `city` | `object` | City. |
| `country` | `object` | Country. |
| `latitude` | `float64` | Latitude. |
| `longitude` | `float64` | Longitude. |
| `channel` | `object` | Channel. |
| `sub_channel` | `object` | Sub channel. |
| `quantity` | `float64` | Quantity. |
| `price` | `float64` | Price. |
| `sales` | `float64` | Sales. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `orders`

Executed rows: 167,756.

| Field | Type | Definition |
|---|---|---|
| `order_id` | `object` | Stable identifier for order. |
| `order_line_id` | `object` | Stable identifier for order line. |
| `source_row_id` | `object` | Stable identifier for the unchanged source CSV row. |
| `source_observed_sales` | `float64` | Source observed sales. |
| `order_date` | `datetime64[ns]` | Order date. |
| `invoice_date` | `datetime64[ns]` | Invoice date. |
| `fulfillment_date` | `datetime64[ns]` | Fulfillment date. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `rep_id` | `object` | Stable identifier for rep. |
| `customer_id` | `object` | Stable identifier for customer. |
| `product_id` | `object` | Stable identifier for product. |
| `territory_id` | `object` | Stable identifier for territory. |
| `quantity` | `float64` | Quantity. |
| `unit_list_price` | `float64` | Unit list price. |
| `gross_sales` | `float64` | Gross sales. |
| `discount_pct` | `float64` | Percentage value for discount. |
| `discount_amount` | `float64` | Discount amount. |
| `net_sales` | `float64` | Net sales. |
| `order_status` | `object` | Order status. |
| `payment_status` | `object` | Payment status. |
| `end_of_period_flag` | `bool` | Boolean indicator for end of period. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `product_master`

Executed rows: 240.

| Field | Type | Definition |
|---|---|---|
| `product_id` | `object` | Stable identifier for product. |
| `product_name` | `object` | Product name. |
| `product_class` | `object` | Product class. |
| `first_observed_period` | `datetime64[ns]` | First observed period. |
| `list_price` | `float64` | List price. |
| `source_sales` | `float64` | Source sales. |
| `source_quantity` | `float64` | Source quantity. |
| `product_category` | `object` | Product category. |
| `launch_date` | `datetime64[ns]` | Launch date. |
| `expected_price_band` | `object` | Expected price band. |
| `margin_pct` | `float64` | Percentage value for margin. |
| `incentive_eligible_flag` | `bool` | Boolean indicator for incentive eligible. |
| `incentive_weight` | `float64` | Incentive weight. |
| `expected_discount_pct` | `float64` | Percentage value for expected discount. |
| `expected_return_rate` | `float64` | Ratio for expected return rate; zero-denominator handling is explicit. |
| `product_complexity_score` | `float64` | Product complexity score. |
| `required_call_intensity` | `int64` | Required call intensity. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `attribute_reference_end_date` | `datetime64[s]` | Attribute reference end date. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `rep_master`

Executed rows: 13.

| Field | Type | Definition |
|---|---|---|
| `rep_id` | `object` | Stable identifier for rep. |
| `rep_name` | `object` | Rep name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `team_id` | `object` | Stable identifier for team. |
| `territory_id` | `object` | Stable identifier for territory. |
| `first_observed_period` | `datetime64[ns]` | First observed period. |
| `last_observed_period` | `datetime64[ns]` | Last observed period. |
| `product_specialization` | `object` | Product specialization. |
| `source_sales` | `float64` | Source sales. |
| `hire_date` | `datetime64[ns]` | Hire date. |
| `tenure_months` | `int32` | Tenure months. |
| `employment_status` | `object` | Employment status. |
| `role_grade` | `object` | Role grade. |
| `historical_performance_band` | `object` | Historical performance band. |
| `standard_field_hours_per_day` | `float64` | Standard field hours per day. |
| `standard_working_days_per_month` | `int64` | Standard working days per month. |
| `baseline_visit_capacity` | `int64` | Baseline visit capacity. |
| `training_hours` | `int64` | Training hours. |
| `administrative_hours` | `int64` | Administrative hours. |
| `leave_days` | `int64` | Leave days. |
| `monthly_available_hours` | `float64` | Monthly available hours. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `attribute_reference_end_date` | `datetime64[s]` | Attribute reference end date. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `rep_targets_quotas`

Executed rows: 364.

| Field | Type | Definition |
|---|---|---|
| `rep_id` | `object` | Stable identifier for rep. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `target_sales` | `float64` | Target sales. |
| `target_units` | `float64` | Target units. |
| `target_priority_product_sales` | `float64` | Target priority product sales. |
| `target_new_customer_sales` | `float64` | Target new customer sales. |
| `target_visit_count` | `int64` | Target visit count. |
| `quota_difficulty_index` | `float64` | Quota difficulty index. |
| `target_revision_flag` | `bool` | Boolean indicator for target revision. |
| `target_effective_date` | `datetime64[ns]` | Target effective date. |
| `target_version` | `int64` | Target version. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `returns_cancellations`

Executed rows: 6,452.

| Field | Type | Definition |
|---|---|---|
| `return_id` | `object` | Stable identifier for return. |
| `order_id` | `object` | Stable identifier for order. |
| `order_line_id` | `object` | Stable identifier for order line. |
| `rep_id` | `object` | Stable identifier for rep. |
| `customer_id` | `object` | Stable identifier for customer. |
| `product_id` | `object` | Stable identifier for product. |
| `original_order_date` | `datetime64[ns]` | Original order date. |
| `return_date` | `datetime64[ns]` | Return date. |
| `return_quantity` | `float64` | Return quantity. |
| `return_amount` | `float64` | Return amount. |
| `cancellation_flag` | `bool` | Boolean indicator for cancellation. |
| `return_reason` | `object` | Return reason. |
| `payout_period` | `datetime64[ns]` | Payout period. |
| `after_incentive_payout_flag` | `bool` | Boolean indicator for after incentive payout. |
| `days_after_order` | `int64` | Days after order. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `team_master`

Executed rows: 4.

| Field | Type | Definition |
|---|---|---|
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `manager_id` | `object` | Stable identifier for manager. |
| `region` | `object` | Region. |
| `active_rep_count` | `int64` | Active rep count. |
| `management_span` | `int64` | Management span. |
| `attribute_reference_end_date` | `datetime64[s]` | Attribute reference end date. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `territory_master`

Executed rows: 8.

| Field | Type | Definition |
|---|---|---|
| `territory_id` | `object` | Stable identifier for territory. |
| `territory_name` | `object` | Territory name. |
| `team_id` | `object` | Stable identifier for team. |
| `team_name` | `object` | Team name. |
| `country` | `object` | Country. |
| `city` | `object` | City. |
| `customer_count` | `int64` | Customer count. |
| `source_sales` | `float64` | Source sales. |
| `source_transaction_count` | `int64` | Source transaction count. |
| `source_rep_count` | `int64` | Source rep count. |
| `source_latitude_centroid` | `float64` | Source latitude centroid. |
| `source_longitude_centroid` | `float64` | Source longitude centroid. |
| `distinct_cities` | `int64` | Distinct cities. |
| `distinct_products` | `int64` | Distinct products. |
| `state` | `object` | State. |
| `region` | `object` | Region. |
| `urbanicity` | `object` | Urbanicity. |
| `territory_potential` | `float64` | Territory potential. |
| `customer_density` | `float64` | Customer density. |
| `travel_complexity_index` | `float64` | Travel complexity index. |
| `average_distance_between_customers` | `float64` | Average distance between customers. |
| `product_complexity_index` | `float64` | Product complexity index. |
| `expected_monthly_workload_hours` | `float64` | Expected monthly workload hours. |
| `expected_rep_capacity_hours` | `float64` | Expected rep capacity hours. |
| `priority_customer_count` | `int64` | Priority customer count. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `attribute_reference_end_date` | `datetime64[s]` | Attribute reference end date. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |

## `travel_expenses`

Executed rows: 21,925.

| Field | Type | Definition |
|---|---|---|
| `expense_id` | `object` | Stable identifier for expense. |
| `rep_id` | `object` | Stable identifier for rep. |
| `visit_id` | `object` | Stable identifier for visit. |
| `period` | `datetime64[ns]` | First calendar day representing the monthly scoring period. |
| `expense_date` | `datetime64[ns]` | Expense date. |
| `expense_category` | `object` | Expense category. |
| `claimed_distance_km` | `float64` | Claimed distance km. |
| `estimated_distance_km` | `float64` | Estimated distance km. |
| `claimed_amount` | `float64` | Claimed amount. |
| `expected_amount` | `float64` | Expected amount. |
| `receipt_available_flag` | `bool` | Boolean indicator for receipt available. |
| `approval_status` | `object` | Approval status. |
| `deviation_pct` | `float64` | Percentage value for deviation. |
| `currency_code` | `object` | Currency code; UNK means the source supplied no reliable currency field. |
| `data_lineage` | `object` | Observed/synthetic provenance category for this row. |
