# Data dictionary and field lineage

## How to read this dictionary

The same column name can be physically present in multiple tables, but its provenance is more important than its location. This run uses the fallback, so even the commercial foundation is synthetic. The lineage classes below keep the layers explicit:

| Code | Lineage | Meaning |
|---|---|---|
| `F` | Source-like fallback commercial | Generated to mimic a possible input record. If a real CSV is supplied later, canonical source columns replace this layer. |
| `M` | Deterministic mapping | Created from commercial/organizational keys with seed `42`; reproducible, but not a real HR/CRM relationship. |
| `E` | Synthetic enrichment | Generated conditionally from commercial performance, opportunity, activity, and workload. It supplements rather than replaces sales. |
| `D` | Derived feature | Calculated from the analytical table, temporal history, or peer/territory aggregates. |
| `L` | Evaluation-only label | Controlled ground truth used after scoring; **never a model input**. |
| `O` | Model/report output | Cluster, score, flag, profile, or explanation created by a fitted method. |

When a provided CSV is used, its successfully mapped canonical fields should be interpreted as original source fields rather than `F`. `provenance.json` and the data-quality report retain the exact source-to-canonical mapping.

## Commercial foundation

Fallback headers are canonicalized to the following internal names.

| Field | Type | Lineage | Definition / unit |
|---|---|---|---|
| `distributor` | string | F | Distributor identifier/name associated with the transaction. |
| `customer` | string | F | Account/customer identifier. It is not asserted to be an HCP. |
| `city` | string | F | Customer/account city. |
| `country` | string | F | Customer/account country. |
| `latitude` | float | F | Approximate account/location latitude in decimal degrees. |
| `longitude` | float | F | Approximate account/location longitude in decimal degrees. |
| `channel` | string | F | Commercial channel, such as hospital, retail, or clinic. |
| `subchannel` | string | F | More detailed account/channel classification. |
| `product_name` | string | F | Product/brand identifier used for aggregation. |
| `product_class` | string | F | Therapeutic or commercial product class. |
| `quantity` | numeric | F | Units on the transaction. If absent in a supplied source, may be derived as `sales / price` and is then derived, not original. |
| `price` | numeric | F | Unit price in the dataset's unspecified currency. If absent, may be derived as `sales / quantity`. |
| `sales` | numeric | F | Transaction sales value in an unspecified currency. Fallback values are approximately `quantity × price` with small variation. If absent, may be derived from quantity and price. |
| `month` | string/date-like | F | Source month value used with year to form the monthly timestamp. |
| `year` | integer | F | Source calendar year. |
| `date` | datetime | D | Canonical first-of-month timestamp derived from the source date or month/year. |
| `sales_manager` | string | F | Manager label in the commercial foundation; synthetic in the fallback. |
| `sales_team` | string | F | Team label in the commercial foundation; synthetic in the fallback. |

The fallback additionally embeds location opportunity and product-demand parameters inside its generator. Those generator parameters are not treated as observed source columns in the analytical table.

## Deterministic representative mapping

| Field | Type | Lineage | Definition / unit |
|---|---|---|---|
| `territory_id` | string | M | Stable territory identifier derived from geographic/organizational keys. |
| `rep_id` | string | M | Synthetic representative ID (`REP_…`) assigned deterministically within the territory/team/manager structure. |
| `manager_id` | string | M | Deterministic synthetic identifier derived from the manager label for future organizational joins. |
| `rep_slot` | integer | M | Zero-based assignment slot within a team/manager/territory combination; used in the persisted mapping table. |
| `rep_tenure_months` | integer | M | Seeded, stable synthetic tenure between 6 and 120 months for a mapped representative. |
| `rep_capacity` | float | M | Seeded, stable relative capacity proxy between 80 and 120 for a mapped representative; it is not literal FTE. |
| `assigned_customer_portfolio` | integer | M/D | Distinct customers assigned to the synthetic representative by the deterministic mapping. |
| customer-to-rep assignment | relationship | M | Stable ownership link used to aggregate customer transactions to a representative; it is not a real CRM assignment. |
| `date` | datetime | D | First-of-month timestamp and month component of the representative behavioral grain. |

The core row grain after mapping and aggregation is approximately `rep_id × product_name × territory_id × month` with geography, manager, and team carried as dimensions.

## Synthetic enrichment

These fields are generated after the commercial foundation and are logically coupled to it.

| Field | Type | Lineage | Definition / unit |
|---|---|---|---|
| `total_calls` | integer | E | Field/customer calls during the analytical month. |
| `unique_customers_contacted` | integer | E | Distinct customers contacted during the month; bounded by the available customer base. |
| `priority_customer_calls` | integer | E | Calls directed to synthetic priority customers. |
| `digital_engagements` | integer | E | Remote/digital customer engagements during the month. |
| `working_days` | integer | E | Synthetic working days available in the month. |
| `travel_distance_km` | float | E | Monthly territory travel proxy in kilometres. |
| `call_plan_adherence_pct` | float | E | Planned-call completion percentage; percentage scale. |
| `customer_coverage_pct` | float | E | `unique_customers_contacted / assigned_customer_portfolio × 100`, capped at 100. |
| `sales_target` | float | E | Monthly sales target in the same unspecified currency as sales. |
| `quantity_target` | float | E | Synthetic monthly unit target, normally `sales_target / average_price`. |
| `target_attainment_pct` | float | E/D | `total_sales / sales_target × 100`, after zero-safe handling; based on a synthetic target. |
| `target_incentive` | float | E | Reference/on-target incentive amount in an unspecified currency. |
| `calculated_incentive` | float | E | Rules-based synthetic incentive driven primarily by target attainment and plan logic. |
| `actual_incentive_paid` | float | E | Synthetic paid amount, normally close to calculated incentive plus recorded adjustments. |
| `incentive_attainment_pct` | float | E/D | `actual_incentive_paid / target_incentive × 100`. |
| `payout_adjustment` | float | E | `actual_incentive_paid - calculated_incentive`. |
| `manual_override_amount` | float | E | Synthetic manually approved component; normally near zero and changed by relevant anomaly scenarios. |

Percentages may exceed 100 when performance or payout exceeds target. Downstream presentation should not silently cap them.

## Aggregated and intermediate support fields

These fields remain useful for auditing formulas even when they are not selected as direct model inputs.

| Field | Lineage | Meaning |
|---|---|---|
| `transaction_count` | D | Number of commercial foundation rows aggregated into the analytical row. |
| `territory_customer_count` | D | Distinct commercial customers observed in the territory. |
| `territory_lat_spread` | D | Standard deviation of customer latitude within the territory. |
| `territory_lon_spread` | D | Standard deviation of customer longitude within the territory. |
| `assigned_customer_portfolio` | M/D | Distinct customers mapped to the representative across the commercial foundation. |
| `opportunity_index_raw` | E/D | Unbounded customer-scale × market-potential proxy before percentile normalization. |
| `workload_index_raw` | E/D | Enrichment-stage workload proxy; the final engineered `workload_index` is recalculated from calls, portfolio, and capacity. |
| `product_sales_share` | D | Product's share of representative-territory-month portfolio sales. |
| `sales_peer_median` | D | Selected hierarchical peer median used to calculate the sales peer deviation. |
| `incentive_peer_median` | D | Selected hierarchical peer median used to calculate the incentive peer deviation. |
| `activity_peer_median` | D | Selected hierarchical peer median used to calculate the activity peer deviation. |

## Engineered model features

The table below documents the default configured features. Formulas are zero-safe; exact grouping keys are implemented in `feature_engineering.py` and should be read with the current configuration.

### Commercial performance

| Feature | Lineage | Meaning |
|---|---|---|
| `total_sales` | D | Sum of sales at the analytical grain. |
| `sales_growth` | D | Period-over-period percentage change within a rep/product/territory series, clipped to `[-10, 10]`; first observation is 0. |
| `rolling_sales_growth` | D | Current sales change versus the shifted mean of up to three prior observations in the same series, clipped to `[-10, 10]`. |
| `total_quantity` | D | Sum of units at the analytical grain. |
| `quantity_growth` | D | Period-over-period percentage change within a rep/product/territory series, clipped to `[-10, 10]`; first observation is 0. |
| `average_price` | D | Mean observed transaction price at the analytical grain, with `sales / quantity` used only as a missing-price fallback. |
| `sales_per_customer` | D | `total_sales / unique_customers`. |
| `sales_per_product` | D | Total representative-territory-month portfolio sales divided by its number of unique products. |
| `sales_per_call` | D | `total_sales / total_calls`. |

### Customer, channel, and product mix

| Feature | Lineage | Meaning |
|---|---|---|
| `unique_customers` | D | Distinct customers represented at the analytical grain. |
| `customer_coverage_pct` | E/D | Synthetic contacted-customer coverage, carried into model features. |
| `customer_concentration` | D | Herfindahl concentration (sum of squared sales shares) across customers; larger values mean less diversification. |
| `channel_mix` | D | Herfindahl concentration of sales across channels; despite the name, larger values mean a narrower mix. |
| `subchannel_mix` | D | Herfindahl concentration of sales across sub-channels; larger values mean a narrower mix. |
| `unique_products` | D | Distinct product count in the applicable rep/month context. |
| `product_concentration` | D | Sum of squared product sales shares (Herfindahl concentration) within representative-territory-month. |
| `dominant_product_share` | D | Largest product's share of relevant representative sales, on a 0–1 scale. |

### Activity and workload

| Feature | Lineage | Meaning |
|---|---|---|
| `calls_per_working_day` | D | `total_calls / working_days`. |
| `calls_per_customer` | D | `total_calls / unique_customers_contacted`. |
| `activity_efficiency` | D | `total_sales / (total_calls + digital_engagements)`. |
| `travel_per_customer` | D | `travel_distance_km / unique_customers_contacted`. |
| `workload_index` | D | `0.55 × total_calls/rep_capacity + 0.45 × assigned_customer_portfolio/rep_capacity`; relative, not hours or FTE. |
| `call_plan_adherence_pct` | E | Synthetic plan-adherence percentage. |
| `rep_capacity` | M | Seeded, deterministic synthetic capacity proxy carried from the representative mapping. |

### Incentive relationships

| Feature | Lineage | Meaning |
|---|---|---|
| `incentive_to_sales_ratio` | D | `actual_incentive_paid / total_sales`. |
| `incentive_per_customer` | D | Actual incentive divided by unique customers. |
| `incentive_per_call` | D | Actual incentive divided by total calls. |
| `incentive_variance` | D | `actual_incentive_paid - calculated_incentive`. |
| `incentive_variance_pct` | D | `(actual_incentive_paid - calculated_incentive) / abs(calculated_incentive)`. |
| `incentive_to_target_ratio` | D | Actual incentive relative to target incentive. |
| `target_attainment_pct` | E/D | Sales attainment against the synthetic target, percentage scale. |

### Peer comparisons

| Feature | Lineage | Meaning |
|---|---|---|
| `sales_vs_peer_median` | D | `(total_sales - peer median) / abs(peer median)`. |
| `incentive_vs_peer_median` | D | `(actual_incentive_paid - peer median) / abs(peer median)`. |
| `activity_vs_peer_median` | D | `(total_calls - peer median) / abs(peer median)`. |
| `sales_zscore_within_peer` | D | `(total_sales - peer mean) / peer standard deviation`. |
| `incentive_zscore_within_peer` | D | `(actual_incentive_paid - peer mean) / peer standard deviation`. |
| `activity_zscore_within_peer` | D | `(total_calls - peer mean) / peer standard deviation`. |

Peer statistics use the first hierarchy level with at least three rows: product × territory × month, then product × sales team × month, then product × month, then month. Global statistics are the final fallback. Small or zero-variance groups receive zero-safe handling; the feature does not imply a statistically representative population.

### Territory and opportunity

| Feature | Lineage | Meaning |
|---|---|---|
| `territory_market_potential` | E/D | Territory sales-potential proxy adjusted by territory customer count relative to the median. |
| `territory_customer_density` | D | Distinct territory customers divided by a latitude/longitude spread-area proxy; not census density. |
| `territory_sales_potential` | E/D | Blend of territory-month total sales and the territory's historical median product-level sales. |
| `opportunity_index` | D | Raw opportunity proxy min-max normalized between its 1st and 99th percentiles, then clipped to 0–1. |
| `market_potential_adjusted_sales` | D | `total_sales / territory_market_potential`. |

Opportunity fields in this benchmark are synthetic/proxy constructs. They are not external market estimates or validated HCP potential.

## Evaluation-only fields

| Field | Type | Lineage | Use |
|---|---|---|---|
| `injected_anomaly_flag` | boolean/integer | L | `1` for a controlled injected scenario and `0` otherwise. Used only after scoring. |
| `anomaly_type` | string | L | Controlled scenario category. Never used as a model feature or tuning input. |
| `anomaly_severity` (when persisted) | float | L | Seeded scenario intensity. Evaluation/audit only; never a feature. |

Label exclusion is a hard leakage boundary. It also covers direct encodings, aliases, and any field derived from the label itself.

## Model and reporting outputs

Output column names may carry a method prefix in the scored CSVs.

| Field family | Lineage | Meaning |
|---|---|---|
| cluster label | O | K-Means cluster integer or DBSCAN cluster integer; DBSCAN `-1` means density noise/review candidate. |
| anomaly score | O | Continuous score oriented so larger values are more unusual and normalized to 0–1 where supported. |
| anomaly flag | O | Thresholded review recommendation at the configured contamination/review budget. It is not the injected label. |
| centroid/neighbor distance | O | K-Means distance to assigned centroid or DBSCAN nearest-neighbor/density distance. |
| top anomaly drivers | O | Human-readable highest-contribution features and peer deviations. |
| PCA coordinates | O | Two-dimensional visualization projection only; models are fit on the configured preprocessed feature space, not the plot. |
| profile/business interpretation | O | Descriptive label derived from cluster statistics, not a fixed class or causal conclusion. |

## Units and missing values

- Currency is intentionally unspecified in the fallback; all sales, target, and incentive monetary fields share the same synthetic unit.
- Distances are kilometres; coordinates are decimal degrees; time is calendar months/days.
- Most fields ending `_pct` use percentage points. The current `incentive_variance_pct` implementation is an explicitly documented exception stored as a unitless fraction. Fields ending `_ratio` and `_share` normally use unitless fractions.
- Divisions by zero produce missing/neutral values through zero-safe logic, never infinity.
- Model preprocessing imputes remaining numeric missingness and applies configured clipping/scaling. Stored business-unit values remain available for explanations.
