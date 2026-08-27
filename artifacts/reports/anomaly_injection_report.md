# Controlled anomaly injection

Injected 250 labeled anomalies across 12 types. Severity varies continuously and the before/after audit is stored in `data/processed/anomaly_injection_audit.csv`.

- `abnormally_high_calls`: 21
- `duplicate_suspicious_activity_pattern`: 20
- `extreme_quantity_spike`: 21
- `extreme_sales_spike`: 21
- `high_incentive_low_target_attainment`: 21
- `high_incentive_weak_sales`: 21
- `large_manual_incentive_override`: 21
- `low_coverage_high_incentive`: 21
- `sales_inconsistent_with_peer_group`: 21
- `sales_inconsistent_with_territory_opportunity`: 21
- `unusual_product_mix`: 20
- `very_high_sales_low_activity`: 21

Evaluation labels (`injected_anomaly_flag`, `anomaly_type`, `anomaly_severity`) are excluded from preprocessing and model features.
