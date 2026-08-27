# Executed methodology

1. Discover and profile source data; use a labeled deterministic fallback only when no qualifying input exists.
2. Create a stable customer-to-representative mapping within sales team, manager, and territory.
3. Aggregate at representative × product × territory × month.
4. Add business-related activity, target, capacity, opportunity, and incentive fields.
5. Inject 5–7% controlled anomalies with variable severity and preserve a before/after audit.
6. Engineer portfolio, growth, efficiency, peer-comparison, and opportunity features.
7. Median-impute and RobustScale numeric features. Fitted quantile clipping (1.0%–99.0%) was applied before scaling to limit extreme leverage without changing saved business-unit values.
8. Tune K-Means and DBSCAN using clustering quality only; evaluate selected configurations against held-out-purpose synthetic labels.
9. Rank anomalies continuously, explain their primary feature drivers, and roll results to representative level.

## Leakage control

The model feature list contains 42 numeric fields. The evaluation-only fields `anomaly_severity, anomaly_type, injected_anomaly_flag` are explicitly rejected by feature validation and never enter preprocessing or clustering.

## Warnings retained from source validation

- No fatal data-quality warnings.

## Important limitations

- Synthetic anomaly labels support a demo benchmark; they do not prove real-world generalisation.
- Model choice based on the same injected label design may be optimistic.
- DBSCAN is deterministic for fixed data and parameters, while its reported stability assesses small input perturbations.
- scikit-learn DBSCAN has no native prediction; the persisted wrapper uses nearest-core assignment within eps for new rows.
- Anomaly flags indicate review priority, not fraud, misconduct, or incorrect payment.
