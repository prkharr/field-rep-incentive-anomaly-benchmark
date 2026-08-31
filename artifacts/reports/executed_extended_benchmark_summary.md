# Executed extended commercial anomaly benchmark

Source: local user-provided `pharma-data.csv`, 167,760 rows × 18 columns.
After 3 exact duplicate removals and 1 invalid-row exclusions: 167,756 transactions.
Actual coverage: 2017-01-01 to 2019-04-01, NOT a complete 2019 year.
Country coverage: {'Germany': {'start': '2017-01-01', 'end': '2019-04-01', 'rows': 127273}, 'Poland': {'start': '2018-01-01', 'end': '2018-12-01', 'rows': 40483}}. Poland's absent 2019 records are NOT evidence of zero staffing/demand.
Grain: Rep x Product Class x Month; 2,184 observations, 124 model features.
TRAIN: through 2018-06-01 (1404 rows); VALIDATION: through 2018-12-01 (468 rows); TEST: 2019-01-01–2019-04-01 (312 rows).
Primary test contains 19 controlled labels; validation contains 29.

## Final TEST comparison (all models, identical population)

| model | Recall@5% | Lift@5% | PR_AUC | F2 | stability | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| PCA Reconstruction | 0.421 | 8.211 | 0.395 | 0.435 | 1.000 | 0.061 |
| Autoencoder | 0.421 | 8.211 | 0.382 | 0.435 | 0.917 | 4.722 |
| Best Ensemble | 0.474 | 9.237 | 0.350 | 0.489 | 1.000 | 12.895 |
| DBSCAN | 0.316 | 6.158 | 0.297 | 0.326 | 1.000 | 1.164 |
| Business Rules | 0.211 | 4.105 | 0.238 | 0.217 | 1.000 | 0.031 |
| K-Means | 0.211 | 4.105 | 0.219 | 0.217 | 0.979 | 3.881 |
| Robust Peer Baseline | 0.158 | 3.079 | 0.165 | 0.163 | 1.000 | 0.063 |
| Isolation Forest | 0.105 | 2.053 | 0.130 | 0.109 | 0.792 | 11.577 |
| Rolling Residual | 0.158 | 3.079 | 0.120 | 0.163 | 1.000 | 1.163 |
| EWMA Residual | 0.158 | 3.079 | 0.117 | 0.163 | 1.000 | 1.163 |
| Seasonal Residual | 0.105 | 2.053 | 0.108 | 0.109 | 1.000 | 1.163 |
| Change-Point / Level Shift | 0.000 | 0.000 | 0.076 | 0.000 | 1.000 | 1.163 |

Classification flags and ranking metrics use exact top review budgets with deterministic ties. They are not calibrated probabilities of misconduct.
Runtime includes parameter search for fitted models; ensemble runtime includes its components. The entire run took 126.6s; package versions, warnings, parameters and seed are in `extended_run_metadata.json`.

## Validation-selected recommendations

- Segmentation: K-Means (clean train/validation clustering quality, independent of anomaly test metrics).
- Best individual anomaly model: PCA Reconstruction.
- Best temporal specialist: EWMA Residual.
- Best interpretable comparator: K-Means (validation utility).
- Primary review architecture: PCA Reconstruction.
- Ensemble: maximum with {'Isolation Forest': 0.3, 'EWMA Residual': 0.2, 'Robust Peer Baseline': 0.2, 'PCA Reconstruction': 0.15, 'Business Rules': 0.15}.
- Material ensemble improvement: False; validation Recall@5% gain -0.103, PR-AUC gain -0.101.
- Keep peer/history/rule explanations alongside the primary score, not as an automated adverse decision.

## Capacity scenarios

| team | country | product_class | allocated_current_fte | required_fte | fte_gap | hiring_priority |
| --- | --- | --- | --- | --- | --- | --- |
| Delta | Germany | Antibiotics | 0.556 | 0.452 | -0.105 | 9.946 |
| Alfa | Germany | Antimalarial | 0.400 | 0.349 | -0.051 | 6.091 |
| Delta | Germany | Antimalarial | 0.546 | 0.459 | -0.087 | 4.983 |
| Charlie | Germany | Analgesics | 0.508 | 0.420 | -0.088 | 4.467 |
| Bravo | Germany | Antipiretics | 0.430 | 0.349 | -0.080 | 3.984 |
| Charlie | Germany | Antipiretics | 0.420 | 0.339 | -0.081 | 3.727 |
| Delta | Germany | Analgesics | 0.693 | 0.558 | -0.135 | 3.697 |
| Bravo | Germany | Antibiotics | 0.416 | 0.331 | -0.084 | 3.329 |

Current FTE summed across business units: 13.00; modeled required FTE: 10.15; sum of positive LOCAL gaps: 0.00.
Local gaps are not a net hiring mandate: reallocation, partial FTE, forecast uncertainty, capacity assumptions and territory constraints matter.
Stale-source business units are explicitly ineligible, with no FTE gap or priority estimate.

## Important benchmark limitations

- PCA wins validation; Autoencoder may lead individual test metrics. The test set does not reselect the winner. Compare the numerical table rather than assuming a complex model is superior.
- The Autoencoder reached its bounded iteration limit; loss curves and early-stopping information are persisted. This is not a claim of fully converged optimization.
- Clean DEMO payout equals its deterministic expected formula. Reconstruction methods can detect deviations from this artificial relationship. This is not proof of performance on real payroll data.
- Some injected aggregate changes intentionally break quantity/price/payout relationships. They are controlled experiments, not a transaction-level fraud simulator.
- The cross-country coverage change introduces a structural historical break. Same-month peer scoring is retrospective; older mixed-country history is not fully comparable with Germany-only 2019.
- Lower prevalence runs use three independent seeds and random type subsets; only five positives per test run make these estimates very uncertain.
- Trend-break family recall here uses level shifts as a limited proxy; gradual trend-change detection needs longer independently labeled histories.
- K-Means uses Euclidean L2 after TRAIN median imputation, signed-log tail compression and RobustScaler. No test fitting or clipping. Its signed-log geometry is intentional and fully persisted.
