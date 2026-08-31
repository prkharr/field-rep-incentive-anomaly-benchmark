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


Models were frozen before test evaluation. Clustering selection uses unsupervised validation quality with train cluster-balance constraints. Other grids use validation ranking utility; final selection adds seed stability. Ensemble must improve validation recall by at least 0.03 AND PR-AUC by 0.02 without more than 0.05 stability loss. Equal percentile/rank averaging are identical here because model scores already represent TRAIN-reference percentile ranks.
These rules are transparent hackathon priorities, not evidence of production optimality. Synthetic labels inform validation only; final test metrics never choose a model or weights.
