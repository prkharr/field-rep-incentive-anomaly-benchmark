# Field Representative Incentive Anomaly Detection

An executed, reproducible K-Means and DBSCAN benchmark for prioritizing unusual pharmaceutical field-representative performance and incentive patterns for business review.

> **Data disclosure:** no qualifying field-sales CSV was available in the workspace. This committed run therefore uses a clearly labeled deterministic fallback commercial dataset. Results are demonstration evidence—not production findings and not evidence of fraud, misconduct, or incorrect payment.

## Executed result

The pipeline completed successfully with seed `42` using Python 3.10.4 and scikit-learn 1.7.2.

| Measure | Executed value |
|---|---:|
| Source-like transaction rows | 9,000 |
| Source columns | 18 |
| Date coverage | Jan 2024–Jun 2025 |
| Countries / cities | 4 / 12 |
| Products / classes | 8 / 8 |
| Synthetic representatives | 36 |
| Analytical rows | 4,161 |
| Analytical grain | Rep × Product × Territory × Month |
| Injected anomalies | 250 (6.01%) |
| Plots generated | 20 |
| Pipeline runtime | ~52 seconds |

Workspace-wide discovery reviewed 135 CSV and 4 XLSX files. No file matched more than one of the expected commercial fields. The two nearest pharma datasets were company-level financial statements and drug-development pipeline records, both rejected as the wrong business grain. See [workspace_data_discovery.md](docs/workspace_data_discovery.md) and [data_quality_report.md](artifacts/reports/data_quality_report.md).

## Final benchmark

These values come from the committed execution—not hand-entered estimates.

| Metric | K-Means | DBSCAN |
|---|---:|---:|
| Selected parameters | `k=3` | `eps=5.2851`, `min_samples=16` |
| Clusters | 3 | 2 (+ noise) |
| Noise | 0.00% | 5.36% |
| Silhouette | 0.248 | 0.909 |
| Davies–Bouldin | 1.330 | 0.114 |
| Stability | 0.999 | 0.997 |
| Precision | 0.656 | 0.659 |
| Recall | 0.656 | 0.588 |
| F1 / F2 | 0.656 / 0.656 | 0.622 / 0.601 |
| PR-AUC | 0.710 | 0.692 |
| ROC-AUC | 0.924 | 0.930 |
| Precision@5% | 0.699 | 0.684 |
| Recall@5% | 0.584 | 0.572 |
| Lift@5% | 11.63× | 11.39× |
| Top-decile capture | 76.4% | 75.2% |

### Model conclusions

- **Best segmentation model: K-Means.** Its weighted segmentation score was approximately `0.60` versus DBSCAN's `0.43`. Selection did not rely on elbow or silhouette alone: it incorporated silhouette, Davies–Bouldin, Calinski–Harabasz, cluster balance, stability, bounded runtime utility, interpretability, and operational usefulness. The `k=2` result was rejected because one cluster exceeded 90% of the population; configurations with sub-1% microclusters were also rejected.
- **Best anomaly-detection model: K-Means.** Its weighted anomaly score was approximately `0.90` versus `0.14`, reflecting the configured mix of recall, precision, F2, PR-AUC, Lift@5%, stability, bounded runtime utility, interpretability, and operational usefulness. DBSCAN retained slightly higher point precision and ROC-AUC, but K-Means captured more injected anomalies and ranked the finite review queue better on the configured priorities.

The weights live in [config.yaml](configs/config.yaml), while every contribution is auditable in `artifacts/metrics/model_selection_contributions.csv`. Synthetic labels were used only for evaluation and final anomaly-model comparison; they were never clustering inputs or unsupervised tuning criteria.

## What the pipeline does

1. Searches configured workspace locations for a defensible pharma commercial CSV and records the search audit.
2. Profiles exact columns, shape, types, nulls, cardinality, statistics, duplicates, dates, geography, products, managers, and teams.
3. Builds deterministic `Sales Team → Manager → Territory → Rep → Customer` mappings.
4. Aggregates transactions to rep × product × territory × month.
5. Adds business-related activity, targets, capacity, opportunity, calculated incentive, actual payout, and adjustment fields without replacing the commercial foundation.
6. Injects 12 controlled anomaly types with variable severity and writes a before/after audit.
7. Engineers 42 numeric commercial, portfolio, activity, incentive, peer, and opportunity features.
8. Median-imputes, fitted-quantile-clips, and RobustScales features; StandardScaler and disabled clipping remain configurable alternatives.
9. Tunes K-Means (`k=2…12`) and DBSCAN (`eps × min_samples`) without anomaly-label leakage.
10. Produces clustering, classification, ranking, stability, profiles, per-row explanations, representative rollups, persisted models, and dashboard-ready artifacts.

K-Means anomaly distance is the ordinary Euclidean (L2) norm from each transformed row to its assigned nearest centroid. It is regression-tested against both the manual formula and scikit-learn's `KMeans.transform`; squared distances reconcile to inertia. The 0–1 score is the empirical percentile of that distance, and feature contributions are shares of squared distance.

## Quick start

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python run_pipeline.py --config configs\config.yaml
python -m pytest -q
streamlit run app.py
```

To use a provided CSV explicitly:

```powershell
python run_pipeline.py --config configs\config.yaml --input C:\path\to\pharma_sales.csv
```

Common column aliases are adapted by name. Missing optional organizational/context fields are documented and defaulted; the original input is never overwritten.

## Dashboard

`app.py` is a read-only Streamlit prototype that loads the generated artifacts without retraining. It includes:

- executive KPIs and weighted model outcome;
- country, city, territory, team, and manager filters with a geographic risk map;
- product/class/channel/sub-channel views;
- K-Means versus DBSCAN benchmark details;
- cluster profiles, PCA, and derived business interpretations;
- a ranked rep/product/month investigation queue with top drivers and CSV export;
- methodology, provenance, limitations, and governance guardrails.

## Key outputs

| Path | Purpose |
|---|---|
| `data/synthetic/fallback_pharma_sales.csv` | Explicit source-like fallback; never represented as provided data |
| `data/processed/rep_mapping.csv` | Deterministic customer/territory/team/manager-to-rep mapping |
| `data/processed/analytical_dataset.csv` | Enriched, injected, feature-engineered analytical rows |
| `data/processed/scored_observations.csv` | Both model clusters/scores/flags, selected score, PCA, and explanations |
| `data/processed/rep_risk_summary.csv` | One highest-risk record plus aggregates per representative |
| `artifacts/metrics/clustering_benchmark.csv` | Requested Metric × K-Means × DBSCAN benchmark |
| `artifacts/metrics/clustering_benchmark_long.csv` | Every tested clustering configuration |
| `artifacts/metrics/anomaly_metrics.csv` | Confusion, ROC/PR, and threshold metrics |
| `artifacts/metrics/ranking_metrics.csv` | Precision, recall, and lift at 1%, 5%, and 10% |
| `artifacts/metrics/cluster_profiles.csv` | Actual cluster statistics and derived interpretations |
| `artifacts/models/` | Fitted preprocessor and final model objects |
| `artifacts/plots/` | All 20 requested diagnostic/business plots |
| `artifacts/reports/anomaly_investigations.csv` | Ranked high-risk observation queue with drivers |
| `artifacts/reports/kmeans_distance_validation.json` | Executed Euclidean/manual/sklearn/inertia reconciliation evidence |
| `notebooks/clustering_benchmark.ipynb` | Lightweight artifact-review notebook |

## Repository layout

```text
field-rep-incentive-anomaly-benchmark/
├── app.py                         # Streamlit dashboard
├── run_pipeline.py                # CLI
├── configs/config.yaml            # Data/model/selection settings
├── data/{raw,synthetic,processed}/
├── artifacts/{metrics,models,plots,reports}/
├── notebooks/clustering_benchmark.ipynb
├── src/field_rep_anomaly/         # Modular production code
├── tests/                         # 21 fast unit tests
└── docs/                          # Methodology, dictionary, discovery, future planning
```

## Tests and reproducibility

The committed run passes **21/21 tests** with no warnings. Tests cover schema adaptation, deterministic generation/mapping, analytical-grain uniqueness, enrichment identities, 5–7% controlled injection, all anomaly types, leakage protection, finite features, StandardScaler/RobustScaler/clipping, exact K-Means/Euclidean distance identities, common model interfaces, scoring bounds/contributions, hand-checked evaluation metrics, cluster-profile reconciliation, and configuration weights.

The source-like fallback SHA-256 is `583fd210513b096ddaba101fc256f52288838fc720241ccdbad28d99f910de71`. Re-running with the same dependencies, configuration, and seed reproduces the source data, mapping, enrichment, labels, and model results; measured runtimes can vary by hardware.

## Limitations and safe use

- An anomaly is a prioritization signal, not a determination of wrongdoing or payout error.
- The fallback and rep identities are synthetic. Results do not describe real employees, customers, or territories.
- Metrics against designed anomalies can be optimistic and depend on injection realism.
- Selection against those same synthetic labels is demo benchmarking, not external validation.
- Peer benchmarks depend on cohort size and the available organizational hierarchy.
- DBSCAN is deterministic for fixed data/parameters, but density results remain scale- and `eps`-sensitive; the persisted wrapper documents its approximate nearest-core out-of-sample assignment.
- Before production use, add governed source controls, temporal validation, review outcomes, drift monitoring, access controls, human escalation rules, and bias/privacy assessment.

Clustering alone must **not** determine sales-force hiring. The reusable path toward territory opportunity, workload, capacity, forecasting, optimization, scenario simulation, and geographic allocation is documented in [future_sales_force_planning.md](docs/future_sales_force_planning.md).

Further detail: [methodology.md](docs/methodology.md) · [data_dictionary.md](docs/data_dictionary.md) · [executed benchmark summary](artifacts/reports/executed_benchmark_summary.md)
