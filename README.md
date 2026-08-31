# Pharma Commercial Review & Field-Force Capacity

Two connected business questions, answered by separate analytical modules:

1. **Which representative performance / incentive observations should be reviewed first, and why?**
2. **Where might commercial workload exceed available field capacity?**

This is an executed hackathon benchmark—not a production fraud detector or an automated hiring system. Real commercial records form the foundation. All incentive amounts are explicitly **simulated**, and all detection labels are **controlled injections**.

## Dataset and scope

The user-provided `pharma-data.csv` contains 167,760 rows and 18 columns: 13 actual sales representatives, 4 managers, 4 teams, 751 customers, 241 products and 6 product classes. Three exact duplicates and one incomplete row are excluded from the modeling copy, leaving 167,756 transactions. The original CSV is unchanged and git-ignored because its redistribution license has not been verified.

**Actual coverage is January 2017–April 2019. Poland ends in December 2018; 2019 contains Germany only.** Missing country records are not evidence of zero demand, zero staffing or poor employee performance. The coverage break is a material limitation for historical anomaly comparisons. Poland units are ineligible for May 2019 FTE-gap recommendations.

The primary grain is **Rep × Product Class × Month**: 2,184 observations, median 72 transactions per observation, and no single-transaction cells. The country-specific candidate has 3,120 observations (median 53 transactions); rep-month has 364. The primary grain retains product-class interpretation while limiting dimensions for only 13 reps. Market and planning views retain country so reviewers can inspect the coverage break.

## Three-layer architecture

- **Commercial behavior:** real identities, transaction aggregation, customer coverage, product/distributor/channel mix, geography, peers and prior-only history.
- **DEMO incentive review:** configurable compensation arithmetic, separate controlled benchmark copies, common model scores and explainable investigation queues.
- **Capacity scenarios:** demand forecasts, workload proxies, historical capacity, fractional rep allocation and FTE scenarios. Anomaly scores never decide hiring.

The extension reuses the original K-Means/DBSCAN model classes, preprocessing interface, evaluation functions and dashboard shell. `run_pipeline.py` remains the legacy synthetic demonstration; **use `run_extended_benchmark.py` for the real-data solution**. Legacy results must not be compared with the new table. See [legacy README](docs/legacy_benchmark_readme.md).

## Features and incentive demonstration

The executed model uses 124 explicitly allowlisted numeric features: sales/quantity/price, transaction value, unique/new/repeat customers, customer loss and concentration, product mix/breadth, geographic dispersion, country/channel/distributor mix, robust peer deviations, ranks and temporal history. Calendar lags 1/2/3/6/12, rolling statistics, MAD, growth, acceleration and personal-history deviations use earlier months only.

Same-month peer comparisons are retrospective after month close. Small cohorts fall back from team/country/class to country/class, class, then month; they do not borrow future observations. Counts are computed from source transactions, not summed across overlapping groups.

DEMO target = prior-three-month median sales × configurable growth (fixed cold-start assumption when history is absent). Expected incentive = base + capped attainment component + above-target accelerator. Actual DEMO payout = expected incentive + configurable adjustment. The arithmetic expressions and constants live in [config.yaml](configs/config.yaml) and use a restricted arithmetic interpreter, not arbitrary code execution. Every incentive field starts with `simulated_`.

## Fair validation and distance

| Partition | Months | Analytical rows |
|---|---|---:|
| Train | Jan 2017–Jun 2018 | 1,404 |
| Validation | Jul–Dec 2018 | 468 |
| Test | Jan–Apr 2019 | 312 |

Preprocessing is fitted on **clean training data only**: median imputation → signed-log tail compression → RobustScaler. No test fitting and no clipping of held-out extremes. Synthetic labels never enter the feature allowlist.

**K-Means uses ordinary Euclidean L2 distance to the assigned nearest centroid in this transformed space.** Executed checks reconcile the manual norm, sklearn distances and training inertia. The metric is not cosine distance and is not computed on unscaled raw commercial values.

The clean commercial/DEMO dataset is never injected. Primary benchmark copies contain approximately 6% injected validation/test anomalies, with no training injections. The injection audit records type, severity, feature, original/injected values, date, representative and seed. Aggregate perturbations include payout, adjustment, commercial, coverage/mix and temporal patterns; sustained events use consecutive months. Three independent 1.5%-prevalence runs test sensitivity without retraining or retuning. Some types have no test support: they are explicitly unavailable, not perfect detections.

Model selection uses validation only. Clustering grids use clean validation quality and balance constraints. Other grids use validation ranking utility; final selection adds seed queue stability. Final test labels are evaluated only after model/ensemble choices are frozen.

## Executed final benchmark

All rows below use the **same 312 held-out observations**. Precision, recall, F1/F2 use a fixed 5% review queue (16 rows), not an optimized test threshold. The full table also contains ROC-AUC, Precision/Recall/Lift at 1%, 5%, 10%, model sizes and interpretation.

<!-- EXTENDED_RESULTS_START -->

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

<!-- EXTENDED_RESULTS_END -->

[Complete benchmark CSV](artifacts/metrics/final_anomaly_model_benchmark.csv) · [Validation results](artifacts/metrics/validation_model_benchmark.csv) · [Executed report](artifacts/reports/executed_extended_benchmark_summary.md)

Runtimes include bounded parameter search for fitted models; the ensemble includes its component costs. Stability is validation top-5% queue overlap across seed refits. A deterministic method's score of 1 does not establish robustness to data perturbation.

### What the benchmark recommends

- **Primary anomaly ranking: PCA Reconstruction**, selected on validation. Use peer/history explanations and the transparent business-rule signals alongside it.
- **Segmentation: K-Means**, with four fitted clusters. Some fitted clusters may have no test members. DBSCAN is a diagnostic density comparator; its high held-out noise rate limits operational segmentation.
- **Temporal specialist: EWMA residual**, selected on validation; keep rolling, seasonal and sustained-shift signals available separately.
- **Autoencoder versus PCA:** compare both validation and test tables. PCA is simpler, faster and seed-stable. The Autoencoder's bounded training reached its iteration limit; convergence/loss information is saved rather than hidden.
- **Ensemble:** equal percentile/rank average, consensus, maximum complementary signal and bounded weighted variants were tested after correlation/overlap analysis. The best candidate did **not** achieve the predefined material improvement over PCA. Do not deploy an ensemble merely because it is more complex.
- **Interpretability:** business rules and robust peer baselines expose recognizable commercial reasons; K-Means and reconstruction methods expose feature contributions. Isolation Forest includes bounded training-median feature ablation for the top 20 review rows.

Clean DEMO payouts follow an exact formula. Reconstruction performance partly reflects detecting violations of that artificial relationship, not proven ability to identify real compensation issues. Only 19 positive test labels make detailed type-level estimates fragile.

## Investigation workflow

1. Open the clean-data queue for actual unsupervised review priorities; use the benchmark queue only to study injected-label performance.
2. Inspect commercial source coverage, actual versus expected history, peer comparisons, DEMO payout deltas, model agreement and feature drivers.
3. Validate against authoritative compensation and commercial records before drawing conclusions.
4. Export review status/comments for a human review process. The dashboard is read-only; it is not a case-management database.

Every model exposes raw score, normalized score/percentile, raw TRAIN-reference threshold, threshold exceedance and separate exact-budget review flag. Higher is more anomalous. Calibration uses TRAIN empirical ranks with bounded monotone tails to preserve extreme ordering; these are not fraud probabilities.

## Temporal methods

Rolling residual uses prior median/MAD; EWMA uses a prior exponentially weighted expectation; seasonal residual compares the exact same month a year earlier; the change detector combines one-sided CUSUM and consecutive same-direction evidence. A separate trend signal compares prior subwindows.

Observed/expected values, residuals, history length, direction, availability and review flags are exported per observation and metric. Earlier test observations can update later expectations in rolling-origin scoring; future observations never enter them. Monthly data cannot establish intra-month timing: quarter-end spikes are only a proxy. See [temporal methodology](artifacts/reports/time_series_methodology.md).

## Capacity and hiring scenarios

Planning grain: **Team × Country × Product Class × Month**.

- Backtest seasonal naive, three-month average and exponential smoothing; select by validation WAPE and report test MAE/RMSE/WAPE/sMAPE/bias.
- Build configurable normalized workload from customer, transaction, city, product and distributor loads.
- Estimate sustainable capacity from the 60th percentile of stable training rep-month workloads, not the historical maximum.
- Allocate each active rep's **single FTE** across served units in proportion to latest workload. Allocations reconcile to 13 reps; overlapping headcounts are never summed as independent capacity.
- Required FTE = forecast workload ÷ sustainable per-rep capacity. Gap = required FTE − allocated FTE.
- Report cautious priority categories and scenario bounds, not hiring instructions. Bounds reflect forecast errors and historical capacity quantiles; they are not statistical confidence intervals.
- Compare raw versus training-bound-winsorized workload estimates. Cleaning can suppress real growth, so this is sensitivity analysis, not a corrected truth.

The forecast horizon is **May 2019**, not today. Germany supports 24 current unit scenarios; 24 Poland units are retained as ineligible due to stale data. Scenarios include demand +10%/+20%, add 1/2 FTE, capacity −10%, product-launch demand and an explicit net-zero donor/receiver reallocation.

The base case assumes all 13 observed reps' capacity is available to the observed Germany scope. Missing Poland records do not verify that assumption: real cross-country assignments may consume part of their time. Validate available FTE before interpreting the modeled spare capacity.

[Capacity results](artifacts/planning/hiring_need_by_business_unit.csv) · [Scenarios](artifacts/planning/hiring_scenarios.csv) · [Forecast backtests](artifacts/planning/forecast_metrics.csv) · [Methodology](artifacts/reports/hiring_need_methodology.md)

## Dashboard

The existing Streamlit application now offers a separate **Real commercial extension** workspace:

Executive overview · Model benchmark · Anomaly investigation · Time-series view · Field-force planning and interactive scenarios · Governance/limitations.

The original synthetic dashboard remains accessible with a prominent legacy warning.

## Reproduce

From the repository root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
# Place your authorized source CSV at data/raw/pharma-data.csv, or pass --input.
python run_extended_benchmark.py --input data/raw/pharma-data.csv --config configs/config.yaml
python -m pytest -q --junitxml=artifacts/reports/extended_tests.xml
streamlit run app.py
```

To extract the supplied ZIP once: `Expand-Archive -LiteralPath 'C:\path\archive (1).zip' -DestinationPath data/raw`. Do not overwrite another dataset unintentionally. Raw source data is not bundled in git. The [run metadata](artifacts/reports/extended_run_metadata.json) records its SHA-256, dimensions, exact periods, feature names, parameters, seed, versions, warnings and runtime. The pipeline refreshes the marked results table above from executed metrics.

## Important outputs

| Location | Contents |
|---|---|
| `data/processed/analytical_dataset.csv` | Clean real commercial + DEMO incentive/features |
| `data/processed/controlled_benchmark_dataset.csv` | Separate injected evaluation copy |
| `data/processed/*scores_long.csv` | Common score contract for both populations |
| `data/processed/scored_observations_all_models.csv` | Wide controlled benchmark scores and drivers |
| `data/processed/*time_series_scores.csv` | Prior-only temporal explanations |
| `artifacts/metrics/` | Final/validation metrics, type/severity recall, sensitivity, correlations, overlap and selection contributions |
| `artifacts/reports/*investigation_queue.csv` | Clean and controlled review queues |
| `artifacts/reports/*all_feature_errors.npz` | Complete aligned per-feature contributions/errors, not only top drivers |
| `artifacts/planning/` | Forecasts, capacity assumptions, eligible/ineligible units, FTE allocation and scenarios |
| `artifacts/models/extended/` | Fitted models, preprocessor, calibrators, formula/scoring manifest and training information |

New modules live in `src/field_rep_anomaly/`: `commercial.py`, `controlled_benchmark.py`, `temporal.py`, `extended_scoring.py`, `extended_pipeline.py`, `extended_dashboard.py`, advanced model wrappers and `planning/capacity.py`.

Tests preserve all original cases and add schema/date/grain checks, label/future leakage protection, training-only preprocessing, score direction/calibration, model reconstruction/persistence, temporal/peer logic, formula safety, ranking/ensemble arithmetic, staffing allocation/scenarios, stale-country protection and every dashboard section.

## Production gaps and next step

For incentive review: actual payouts, targets/quotas, compensation rules, adjustment approvals, calls/activity, territory assignments, working days/leave, approved exceptions and adjudicated review outcomes.

For capacity: customer/HCP potential, required call frequency, territory boundaries, travel times, vacancies, tenure, working capacity, hiring cost, ramp-up time, launches and access restrictions.

Next: validate the coverage discontinuity and workload assumptions with a commercial stakeholder, then replace DEMO incentives with governed actual records and evaluate against independently reviewed cases. [Production data gaps](artifacts/reports/production_data_gaps.md)
