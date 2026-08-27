# Methodology

## Purpose and interpretation

This repository benchmarks K-Means and DBSCAN for finding unusual patterns in pharmaceutical field-representative performance and incentive payouts. Its analytical question is whether a representative/product/territory/month behaves differently enough from comparable observations to merit review.

An anomaly is a **review signal**, not evidence of fraud, misconduct, or an incorrect payment. Data-quality problems, territory structure, product launches, one-time customer events, legitimate manual adjustments, and other operational causes can all produce high anomaly scores. A business investigator must make the final determination.

## Provenance for this run

Workspace-wide discovery inspected 135 CSV and 4 XLSX files and found no qualifying field-sales dataset. No file matched more than one of the 17 expected commercial columns. Two pharma-adjacent candidates were rejected because one contained company financials and the other drug-development pipeline records, not field sales. The benchmark therefore activates the seeded fallback transaction generator. See `workspace_data_discovery.md` for the audit summary.

This distinction matters throughout the architecture:

1. **Source-like fallback commercial fields** mimic the expected input schema but are synthetic in this run.
2. **Deterministic mapping fields** construct territories and representative/customer assignments using stable keys and the configured seed.
3. **Synthetic enrichment fields** add logically related activity, capacity, targets, and incentives without replacing commercial sales.
4. **Derived features** summarize behavior, peer position, opportunity, workload, and payout relationships.
5. **Evaluation-only labels** identify controlled injected anomalies and are excluded from preprocessing and modeling.

## End-to-end architecture

```text
Workspace discovery / explicit input
                │
                ▼
    Canonical schema + provenance
                │
                ├──► source-level quality report
                │
                ▼
Deterministic rep / territory mapping
                │
                ▼
Related synthetic activity, target,
capacity, and incentive enrichment
                │
                ▼
Rep × product × territory × month grain
                │
                ▼
Feature engineering + controlled labels
                │
                ▼
Imputation → optional clipping → scaling
                │
          ┌─────┴─────┐
          ▼           ▼
       K-Means      DBSCAN
          │           │
          └─────┬─────┘
                ▼
Clustering, ranking, stability, and
ground-truth anomaly evaluation
                │
                ▼
Weighted selection, profiles, reasons,
plots, persisted artifacts, dashboard
```

The stages are kept in separate modules so a new clustering estimator can implement the common model interface without rewriting discovery, feature engineering, evaluation, or reporting.

## 1. Discovery, schema adaptation, and data quality

Column names are never assumed by position. Known aliases are normalized to canonical snake-case names. The adapter can derive sales from quantity and price, price from sales and quantity, or quantity from sales and price when two of the three exist. It derives a monthly timestamp from date/month/year fields and records every original-to-canonical mapping, default, and derivation.

Validation requires usable customer, product, sales, date, city, and country information, at least 100 rows, and no more than 20% missing sales. Negative sales, duplicates, and partially invalid dates generate visible warnings rather than being silently erased. Before enrichment, the quality report records:

- exact shape and data types;
- null counts and percentages;
- cardinality and descriptive statistics;
- exact duplicate count;
- date range and distinct months; and
- geographic, product, manager, team, channel, and sub-channel coverage.

Generated processed data and artifacts are excluded from discovery to prevent feedback loops.

## 2. Representative mapping and analytical grain

Transaction rows are not clustered directly. Customers are assigned to synthetic representatives inside a territory/team/manager structure using stable keys and seed `42`; repeated runs with the same data and configuration reproduce the mapping. The primary analytical unit is approximately:

> field representative × product × territory/geography × calendar month

At that level the pipeline aggregates sales, units, customers, product/channel mix, activity, targets, incentives, capacity, and geographic/opportunity context. This is the closest defensible representation of representative behavior when real representative IDs are absent.

Synthetic IDs are linkage devices only. They do not represent real employees, reporting lines, ownership, or actual customer assignments.

## 3. Synthetic enrichment

The enrichment layer preserves commercial sales and adds missing field-force concepts. Random variation is conditional on existing commercial and organizational context rather than independent noise:

- more customers, larger territories, and travel burden generally imply more workload and calls;
- coverage and activity generally move with commercial performance;
- targets are related to prior/current opportunity and expected sales/quantity;
- calculated incentive generally increases with attainment;
- actual payout is normally close to calculated incentive except for adjustments; and
- capacity reflects working time, territory workload, and representative characteristics.

This layer includes representative tenure/capacity, calls and digital engagement, working days and travel, plan adherence and coverage, sales/quantity targets, target and calculated incentives, actual payout, and adjustment/override amounts. It enables a working benchmark, but it must be replaced or calibrated against CRM, HR, target-setting, and compensation records before production use.

## 4. Feature engineering

Features are grouped into:

- **commercial performance:** sales, quantity, weighted average price, temporal growth, productivity ratios;
- **customer and coverage:** customer counts, coverage, concentration, and channel/sub-channel mix;
- **product behavior:** product breadth, concentration, and dominant share;
- **activity and workload:** calls per day/customer, efficiency, travel burden, capacity, and workload;
- **incentive relationships:** payout-to-sales, payout-to-target, payout per customer/call, and actual-versus-calculated variance;
- **peer comparisons:** deviations from medians and within-peer z-scores; and
- **territory opportunity:** market potential, density, sales potential, opportunity index, and opportunity-adjusted sales.

Temporal features use prior-period information within the relevant behavioral series. Peer statistics use sensible product/territory/team/manager cohorts and zero-safe calculations. Undefined ratios and small-group statistics become missing and are handled by the preprocessing pipeline; they are not converted to infinite values. The configured model feature list is in `configs/config.yaml` and its meanings are in `data_dictionary.md`.

Identifiers, raw text categories, anomaly labels, and anomaly type are not model features.

## 5. Controlled anomaly injection

Approximately 6% of analytical rows are selected with seed-controlled sampling and variable severity. Scenarios span:

1. high incentive with weak sales;
2. high incentive with low attainment;
3. very high sales with unusually low activity;
4. extreme sales spike;
5. extreme quantity spike;
6. abnormally high calls;
7. low coverage with high incentive;
8. large manual override;
9. sales inconsistent with territory opportunity;
10. sales inconsistent with peer group;
11. unusual product mix; and
12. duplicate/suspicious activity pattern.

Injection changes relevant analytical values and then features affected by those values are recalculated as needed. `injected_anomaly_flag`, `anomaly_type`, and any stored severity field exist only to evaluate recovery. They are explicitly denied access to feature selection, preprocessing, tuning, clustering, thresholding, and explanations.

The label is artificial ground truth: it measures recovery of these designed scenarios, not detection of every real-world anomaly.

## 6. Preprocessing and leakage controls

The reusable preprocessing stage selects only configured numeric features and applies:

1. finite-value normalization (`±∞` becomes missing);
2. fit-time median missing-value imputation;
3. optional quantile clipping, configured at the 1st and 99th percentiles; and
4. either `RobustScaler` (default) or `StandardScaler`.

Fitted transforms are reused for scoring and persisted with the model artifacts. Clipping is useful for preventing a single extreme dimension from dominating Euclidean distance; it also limits sensitivity to the most extreme injected values, so the option and bounds must be reported with results.

There is no supervised label fitting. Controlled labels are joined only after model scores exist for evaluation.

## 7. K-Means benchmark

K-Means is tested for `k = 2…12` with multiple centroid initializations. Every configuration records inertia, silhouette, Davies–Bouldin, Calinski–Harabasz, cluster-size extremes and balance, runtime, and stability. Stability is the mean pairwise Adjusted Rand Index across repeated fits with different seeded initializations.

The selected `k` is **not** chosen from the elbow alone. Selection considers separation, compactness, stability, cluster balance, runtime, operational interpretability, and whether the profiles form defensible business cohorts.

For each row, K-Means uses ordinary Euclidean (L2) distance in the fitted imputed, quantile-clipped, RobustScaled feature space. If `c(i)` is the nearest assigned cluster, the distance is `d_i = sqrt(sum_j((x_ij - centroid_c(i),j)^2))`. This equals the assigned column of scikit-learn's `KMeans.transform`; the sum of `d_i^2` equals fitted inertia. The 0–1 anomaly score is the empirical percentile of `d_i` among training distances, so taking the square root instead of squared distance does not change ranking. Explanation contributions are each transformed feature's squared residual divided by the row's total squared centroid distance; they therefore sum to 1 for every non-zero-distance row. Business-unit cluster- and peer-median differences supplement those transformed-space contributions.

K-Means is useful for segmentation and ranking but assumes roughly compact, centroid-representable structure. A distant point can be unusual without belonging to a small cluster, while a small cluster can be legitimate.

## 8. DBSCAN benchmark

DBSCAN tests combinations of `eps` and `min_samples`. Candidate `eps` values are informed by k-nearest-neighbor distance quantiles; the saved k-distance curve is a diagnostic, not an automatic proof of the optimum.

Each configuration records cluster count excluding noise, noise percentage, valid separation metrics, cluster-size distribution and balance, runtime, and stability/sensitivity. Configurations are rejected when they produce one undifferentiated cluster, all noise, no valid separation, operationally excessive noise (the default maximum is 55%), or a smallest cluster below the configured floor. DBSCAN label `-1` is a primary review candidate. The continuous score ranks the `min_samples`-neighbor distance empirically and reserves the upper score band (0.75–1.00) for noise while retaining density ranking within noise and non-noise populations.

Scikit-learn DBSCAN has no native out-of-sample `predict`. For interface compatibility, this implementation assigns a new row to the nearest fitted core sample only when that sample lies within `eps`; otherwise it returns `-1`. That approximation is disclosed in the persisted model parameters and is another reason to re-fit/revalidate DBSCAN after material data drift.

DBSCAN is deterministic for fixed ordered input, preprocessing, and parameters. That does not mean it is stable: small perturbations, sampling changes, or a slightly different `eps` can change density connectivity. The reported stability is the mean Adjusted Rand Index between the fixed-data solution and seeded Gaussian perturbations of the scaled matrix (default standard deviation 0.02); it is not repeated-random-initialization stability.

## 9. Evaluation

### Clustering quality

Both methods are assessed with number of clusters, noise percentage, silhouette, Davies–Bouldin, Calinski–Harabasz, smallest/largest cluster percentage, cluster balance, runtime, and an Adjusted-Rand-Index-based stability measure where applicable. Metrics that are mathematically undefined—for example silhouette with fewer than two non-noise clusters—remain unavailable rather than being replaced with favorable values.

### Anomaly recovery

After scoring, the controlled labels support precision, recall, F1, F2, specificity, balanced accuracy, ROC-AUC, PR-AUC, and the full confusion matrix. Ranking is evaluated at 1%, 5%, and 10% with precision, recall, and lift, plus anomaly capture in the top decile.

Ranking metrics are especially relevant because review capacity is limited. A high overall classification metric does not guarantee that the first investigation queue is useful. PR-AUC and lift should be emphasized under the intentionally imbalanced label rate; ROC-AUC remains contextual.

Threshold-dependent metrics use the configured contamination/review policy. Comparing methods is valid only when the review budget and score direction are aligned.

## 10. Model selection and profiling

Selection produces two conclusions:

- **best segmentation model**, weighted toward cluster quality, balance, stability, interpretability, and usefulness; and
- **best anomaly-detection model**, weighted toward recall, precision, F2, PR-AUC, lift at 5%, stability, runtime, interpretability, and usefulness.

Weights are explicit in `configs/config.yaml`. Metrics are normalized in the appropriate direction (for example, lower Davies–Bouldin and runtime are better), and invalid configurations are excluded before ranking. This prevents a single silhouette number—or any one metric—from deciding both use cases.

Cluster profiles are derived from observed cluster statistics, never hardcoded labels. Profiles report population, sales, quantity, customers, activity, attainment, incentive, opportunity, anomaly rate, dominant product/geography, and differences from population baselines. Business interpretations are descriptive summaries, not causal claims.

## 11. Explainability and investigation output

K-Means reasons combine centroid distance contribution, standardized deviations, cluster median differences, and peer median differences. DBSCAN reasons combine noise/local-density evidence, neighbor distance, unusual standardized features, and peer differences. The ranked investigation output carries the representative, territory, product, manager, team, commercial values, score, selected method, and top drivers.

No SHAP explanation is claimed. SHAP would require a clearly labeled supervised model or surrogate, which is outside this benchmark.

## Reproducibility

- Default seed: `42`.
- Configuration is centralized in `configs/config.yaml`.
- Discovery and schema decisions are persisted in provenance.
- Preprocessing and model objects are persisted.
- Full tuning results are written in machine-readable long format.
- Summary metrics are populated from execution outputs; no metric should be hand-entered.
- Re-running with the same software versions, input, row order, and configuration should reproduce deterministic mapping and closely reproduce numerical results.

Minor floating-point or parallel-runtime differences can occur across platforms and dependency versions.

## Limitations

1. **No original field-sales data was available.** Commercial transactions, representative mappings, enrichments, and labels in the executed fallback run are synthetic.
2. **Injected-label metrics are scenario-specific.** They can overstate generalization to novel, subtle, or business-legitimate anomalies.
3. **No independent real-world holdout exists.** This is a technical benchmark, not a production validation study.
4. **Synthetic representatives are not people.** Team, manager, territory, tenure, capacity, and customer ownership do not describe actual employees.
5. **Aggregation can hide detail.** Monthly rep/product/territory aggregation may mask within-month duplicate, timing, or transaction-level issues.
6. **Peer comparisons inherit peer definitions.** Sparse or heterogeneous cohorts can make medians and z-scores unstable.
7. **Distance is representation-dependent.** Scaling, clipping, correlated features, and feature selection affect both methods and their explanations.
8. **K-Means has geometric assumptions.** Non-spherical structure and varying cluster density can be misrepresented.
9. **DBSCAN has global-density limitations.** A single `eps` can struggle with territories of very different density, and noise rate can move abruptly.
10. **Payout rules are simplified.** Real compensation plans contain eligibility, caps, accelerators, gates, splits, exceptions, and approval workflows not modeled here.
11. **Opportunity and workload are proxies.** External market potential, HCP/account universe, call-frequency policy, travel network, vacancies, and product eligibility are absent.
12. **Scores are not causal or adjudicative.** They do not prove why an observation occurred and must not automatically trigger disciplinary or payment action.
13. **Production controls are not complete.** Deployment would require access controls, privacy review, monitoring, drift checks, investigation feedback, audit retention, and jurisdiction-specific compliance review.
