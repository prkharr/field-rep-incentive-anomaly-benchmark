# Model card — finalized PCA commercial-review ranker

## Intended use

The model ranks representative-month observations for human commercial review. It answers “what should a manager validate first?” It does not estimate fraud probability, prove misconduct, determine compensation, or make employment decisions. The independent deterministic capacity module answers a separate workload question.

## Architecture

The repository’s finalized architecture remains **PCA Reconstruction**. The expanded version refits the same standalone reconstruction detector on the clean rep-month feature store. It is not PCA followed by another anomaly detector. Existing K-Means/DBSCAN/Isolation Forest/Autoencoder/ensemble artifacts remain legacy or comparison evidence and are not silently substituted.

Preprocessing is fitted only on clean chronological training rows:

1. median imputation;
2. signed `log1p` tail compression;
3. robust scaling;
4. PCA retaining 95% cumulative variance.

Raw anomaly score is mean per-feature squared reconstruction error. The 0–1 manager score is a clean-training empirical percentile with monotone tails. Per-feature squared reconstruction errors provide non-causal contribution evidence. SHAP is not claimed or used.

## Data and split

Scoring grain is representative × month. Training ends June 2018, validation ends December 2018, and January–April 2019 is the final test. PCA and preprocessing fit only clean training data. The raw operational threshold is the unlabeled validation score at the configured 5% manager review capacity. That one frozen raw threshold determines manager-queue membership in every period; the exact top-1/5/10% test selections exist only as explicitly named ranking metrics. Final test scores or labels never select or tune the operational threshold.

Controlled ground truth is kept separately. Commercial labels support anomaly evaluation; capacity-overload truth supports the independent capacity evaluation. Capacity-mutated calendar rows are replaced with the clean calendar before the commercial PCA is scored, so capacity cases are neither mislabeled commercial negatives nor label proxies. No ground-truth, injection ID, anomaly type, severity or correlated-case flag enters model features.

## Feature families

The allowlist covers sales/history, target and incentive reconciliation, customer concentration/mix, product mix, orders/timing, discounts, returns, visits, CRM, travel/expenses, commercially similar peers, territory/tenure/product/channel adjustments and clean capacity/calendar signals. Earlier-history features shift before rolling. Behavior-derived customer/territory potential and product-price baselines are recomputed from cumulative prior periods at each scoring date; descriptive all-history master values do not enter those early-period features. Explicitly named post-period return features are available only for post-payout review.

## Evaluation

The executed artifact reports ROC-AUC, average precision, precision/recall/F1, confusion counts, precision/recall/lift at 1%, 5% and 10%, detection by type/severity/organization, period stability, false-positive drivers and clean-versus-injected distributions. Every configured type appears in the group-metric table; types without final-holdout support are explicitly `no_final_test_support`/N-A instead of being reported as zero. The two capacity types are evaluated by their separate deterministic rules. Results measure recovery of controlled synthetic cases, not production misconduct detection. Small held-out support makes type-level estimates uncertain.

## Explanations

Each queue row receives a deterministic primary reason code, manager sentence, secondary reason, three drivers with actual/peer values and percentiles when available, and a recommended validation action. Contributions identify reconstruction mismatch; they are not causal statements.

## Responsible use

- Use “review candidate,” “unusual observation,” “incentive anomaly,” or “behavior requiring validation,” never “fraudulent representative.”
- Validate governed orders, policies, adjustments, returns, CRM and expense records with the relevant manager and representative.
- Do not use injected labels as employee labels.
- Do not take punitive action without independent investigation.
- Do not use capacity scores as automated hiring, termination, performance-rating or territory-assignment decisions.

## Known limitations

All new incentives, quotas, discounts, visits, CRM, expenses, capacity inputs and evaluation labels are synthetic or derived. Currency is unknown. Geographic coordinates are synthetic. Customer potential, travel and workload assumptions require stakeholder validation. Historical market coverage changes can create unusual signals unrelated to representative behavior.
