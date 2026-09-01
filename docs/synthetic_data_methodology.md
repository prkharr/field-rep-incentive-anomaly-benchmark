# Relational synthetic-data methodology

## Purpose and provenance

The commercial-review extension starts from the unchanged local `pharma-data.csv`. Observed sales identities, quantities, prices, sales values, months, representatives, managers, teams, customers, products and geography provide the relational foundation. A deterministic day within the observed month is added because the source provides only month/year.

Every generated table carries one of four lineage values:

- `source_observed`: a fact retained from the supplied CSV;
- `synthetic_derived`: a deterministic or seeded attribute derived from observed structure;
- `synthetic_normal`: a controlled simulation of reasonable commercial behavior;
- `synthetic_injected`: a separate controlled evaluation perturbation.

The source has no reliable currency field. No currency conversion is performed. Generated monetary datasets use `currency_code=UNK`, meaning “currency not supplied.” Synthetic coordinates are seeded offsets around territory centroids and are not real addresses or externally geocoded locations.

## Reproducibility and relationships

`configs/synthetic_data.yaml` is the single configuration contract. Seed 42 is the default and can be changed. Stable SHA-256 identifiers link representatives, managers, teams, territories, customers, products, transactions, orders, visits and expenses. The complete pipeline validates primary-key uniqueness, foreign keys, date chronology, payout timing and source-file fingerprint preservation.

The generator creates:

- representative, manager, team, customer, product and territory masters;
- rep-month targets and quotas;
- versioned incentive-policy tiers;
- source-anchored orders and synthetic discount detail;
- source-derived and synthetic returns/cancellations;
- physical visits, CRM interactions and travel/expenses;
- centralized incentive calculations and payouts;
- a rep-month capacity calendar and customer-coverage drilldown.

Targets combine earlier rep sales, cumulative prior-only priority-product mix, prior-only seasonality, as-of territory potential, tenure and seeded variation. Priority-product targets inherit the representative's prior eligible high-weight product share rather than an unrelated random percentage. They are not fixed multiples of current-period sales. Descriptive master potentials are frozen to the configured reference window, while scoring and target features rebuild behavior-derived potential from information available strictly before each period. Normal discounts depend on product expectations, customer segment, channel and volume. Seeded visit selection is positively related to priority, required cadence and next-period sales opportunity, and negatively related to simulated route distance. Next-period sales is used only to create this synthetic causal relationship and is never exposed to the feature store. The relationship remains deliberately imperfect. Normal visits are capped at the configured 84 per rep-month and scheduled in non-overlapping four-per-day slots; controlled overlap/impossible-travel cases remain distinct. Expenses vary with simulated route distance. Normal incentive payments are calculated only through the versioned policy module and reconcile within the explicit tolerance. Product-master eligibility/weights are seeded exogenous plan attributes, not values inferred from later product sales; the effective rule's `product_weight` is an additional plan-level multiplier. The numeric discount threshold, unapproved-discount scaling, eligibility floor, payout rate, decelerator, accelerator, bonus, clawback, cap and delay are all consumed from or described by the effective policy row. Below the documented 50% minimum attainment, all payout components are zero. Eligible observations below 100% attainment apply the tier's decelerator to the version payout rate; above-target incremental payout applies the applicable accelerator.

## Clean and injected layers

Clean tables are never modified in place. The injected layer is a deep copy. The separate `anomaly_ground_truth` table records injection identity, entity, rep, period, type/category/severity, affected dataset and record IDs, original/injected values and expected signals. Labels and injection metadata are excluded categorically from the model feature allowlist.

The benchmark covers 22 required scenarios: peer payout outliers, wrong accelerator tier, duplicated adjustments, unsupported overrides, period-end spikes, post-payout returns, threshold-crossing discounts, low-volume customer bursts, customer concentration, incentivized-product mix shifts, short visits, impossible travel/overlap, sales without support, activity without engagement, inflated distance, duplicate expenses, unusual returns, late target revisions, capacity overload, priority-customer undercoverage, territory-potential-explained performance and a correlated sales/discount/return case.

Configured commercial anomalies are concentrated in approximately 5% of rep-periods, affected order lines in approximately 1.5%, and controlled overload truth in approximately 11% of capacity rows. Multiple scenario types or intrinsically multi-signal mutations share selected rep-periods so roughly one fifth of anomalous commercial cases contain correlated evidence. Persistent undercoverage spans three consecutive calendar periods and updates both calendar and customer drilldown records. Magnitudes differ by low, medium and high severity and include boundary-near cases. Order timing changes cascade to invoice, fulfillment and linked-return chronology; all injected arithmetic is reconciled before incentive recalculation.

## Capacity methodology

The existing count-based workload method is preserved as `legacy_normalized_workload_index`:

```text
0.40 × customers / training median customers
+ 0.25 × transactions / training median transactions
+ 0.15 × cities / training median cities
+ 0.10 × products / training median products
+ 0.10 × distributors / training median distributors
```

The new transparent hours layer subtracts leave, holidays, training, administration and meetings from rostered hours. Customer obligations use the dominant owner observed in that exact period with an explicit master fallback and no future fill. Cadence-aligned visit/travel hours, observed activity, and excess-service diagnostics are attributed to the actual visited territory and representative; uncovered obligations remain with the period owner and the owned customer's territory. A dominant territory is still shown for compact rep-period reporting. A separate rep-territory-period fact carries that exact geographic core workload while allocating shared roster availability and non-workload quantities by normalized rep-period transaction shares (with workload and roster-territory fallbacks for idle periods), and enforces conservation of every additive quantity back to the rep-period calendar. The preparation/follow-up buffer follows exact core workload. A controlled injected overload that exists only in the compact calendar is assigned deterministically to the dominant territory and explicitly marked as a residual allocation. Territory-period utilization and risk are recomputed from the resulting numerators and denominators rather than from dominant-territory labels. Controlled overload truth remains rep-period truth, so territory comparison is disclosed as an allocation-sensitivity diagnostic, never independent territory-ranking agreement. Required hours combine cadence-aligned visits, planned travel, uncovered required customer calls and the workload buffer. Required and available FTE use the same nominal monthly full-time-hour denominator. Capacity risk is a review signal for workload balancing, territory redesign, resource sharing or further staffing analysis—not an automated hiring decision.

Numeric workload/utilization MAE in the controlled evaluation is an arithmetic reconciliation to the injected deterministic values, not an independent predictive-accuracy estimate. Clean-versus-injected deltas and overload ranking/classification are reported separately.

## Limitations

These synthetic systems simplify real contracts, approvals, route constraints, customer access, product launches and employment availability. Source coverage changes between Poland and Germany. The benchmark is suitable for controlled pipeline testing, not production payroll, misconduct findings or punitive employee decisions.
