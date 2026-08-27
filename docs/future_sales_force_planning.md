# Future sales-force planning

## Decision boundary

**Clustering alone should NOT be used to decide hiring requirements.**

Clusters can describe similar territory/product workload and performance patterns. They cannot, by themselves, estimate future demand, translate demand into required work, measure available capacity, respect assignment constraints, or quantify uncertainty. A cluster must never be interpreted as “hire” or “do not hire.”

This repository does not build a hiring model. It preserves a reusable analytical foundation for a later, separately validated planning system that combines:

- clustering for descriptive segmentation;
- forecasting for future product/territory demand;
- capacity modeling for work that current representatives can deliver;
- constrained optimization for headcount and geographic allocation;
- scenario simulation for uncertain launches, attrition, budgets, and productivity; and
- geographic analysis for customer density, boundaries, and travel time.

## Target decision flow

```text
Region × Product
      │
      ▼
Market and customer/HCP opportunity
      │
      ▼
Product demand forecast + service policy
      │
      ▼
Required workload (calls, time, travel)
      │
      ▼
Available capacity of eligible current reps
      │
      ▼
Coverage gap with uncertainty
      │
      ▼
Constrained headcount and territory optimization
      │
      ▼
Scenario-specific hiring recommendation
      │
      ▼
Human, finance, HR, legal, and compliance review
```

The recommendation should be product-specific and time-bounded, state assumptions, provide a range rather than false precision, and distinguish permanent hiring from territory redesign, vacancy fill, contractor capacity, digital coverage, or changes to service policy.

## How the current architecture can be reused

### 1. Common canonical schema and lineage

The discovery and schema-adaptation layer already standardizes geography, product, customer, sales, manager, and team fields while preserving source mappings. Future source adapters can add CRM calls, roster/capacity, market opportunity, HCP/account segmentation, vacancies, costs, and territory boundaries. The lineage separation in `data_dictionary.md` is essential: actual planning must not confuse synthetic proxies with operational facts.

### 2. Defensible planning grain

The current representative × product × territory × month grain is a useful bridge between customer activity and headcount planning. A mature system would maintain at least three linked grains:

- customer/HCP × product × month for opportunity and service requirements;
- representative × product × territory × month for productivity and available capacity; and
- territory/region × product × planning period for demand, gaps, and decisions.

Aggregations must be additive or explicitly weighted so plans reconcile from account through national level.

### 3. Reusable features

The repository preserves geography, territory, product, customer count/density, sales, growth, opportunity, workload, capacity, and rep productivity. Activity efficiency, coverage, travel burden, product mix, team/manager structure, and peer context can become diagnostics or covariates.

Anomaly flags should generally be used as data-quality/review indicators—not as negative productivity labels. Suspected anomalies must be resolved or sensitivity-tested before they influence capacity estimates.

### 4. Modular evaluation and reporting

The common model interface, configuration, artifact layout, profiling, and dashboard patterns can be extended to forecasting and optimization components. Planning outputs should have separate validation metrics and model cards; clustering metrics such as silhouette are not hiring-model validation metrics.

## Required modeling workflow

### Territory segmentation

Use clustering to identify descriptive cohorts such as dense mature metros, dispersed growth territories, specialty-product territories, or high-digital-engagement regions. Candidate features include customer density, product mix, opportunity, service intensity, travel burden, and historical productivity.

Segmentation can support differentiated assumptions—for example, call duration or travel efficiency—but each assumption must be estimated from real data. Stability, geographic coherence, interpretability, and temporal drift matter more than a visually pleasing partition. Territory boundaries should not be redrawn from clusters without constraint and feasibility analysis.

### Product/territory forecasting

Forecast demand or opportunity at region/territory × product × month/quarter. Depending on data availability, targets may include sales, prescriptions, eligible patients, new-to-brand starts, priority-account opportunity, or required contacts.

Useful signals may include historical trend and seasonality, product lifecycle, launch timing, indication changes, market access/formulary status, competitor events, epidemiology, account potential, channel shift, macro factors, and planned marketing. Use rolling-origin backtests and report forecast bias plus MAE/WAPE or a target-appropriate metric. Produce quantiles or prediction intervals so headcount is evaluated under uncertainty rather than from a single point forecast.

Avoid using incentive payout as a demand driver unless a causal/operational rationale and leakage-safe timing are established.

### Workload modeling

Translate opportunity and service policy into work:

```text
required customer calls
× average contact preparation/contact/follow-up time
+ travel time
+ territory administration
= required workload hours
```

Service policy should specify account segment, product eligibility, channel, desired call frequency, minimum coverage, and acceptable delay. Geographic routing or travel-time matrices should replace straight-line distance where practical. Digital engagements need their own time and effectiveness assumptions rather than being treated as free capacity.

Workload models should distinguish demand-driven contact need from historical calls: historical under-coverage must not become the new “normal.”

### Capacity estimation

Estimate current capacity from rostered, eligible FTE and time availability:

```text
gross workdays
− leave, training, meetings, administration, and expected vacancy
= field-available days
× sustainable productive hours/day
= gross contact capacity
− geography/travel burden
= usable territory/product capacity
```

Calibrate call time, travel time, ramp-up, tenure, product expertise, part-time status, leave, vacancy, manager span, compliance limits, and non-selling duties from operational records. Use representative-level estimates only when privacy and fairness review permits; otherwise use robust cohort estimates. Do not set “capacity” to the observed output of overworked representatives.

### Coverage-gap and headcount analysis

For territory `t`, product `p`, and period `h`:

```text
coverage_gap[t,p,h] = required_workload[t,p,h] - usable_capacity[t,p,h]
```

Positive gaps may suggest incremental capacity; negative gaps may suggest redeployment potential. A simple first diagnostic is:

```text
unconstrained_incremental_FTE = max(0, coverage_gap / sustainable_capacity_per_FTE)
```

That value is not yet a hiring recommendation. It must be passed through assignment constraints, minimum viable territory size, vacancy and attrition plans, budget, ramp time, alternative channels, and forecast uncertainty. Report low/base/high FTE ranges and the probability of missing service levels.

### Constrained optimization

A mature allocation model can use mixed-integer or network optimization to minimize a weighted combination of uncovered priority opportunity, travel, cost, territory disruption, and imbalance. Decision variables can represent hires by location/product, representative-to-territory assignments, shared territories, or digital/contract coverage.

Constraints may include:

- product certification and specialty eligibility;
- maximum sustainable workload and travel;
- contiguous or practical geographic assignments;
- required customer coverage and call frequency;
- manager span of control;
- territory stability/change limits;
- launch dates and hiring/ramp lead times;
- budget, approved positions, and minimum/maximum team size;
- labor, works-council, privacy, and pharmaceutical compliance rules; and
- fairness checks so protected characteristics or close proxies do not drive opportunity allocation.

Optimization should expose infeasibility and unmet demand; it must not silently relax material constraints.

### Scenario simulation

The same model should be run across explicit scenarios:

- base, downside, and upside demand;
- launch delay or indication expansion;
- high/low competitor impact;
- vacancy and attrition shocks;
- different hiring dates and ramp curves;
- changed call-frequency or coverage policy;
- productivity improvement or deterioration;
- budget caps;
- digital versus in-person channel substitution; and
- territory-boundary or manager-span changes.

For each scenario, report demand, workload, capacity, uncovered opportunity, recommended FTE, cost, expected service level, travel, disruption, and confidence/risk. A robust plan performs acceptably across plausible scenarios rather than optimizing one forecast exactly.

### Geographic allocation

Geography should progress from the current latitude/longitude and customer-density proxies to validated account geocodes, territory polygons, road-network travel times, rep home/base locations where legally permitted, accessibility, and cross-border constraints. Customer/HCP-level outputs should be aggregated or access-controlled to protect privacy.

Geographic visualization supports human review, but map appearance is not evidence of operational feasibility. Route and boundary solutions require local field validation.

## Additional data required before planning

| Domain | Minimum production-grade additions |
|---|---|
| Customer/HCP opportunity | Validated account/HCP universe, specialty/segment, potential, eligibility/consent, product relevance, access restrictions |
| Commercial demand | Prescriptions/units/sales with trustworthy dates, lifecycle and launch assumptions, market access, competitor and epidemiology signals |
| Activity/service | CRM calls and outcomes, channel, duration, preparation/follow-up, call-frequency policy, coverage targets |
| Geography | Valid geocodes, territory polygons, road travel-time matrix, accessibility and boundary constraints |
| Workforce | Real roster and FTE, product eligibility, vacancies, leave, tenure/ramp, working pattern, base location where permitted, manager structure |
| Economics | Salary/benefit/vehicle/travel costs, contractor and digital alternatives, budget and approved-headcount constraints |
| Governance | Data owners, retention/access rules, privacy impact assessment, labor/legal rules, approval workflow and decision audit trail |

Synthetic enrichment should be removed or clearly quarantined when these operational sources arrive. It is useful for software testing, not workforce decisions.

## Validation and governance gates

A planning system should not be released solely because it runs. Minimum gates include:

1. **Data reconciliation:** customer, sales, activity, and roster totals reconcile to authoritative systems.
2. **Forecast backtesting:** rolling-origin performance, bias by product/region, and interval coverage are acceptable.
3. **Capacity calibration:** predicted sustainable calls/hours match time studies and field-manager review without encoding overwork.
4. **Policy validation:** service-level and call-frequency assumptions are approved by commercial, medical, legal, and compliance stakeholders.
5. **Optimization verification:** constraints, infeasibility behavior, and cost calculations pass unit and scenario tests.
6. **Fairness and privacy review:** protected attributes and problematic proxies are excluded or governed; access is role-based.
7. **Human review:** local managers can challenge data and constraints, but overrides require a reason code and audit trail.
8. **Monitoring:** forecast drift, workload error, vacancy, coverage, plan stability, and realized outcomes are tracked after each planning cycle.

## Suggested future modules

The existing package could later add, without changing the current benchmark's purpose:

```text
src/field_rep_anomaly/planning/
├── opportunity.py
├── forecasting.py
├── service_policy.py
├── workload.py
├── capacity.py
├── gap_analysis.py
├── optimization.py
├── scenarios.py
├── geography.py
└── planning_evaluation.py
```

These components should consume versioned inputs and write separate planning artifacts. They should not overwrite anomaly-benchmark scores, labels, or metrics.

## Appropriate final recommendation format

A mature output should read like:

> Under the base-demand and approved-service scenario for Product X in Region Y over the next two quarters, forecast required workload exceeds current eligible capacity by 1.4–2.2 sustainable FTE after expected vacancy and ramp. The constrained plan recommends two hires near locations A/B, or one hire plus expanded digital coverage, subject to budget, compliance, and local feasibility review.

It should never read:

> This territory belongs to Cluster 4, therefore hire two representatives.
