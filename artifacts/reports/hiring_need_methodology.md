# Field-force capacity scenario methodology

Independent team × country × product-class × month planning, using real commercial measures. Backtest seasonal naive (lag-12, falling back to prior-three mean when unavailable), three-month mean, and alpha=.3 exponential smoothing. Select each metric's method by VALIDATION WAPE; report untouched TEST MAE/RMSE/WAPE/sMAPE/bias. Final forecast is one month beyond the actual data, May 2019—not a current 2026 hiring forecast.

Country coverage is incomplete: Poland stops in December 2018. Units lacking latest-month observations are marked ineligible with missing FTE/priority outputs, not assumed to have zero staffing. Only Germany supports May 2019 capacity scenarios. Test forecast errors likewise cover Germany only.

Normalize workload components using TRAIN per-rep-unit median loads; weights are configurable. Rep-unit loads sum to rep-month workload. Sustainable capacity is the training stable-period 60th percentile (absolute month-on-month workload growth <=50%), not the historical maximum. Missing actual working days, calls and vacancies make this a workload proxy.

Reps serve multiple units. Allocate each active rep's ONE FTE across their latest month's units in proportion to observed workload; allocations reconcile to the actual rep count. Distinct headcounts are shown but are NOT summed as available capacity. Required FTE=forecast workload/sustainable capacity. Gap=required FTE−allocated FTE. Capacity utilization=required FTE/allocated FTE.

The base case assumes all observed reps' capacity is available to the observed Germany scope. Missing Poland records do not establish that cross-country assignments ceased. This assumption can overstate available capacity; validate actual time/territory allocations before interpreting spare capacity.

Scenario bounds combine validation 80th-percentile absolute forecast error with training capacity 40th/80th quantiles; these are NOT confidence intervals. Hiring priority combines growth, utilization, gap, customer/geographic/product pressure; high forecast uncertainty discounts the score. Never use anomaly scores as hiring decisions. Forecast customer breadth and sales reflect observed demand, not total addressable opportunity.

Raw-vs-cleaned sensitivity clips workload to unit TRAIN 5th/95th percentiles with capacity held fixed; this can suppress real structural growth and is a sensitivity check, not truth. Scenarios are independent unit what-ifs. Reallocation explicitly debits a donor and credits a receiver by the same bounded fractional FTE, so headcount is conserved. Validate travel feasibility and staffing constraints before considering any action.
