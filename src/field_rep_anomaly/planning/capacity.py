"""Interpretable one-month forecasts and non-double-counted FTE allocation.

Observed reps span country/class cells. Allocate their single FTE proportionally to
their latest observed workload; do not count a full rep in every planning cell.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..commercial import aggregate, PLAN_GRAIN

UNIT = PLAN_GRAIN[:-1]
METHODS = ['seasonal_naive', 'moving_average', 'exponential_smoothing']


def forecast(history, method, next_date=None):
    s = history.dropna()
    if s.empty:
        return 0.0
    if method == 'seasonal_naive' and next_date is not None:
        prior_year = next_date - pd.DateOffset(years=1)
        if prior_year in s.index:
            return max(0.0, float(s.loc[prior_year]))
    if method == 'exponential_smoothing':
        return max(0.0, float(s.ewm(alpha=.3, adjust=False).mean().iloc[-1]))
    return max(0.0, float(s.tail(3).mean()))


def forecast_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {'MAE': float(np.mean(abs(p-y))), 'RMSE': float(np.sqrt(np.mean((p-y)**2))),
            'WAPE': float(np.sum(abs(p-y)) / max(np.sum(abs(y)), 1)),
            'sMAPE': float(np.mean(2*abs(p-y)/np.maximum(abs(p)+abs(y), 1))),
            'bias': float(np.mean(p-y))}


def required_fte(workload, per_rep_capacity, allocated_fte):
    if per_rep_capacity <= 0:
        raise ValueError('Capacity must be positive')
    need = workload / per_rep_capacity
    return {'required_fte': need, 'fte_gap': need - allocated_fte,
            'additional_fte_need': max(0, need - allocated_fte),
            'current_capacity': allocated_fte * per_rep_capacity,
            'utilization': need / allocated_fte if allocated_fte > 0 else (np.inf if need > 0 else 0.0)}


def scenario(workload, capacity_per_rep, fte, demand_change=0.0, add_reps=0.0, capacity_change=0.0):
    before = required_fte(workload, capacity_per_rep, fte)
    after = required_fte(workload*(1+demand_change), capacity_per_rep*(1+capacity_change), fte+add_reps)
    return {'workload_before': workload, 'capacity_before': before['current_capacity'],
            'utilization_before': before['utilization'], 'fte_gap_before': before['fte_gap'],
            'workload_after': workload*(1+demand_change), 'capacity_after': after['current_capacity'],
            'utilization_after': after['utilization'], 'remaining_gap': after['fte_gap']}


def run_planning(transactions, config, train_end, validation_end, root):
    cfg = config['planning']
    weights = cfg['workload_weights']
    metric_names = list(dict.fromkeys(['total_sales', 'distinct_customers', 'transaction_count', 'distinct_products'] + list(weights)))
    units = aggregate(transactions, PLAN_GRAIN)
    # Per-rep × unit workload is additive; repeated customer/class service is intentional.
    rep_units = aggregate(transactions, ['representative'] + PLAN_GRAIN)
    training = rep_units.date <= train_end
    scales = rep_units.loc[training, list(weights)].median().clip(lower=1)
    rep_units['workload'] = sum(weights[m] * rep_units[m] / scales[m] for m in weights)
    workloads = rep_units.groupby(PLAN_GRAIN, observed=True).workload.sum().reset_index()
    units = units.merge(workloads, on=PLAN_GRAIN, validate='one_to_one')
    rep_month = rep_units.groupby(['representative', 'date'], observed=True).workload.sum().reset_index()
    rep_month['growth'] = rep_month.groupby('representative').workload.pct_change(fill_method=None)
    stable = rep_month.loc[(rep_month.date <= train_end) & rep_month.growth.abs().le(cfg['stable_growth_limit']), 'workload']
    fallback = len(stable) < 10
    if fallback:
        stable = rep_month.loc[rep_month.date <= train_end, 'workload']
    capacity = float(stable.quantile(cfg['capacity_quantile']))
    cap_low, cap_high = [float(stable.quantile(q)) for q in cfg['uncertainty_quantiles']]
    records = []
    all_metrics = metric_names + ['workload']
    for key, g in units.groupby(UNIT, observed=True):
        g = g.sort_values('date').set_index('date')
        for metric in all_metrics:
            for date, observed in g[metric].items():
                if date <= train_end:
                    continue
                hist = g.loc[g.index < date, metric]
                for method in METHODS:
                    records.append({**dict(zip(UNIT, key)), 'date': date, 'metric': metric, 'method': method,
                                    'split': 'validation' if date <= validation_end else 'test',
                                    'observed': observed, 'prediction': forecast(hist, method, date)})
    backtest = pd.DataFrame(records)
    summaries = []
    for (metric, method, split), g in backtest.groupby(['metric', 'method', 'split']):
        summaries.append({'metric': metric, 'method': method, 'split': split,
                          **forecast_metrics(g.observed, g.prediction)})
    metrics = pd.DataFrame(summaries)
    selected = metrics.query("split == 'validation'").sort_values('WAPE').drop_duplicates('metric').set_index('metric').method.to_dict()
    latest = rep_units.date.max()
    allocation = rep_units.loc[rep_units.date.eq(latest)].copy()
    allocation['allocated_fte'] = allocation.workload / allocation.groupby('representative').workload.transform('sum')
    ftes = allocation.groupby(UNIT).allocated_fte.sum()
    assert np.isclose(ftes.sum(), allocation.representative.nunique())
    rows = []
    assumptions = [{'assumption': 'sustainable_capacity_per_rep', 'value': capacity, 'basis': 'Training stable rep-month 60th percentile'},
                   {'assumption': 'capacity_low', 'value': cap_low, 'basis': 'Training 40th percentile (scenario bound, not CI)'},
                   {'assumption': 'capacity_high', 'value': cap_high, 'basis': 'Training 80th percentile (scenario bound, not CI)'},
                   {'assumption': 'stable_training_period_count', 'value': len(stable), 'basis': 'Absolute monthly workload growth <= configured limit'},
                   {'assumption': 'stable_period_fallback', 'value': int(fallback), 'basis': 'All training periods if fewer than 10 stable samples'}]
    assumptions += [{'assumption': f'{m}_scale', 'value': float(scales[m]), 'basis': f'Training per-rep-unit median; weight={weights[m]}'} for m in weights]
    for key, g in units.groupby(UNIT, observed=True):
        g = g.sort_values('date').set_index('date')
        future = latest + pd.offsets.MonthBegin(1)
        if g.index.max() < latest:
            rows.append({**dict(zip(UNIT,key)), 'forecast_date':future,
                         'last_observed_month':g.index.max(),
                         'coverage_note':'Stale source coverage: missing months are NOT zero demand or zero staffing',
                         'recommendation':'Insufficient recent source coverage — validate data',
                         'planning_eligible':False})
            continue
        raw_work = forecast(g.workload, selected['workload'], future)
        lo, hi = g.loc[g.index <= train_end, 'workload'].quantile(cfg['winsor_quantiles'])
        clean_work = forecast(g.workload.clip(lo, hi), selected['workload'], future)
        allocated = float(ftes.get(key, 0))
        current_reps = int(allocation.loc[(allocation[UNIT] == pd.Series(dict(zip(UNIT, key)))).all(axis=1)].representative.nunique())
        unit_errors = backtest[(backtest[UNIT] == pd.Series(dict(zip(UNIT, key)))).all(axis=1)
                              & backtest.metric.eq('workload') & backtest.method.eq(selected['workload']) & backtest.split.eq('validation')]
        err = float(abs(unit_errors.prediction - unit_errors.observed).quantile(.8)) if len(unit_errors) else 0
        row = {**dict(zip(UNIT, key)), 'forecast_date': future, 'forecast_workload': raw_work,
               'planning_eligible':True,
               'last_observed_month': g.index.max(),
               'coverage_note': 'No latest-month rep allocation; validate territory continuity' if allocated == 0 else 'Observed latest-month rep allocation',
               'current_active_reps': current_reps, 'allocated_current_fte': allocated,
               'capacity_per_rep': capacity, **required_fte(raw_work, capacity, allocated),
               'required_fte_low': max(0, raw_work-err)/max(cap_high, 1e-9),
               'required_fte_high': (raw_work+err)/max(cap_low, 1e-9),
               'forecast_uncertainty': err/max(raw_work, 1e-9),
               'confidence_note': 'Scenario interval from validation absolute error and historical capacity quantiles; not a statistical confidence interval',
               'raw_required_fte': raw_work/capacity, 'cleaned_required_fte': clean_work/capacity,
               'cleaning_difference': (clean_work-raw_work)/capacity,
               'cleaning_explanation': 'Workload winsorized to unit-specific TRAIN 5th/95th percentiles; capacity held fixed',
               'forecast_method': selected['workload']}
        for metric in metric_names:
            row['forecast_' + metric] = forecast(g[metric], selected[metric], future)
        row['demand_growth'] = row['forecast_total_sales']/max(abs(g.total_sales.tail(3).mean()), 1)-1
        row['customer_growth'] = row['forecast_distinct_customers']/max(g.distinct_customers.tail(3).mean(), 1)-1
        component = {'utilization': np.clip((row['utilization']-.7)/.8, 0, 1),
                     'gap': np.clip(row['additional_fte_need']/max(allocated, .1), 0, 1),
                     'demand_growth': np.clip(row['demand_growth']/.2, 0, 1),
                     'customer_growth': np.clip(row['customer_growth']/.2, 0, 1),
                     'geographic_load': np.clip(row['forecast_distinct_cities']/max(g.distinct_cities.median(),1)-1, 0, 1),
                     'product_opportunity': np.clip(row['forecast_distinct_products']/max(g.distinct_products.median(),1)-1, 0, 1),
                     'workload_pressure': np.clip(raw_work/max(g.workload.tail(3).mean(), 1e-9)-1, 0, 1)}
        raw_priority = 100*sum(cfg['priority_weights'][k]*v for k,v in component.items())
        row['hiring_priority'] = raw_priority * (1 - .25 * min(row['forecast_uncertainty'], 1))
        row['recommendation'] = next(label for upper, label in [(30,'No modeled capacity pressure'), (50,'Monitor'), (70,'Validate capacity assumptions'), (85,'Capacity pressure indicated'), (100,'Additional field-capacity scenario strongly indicated')] if row['hiring_priority'] <= upper)
        rows.append(row)
    results = pd.DataFrame(rows).sort_values('hiring_priority', ascending=False)
    scenarios = []
    specs = [('Base case', 0, 0, 0), ('Demand +10%', .1, 0, 0), ('Demand +20%', .2, 0, 0),
             ('Add 1 representative', 0, 1, 0), ('Add 2 representatives', 0, 2, 0),
             ('Capacity -10%', 0, 0, -.1), ('Product launch +30%', .3, 0, 0)]
    for _, row in results.iterrows():
        if not row.planning_eligible:
            continue
        for name, demand, add, change in specs:
            scenarios.append({**row[UNIT].to_dict(), 'scenario': name,
                              'scenario_change': f'demand={demand:+.0%}; FTE={add:+}; capacity={change:+.0%}',
                              **scenario(row.forecast_workload, capacity, row.allocated_current_fte, demand, add, change)})
    # One explicit transfer pair, not an invented extra rep in every cell.
    receiver = results.iloc[0]
    donors = results.loc[results.team.eq(receiver.team) & results.planning_eligible].iloc[1:].sort_values('utilization')
    if len(donors):
        donor = donors.iloc[0]
        amount = min(1.0, float(donor.allocated_current_fte))
        for row, delta in [(receiver, amount), (donor, -amount)]:
            scenarios.append({**row[UNIT].to_dict(), 'scenario': 'Reallocate 1 representative (bounded by donor FTE)',
                              'scenario_change': f'Net-zero paired transfer; allocated FTE change={delta:+.4f}',
                              **scenario(row.forecast_workload, capacity, row.allocated_current_fte, add_reps=delta)})
    folder = root/'artifacts/planning'
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {'hiring_need_by_business_unit': results, 'hiring_scenarios': pd.DataFrame(scenarios),
               'capacity_assumptions': pd.DataFrame(assumptions), 'anomaly_cleaning_sensitivity': results[UNIT + ['raw_required_fte','cleaned_required_fte','cleaning_difference','cleaning_explanation']],
               'forecast_backtest': backtest, 'forecast_metrics': metrics,
               'fte_allocation': allocation[['representative']+PLAN_GRAIN+['workload','allocated_fte']]}
    for name, df in outputs.items():
        df.to_csv(folder/f'{name}.csv', index=False)
    return results, metrics
