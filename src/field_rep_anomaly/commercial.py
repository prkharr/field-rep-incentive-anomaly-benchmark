"""Real commercial aggregation, calendar-safe features and explicitly DEMO payouts.

This adapter preserves actual rep identities; it does not use legacy synthetic mapping.
Signed sales/quantity (returns) are retained. Concentration uses absolute sales shares.
"""
from __future__ import annotations

import calendar
import ast
import operator
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    'Distributor': 'distributor', 'Customer Name': 'customer', 'City': 'city',
    'Country': 'country', 'Latitude': 'latitude', 'Longitude': 'longitude',
    'Channel': 'channel', 'Sub-channel': 'subchannel', 'Product Name': 'product',
    'Product Class': 'product_class', 'Quantity': 'quantity', 'Price': 'price',
    'Sales': 'sales', 'Month': 'month', 'Year': 'year',
    'Name of Sales Rep': 'representative', 'Manager': 'manager', 'Sales Team': 'team',
}
GRAIN = ['representative', 'product_class', 'date']
SERIES = GRAIN[:-1]
PLAN_GRAIN = ['team', 'country', 'product_class', 'date']
METRICS = ['total_sales', 'total_quantity', 'distinct_customers', 'simulated_actual_payout']
BASE_FEATURES = [
    'total_sales', 'total_quantity', 'transaction_count', 'average_price', 'median_price',
    'average_transaction_value', 'sales_per_transaction', 'distinct_customers',
    'new_customers', 'repeat_customer_ratio', 'sales_per_customer', 'customer_concentration',
    'customer_growth', 'customer_loss_rate', 'distinct_products', 'product_concentration',
    'dominant_product_share', 'product_class_share', 'product_mix_change', 'product_breadth_change',
    'distinct_cities', 'geographic_spread', 'latitude_dispersion', 'longitude_dispersion',
    'country_mix', 'city_concentration', 'distributor_count', 'distributor_concentration',
    'channel_mix', 'subchannel_mix', 'distributor_mix_change',
    'simulated_target_attainment_pct', 'simulated_expected_incentive', 'simulated_actual_payout',
    'simulated_adjustment', 'simulated_payout_delta', 'simulated_payout_delta_pct',
    'simulated_payout_to_sales_ratio',
]


def safe_ratio(a, b):
    return np.asarray(a, float) / np.maximum(np.abs(np.asarray(b, float)), 1.0)


def monthly_date(year: pd.Series, month: pd.Series) -> pd.Series:
    names = {v.lower(): i for i, v in enumerate(calendar.month_name) if v}
    names.update({v.lower(): i for i, v in enumerate(calendar.month_abbr) if v})
    numeric = pd.to_numeric(month, errors='coerce')
    numeric = numeric.fillna(month.astype(str).str.strip().str.lower().map(names))
    return pd.to_datetime(dict(year=pd.to_numeric(year, errors='coerce'), month=numeric, day=1), errors='coerce')


def load_commercial(path: str | Path):
    raw = pd.read_csv(path)
    missing = set(ALIASES) - set(raw)
    if missing:
        raise ValueError(f'Missing Kaggle columns: {sorted(missing)}')
    duplicates = int(raw.duplicated().sum())
    d = raw.drop_duplicates().rename(columns=ALIASES).copy()
    d['date'] = monthly_date(d['year'], d['month'])
    required = ['date', 'representative', 'product_class', 'sales', 'quantity', 'price']
    for c in ['sales', 'quantity', 'price', 'latitude', 'longitude']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    invalid = d[required].isna().any(axis=1) | ~np.isfinite(d[['sales', 'quantity', 'price']]).all(axis=1)
    invalid_count = int(invalid.sum())
    d = d.loc[~invalid].copy()
    for c in ['customer', 'city', 'country', 'distributor', 'team', 'manager', 'product', 'channel', 'subchannel']:
        d[c] = d[c].fillna('Unknown').astype(str).str.strip()
    report = {
        'source': Path(path).name, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'raw_columns': len(raw.columns), 'duplicates_removed': duplicates,
        'invalid_rows_removed': invalid_count, 'clean_rows': len(d),
        'missing_by_column': raw.isna().sum().to_dict(),
        'cardinalities': d[list(ALIASES.values())].nunique().to_dict(),
        'date_min': str(d.date.min().date()), 'date_max': str(d.date.max().date()),
        'nonpositive_sales_retained': int(d.sales.le(0).sum()),
        'sales_total': float(d.sales.sum()),
        'country_date_coverage': {country:{'start':str(g.date.min().date()),'end':str(g.date.max().date()),'rows':len(g)} for country,g in d.groupby('country')},
    }
    return d, report


def aggregate(d: pd.DataFrame, grain: list[str]) -> pd.DataFrame:
    """Actual unique counts, not sums of lower-grain unique counts."""
    g = d.groupby(grain, sort=True, observed=True)
    out = g.agg(total_sales=('sales', 'sum'), total_quantity=('quantity', 'sum'),
                transaction_count=('sales', 'size'), average_price=('price', 'mean'),
                median_price=('price', 'median'), distinct_customers=('customer', 'nunique'),
                distinct_products=('product', 'nunique'), distinct_cities=('city', 'nunique'),
                distributor_count=('distributor', 'nunique'), active_reps=('representative', 'nunique'),
                latitude_dispersion=('latitude', 'std'), longitude_dispersion=('longitude', 'std')).reset_index()
    return out


def _distribution(part, column):
    weights = part.groupby(column, observed=True).sales.apply(lambda s: s.abs().sum())
    if weights.sum() <= 0:
        weights = part[column].value_counts().astype(float)
    return (weights / weights.sum()).to_dict()


def _mix_distance(current, previous):
    if previous is None:
        return 0.0
    return sum(abs(current.get(k, 0) - previous.get(k, 0)) for k in current.keys() | previous.keys()) / 2


def build_population(d: pd.DataFrame):
    grain_rows = []
    for keys in [GRAIN, ['representative', 'product_class', 'country', 'date'], ['representative', 'date']]:
        sizes = d.groupby(keys, observed=True).size()
        grain_rows.append({'grain': ' x '.join(keys), 'observations': len(sizes),
                           'median_transactions': float(sizes.median()),
                           'single_transaction_pct': float(sizes.eq(1).mean() * 100)})
    base = aggregate(d, GRAIN)
    extra = []
    histories = {}
    for key, part in d.sort_values('date').groupby(GRAIN, sort=True, observed=True):
        rep, cls, date = key
        seen, prior_customers, last_product, last_distributor = histories.get((rep, cls), (set(), set(), None, None))
        customers = set(part.customer)
        distributions = {name: _distribution(part, name) for name in ['customer', 'product', 'city', 'country', 'distributor', 'channel', 'subchannel']}
        row = dict(zip(GRAIN, key))
        row.update(manager=part.manager.mode().iloc[0], team=part.team.mode().iloc[0],
                   country='|'.join(sorted(part.country.unique())),
                   peer_country=max(distributions['country'], key=distributions['country'].get),
                   new_customers=len(customers - seen),
                   repeat_customer_ratio=len(customers & seen) / max(1, len(customers)),
                   customer_loss_rate=len(prior_customers - customers) / max(1, len(prior_customers)),
                   product_mix_change=_mix_distance(distributions['product'], last_product),
                   distributor_mix_change=_mix_distance(distributions['distributor'], last_distributor))
        for name in ['customer', 'product', 'city', 'distributor']:
            row[name + '_concentration'] = sum(v*v for v in distributions[name].values())
        row['dominant_product_share'] = max(distributions['product'].values())
        for name in ['country', 'channel', 'subchannel']:
            row[name + '_mix'] = 1 - sum(v*v for v in distributions[name].values())
        extra.append(row)
        histories[(rep, cls)] = (seen | customers, customers, distributions['product'], distributions['distributor'])
    base = base.merge(pd.DataFrame(extra), on=GRAIN, validate='one_to_one')
    base['geographic_spread'] = np.hypot(base.latitude_dispersion.fillna(0), base.longitude_dispersion.fillna(0))
    base = base.sort_values(GRAIN).reset_index(drop=True)
    base['observation_id'] = ['obs_' + hashlib.sha256('|'.join(map(str, row)).encode()).hexdigest()[:16] for row in base[GRAIN].itertuples(index=False, name=None)]
    return base, pd.DataFrame(grain_rows), {
        'rep_month_rollup': aggregate(d, ['representative', 'date']),
        'market_view': aggregate(d, ['product_class', 'country', 'date']),
        'planning_view': aggregate(d, PLAN_GRAIN),
    }


def _calendar_history(frame, column, fn):
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(SERIES, observed=True):
        idx = pd.date_range(group.date.min(), group.date.max(), freq='MS')
        values = group.set_index('date')[column].reindex(idx)
        result.loc[group.index] = fn(values).reindex(group.date).to_numpy()
    return result


def evaluate_formula(expression, variables):
    """Small arithmetic language, no eval, attributes, imports or arbitrary calls."""
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
    funcs={'maximum':np.maximum,'minimum':np.minimum,'abs':np.abs}
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and isinstance(node.value,(float,int)):return node.value
        if isinstance(node,ast.Name) and node.id in variables:return variables[node.id]
        if isinstance(node,ast.BinOp) and type(node.op) in ops:return ops[type(node.op)](visit(node.left),visit(node.right))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -visit(node.operand)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in funcs and not node.keywords:
            return funcs[node.func.id](*[visit(a) for a in node.args])
        raise ValueError('Unsupported demo incentive formula expression')
    return visit(ast.parse(expression,mode='eval'))


def add_demo_incentives(base, config):
    f = base.copy()
    prior = _calendar_history(f, 'total_sales', lambda s: s.shift().rolling(config['target_history_months'], min_periods=1).median())
    variables={**{k:v for k,v in config.items() if isinstance(v,(int,float))},
               **{k:f[k] for k in f.columns},'historical_target':prior.fillna(config['cold_start_target'])}
    for name,formula in config['formulas'].items():
        f[name]=evaluate_formula(formula,variables)
        variables[name]=f[name]
    return f


def peer_signals(f, column, minimum=4):
    median = pd.Series(np.nan, index=f.index)
    mad = median.copy()
    pct = median.copy()
    cohort = pd.Series('', index=f.index, dtype=object)
    levels = [['product_class', 'peer_country', 'team', 'date'],
              ['product_class', 'peer_country', 'date'], ['product_class', 'date'], ['date']]
    for level in levels:
        g = f.groupby(level, observed=True)[column]
        med = g.transform('median')
        variation = (f[column] - med).abs().groupby([f[k] for k in level], observed=True).transform('median')
        use = median.isna() & g.transform('size').ge(minimum)
        median.loc[use], mad.loc[use] = med[use], variation[use]
        pct.loc[use] = g.rank(pct=True)[use]
        cohort.loc[use] = '|'.join(level)
    # Tiny months fall back to same-month median, never a future global median.
    fallback = f.groupby('date', observed=True)[column].transform('median')
    median = median.fillna(fallback)
    mad = mad.fillna(0)
    scale = np.maximum(1.4826 * mad, np.maximum(median.abs() * 0.10, 1.0))
    return median, (f[column] - median) / scale, pct.fillna(.5), cohort.replace('', 'date_sparse')


def engineer_commercial(f, minimum_peer=4, incentive_config=None):
    f = f.sort_values(GRAIN).reset_index(drop=True).copy()
    f['average_transaction_value'] = safe_ratio(f.total_sales, f.transaction_count)
    f['sales_per_transaction'] = f.average_transaction_value
    f['sales_per_customer'] = safe_ratio(f.total_sales, f.distinct_customers)
    f['product_class_share'] = safe_ratio(f.total_sales, f.groupby(['representative', 'date']).total_sales.transform('sum'))
    f['simulated_payout_delta'] = f.simulated_actual_payout - f.simulated_expected_incentive
    f['simulated_payout_delta_pct'] = safe_ratio(f.simulated_payout_delta, f.simulated_expected_incentive) * 100
    f['simulated_payout_to_sales_ratio'] = safe_ratio(f.simulated_actual_payout, f.total_sales)
    if incentive_config is not None:
        for c in ['simulated_payout_delta','simulated_payout_delta_pct','simulated_payout_to_sales_ratio']:
            f[c]=evaluate_formula(incentive_config['formulas'][c],{k:f[k] for k in f})
    features = BASE_FEATURES.copy()
    derived = {}
    for metric in METRICS:
        for lag in [1, 2, 3, 6, 12]:
            derived[f'{metric}_lag_{lag}'] = _calendar_history(f, metric, lambda s, n=lag: s.shift(n))
        for w in [3, 6]:
            derived[f'{metric}_rolling_mean_{w}'] = _calendar_history(f, metric, lambda s, n=w: s.shift().rolling(n, min_periods=1).mean())
        for stat in ['median', 'std']:
            derived[f'{metric}_rolling_{stat}'] = _calendar_history(f, metric, lambda s, stat=stat: getattr(s.shift().rolling(6, min_periods=2), stat)())
        derived[f'{metric}_rolling_MAD'] = _calendar_history(f, metric, lambda s: s.shift().rolling(6, min_periods=3).apply(lambda a: np.nanmedian(np.abs(a - np.nanmedian(a))), raw=True))
        prior = derived[f'{metric}_lag_1']
        derived[f'{metric}_month_over_month_growth'] = safe_ratio(f[metric] - prior, prior)
        year = derived[f'{metric}_lag_12']
        derived[f'{metric}_year_over_year_growth'] = safe_ratio(f[metric] - year, year)
        derived[f'{metric}_trend'] = safe_ratio(prior - derived[f'{metric}_lag_3'], derived[f'{metric}_lag_3'])
        derived[f'{metric}_acceleration'] = derived[f'{metric}_month_over_month_growth'] - safe_ratio(prior - derived[f'{metric}_lag_2'], derived[f'{metric}_lag_2'])
        derived[f'{metric}_history_deviation'] = safe_ratio(f[metric] - derived[f'{metric}_rolling_median'], np.maximum(1.4826 * derived[f'{metric}_rolling_MAD'], np.abs(derived[f'{metric}_rolling_median']) * .1))
        med, z, pct, _ = peer_signals(f, metric, minimum_peer)
        derived[f'{metric}_peer_z'], derived[f'{metric}_peer_percentile'] = z, pct
        derived[f'{metric}_peer_median'] = med
        for cohort in [['team', 'date'], ['product_class', 'date'], ['peer_country', 'product_class', 'date']]:
            name = f'{metric}_' + '_'.join(cohort[:-1]) + '_relative'
            cm = f.groupby(cohort)[metric].transform('median')
            derived[name] = safe_ratio(f[metric] - cm, cm)
    f = pd.concat([f, pd.DataFrame(derived, index=f.index)], axis=1)
    f['customer_growth'] = f.distinct_customers_month_over_month_growth
    prior_products = _calendar_history(f, 'distinct_products', lambda s: s.shift())
    f['product_breadth_change'] = safe_ratio(f.distinct_products - prior_products, prior_products)
    f['representative_rank'] = f.groupby(['product_class', 'date']).total_sales.rank(ascending=False, method='average')
    f['rank_change'] = f.representative_rank - _calendar_history(f, 'representative_rank', lambda s: s.shift())
    features += list(derived) + ['representative_rank', 'rank_change']
    # A label-independent allowlist is the only model input interface.
    if any('injected' in c or c in ['anomaly_type', 'severity', 'seed'] for c in features):
        raise ValueError('Benchmark-label leakage')
    f[features] = f[features].replace([np.inf, -np.inf], np.nan)
    return f.copy(), features
