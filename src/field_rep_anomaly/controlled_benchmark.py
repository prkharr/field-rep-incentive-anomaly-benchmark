"""Separate, audited benchmark copies; labels never feed feature selection."""
import numpy as np
import pandas as pd

from .commercial import SERIES

TYPES = [
    ('payout_spike', 'simulated_actual_payout', 'up'),
    ('payout_reduction', 'simulated_actual_payout', 'down'),
    ('payout_mismatch', 'simulated_actual_payout', 'up'),
    ('adjustment_anomaly', 'simulated_adjustment', 'adjustment'),
    ('sales_spike', 'total_sales', 'up'),
    ('quantity_spike', 'total_quantity', 'up'),
    ('quantity_mismatch', 'total_quantity', 'down'),
    ('price_anomaly', 'average_price', 'up'),
    ('customer_burst', 'distinct_customers', 'up'),
    ('customer_collapse', 'distinct_customers', 'down'),
    ('product_mix_shift', 'product_concentration', 'bounded'),
    ('distributor_shift', 'distributor_concentration', 'bounded'),
    ('geographic_change', 'distinct_cities', 'up'),
    ('peer_divergence', 'total_sales', 'up'),
    ('inactivity_burst', 'total_sales', 'sequence'),
    ('upward_level_shift', 'total_sales', 'level_up'),
    ('downward_level_shift', 'total_sales', 'level_down'),
    ('end_period_spike', 'total_sales', 'quarter'),
]


def inject_benchmark(clean, rate, seed, train_end, validation_end):
    """Inject validation/test independently, with untouched training and exact budgets.

    Stateful anomalies cover consecutive months in ONE split. Monthly data cannot
    identify intra-month end-of-period timing; the last type uses quarter-end months.
    """
    f = clean.copy(deep=True)
    for _, feature, _ in TYPES:
        f[feature] = f[feature].astype(float)
    f['injected_anomaly_flag'] = False
    f['anomaly_type'] = 'none'
    f['severity'] = 'none'
    rng = np.random.default_rng(seed)
    audit = []
    ranges = [('validation', (f.date > train_end) & (f.date <= validation_end)), ('test', f.date > validation_end)]
    for split, mask in ranges:
        available = set(f.index[mask])
        budget = int(np.ceil(len(available) * rate))
        used = 0
        order = list(range(len(TYPES)))
        if budget < len(TYPES):
            rng.shuffle(order)
        # Primary validation has enough observations to cover every injection type.
        for step in range(budget):
            if used >= budget:
                break
            kind, feature, mode = TYPES[order[step % len(order)]]
            if mode in ['sequence','level_up','level_down'] and budget-used<2:
                kind,feature,mode=TYPES[-1]  # Never label a singleton as sustained.
            candidates = sorted(available)
            if mode == 'quarter':
                quarter = [i for i in candidates if f.at[i, 'date'].month in [3, 6, 9, 12]]
                candidates = quarter or candidates
            rng.shuffle(candidates)
            chosen = [candidates[0]]
            if mode in ['sequence', 'level_up', 'level_down'] and budget - used >= 2:
                for i in candidates:
                    row = f.loc[i]
                    follow = f.index[mask & f.representative.eq(row.representative) & f.product_class.eq(row.product_class)
                                     & f.date.eq(row.date + pd.offsets.MonthBegin(1))].tolist()
                    if follow and follow[0] in available:
                        chosen = [i, follow[0]]
                        break
            severity = ['low', 'medium', 'high'][step % 3]
            strength = {'low': .6, 'medium': 1.5, 'high': 3.0}[severity]
            for pos, idx in enumerate(chosen):
                old = float(f.at[idx, feature])
                if mode in ['down', 'level_down']:
                    new = old / (1 + strength * 2)
                elif mode == 'bounded':
                    new = (old + (1 - old) * min(.95, .25 + strength * .2)
                           if old < .95 else max(.05, old - (.2 + strength * .15)))
                elif mode == 'adjustment':
                    new = float(f.at[idx, 'simulated_expected_incentive']) * strength
                elif mode == 'sequence':
                    new = old * (.05 if pos == 0 else 1 + strength * 2)
                else:
                    new = max(abs(old), 1) * (1 + strength)
                if feature in ['distinct_customers', 'distinct_cities']:
                    new = float(round(new))
                changes = {feature: new}
                if mode == 'adjustment':
                    changes['simulated_actual_payout'] = float(f.at[idx, 'simulated_actual_payout']) + new - old
                if kind == 'price_anomaly':
                    changes['median_price'] = float(f.at[idx, 'median_price']) * (1 + strength)
                for field, value in changes.items():
                    original = float(f.at[idx, field])
                    f.at[idx, field] = value
                    audit.append({'observation_id': f.at[idx, 'observation_id'], 'representative': f.at[idx, 'representative'],
                                  'product_class': f.at[idx, 'product_class'], 'period': f.at[idx, 'date'],
                                  'split': split, 'anomaly_type': kind, 'severity': severity, 'affected_feature': field,
                                  'original_value': original, 'injected_value': value, 'seed': seed})
                f.loc[idx, ['injected_anomaly_flag', 'anomaly_type', 'severity']] = [True, kind, severity]
                available.remove(idx)
                used += 1
    return f, pd.DataFrame(audit)
