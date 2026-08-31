"""Walk-forward residuals and one-sided CUSUM; no future observations in baselines."""
import numpy as np
import pandas as pd

from .commercial import METRICS, SERIES, peer_signals

TEMPORAL_NAMES = ['Rolling Residual', 'EWMA Residual', 'Seasonal Residual', 'Change-Point / Level Shift']


def temporal_scores(frame, settings):
    scores = pd.DataFrame(0.0, index=frame.index, columns=TEMPORAL_NAMES)
    details = []
    for _, group in frame.groupby(SERIES, observed=True):
        for metric in METRICS:
            history, dates = [], []
            ewma, up, down = None, 0.0, 0.0
            for i, row in group.sort_values('date').iterrows():
                observed = float(row[metric])
                # Missing calendar months do not masquerade as lag-12 observations.
                h = np.asarray(history[-settings['window']:], float)
                enough = len(h) >= settings['min_history']
                expected = float(np.median(h)) if len(h) else np.nan
                mad = float(np.median(np.abs(h - expected))) if len(h) else np.nan
                scale = max(1.4826 * mad, abs(expected) * .1, 1) if len(h) else 1
                seasonal = dict(zip(dates, history)).get(row.date - pd.DateOffset(years=1), np.nan)
                residual = (observed - expected) / scale if enough else 0
                up = max(0.0, up + residual - settings['cusum_drift']) if enough else 0
                down = max(0.0, down - residual - settings['cusum_drift']) if enough else 0
                # Confirmation requires two consecutive same-direction deviations.
                previous_z = (history[-1] - np.median(history[-settings['window']-1:-1])) / scale if len(history) > settings['min_history'] else 0
                level = max(up, down) if residual * previous_z > 0 and abs(previous_z) > 1 else 0
                trend = (np.mean(h[-3:]) - np.mean(h[:-3])) / scale if len(h) >= 6 else 0
                options = [
                    ('Rolling Residual', expected, abs(residual), 'point_anomaly'),
                    ('EWMA Residual', ewma, abs(observed - ewma) / scale if enough else 0, 'point_anomaly'),
                    ('Seasonal Residual', seasonal, abs(observed - seasonal) / scale if np.isfinite(seasonal) else 0, 'seasonal_anomaly'),
                    ('Change-Point / Level Shift', expected, level, 'level_shift'),
                ]
                for name, exp, score, kind in options:
                    scores.at[i, name] = max(scores.at[i, name], score)
                    details.append({'observation_id': row.observation_id, 'representative': row.representative,
                                    'product_class': row.product_class, 'date': row.date, 'metric': metric,
                                    'model': name, 'observed': observed, 'expected': exp,
                                    'residual': observed - exp if exp is not None else np.nan,
                                    'normalized_residual': (observed - exp) / scale if exp is not None else np.nan,
                                    'direction': 'up' if exp is not None and observed > exp else 'down',
                                    'anomaly_type': kind, 'score': score, 'trend_score': abs(trend),
                                    'history_length': len(history), 'available': bool(enough and exp is not None and np.isfinite(exp))})
                history.append(observed)
                dates.append(row.date)
                ewma = observed if ewma is None else settings['ewma_alpha'] * observed + (1 - settings['ewma_alpha']) * ewma
    return scores, pd.DataFrame(details)


def robust_peer(frame, minimum=4):
    deviations, details = [], []
    for metric in METRICS:
        med, z, pct, cohort = peer_signals(frame, metric, minimum)
        deviations.append(z.abs().to_numpy())
        details.append(pd.DataFrame({'observation_id': frame.observation_id, 'metric': metric, 'observed': frame[metric],
                                     'expected': med, 'robust_z': z, 'peer_percentile': pct, 'cohort': cohort}))
    return np.max(deviations, axis=0), pd.concat(details, ignore_index=True)


def business_rules(frame):
    f = frame
    definitions = {
        'payout_above_expected': (f.simulated_payout_delta_pct.abs() / 25, 'DEMO payout differs from formula expectation'),
        'large_adjustment': (f.simulated_adjustment.abs() / f.simulated_expected_incentive.clip(lower=1) / .25, 'DEMO adjustment exceeds 25% of expected payout'),
        'payout_rising_sales_falling': ((f.simulated_actual_payout_month_over_month_growth.clip(lower=0) * (-f.total_sales_month_over_month_growth).clip(lower=0)) / .1, 'DEMO payout growth accompanies commercial decline'),
        'payout_to_sales': ((f.simulated_payout_to_sales_ratio - .05).clip(lower=0) / .05, 'DEMO payout exceeds 5% of net sales'),
        'attainment_without_coverage': (((f.simulated_target_attainment_pct - 150).clip(lower=0) / 100) * f.customer_growth.fillna(0).le(0), 'High DEMO attainment without customer growth'),
        'single_customer_sales_spike': (f.total_sales_month_over_month_growth.clip(lower=0) * f.customer_concentration / .25, 'Sales growth is concentrated in few customers'),
        'product_concentration': ((f.product_concentration - .5).clip(lower=0) / .5, 'High product concentration'),
        'coverage_expansion': (f.customer_growth.clip(lower=0) / .5, 'Sudden customer expansion'),
    }
    out = []
    for rule, (severity, explanation) in definitions.items():
        severity = severity.fillna(0).clip(lower=0)
        out.append(pd.DataFrame({'observation_id': f.observation_id, 'rule_name': rule, 'flag': severity.ge(1),
                                 'normalized_severity': severity / (1 + severity), 'explanation': explanation}))
    details = pd.concat(out, ignore_index=True)
    score = details.groupby('observation_id').normalized_severity.max().reindex(f.observation_id).to_numpy()
    flagged = pd.Series(score >= .5, index=f.index).astype(float)
    repeated = flagged.groupby([f[k] for k in SERIES]).transform(lambda s: s.shift().rolling(3, min_periods=1).sum()).fillna(0)
    repeat_detail = pd.DataFrame({'observation_id': f.observation_id, 'rule_name': 'repeated_unusual_periods',
                                  'flag': repeated.ge(2), 'normalized_severity': repeated / 3,
                                  'explanation': 'Repeated prior rule-triggering periods (not confirmed misconduct)'})
    return np.maximum(score, repeated.to_numpy() / 3), pd.concat([details, repeat_detail], ignore_index=True)
