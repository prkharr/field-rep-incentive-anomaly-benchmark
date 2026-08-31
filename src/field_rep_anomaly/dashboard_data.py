"""Dashboard semantic layer: read executed clean artifacts, never fit or rescore.

Only the five allowlisted manager CSVs and their metadata are written. Technical
artifacts, raw data, calibration, model selection and evaluation labels are untouched.
Run ``python -m field_rep_anomaly.dashboard_data`` to refresh without retraining.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from .dashboard_capacity_data import build_capacity_base_dataset, build_capacity_scenario_dataset

DEFAULT_PRIORITY = {'very_high_percentile': .99, 'elevated_percentile': .95, 'multiple_support_count': 2}
DATASETS = {
    'dashboard_anomaly_review.csv': ['representative', 'product_class', 'month'],
    'dashboard_rep_summary.csv': ['representative'],
    'dashboard_capacity_base.csv': ['team', 'country', 'product_class'],
    'dashboard_capacity_scenarios.csv': ['team', 'country', 'product_class', 'scenario_name'],
    'dashboard_model_summary.csv': ['model'],
}
MODEL_LABELS = {
    'PCA Reconstruction': 'Pattern deviation', 'EWMA Residual': 'Recent trend deviation',
    'Robust Peer Baseline': 'Peer deviation', 'Business Rules': 'Business rule indicator',
    'K-Means': 'Commercial behavior segment', 'DBSCAN': 'Local density deviation',
    'Autoencoder': 'Nonlinear pattern deviation', 'Isolation Forest': 'Tree-based unusual pattern',
    'Rolling Residual': 'Recent historical deviation', 'Seasonal Residual': 'Same-month historical deviation',
    'Change-Point / Level Shift': 'Sustained change indicator', 'Best Ensemble': 'Combined comparison signal',
}
FIELD_LABELS = {
    'total_sales': 'Sales', 'total_quantity': 'Quantity', 'distinct_customers': 'Customer coverage',
    'distinct_products': 'Product breadth', 'distinct_cities': 'City coverage',
    'simulated_actual_payout': 'DEMO payout', 'simulated_expected_incentive': 'DEMO expected incentive',
    'simulated_adjustment': 'DEMO adjustment', 'simulated_payout_delta': 'DEMO payout difference',
    'simulated_payout_delta_pct': 'DEMO payout difference (%)',
    'simulated_payout_to_sales_ratio': 'DEMO payout relative to sales',
    'simulated_target_attainment_pct': 'DEMO target attainment (%)',
    'country_mix': 'Country mix diversity', 'channel_mix': 'Channel mix diversity',
    'subchannel_mix': 'Subchannel mix diversity', 'geographic_spread': 'Geographic dispersion',
}


def manager_feature_label(feature):
    if pd.isna(feature):
        return None
    if feature in FIELD_LABELS:
        return FIELD_LABELS[feature]
    suffixes = {'_history_deviation': 'personal history deviation', '_peer_z': 'peer deviation',
                '_peer_percentile': 'peer percentile', '_peer_median': 'peer median',
                '_month_over_month_growth': 'month-over-month change', '_year_over_year_growth': 'year-over-year change',
                '_rolling_MAD': 'prior historical variation', '_trend': 'prior trend', '_acceleration': 'change in growth'}
    for suffix, label in suffixes.items():
        if feature.endswith(suffix):
            return f'{manager_feature_label(feature[:-len(suffix)])}: {label}'
    for prefix, label in FIELD_LABELS.items():
        if feature.startswith(prefix + '_'):
            return label + ': ' + feature[len(prefix)+1:].replace('_', ' ')
    return feature.removeprefix('simulated_').replace('_', ' ').capitalize()


def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors='coerce') if column in frame else pd.Series(np.nan, index=frame.index)


def _bool(values):
    """Parse booleans without treating the string 'False' as truthy."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype('boolean')
    if not values.dropna().isin([True, False, 0, 1, 'True', 'False', 'true', 'false']).all():
        raise ValueError('Invalid persisted boolean values')
    return values.replace({'True': True, 'False': False, 'true': True, 'false': False}).astype('boolean')


def _unique(frame, keys, label):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f'{label} must be unique and non-null at {keys}')


def _month(values):
    return pd.to_datetime(values, errors='raise').dt.to_period('M').dt.to_timestamp()


def _integer_inference(values):
    """Only expose a count if the source arithmetic identifies an integer count."""
    valid = values.notna() & values.ge(0) & np.isclose(values, values.round(), atol=1e-7, rtol=0)
    return values.round().where(valid).astype('Int64')


def _strongest(frame, value_column, metric_column='metric'):
    f = frame.copy()
    f['_magnitude'] = pd.to_numeric(f[value_column], errors='coerce').abs()
    return (f.dropna(subset=['_magnitude']).sort_values(
        ['observation_id', '_magnitude', metric_column], ascending=[True, False, True], kind='stable')
        .drop_duplicates('observation_id').set_index('observation_id'))


def build_anomaly_dashboard_dataset(analytical, scores, pca_contributions, peer_explanations,
                                    temporal_scores, rule_signals, queue, settings=None):
    """Rep × class × month. TRAIN percentiles and persisted queue flags are reused.

    Review ranks are snapshot-wide display ranks (not historical model features).
    Supporting signal count uses three families: peer, EWMA and any business rule.
    These are corroborating views, not statistically independent evidence.
    """
    policy = {**DEFAULT_PRIORITY, **(settings or {})}
    if not 0 <= policy['elevated_percentile'] <= policy['very_high_percentile'] <= 1:
        raise ValueError('Dashboard percentiles must satisfy 0 <= elevated <= very high <= 1')
    if policy['multiple_support_count'] < 1:
        raise ValueError('multiple_support_count must be positive')
    source = analytical.copy(deep=True)
    _unique(source, ['observation_id'], 'Analytical observations')
    source['date'] = _month(source.date)
    _unique(source, ['representative', 'product_class', 'date'], 'Analytical grain')
    source = source.set_index('observation_id', drop=False)
    if 'population' in scores and not scores.population.eq('clean').all():
        raise ValueError('Manager dashboard requires clean scores, not controlled benchmark scores')
    if 'population' in queue and not queue.population.eq('clean').all():
        raise ValueError('Manager dashboard requires a clean investigation queue')
    _unique(scores, ['observation_id', 'model_name'], 'Persisted model scores')
    out = pd.DataFrame(index=source.index)
    out['observation_id'] = source.observation_id
    for c in ['representative', 'manager', 'team', 'country', 'product_class']:
        out[c] = source[c] if c in source else None
    out['month'] = source.date.dt.strftime('%Y-%m-%d')
    out['source_partition'] = source['split'] if 'split' in source else None
    mapping = {
        'sales': 'total_sales', 'quantity': 'total_quantity', 'transaction_count': 'transaction_count',
        'transaction_value': 'average_transaction_value', 'unique_customers': 'distinct_customers',
        'new_customers': 'new_customers', 'product_breadth': 'distinct_products', 'city_coverage': 'distinct_cities',
        'geographic_spread': 'geographic_spread', 'distributor_count': 'distributor_count',
        'distributor_concentration': 'distributor_concentration', 'distributor_mix_change': 'distributor_mix_change',
        'channel_mix': 'channel_mix', 'subchannel_mix': 'subchannel_mix',
        'simulated_target': 'simulated_target_sales', 'simulated_expected_incentive': 'simulated_expected_incentive',
        'simulated_actual_payout': 'simulated_actual_payout', 'simulated_adjustment': 'simulated_adjustment',
        'simulated_payout_delta': 'simulated_payout_delta', 'simulated_attainment': 'simulated_target_attainment_pct',
    }
    for target, original in mapping.items():
        out[target] = _numeric(source, original)
    out['repeat_customers'] = _integer_inference(_numeric(source, 'distinct_customers') * _numeric(source, 'repeat_customer_ratio'))
    prior = source[['representative', 'product_class', 'date', 'distinct_customers']].copy()
    prior['date'] += pd.offsets.MonthBegin(1)
    lag = source[['representative', 'product_class', 'date']].merge(
        prior.rename(columns={'distinct_customers': 'prior_customers'}),
        on=['representative', 'product_class', 'date'], how='left', validate='one_to_one')
    # Reset the merge's row index explicitly; never infer customer loss from a future row.
    lost = pd.Series(lag.prior_customers.to_numpy(), index=source.index) * _numeric(source, 'customer_loss_rate')
    out['lost_customers'] = _integer_inference(lost)
    out['distributor_mix_summary'] = [
        f'{int(n)} distributors; concentration index {h:.3f}; mix change {m:.3f}'
        if pd.notna(n) and pd.notna(h) and pd.notna(m) else None
        for n,h,m in zip(out.distributor_count, out.distributor_concentration, out.distributor_mix_change)]
    out['channel_summary'] = [
        f'Channel diversity index {a:.3f}; subchannel diversity index {b:.3f}' if pd.notna(a) and pd.notna(b) else None
        for a,b in zip(out.channel_mix, out.subchannel_mix)]

    aligned = {}
    for name in ['PCA Reconstruction', 'EWMA Residual', 'Robust Peer Baseline', 'K-Means', 'DBSCAN', 'Autoencoder', 'Isolation Forest']:
        part = scores.loc[scores.model_name.eq(name)].set_index('observation_id')
        if name == 'PCA Reconstruction' and not source.index.isin(part.index).all():
            raise ValueError('PCA scores must cover every analytical observation')
        aligned[name] = part.reindex(source.index)
    pca = aligned['PCA Reconstruction']
    out['pca_raw_score'] = _numeric(pca, 'raw_score')
    out['pca_score_percentile'] = _numeric(pca, 'anomaly_score')
    out['pca_review_flag'] = _bool(pca.anomaly_flag)
    out['pca_threshold_exceedance'] = _bool(pca.threshold_flag)
    out['pca_raw_threshold'] = _numeric(pca, 'threshold')
    if not out.pca_score_percentile.between(0,1).all() or not np.isfinite(out.pca_raw_score).all():
        raise ValueError('PCA scores must be finite and percentiles must be in [0,1]')
    for name, prefix in [('EWMA Residual','ewma'),('Robust Peer Baseline','robust_peer'),
                         ('K-Means','kmeans'),('Autoencoder','autoencoder'),('Isolation Forest','isolation_forest')]:
        out[prefix + ('_score' if prefix=='ewma' else '_percentile')] = _numeric(aligned[name], 'anomaly_score')
    out['temporal_review_flag'] = _bool(aligned['EWMA Residual'].anomaly_flag)
    out['robust_peer_flag'] = _bool(aligned['Robust Peer Baseline'].anomaly_flag)
    out['peer_flag'] = out.robust_peer_flag
    out['kmeans_distance'] = _numeric(aligned['K-Means'], 'raw_score')
    _unique(queue, ['observation_id'], 'Clean investigation queue')
    q = queue.set_index('observation_id').reindex(source.index)
    out['kmeans_cluster'] = _numeric(q, 'K-Means cluster').astype('Int64')
    out['dbscan_cluster'] = _numeric(q, 'DBSCAN cluster').astype('Int64')
    out['dbscan_noise'] = _bool(q['DBSCAN noise']) if 'DBSCAN noise' in q else pd.NA

    contrib = pca_contributions.copy()
    if 'model' in contrib:
        contrib = contrib[contrib.model.eq('PCA Reconstruction')]
    _unique(contrib, ['observation_id','feature'], 'PCA contributions')
    contrib = contrib.sort_values(['observation_id','contribution','feature'], ascending=[True,False,True], kind='stable')
    contrib['driver_number'] = contrib.groupby('observation_id').cumcount()+1
    for n in range(1,4):
        driver = contrib[contrib.driver_number.eq(n)].set_index('observation_id')
        out[f'top_driver_{n}_feature'] = source.index.map(driver.feature)
        out[f'top_driver_{n}'] = out[f'top_driver_{n}_feature'].map(manager_feature_label)
        out[f'top_driver_{n}_contribution'] = source.index.map(driver.contribution)

    peers = _strongest(peer_explanations, 'robust_z')
    out['strongest_peer_deviation_metric'] = pd.Series(source.index.map(peers.metric),index=source.index).map(manager_feature_label)
    out['strongest_peer_deviation_value'] = source.index.map(peers.robust_z)
    out['peer_comparison_cohort'] = source.index.map(peers.cohort)
    history_columns = [m+'_history_deviation' for m in ['total_sales','total_quantity','distinct_customers','simulated_actual_payout'] if m+'_history_deviation' in source]
    if history_columns:
        history = source.reset_index(drop=True)[['observation_id']+history_columns].melt(id_vars='observation_id',var_name='metric',value_name='deviation')
        best = _strongest(history, 'deviation')
        out['strongest_history_deviation_metric'] = pd.Series(source.index.map(best.metric),index=source.index).map(manager_feature_label)
        out['strongest_history_deviation_value'] = source.index.map(best.deviation)
    else:
        out['strongest_history_deviation_metric'] = None
        out['strongest_history_deviation_value'] = np.nan
    ewma = temporal_scores[temporal_scores.model.eq('EWMA Residual')].copy()
    _unique(ewma, ['observation_id','metric'], 'EWMA details')
    strongest = _strongest(ewma, 'score')
    for output,original in [('temporal_metric_feature','metric'),('temporal_observed','observed'),
                            ('temporal_expected','expected'),('temporal_direction','direction'),
                            ('temporal_history_length','history_length'),('temporal_available','available')]:
        out[output] = source.index.map(strongest[original])
    out['temporal_metric'] = out.temporal_metric_feature.map(manager_feature_label)
    unavailable = ~_bool(out.temporal_available).fillna(False) | out.temporal_expected.isna()
    out.loc[unavailable,'temporal_direction'] = None
    out['temporal_history_length'] = pd.to_numeric(out.temporal_history_length,errors='coerce').astype('Int64')
    sales_history = ewma[ewma.metric.eq('total_sales')].set_index('observation_id')
    for output,original in [('ewma_sales_observed','observed'),('ewma_sales_expected','expected'),
                            ('ewma_sales_raw_score','score'),('ewma_sales_history_length','history_length'),('ewma_sales_available','available')]:
        out[output] = source.index.map(sales_history[original])
    rules = rule_signals.copy()
    rules['flag'] = _bool(rules.flag)
    _unique(rules, ['observation_id','rule_name'], 'Business rule signals')
    out['business_rule_flag'] = source.index.map(rules.groupby('observation_id').flag.any()).astype('boolean')
    flagged = rules[rules.flag.fillna(False)].sort_values(['observation_id','rule_name'])
    out['business_rule_summary'] = source.index.map(flagged.groupby('observation_id').explanation.agg('; '.join))
    out['number_of_supporting_signals'] = out[['temporal_review_flag','robust_peer_flag','business_rule_flag']].fillna(False).astype(int).sum(axis=1)
    high = out.pca_review_flag.fillna(False) | (out.pca_score_percentile.ge(policy['very_high_percentile']) & out.number_of_supporting_signals.ge(1))
    medium = out.pca_score_percentile.ge(policy['elevated_percentile']) | out.number_of_supporting_signals.ge(policy['multiple_support_count'])
    out['review_priority'] = np.select([high.to_numpy(bool),medium.to_numpy(bool)],['High','Medium'],default='Low')
    pca_evidence = out.pca_review_flag.fillna(False) | out.pca_threshold_exceedance.fillna(False) | out.pca_score_percentile.ge(policy['elevated_percentile'])
    out['model_agreement_summary'] = [
        ' + '.join(names) if names and names!=['PCA'] else 'PCA only' if names else 'No elevated signals'
        for names in [[name for name,flag in [('PCA',p),('Peer',r),('Temporal',t),('Business Rule',b)] if flag]
                      for p,r,t,b in zip(pca_evidence, out.robust_peer_flag.fillna(False),out.temporal_review_flag.fillna(False),out.business_rule_flag.fillna(False))]]
    out = out.reset_index(drop=True).sort_values(['pca_raw_score','observation_id'],ascending=[False,True],kind='stable').reset_index(drop=True)
    out['pca_rank'] = np.arange(1,len(out)+1)
    out['review_rank'] = out.pca_rank
    return out


def build_rep_summary_dataset(anomaly, rep_month_rollup):
    """As-of latest anomaly month; exact calendar windows, never summed unique counts."""
    a = anomaly.copy(deep=True)
    a['month'] = _month(a.month)
    r = rep_month_rollup.copy(deep=True)
    r['date'] = _month(r.date)
    _unique(r, ['representative','date'], 'Direct rep-month rollup')
    rows = []
    for rep, group in a.groupby('representative',sort=True):
        latest = group.month.max()
        latest_rows = group[group.month.eq(latest)].sort_values(['pca_score_percentile','product_class'],ascending=[False,True],kind='stable')
        strongest = latest_rows.iloc[0]
        hist = r[(r.representative==rep)&r.date.le(latest)].set_index('date')
        def window_sum(end):
            months = pd.date_range(end=end,periods=3,freq='MS')
            values = hist.total_sales.reindex(months)
            return values.sum(min_count=3)
        recent, previous = window_sum(latest), window_sum(latest-pd.offsets.MonthBegin(3))
        customers = hist.distinct_customers.reindex([latest,latest-pd.offsets.MonthBegin(3)])
        countries = set()
        for country in latest_rows.country.dropna() if 'country' in latest_rows else []:
            countries.update(str(country).split('|'))
        priority = next(p for p in ['High','Medium','Low'] if latest_rows.review_priority.eq(p).any())
        row = {'representative':rep, 'manager':strongest.get('manager'), 'team':strongest.get('team'),
               'primary_country':next(iter(countries)) if len(countries)==1 else None,
               'latest_month_available':latest.strftime('%Y-%m-%d'), 'total_observations':len(group),
               'high_priority_review_count':int(group.review_priority.eq('High').sum()),
               'medium_priority_review_count':int(group.review_priority.eq('Medium').sum()),
               'top_5_percent_review_count':int(group.pca_review_flag.sum()),
               'maximum_pca_percentile':group.pca_score_percentile.max(),
               'mean_pca_percentile':group.pca_score_percentile.mean(),
               'latest_pca_percentile':latest_rows.pca_score_percentile.max(),
               'latest_review_priority':priority, 'strongest_recent_driver':strongest.get('top_driver_1'),
               'temporal_flag_count':int(group.temporal_review_flag.sum()),
               'business_rule_flag_count':int(group.business_rule_flag.sum()),
               'peer_flag_count':int(group.peer_flag.sum()),
               'model_agreement_high_count':int(group.number_of_supporting_signals.ge(2).sum()),
               'total_sales':hist.total_sales.sum(min_count=1), 'recent_3m_sales':recent, 'prior_3m_sales':previous,
               'sales_growth_3m':(recent-previous)/abs(previous) if pd.notna(previous) and previous!=0 and pd.notna(recent) else np.nan,
               'unique_customers_latest':customers.iloc[0], 'customer_change_3m':customers.iloc[0]-customers.iloc[1]}
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_summary_dataset(benchmark, selection):
    _unique(benchmark, ['model'], 'Executed model benchmark')
    roles = {'PCA Reconstruction':'Primary anomaly ranking','K-Means':'Segmentation','DBSCAN':'Density diagnostic',
             'EWMA Residual':'Temporal specialist','Business Rules':'Transparent rule evidence',
             'Robust Peer Baseline':'Peer comparison','Autoencoder':'Reconstruction comparator',
             'Isolation Forest':'Tree-based anomaly comparator','Best Ensemble':'Benchmark comparator only',
             'Rolling Residual':'Temporal comparator','Seasonal Residual':'Seasonal comparator',
             'Change-Point / Level Shift':'Sustained-change comparator'}
    interpretations = {
        'PCA Reconstruction':'Linear reconstruction deviation from training patterns; review signal, not a probability.',
        'K-Means':'Commercial behavior groups with Euclidean distance from the assigned centroid.',
        'DBSCAN':'Density diagnostic; out-of-sample nearest-core approximation can mark many rows as noise.',
        'EWMA Residual':'Current values compared with prior exponentially weighted expectations.',
        'Business Rules':'Transparent commercial and DEMO payout checks requiring business validation.',
        'Robust Peer Baseline':'Same-period median/MAD comparisons within available peer cohorts.',
        'Autoencoder':'Nonlinear reconstruction comparator; consult the recorded convergence limitations.',
        'Isolation Forest':'Tree-based unusual-pattern ranking relative to training observations.',
        'Best Ensemble':'Combined signals evaluated on the same controlled benchmark; selection remains validation-based.',
        'Rolling Residual':'Deviation from prior rolling median and robust variation.',
        'Seasonal Residual':'Deviation from the same calendar month one year earlier, where available.',
        'Change-Point / Level Shift':'Consecutive same-direction deviations and accumulated change evidence.',
    }
    out = pd.DataFrame({'model':benchmark.model})
    out['role'] = out.model.map(roles).fillna('Benchmark comparator')
    out['manager_facing_label'] = out.model.map(MODEL_LABELS).fillna(out.model)
    primary = selection.get('recommended_model')
    out['selected_for_primary_use'] = out.model.eq(primary)
    out.loc[out.selected_for_primary_use,'role'] = 'Primary anomaly ranking'
    out.loc[out.model.eq('PCA Reconstruction')&~out.selected_for_primary_use,'role'] = 'Reconstruction comparator'
    metric_map = {'recall_at_5pct':'Recall@5%','lift_at_5pct':'Lift@5%','precision_at_5pct':'Precision@5%',
                  'pr_auc':'PR_AUC','f1':'F1','f2':'F2','stability':'stability','runtime_seconds':'runtime_seconds'}
    for target,source in metric_map.items():
        out[target] = _numeric(benchmark,source)
    out['business_interpretation'] = out.model.map(interpretations).fillna('Executed benchmark comparison signal.')
    return out


def write_dashboard_metadata(root, output_dir, datasets, run_metadata, selection, settings):
    root,output_dir = Path(root),Path(output_dir)
    def git(*args):
        try:
            return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            return None
    try:
        version = importlib.metadata.version('field-rep-incentive-anomaly-benchmark')
    except importlib.metadata.PackageNotFoundError:
        version = None
    base = datasets['dashboard_capacity_base.csv']
    schema = {}
    for name,df in datasets.items():
        schema[name] = {'row_count':len(df),'grain':DATASETS[name],
                        'columns':[{'name':c,'dtype':str(df[c].dtype),'null_count':int(df[c].isna().sum())} for c in df]}
    metadata = {
        'schema_version':'1.0.0', 'generated_timestamp':datetime.now(timezone.utc).isoformat(),
        'source_csv_sha256':run_metadata['sha256'], 'source_row_count':run_metadata['raw_rows'],
        'clean_source_row_count':run_metadata['clean_rows'], 'modeling_row_count':run_metadata['analytical_rows'],
        'source_date_range':{'start':run_metadata['date_min'],'end':run_metadata['date_max']},
        'primary_analytical_grain':'Representative × Product Class × Month',
        'dashboard_files_generated':list(datasets)+['dashboard_metadata.json'], 'datasets':schema,
        'selected_primary_anomaly_model':selection['recommended_model'], 'selected_temporal_model':selection['best_temporal'],
        'selected_capacity_forecast_methodology':sorted(base.selected_forecast_method.dropna().unique().tolist()),
        'planning_horizon':sorted(base.forecast_horizon.dropna().unique().tolist()),
        'known_poland_coverage_limitation':'Poland source coverage ends December 2018. Its May 2019 units remain ineligible; missing records are not zero demand or zero staffing.',
        'country_date_coverage':run_metadata.get('country_date_coverage'), 'feature_count':run_metadata['feature_count'],
        'seed':run_metadata['seed'], 'pipeline_version':version, 'git_commit':git('rev-parse','HEAD'),
        'git_worktree_dirty':bool(git('status','--porcelain')), 'review_priority_policy':settings,
        'percentile_scale':'0–1 throughout all five dashboard CSVs',
        'model_labels':MODEL_LABELS,
        'definitions':{
            'review_priority':'High: persisted PCA exact-5% flag, or TRAIN percentile >= very_high with >=1 support. Medium: percentile >= elevated, or >= multiple_support_count supporting families. Otherwise Low. Thresholds are presentation rules, not test-optimized.',
            'review_rank':'Snapshot-wide PCA raw-score descending, ties observation_id ascending. Retrospective display rank; never a historical model input.',
            'pca_review_flag':'Exact budget flag already executed within each train/validation/test partition, not a new global top-5% selection.',
            'pca_threshold_exceedance':'Persisted boolean exceedance of the raw-score TRAIN-reference threshold.',
            'supporting_signals':'Three supporting families: EWMA exact-budget flag, robust-peer exact-budget flag, and any executed business-rule binary flag. Count excludes PCA and correlated model comparators. Not statistical independence or certainty.',
            'transaction_value':'Mean transaction value, copied from average_transaction_value; sales is the monthly total.',
            'repeat_customers':'unique_customers × persisted repeat_customer_ratio, only when it identifies an integer count.',
            'lost_customers':'Previous calendar-month same rep/class customer count × current loss rate; null without the previous month or integer-identifying arithmetic.',
            'top_driver_contribution':'Unmodified per-feature squared reconstruction error in transformed model space, NOT a percentage. Top three exported PCA features; technical feature ids retained.',
            'strongest_deviation':'Largest absolute signed peer robust-z or personal-history deviation; original sign retained.',
            'temporal_context':'Strongest EWMA metric by persisted raw signal, with separate SALES-only observed/expected fields for charts; unavailable direction remains null.',
            'channel_summary':'Persisted diversity indices only; channel/distributor names were not retained and are not invented.',
            'latest_rep_percentile':'Maximum across product-class observations in that representative’s latest month; latest priority is the highest latest-month category.',
            'strongest_recent_driver':'Top PCA contributor of the highest-percentile product-class row in the representative’s latest month.',
            'model_agreement_high_count':'Representative-level count of observations with at least two of the three supporting signal families; not a count of confirmed outcomes.',
            'recent_3m_sales':'Latest available month plus two preceding calendar months; prior_3m_sales is the preceding three. Incomplete windows remain null. Growth=(recent-prior)/abs(prior); zero prior denominator is null.',
            'unique_customers_latest':'Direct rep-month unique count, never summed across product classes.',
            'customer_change_3m':'Latest direct rep-month unique count minus the exact three-calendar-month-earlier count; null if either is unavailable.',
            'primary_country':'Latest-month singleton country only; multi-country ambiguity remains null.',
            'fte_availability':'Base case assumes 13 observed representatives are available to the observed Germany scope; missing Poland records do not verify their actual cross-country availability.',
            'capacity_priority':'Signed modeled FTE gap with numerical tolerance, not a hiring or employment decision.',
            'forecast_scenario_bounds':'Scenario bounds reconstructed only from persisted forecast uncertainty/capacity bounds; not confidence intervals.',
        },
        'governance':['All incentive amounts remain simulated_ fields.',
                      'Manager observation files contain no controlled injection labels.',
                      'An unusual pattern is a review signal, not an automatic finding.',
                      'Model-summary metrics come from controlled benchmark labels and are not real-world detection accuracy.',
                      'Models are not fitted or rescored during dashboard export.'],
    }
    path=output_dir/'dashboard_metadata.json'
    path.write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return metadata


def _validate_dashboard(datasets):
    forbidden = {'anomaly_type','anomaly_severity','severity','anomaly_label','injected_label','injected_type','injected_severity'}
    for name,df in datasets.items():
        if df.empty:
            raise ValueError(f'{name} is empty')
        _unique(df, DATASETS[name], name)
        bad=[c for c in df if c.lower() in forbidden or 'injected' in c.lower() or 'fraud' in c.lower()]
        if bad:
            raise ValueError(f'Forbidden manager-facing fields: {bad}')
        incentive=[c for c in df if any(s in c.lower() for s in ['incentive','payout','attainment','adjustment','quota']) and not c.startswith('simulated_')]
        if incentive:
            raise ValueError(f'Incentive fields must keep simulated_ prefix: {incentive}')


def build_all_dashboard_datasets(root, output_dir=None, settings=None):
    """Read-only with respect to all inputs. Write manager files only to output_dir."""
    root=Path(root).resolve()
    destination=Path(output_dir).resolve() if output_dir else root/'data/dashboard'
    protected=[root/'data/processed',root/'data/raw',root/'artifacts']
    if any(destination==p or p in destination.parents for p in protected):
        raise ValueError('Dashboard output directory must not overwrite technical/source artifacts')
    def read(rel):
        return pd.read_csv(root/rel)
    run_metadata=json.loads((root/'artifacts/reports/extended_run_metadata.json').read_text(encoding='utf-8'))
    selection=json.loads((root/'artifacts/reports/extended_model_selection.json').read_text(encoding='utf-8'))
    config_path=root/'configs/config.yaml'
    configured=yaml.safe_load(config_path.read_text(encoding='utf-8')).get('dashboard',{}) if config_path.exists() else {}
    policy={**DEFAULT_PRIORITY,**configured,**(settings or {})}
    anomaly=build_anomaly_dashboard_dataset(read('data/processed/analytical_dataset.csv'),read('data/processed/clean_scores_long.csv'),
        read('artifacts/reports/clean_pca_reconstruction_contributions.csv'),read('artifacts/reports/clean_peer_explanations.csv'),
        read('data/processed/clean_time_series_scores.csv'),read('artifacts/reports/clean_rule_signals.csv'),
        read('artifacts/reports/clean_investigation_queue.csv'),policy)
    capacity=build_capacity_base_dataset(read('artifacts/planning/hiring_need_by_business_unit.csv'),
        read('artifacts/planning/forecast_metrics.csv'),read('artifacts/planning/capacity_assumptions.csv'),
        read('artifacts/planning/forecast_backtest.csv'),read('data/processed/planning_view.csv'),
        read('artifacts/planning/fte_allocation.csv'),read('artifacts/planning/anomaly_cleaning_sensitivity.csv'))
    datasets={
        'dashboard_anomaly_review.csv':anomaly,
        'dashboard_rep_summary.csv':build_rep_summary_dataset(anomaly,read('data/processed/rep_month_rollup.csv')),
        'dashboard_capacity_base.csv':capacity,
        'dashboard_capacity_scenarios.csv':build_capacity_scenario_dataset(read('artifacts/planning/hiring_scenarios.csv'),capacity),
        'dashboard_model_summary.csv':build_model_summary_dataset(read('artifacts/metrics/final_anomaly_model_benchmark.csv'),selection),
    }
    _validate_dashboard(datasets)
    destination.mkdir(parents=True,exist_ok=True)
    for name,df in datasets.items():
        df.to_csv(destination/name,index=False,lineterminator='\n')
    write_dashboard_metadata(root,destination,datasets,run_metadata,selection,policy)
    print('Dashboard datasets generated:',flush=True)
    for name,df in datasets.items():
        print(f'  {destination/name} — {len(df):,} rows; grain: '+ ' × '.join(DATASETS[name]),flush=True)
    print(f'  {destination / "dashboard_metadata.json"} — provenance, schemas and definitions',flush=True)
    return datasets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    build_all_dashboard_datasets(args.root,args.output_dir)


if __name__=='__main__':
    main()
