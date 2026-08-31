"""Read-only semantic dashboard, with an explicitly separate technical workspace.

The default manager workspace reads only the five CSVs and metadata JSON in
``data/dashboard``. Controlled-injection artifacts are read only after selecting
the clearly labelled benchmark population. Neither workspace fits a model.
"""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DASHBOARD_FILES = (
    'dashboard_anomaly_review.csv',
    'dashboard_rep_summary.csv',
    'dashboard_capacity_base.csv',
    'dashboard_capacity_scenarios.csv',
    'dashboard_model_summary.csv',
    'dashboard_metadata.json',
)
PAGES = (
    'Executive Overview', 'Model Benchmark', 'Anomaly Investigation',
    'Time-Series View', 'Field-Force Planning', 'Governance / Limitations',
)


@st.cache_data
def _read(path, modified):
    return pd.read_csv(path)


def render_extended(root, render_table, section):
    """Route to manager-ready data unless the technical workspace is explicit."""
    st.title('Commercial review & field-force capacity')
    st.caption('Historical 2017–April 2019 commercial data · DEMO incentives · Human review required')
    page = st.sidebar.radio('Extended workspace', list(PAGES))
    population = st.sidebar.selectbox(
        'Scoring population', ['clean', 'benchmark'],
        format_func=lambda x: 'Clean commercial / DEMO incentives' if x == 'clean'
        else 'Technical: controlled injected benchmark',
    )
    st.sidebar.caption('Capacity planning is independent of anomaly scores.')
    if st.sidebar.button('Refresh extended artifacts'):
        st.cache_data.clear()
        st.rerun()
    if population == 'clean':
        _render_manager(root, render_table, section, page)
        return
    st.warning(
        'Technical benchmark workspace: controlled injections and their labels '
        'are artificial evaluation cases, not observations of misconduct. '
        'The default manager workspace is separate and contains no injection labels.'
    )
    _render_technical(root, render_table, section, page)


def _true(values):
    """Handle CSV booleans consistently without treating the text 'False' as true."""
    return values.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])


def _columns(frame, names):
    """Optional context is omitted, never fabricated for presentation."""
    return frame.loc[:, [name for name in names if name in frame.columns]]


def _render_manager(root, render_table, section, page):
    directory = root / 'data' / 'dashboard'
    missing = [name for name in DASHBOARD_FILES if not (directory / name).exists()]
    if missing:
        st.info(
            'Dashboard-ready datasets are not yet complete. Generate them with '
            'the extended pipeline or the dashboard-data-only builder, then refresh. '
            'This dashboard does not train models or fall back to technical data.'
        )
        st.caption('Missing: ' + ', '.join(missing))
        return
    meta = json.loads((directory / 'dashboard_metadata.json').read_text(encoding='utf-8'))

    def table(name):
        path = directory / name
        return _read(str(path), path.stat().st_mtime_ns)

    primary = meta.get('selected_primary_anomaly_model', 'PCA Reconstruction')
    st.sidebar.caption('Primary review model: ' + str(primary))
    st.sidebar.caption('Manager data source: data/dashboard only')
    if page == 'Executive Overview':
        section('Two separate business questions',
                'What should we review first? Where does modeled workload exceed capacity?')
        reps = table('dashboard_rep_summary.csv')
        queue = table('dashboard_anomaly_review.csv')
        planning = table('dashboard_capacity_base.csv')
        eligible = planning[_true(planning.eligible_for_capacity_recommendation)]
        cols = st.columns(4)
        cols[0].metric('Representatives', len(reps))
        cols[1].metric('Net commercial sales', f'{reps.total_sales.sum():,.0f}')
        cols[2].metric('High-priority observations', int(queue.review_priority.eq('High').sum()))
        cols[3].metric('Eligible planning units', len(eligible))
        st.caption(
            'Sales currency is not supplied. Review priorities describe unusual patterns, '
            'not probabilities or investigation outcomes. Incentives are simulated DEMO fields.'
        )
        st.subheader('Representative overview')
        render_table(_columns(reps, [
            'representative', 'manager', 'team', 'primary_country', 'latest_month_available',
            'high_priority_review_count', 'top_5_percent_review_count',
            'latest_pca_percentile', 'latest_review_priority', 'strongest_recent_driver',
            'total_sales', 'recent_3m_sales', 'sales_growth_3m',
        ]), height=330)
        recent = queue
        if 'source_partition' in queue:
            recent = queue[queue.source_partition.eq('test')]
        st.subheader('Highest-priority reviews — final historical period')
        render_table(_columns(recent.sort_values('review_rank'), [
            'review_rank', 'representative', 'product_class', 'month', 'sales',
            'pca_score_percentile', 'review_priority', 'top_driver_1', 'model_agreement_summary',
        ]).head(12), height=350)
        st.subheader('Modeled capacity pressure — May 2019')
        render_table(_columns(eligible.sort_values('fte_gap', ascending=False), [
            'team', 'country', 'product_class', 'allocated_fte', 'required_fte',
            'fte_gap', 'capacity_priority',
        ]).head(10), height=340)
        st.caption('Poland units remain ineligible because source coverage ends in December 2018.')
    elif page == 'Model Benchmark':
        section('Same population, frozen models',
                'Executed test metrics from controlled-injection evaluation; choices use training/validation only.')
        comparison = table('dashboard_model_summary.csv')
        render_table(_columns(comparison, [
            'model', 'manager_facing_label', 'role', 'recall_at_5pct', 'lift_at_5pct',
            'precision_at_5pct', 'pr_auc', 'f1', 'f2', 'stability', 'runtime_seconds',
            'selected_for_primary_use', 'business_interpretation',
        ]), height=480)
        st.plotly_chart(px.bar(
            comparison, x='model', y=['recall_at_5pct', 'pr_auc', 'f2'], barmode='group',
            labels={'value': 'Metric value (0–1)', 'variable': 'Metric'},
        ), use_container_width=True)
        st.info('Primary anomaly ranking: ' + str(primary) + '. Model roles are preserved from executed selection.')
        st.caption(
            'Small controlled test samples do not establish production detection performance. '
            'Stability is the executed seed-refit review-queue overlap, not a guarantee under drift.'
        )
        st.download_button('Download model summary', comparison.to_csv(index=False),
                           'dashboard_model_summary.csv', 'text/csv')
    elif page == 'Anomaly Investigation':
        section('Prioritized review queue',
                'Pattern deviation is the primary ranking; supporting signals supply review context.')
        queue = table('dashboard_anomaly_review.csv').sort_values('review_rank')
        filtered = queue.copy()
        cols = st.columns(3)
        for i, column in enumerate(['representative', 'manager', 'team', 'country', 'product_class', 'month']):
            if column not in queue:
                continue
            chosen = cols[i % 3].multiselect(
                column.replace('_', ' ').title(), sorted(queue[column].dropna().astype(str).unique()),
                key='manager_filter_' + column,
            )
            if chosen:
                filtered = filtered[filtered[column].astype(str).isin(chosen)]
        if 'source_partition' in queue:
            partitions = ['test', 'validation', 'train', 'all']
            period = st.selectbox('Period partition', [
                value for value in partitions if value == 'all' or value in queue.source_partition.unique()
            ])
            if period != 'all':
                filtered = filtered[filtered.source_partition.eq(period)]
        priorities = st.multiselect('Review priority', ['High', 'Medium', 'Low'])
        if priorities:
            filtered = filtered[filtered.review_priority.isin(priorities)]
        render_table(_columns(filtered, [
            'review_rank', 'review_priority', 'representative', 'manager', 'team', 'country',
            'product_class', 'month', 'sales', 'quantity', 'unique_customers',
            'pca_raw_score', 'pca_score_percentile', 'pca_review_flag',
            'top_driver_1', 'top_driver_1_contribution', 'top_driver_2', 'top_driver_3',
            'strongest_peer_deviation_metric', 'strongest_peer_deviation_value',
            'strongest_history_deviation_metric', 'strongest_history_deviation_value',
            'ewma_score', 'temporal_review_flag', 'business_rule_flag', 'peer_flag',
            'number_of_supporting_signals', 'model_agreement_summary', 'kmeans_cluster',
            'simulated_target', 'simulated_attainment', 'simulated_expected_incentive',
            'simulated_actual_payout', 'simulated_adjustment', 'simulated_payout_delta',
        ]), height=520)
        st.download_button('Download filtered review queue', filtered.to_csv(index=False),
                           'dashboard_anomaly_review_filtered.csv', 'text/csv')
        st.caption(
            'PCA percentiles use the frozen training reference and range from 0 to 1. '
            'Exact-budget flags and deterministic priority rules are not risk probabilities. '
            'All incentive fields are simulated. This is a read-only review aid.'
        )
    elif page == 'Time-Series View':
        section('Observed sales versus prior-only expectation',
                'Recent trend deviation supplies temporal context alongside the PCA review ranking.')
        queue = table('dashboard_anomaly_review.csv')
        cols = st.columns(2)
        rep = cols[0].selectbox('Representative', sorted(queue.representative.dropna().unique()))
        rep_rows = queue[queue.representative.eq(rep)]
        product_class = cols[1].selectbox('Product class', sorted(rep_rows.product_class.dropna().unique()))
        group = rep_rows[rep_rows.product_class.eq(product_class)].sort_values('month')
        if {'ewma_sales_observed', 'ewma_sales_expected'}.issubset(group.columns):
            fig = go.Figure()
            fig.add_scatter(x=group.month, y=group.ewma_sales_observed,
                            name='Observed sales', mode='lines+markers')
            fig.add_scatter(x=group.month, y=group.ewma_sales_expected,
                            name='Prior-only EWMA expectation', mode='lines')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Sales EWMA detail is unavailable in this semantic export; available temporal context is shown below.')
        if 'ewma_sales_raw_score' in group:
            st.plotly_chart(px.line(group, x='month', y='ewma_sales_raw_score',
                                   title='Sales-only EWMA deviation score'), use_container_width=True)
        render_table(_columns(group, [
            'month', 'sales', 'ewma_sales_observed', 'ewma_sales_expected', 'ewma_sales_raw_score',
            'ewma_sales_history_length', 'ewma_sales_available', 'ewma_score',
            'temporal_review_flag', 'temporal_available', 'temporal_metric', 'temporal_observed',
            'temporal_expected', 'temporal_direction', 'temporal_history_length',
            'pca_score_percentile', 'review_priority',
        ]))
        st.caption(
            'Sales is shown on one consistent scale across months. The observation-level temporal '
            'flag may be driven by another EWMA metric. Expectations use earlier months only; '
            'missing history stays unavailable.'
        )
    elif page == 'Field-Force Planning':
        section('Capacity scenarios, not hiring decisions',
                'May 2019 planning horizon. Select an already executed scenario; no calculation or refitting occurs here.')
        planning = table('dashboard_capacity_base.csv')
        scenarios = table('dashboard_capacity_scenarios.csv')
        eligible = planning[_true(planning.eligible_for_capacity_recommendation)]
        cols = st.columns(3)
        cols[0].metric('Eligible allocated FTE', f'{eligible.allocated_fte.sum():.2f}')
        cols[1].metric('Modeled required FTE', f'{eligible.required_fte.sum():.2f}')
        cols[2].metric('Ineligible planning units', len(planning) - len(eligible))
        st.warning(
            'Poland source records stop in December 2018. May 2019 units remain ineligible; '
            'missing records are not zero demand or zero staffing.'
        )
        render_table(_columns(planning, [
            'team', 'country', 'product_class', 'forecast_horizon', 'eligible_for_capacity_recommendation',
            'forecast_workload', 'selected_forecast_method', 'allocated_fte',
            'sustainable_workload_per_rep', 'required_fte', 'fte_gap', 'required_fte_lower',
            'required_fte_upper', 'fte_gap_lower', 'fte_gap_upper', 'capacity_priority',
        ]), height=450)
        scenario_name = st.selectbox('Planning scenario', list(scenarios.scenario_name.drop_duplicates()))
        selected = scenarios[scenarios.scenario_name.eq(scenario_name)]
        if 'scenario_description' in selected and len(selected):
            st.caption(str(selected.scenario_description.iloc[0]))
        render_table(_columns(selected, [
            'team', 'country', 'product_class', 'scenario_name', 'forecast_workload',
            'allocated_fte', 'sustainable_capacity_per_rep', 'required_fte', 'fte_gap',
            'capacity_priority', 'eligible_for_capacity_recommendation',
            'source_unit', 'target_unit', 'fte_reallocated',
        ]), height=380)
        with st.expander('Forecast and workload context'):
            render_table(_columns(planning, [
                'team', 'country', 'product_class', 'selected_forecast_method',
                'forecast_error_metric_used_for_selection', 'validation_wape', 'test_wape',
                'forecast_lower_scenario', 'forecast_upper_scenario', 'customer_load',
                'transaction_load', 'geography_load', 'product_load', 'distributor_load',
                'workload_score_raw', 'workload_score_winsorized', 'latest_observed_workload',
                'recent_workload_growth',
            ]))
        st.caption(
            'Fractional allocation prevents double-counting representatives. Capacity and scenario '
            'bounds are assumptions, not verified staffing or statistical confidence intervals.'
        )
        st.download_button('Download capacity base', planning.to_csv(index=False),
                           'dashboard_capacity_base.csv', 'text/csv')
        st.download_button('Download scenarios', scenarios.to_csv(index=False),
                           'dashboard_capacity_scenarios.csv', 'text/csv')
    else:
        section('Provenance and limitations', 'The manager workspace consumes the semantic data layer only.')
        st.warning('Review signals require human investigation. Capacity scenarios are not automated employment decisions.')
        st.markdown(
            'PCA ranks unusual commercial and simulated-incentive patterns. Supporting models '
            'provide peer, trend, rule, or segmentation context; agreement does not establish certainty. '
            'Benchmark metrics evaluate controlled examples, not verified real-world cases.\n\n'
            'All incentive amounts are DEMO simulations. Historical expectations use prior-only '
            'information, and percentiles use the frozen training reference. Capacity planning is '
            'separate from anomaly priority and requires actual availability and coverage checks.'
        )
        st.info(str(meta.get('known_poland_coverage_limitation',
                            'Poland source coverage ends in December 2018; May 2019 units are ineligible.')))
        st.json(meta)


def _render_technical(root, render_table, section, page):
    """Legacy detailed artifacts are confined to explicit benchmark mode."""
    population = 'benchmark'
    def table(rel):
        p=root/rel
        return _read(str(p),p.stat().st_mtime_ns)
    meta_path=root/'artifacts/reports/extended_run_metadata.json'
    if not meta_path.exists():
        st.info('Extended execution is in progress. Refresh once the benchmark completes.')
        return
    meta=json.loads(meta_path.read_text())
    choice=json.loads((root/'artifacts/reports/extended_model_selection.json').read_text())
    queue=table(f'artifacts/reports/{population}_investigation_queue.csv')
    comparison=table('artifacts/metrics/final_anomaly_model_benchmark.csv')
    planning=table('artifacts/planning/hiring_need_by_business_unit.csv')
    st.sidebar.caption('Selected architecture: '+choice['recommended_model'])
    if page=='Executive Overview':
        section('Two separate business questions','What should we review first? Where does modeled workload exceed capacity?')
        cols=st.columns(4)
        cols[0].metric('Actual representatives',meta['cardinalities']['representative'])
        cols[1].metric('Net commercial sales',f"{meta['sales_total']:,.0f}")
        cols[2].metric('Actual customers',meta['cardinalities']['customer'])
        scores=table(f'data/processed/{population}_scores_long.csv')
        flags=scores[(scores.model_name==choice['recommended_model'])&(scores.split=='test')].anomaly_flag.sum()
        cols[3].metric('Test-period review queue',int(flags))
        st.caption('Sales currency is not supplied. Review queue is an exact 5% investigation budget, not confirmed fraud.')
        st.subheader('Highest-priority test-period reviews')
        show=['representative','product_class','date','total_sales','selected_score','model_agreement_count','top_drivers']
        render_table(queue[queue.split=='test'][show].head(12),height=350)
        st.subheader('Highest modeled capacity pressure — May 2019 scenario')
        render_table(planning[['team','country','product_class','allocated_current_fte','required_fte','fte_gap','hiring_priority','recommendation']].head(10),height=340)
    elif page=='Model Benchmark':
        section('Same population, frozen models','Test: Jan–Apr 2019. All choices made using training/validation; controlled labels only.')
        cols=['model','Recall@5%','Lift@5%','PR_AUC','F2','stability','runtime_seconds','recommended_role']
        render_table(comparison[cols],height=480)
        st.plotly_chart(px.bar(comparison,x='model',y=['Recall@5%','PR_AUC','F2'],barmode='group'),use_container_width=True)
        st.info(f"Validation-selected: {choice['recommended_model']}. Ensemble material improvement: {choice['ensemble_material_improvement']}.")
        st.caption('Stability is seed-refit validation top-5% queue overlap; deterministic models are seed-invariant, not proven perturbation-robust. Fitted-model runtime includes bounded grid search.')
        with st.expander('Validation, correlations, overlap and clustering-only evidence'):
            render_table(table('artifacts/metrics/validation_model_benchmark.csv'))
            render_table(table('artifacts/metrics/model_score_correlations.csv'))
            render_table(table('artifacts/metrics/model_topk_overlap.csv'))
            render_table(table('artifacts/metrics/extended_clustering_benchmark.csv'))
        st.download_button('Download complete benchmark',comparison.to_csv(index=False),'final_anomaly_model_benchmark.csv','text/csv')
    elif page=='Anomaly Investigation':
        section('Prioritized review queue','Simulated payouts and benchmark labels must not be mistaken for observed compensation or misconduct.')
        filtered=queue.copy()
        cols=st.columns(3)
        for i,c in enumerate(['representative','manager','team','country','product_class','date']):
            chosen=cols[i%3].multiselect(c.replace('_',' ').title(),sorted(queue[c].astype(str).unique()),key='filter_'+c)
            if chosen:
                filtered=filtered[filtered[c].astype(str).isin(chosen)]
        horizon=st.selectbox('Period partition',['test','validation','train','all'])
        if horizon!='all':filtered=filtered[filtered.split==horizon]
        columns=['representative','manager','team','country','product_class','date','total_sales','total_quantity','distinct_customers',
                 'simulated_target_attainment_pct','simulated_expected_incentive','simulated_actual_payout','simulated_adjustment',
                 'model_agreement_count','top_drivers','temporal_observed','temporal_expected','temporal_difference']
        columns += [c for c in filtered if c.endswith(' score')]
        if population=='benchmark':columns += ['anomaly_type','severity']
        columns += ['review_status','reviewer_comments']
        render_table(filtered[columns],height=520)
        st.download_button('Download filtered review queue',filtered[columns].to_csv(index=False),f'{population}_review_queue.csv','text/csv')
        st.caption('Read-only dashboard. Reviewer status/comments are blank export fields, not a persisted workflow system.')
    elif page=='Time-Series View':
        section('Actual versus prior-only expectation','Rolling residual, EWMA, same-month last-year, and sustained-shift signals')
        ts=table(f'data/processed/{population}_time_series_scores.csv')
        cols=st.columns(4)
        rep=cols[0].selectbox('Representative',sorted(ts.representative.unique()))
        cls=cols[1].selectbox('Product class',sorted(ts.product_class.unique()))
        metric=cols[2].selectbox('Metric',sorted(ts.metric.unique()))
        method=cols[3].selectbox('Temporal model',sorted(ts.model.unique()))
        g=ts[(ts.representative==rep)&(ts.product_class==cls)&(ts.metric==metric)&(ts.model==method)].sort_values('date')
        fig=go.Figure()
        fig.add_scatter(x=g.date,y=g.observed,name='Observed',mode='lines+markers')
        fig.add_scatter(x=g.date,y=g.expected,name='Prior-only expected',mode='lines')
        high=g[g.score>=3]
        fig.add_scatter(x=high.date,y=high.observed,name='Residual ≥3 (visual screen)',mode='markers',marker=dict(color='red',size=10))
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(px.line(g,x='date',y=['score','trend_score'],title='Point/shift score and separate historical trend signal'),use_container_width=True)
        render_table(g[['date','observed','expected','residual','normalized_residual','direction','anomaly_type','history_length','available']])
    elif page=='Field-Force Planning':
        from .planning.capacity import scenario
        section('Capacity scenarios, not hiring decisions','Forecast horizon: May 2019. Assumptions require validation against calls, travel, working days and territory rules.')
        columns=['team','country','product_class','forecast_workload','current_active_reps','allocated_current_fte','current_capacity','utilization','required_fte','fte_gap','required_fte_low','required_fte_high','hiring_priority','recommendation','coverage_note']
        render_table(planning[columns],height=450)
        eligible=planning[planning.planning_eligible].copy()
        labels=eligible.team+' / '+eligible.country+' / '+eligible.product_class
        unit=st.selectbox('Scenario business unit',labels.tolist())
        row=eligible.loc[labels==unit].iloc[0]
        a,b,c=st.columns(3)
        demand=a.slider('Demand change (%)',-30,50,10)/100
        add=b.slider('Additional representative FTE',0,2,0)
        cap=c.slider('Capacity change (%)',-20,20,0)/100
        result=scenario(row.forecast_workload,row.capacity_per_rep,row.allocated_current_fte,demand,add,cap)
        render_table(pd.DataFrame([result]),height=140)
        st.caption('Independent unit scenario. Adding FTE here is a what-if, not an approved hiring action. Reallocation is a paired donor/receiver scenario in the exported table.')
        with st.expander('Raw/cleaned sensitivity, capacity assumptions and forecast accuracy'):
            render_table(table('artifacts/planning/anomaly_cleaning_sensitivity.csv'))
            render_table(table('artifacts/planning/capacity_assumptions.csv'))
            render_table(table('artifacts/planning/forecast_metrics.csv'))
        st.download_button('Download scenarios',table('artifacts/planning/hiring_scenarios.csv').to_csv(index=False),'hiring_scenarios.csv','text/csv')
    else:
        st.warning('Review signals are not fraud findings. Capacity scenarios are not automated employment decisions.')
        for title,file in [('Production data gaps','production_data_gaps.md'),('Temporal methodology','time_series_methodology.md'),('Capacity methodology','hiring_need_methodology.md')]:
            with st.expander(title,expanded=title=='Production data gaps'):
                st.markdown((root/'artifacts/reports'/file).read_text(encoding='utf-8'))
        st.json({k:meta[k] for k in ['source','raw_rows','clean_rows','date_min','date_max','analytical_rows','feature_count','train_end','validation_end','test_start','test_end','seed','runtime_seconds']})
