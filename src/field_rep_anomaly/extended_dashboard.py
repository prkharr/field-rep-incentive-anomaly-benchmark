"""New real-data views reuse the existing Streamlit shell and rendering helpers."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .planning.capacity import scenario


@st.cache_data
def _read(path, modified):
    return pd.read_csv(path)


def render_extended(root, render_table, section):
    def table(rel):
        p=root/rel
        return _read(str(p),p.stat().st_mtime_ns)
    meta_path=root/'artifacts/reports/extended_run_metadata.json'
    if not meta_path.exists():
        st.info('Extended execution is in progress. Refresh once the benchmark completes.')
        return
    meta=json.loads(meta_path.read_text())
    choice=json.loads((root/'artifacts/reports/extended_model_selection.json').read_text())
    st.title('Commercial review & field-force capacity')
    st.caption('Historical 2017–April 2019 commercial data · DEMO incentives · Human review required')
    page=st.sidebar.radio('Extended workspace',['Executive Overview','Model Benchmark','Anomaly Investigation','Time-Series View','Field-Force Planning','Governance / Limitations'])
    population=st.sidebar.selectbox('Scoring population',['clean','benchmark'],format_func=lambda x:'Clean commercial / DEMO incentives' if x=='clean' else 'Controlled injected benchmark')
    queue=table(f'artifacts/reports/{population}_investigation_queue.csv')
    comparison=table('artifacts/metrics/final_anomaly_model_benchmark.csv')
    planning=table('artifacts/planning/hiring_need_by_business_unit.csv')
    st.sidebar.caption('Selected architecture: '+choice['recommended_model'])
    st.sidebar.caption('Capacity planning is independent of anomaly scores.')
    if st.sidebar.button('Refresh extended artifacts'):
        st.cache_data.clear();st.rerun()
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
