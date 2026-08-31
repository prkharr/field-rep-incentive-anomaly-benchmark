"""Exercise semantic-only manager pages and explicit technical separation."""
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def semantic_only_root(tmp_path):
    """A distributable manager layer with no raw, processed, or technical files."""
    directory = tmp_path / 'data' / 'dashboard'
    directory.mkdir(parents=True)
    anomaly = pd.DataFrame([
        {
            'representative': 'Example representative', 'manager': 'Example manager',
            'team': 'Example team', 'country': 'Germany', 'product_class': 'Example class',
            'month': month, 'source_partition': 'test', 'sales': sales,
            'pca_raw_score': score, 'pca_score_percentile': score, 'pca_review_flag': score > .95,
            'pca_rank': rank, 'review_rank': rank, 'review_priority': priority,
            'top_driver_1': 'Sales pattern', 'model_agreement_summary': 'PCA only',
            'ewma_sales_observed': sales, 'ewma_sales_expected': expected,
            'ewma_sales_raw_score': 1.0, 'ewma_sales_history_length': 12,
            'ewma_sales_available': True, 'ewma_score': .6, 'temporal_review_flag': False,
            'temporal_available': True, 'temporal_history_length': 12,
        }
        for month, sales, expected, score, rank, priority in [
            ('2019-03-01', 100.0, 90.0, .7, 2, 'Low'),
            ('2019-04-01', 200.0, 110.0, .99, 1, 'High'),
        ]
    ])
    anomaly.to_csv(directory / 'dashboard_anomaly_review.csv', index=False)
    pd.DataFrame([{
        'representative': 'Example representative', 'manager': 'Example manager',
        'team': 'Example team', 'primary_country': 'Germany',
        'latest_month_available': '2019-04-01', 'total_observations': 2,
        'high_priority_review_count': 1, 'top_5_percent_review_count': 1,
        'latest_pca_percentile': .99, 'latest_review_priority': 'High', 'total_sales': 300.0,
    }]).to_csv(directory / 'dashboard_rep_summary.csv', index=False)
    base = pd.DataFrame([
        {
            'team': 'Example team', 'country': country, 'product_class': 'Example class',
            'forecast_horizon': '2019-05-01', 'eligible_for_capacity_recommendation': eligible,
            'forecast_workload': 10.0 if eligible else None,
            'allocated_fte': 1.0 if eligible else None,
            'required_fte': .8 if eligible else None, 'fte_gap': -.2 if eligible else None,
            'capacity_priority': 'Balanced' if eligible else 'Ineligible / Stale Coverage',
        }
        for country, eligible in [('Germany', True), ('Poland', False)]
    ])
    base.to_csv(directory / 'dashboard_capacity_base.csv', index=False)
    scenarios = base.assign(scenario_name='Base', scenario_description='Executed base assumptions.')
    scenarios.to_csv(directory / 'dashboard_capacity_scenarios.csv', index=False)
    pd.DataFrame([{
        'model': 'PCA Reconstruction', 'manager_facing_label': 'Pattern deviation',
        'role': 'Primary anomaly ranking', 'recall_at_5pct': .4, 'lift_at_5pct': 8.0,
        'precision_at_5pct': .5, 'pr_auc': .4, 'f1': .4, 'f2': .4,
        'stability': 1.0, 'runtime_seconds': .1, 'selected_for_primary_use': True,
        'business_interpretation': 'Ranks unusual patterns for human review.',
    }]).to_csv(directory / 'dashboard_model_summary.csv', index=False)
    (directory / 'dashboard_metadata.json').write_text(json.dumps({
        'selected_primary_anomaly_model': 'PCA Reconstruction',
        'selected_temporal_model': 'EWMA Residual',
        'known_poland_coverage_limitation': 'Poland coverage ends in December 2018.',
    }), encoding='utf-8')
    return tmp_path


def _semantic_app(root):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_string(
        'from pathlib import Path\n'
        'import streamlit as st\n'
        'from field_rep_anomaly.extended_dashboard import render_extended\n'
        'def table(frame, height=430):\n'
        '    st.dataframe(frame, height=height)\n'
        'def section(title, caption):\n'
        '    st.subheader(title)\n'
        '    st.caption(caption)\n'
        f'render_extended(Path({str(root)!r}), table, section)\n',
        default_timeout=60,
    ).run()


def test_manager_pages_read_only_dashboard_datasets(semantic_only_root, monkeypatch):
    pytest.importorskip('streamlit')
    from field_rep_anomaly import extended_dashboard

    reads = []
    expected = set(extended_dashboard.DASHBOARD_FILES) - {'dashboard_metadata.json'}

    def read_semantic_only(path, modified):
        path = Path(path)
        assert path.parent == semantic_only_root / 'data' / 'dashboard'
        assert path.name in expected
        reads.append(path.name)
        return pd.read_csv(path)

    monkeypatch.setattr(extended_dashboard, '_read', read_semantic_only)
    app = _semantic_app(semantic_only_root)
    assert not app.exception, [e.message for e in app.exception]
    assert next(s for s in app.sidebar.selectbox if s.label == 'Scoring population').value == 'clean'
    for page in extended_dashboard.PAGES[1:]:
        next(r for r in app.sidebar.radio if r.label == 'Extended workspace').set_value(page).run()
        assert not app.exception, [e.message for e in app.exception]
        for table in app.dataframe:
            assert not {'injected_label', 'anomaly_label', 'injected_type', 'injected_severity'}.intersection(table.value.columns)
    assert set(reads) == expected


def test_missing_semantic_layer_does_not_fall_back_to_technical_data(tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from field_rep_anomaly import extended_dashboard

    def fail_read(*args):
        pytest.fail('Missing semantic data must not trigger technical artifact reads')

    monkeypatch.setattr(extended_dashboard, '_read', fail_read)
    app = _semantic_app(tmp_path)
    assert not app.exception, [e.message for e in app.exception]
    assert any('Dashboard-ready datasets are not yet complete' in item.value for item in app.info)


def test_all_extended_dashboard_sections():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    if not (root/'data/dashboard/dashboard_metadata.json').exists():
        pytest.skip('Requires executed dashboard-ready artifacts')
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    assert not app.exception, [e.message for e in app.exception]
    for name in ['Model Benchmark','Anomaly Investigation','Time-Series View','Field-Force Planning','Governance / Limitations']:
        navigation=next(r for r in app.sidebar.radio if r.label=='Extended workspace')
        navigation.set_value(name).run()
        assert not app.exception, [e.message for e in app.exception]
    # Technical labels can only be reached by explicitly changing population.
    if not (root/'artifacts/reports/extended_run_metadata.json').exists():
        return
    population=next(s for s in app.sidebar.selectbox if s.label=='Scoring population')
    population.set_value('benchmark').run()
    for name in ['Executive Overview','Model Benchmark','Anomaly Investigation','Time-Series View','Field-Force Planning','Governance / Limitations']:
        navigation=next(r for r in app.sidebar.radio if r.label=='Extended workspace')
        navigation.set_value(name).run()
        assert not app.exception, [e.message for e in app.exception]
        assert any('Technical benchmark workspace' in item.value for item in app.warning)
