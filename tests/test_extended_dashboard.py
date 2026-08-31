"""Execute every extended Streamlit section against generated artifacts."""
from pathlib import Path
import pytest


def test_all_extended_dashboard_sections():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/reports/extended_run_metadata.json').exists():
        pytest.skip('Requires executed extended benchmark artifacts')
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    assert not app.exception, [e.message for e in app.exception]
    for name in ['Model Benchmark','Anomaly Investigation','Time-Series View','Field-Force Planning','Governance / Limitations']:
        navigation=next(r for r in app.sidebar.radio if r.label=='Extended workspace')
        navigation.set_value(name).run()
        assert not app.exception, [e.message for e in app.exception]
    # Both dataset populations render the same interface, without mixing labels.
    population=next(s for s in app.sidebar.selectbox if s.label=='Scoring population')
    population.set_value('benchmark').run()
    navigation=next(r for r in app.sidebar.radio if r.label=='Extended workspace')
    navigation.set_value('Anomaly Investigation').run()
    assert not app.exception, [e.message for e in app.exception]
