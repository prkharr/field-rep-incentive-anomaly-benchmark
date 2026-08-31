"""Contract, causal-leakage, reconstruction and capacity regression tests."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import warnings
from sklearn.exceptions import ConvergenceWarning
import yaml
from threadpoolctl import threadpool_limits

from field_rep_anomaly.commercial import (
    ALIASES, GRAIN, monthly_date, load_commercial, build_population,
    add_demo_incentives, engineer_commercial, peer_signals, _calendar_history,
    evaluate_formula,
)
from field_rep_anomaly.controlled_benchmark import inject_benchmark
from field_rep_anomaly.models.isolation_forest import IsolationForestAnomaly
from field_rep_anomaly.models.autoencoder import AutoencoderAnomaly
from field_rep_anomaly.models.pca_reconstruction import PCAReconstruction
from field_rep_anomaly.extended_scoring import PercentileCalibrator, metrics, ensemble_scores
from field_rep_anomaly.temporal import temporal_scores, business_rules
from field_rep_anomaly.preprocessing import fit_preprocessor
from field_rep_anomaly.evaluation import top_fraction_flags
from field_rep_anomaly.planning.capacity import required_fte, scenario, forecast, forecast_metrics, run_planning

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def cfg():
    return yaml.safe_load((ROOT/'configs/config.yaml').read_text())['extended']


@pytest.fixture(scope='module')
def commercial_fixture():
    rows=[]
    for rep in range(5):
        for i,date in enumerate(pd.date_range('2017-01-01',periods=28,freq='MS')):
            for cls in ['A','B']:
                for customer in range(3):
                    rows.append(dict(representative=f'R{rep}',manager='Manager',team='Team',country='Country',
                                     date=date,product_class=cls,customer=f'C{customer}',product=f'P{customer}',
                                     distributor='D',channel='Retail',subchannel='Pharmacy',city=f'City{customer}',
                                     latitude=customer,longitude=customer,quantity=10+rep+i,price=5,
                                     sales=(10+rep+i)*5))
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def demo(commercial_fixture,cfg):
    base,_,_=build_population(commercial_fixture)
    return add_demo_incentives(base,cfg['demo_incentives'])


def test_monthly_date_names_numbers_and_invalid():
    d=monthly_date(pd.Series([2017,2018,2019,2019]),pd.Series(['January','Feb','3','bad']))
    assert list(d.iloc[:3])==list(pd.to_datetime(['2017-01-01','2018-02-01','2019-03-01']))
    assert pd.isna(d.iloc[3])


def test_schema_and_exact_duplicate_handling(tmp_path):
    row={c:'value' for c in ALIASES}
    row.update({'Sales':100,'Quantity':2,'Price':50,'Latitude':1,'Longitude':2,'Month':'March','Year':2019})
    p=tmp_path/'source.csv';pd.DataFrame([row,row]).to_csv(p,index=False)
    d,report=load_commercial(p)
    assert len(d)==1 and report['duplicates_removed']==1
    assert d.representative.iloc[0]=='value'
    pd.DataFrame({'wrong':[1]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='Missing Kaggle'):
        load_commercial(p)


def test_analytical_grain_and_unique_count(commercial_fixture):
    f,grains,rollups=build_population(commercial_fixture)
    assert not f.duplicated(GRAIN).any()
    assert np.isclose(f.total_sales.sum(),commercial_fixture.sales.sum())
    assert (rollups['rep_month_rollup'].distinct_customers==3).all()
    assert len(grains)==3


def test_simulated_formula_and_cold_start(demo,cfg):
    p=cfg['demo_incentives'];row=demo.iloc[0]
    assert row.simulated_target_sales==pytest.approx(p['cold_start_target']*(1+p['target_growth']))
    a=max(row.total_sales,0)/row.simulated_target_sales
    expected=p['base_incentive']+p['attainment_component']*min(a,p['attainment_cap'])+p['accelerator_amount']*max(0,a-p['accelerator_threshold'])
    assert row.simulated_expected_incentive==pytest.approx(expected)
    np.testing.assert_allclose(demo.simulated_actual_payout,demo.simulated_expected_incentive)


def test_config_formula_is_executable_but_not_arbitrary_code():
    result=evaluate_formula('base + maximum(x - 1, 0) * factor',{'base':10,'x':np.array([0,2]),'factor':5})
    np.testing.assert_allclose(result,[10,15])
    with pytest.raises(ValueError):
        evaluate_formula('__import__("os").getcwd()',{})


def test_features_no_future_leakage(demo):
    f,features=engineer_commercial(demo)
    changed=demo.copy();date=demo.date.max()
    changed.loc[changed.date==date,'total_sales']*=100
    g,_=engineer_commercial(changed)
    pd.testing.assert_frame_equal(f.loc[f.date<date,features],g.loc[g.date<date,features])
    np.testing.assert_allclose(f.loc[f.date==date,'total_sales_lag_1'],g.loc[g.date==date,'total_sales_lag_1'])


def test_calendar_lag_does_not_bridge_missing_month(demo):
    f=demo[(demo.representative=='R0')&(demo.product_class=='A')].head(3)
    f=f[f.date.dt.month!=2]
    lag=_calendar_history(f,'total_sales',lambda s:s.shift())
    assert pd.isna(lag.iloc[-1])


def test_no_label_leakage(demo,cfg):
    f=demo.copy();f['injected_anomaly_flag']=True;f['anomaly_type']='secret';f['severity']='high'
    first,features=engineer_commercial(f)
    f['injected_anomaly_flag']=False;f['anomaly_type']='other'
    second,_=engineer_commercial(f)
    pd.testing.assert_frame_equal(first[features],second[features])
    assert not set(['injected_anomaly_flag','anomaly_type','severity']) & set(features)


def test_injection_copy_budget_audit_and_train_untouched(demo,cfg):
    original=demo.copy(deep=True)
    f,a=inject_benchmark(demo,.06,42,pd.Timestamp(cfg['train_end']),pd.Timestamp(cfg['validation_end']))
    pd.testing.assert_frame_equal(demo,original)
    train=demo.date<=pd.Timestamp(cfg['train_end'])
    pd.testing.assert_frame_equal(f.loc[train,demo.columns],demo.loc[train],check_dtype=False)
    for mask in [(f.date>cfg['train_end'])&(f.date<=cfg['validation_end']), f.date>cfg['validation_end']]:
        assert f.loc[mask,'injected_anomaly_flag'].sum()==int(np.ceil(mask.sum()*.06))
    assert set(['original_value','injected_value','affected_feature','seed','severity'])<=set(a)
    assert (a.original_value!=a.injected_value).all()


def test_training_only_preprocessor():
    train=pd.DataFrame({'x':[0,1,2,3]})
    p,X=fit_preprocessor(train,{'features':['x'],'scaler':'robust','clip_outliers':False})
    center=p.pipeline.named_steps['scaler'].center_.copy()
    transformed=p.transform(pd.DataFrame({'x':[1000]}))
    np.testing.assert_array_equal(center,p.pipeline.named_steps['scaler'].center_)
    assert transformed[0,0]>100


def test_calibration_direction_and_no_saturation():
    cal=PercentileCalibrator().fit([0,0,1,2,3])
    scores=cal.transform([-100,-.001,0,.001,1,2,3,3.001,100,1000])
    assert np.all(np.diff(scores)>=0)
    assert scores[-1]>scores[-2] and ((scores>=0)&(scores<=1)).all()


@pytest.mark.parametrize('model',[IsolationForestAnomaly(n_estimators=20),PCAReconstruction(n_components=.9),AutoencoderAnomaly(hidden_layer_sizes=(8,2,8),max_iter=100)])
def test_continuous_model_reconstruction_and_persistence(model,tmp_path):
    rng=np.random.default_rng(7);X=rng.normal(size=(100,5))
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X)
    assert all(issubclass(w.category,ConvergenceWarning) for w in caught)
    scores=model.raw_score(np.array([[0,0,0,0,0],[100,100,100,100,100]]))
    assert scores.shape==(2,) and np.isfinite(scores).all()
    assert scores[1]>scores[0]
    p=tmp_path/'model.joblib';joblib.dump(model,p)
    np.testing.assert_allclose(joblib.load(p).raw_score(X),model.raw_score(X))
    if not isinstance(model,IsolationForestAnomaly):
        np.testing.assert_allclose(model.raw_score(X),model.contributions(X).mean(axis=1))


def test_isolation_forest_deterministic():
    X=np.random.default_rng(1).normal(size=(80,4))
    a=IsolationForestAnomaly(random_state=42,n_estimators=20).fit(X)
    b=IsolationForestAnomaly(random_state=42,n_estimators=20).fit(X)
    np.testing.assert_array_equal(a.raw_score(X),b.raw_score(X))


def test_temporal_expectation_and_future_invariance(demo,cfg):
    f,_=engineer_commercial(demo)
    scores,details=temporal_scores(f,cfg['temporal'])
    g=f.copy();g.loc[g.date==g.date.max(),'total_sales']*=100
    s2,d2=temporal_scores(g,cfg['temporal'])
    pd.testing.assert_frame_equal(scores[f.date<f.date.max()],s2[f.date<f.date.max()])
    target=details[(details.representative=='R0')&(details.product_class=='A')&(details.metric=='total_sales')&(details.model=='Rolling Residual')].iloc[3]
    expected=f[(f.representative=='R0')&(f.product_class=='A')].total_sales.iloc[:3].median()
    assert target.expected==expected
    seasonal=details[details.model=='Seasonal Residual']
    assert (seasonal.loc[seasonal.history_length<12,'score']==0).all()


def test_level_shift_requires_sustained_change(demo,cfg):
    f=demo[(demo.representative=='R0')&(demo.product_class=='A')].copy().reset_index(drop=True)
    f['total_quantity']=100.;f['distinct_customers']=10.;f['simulated_actual_payout']=1000.
    f['total_sales']=100.;f.loc[18:,'total_sales']=1000.
    scores,_=temporal_scores(f,cfg['temporal'])
    assert scores.loc[19:,'Change-Point / Level Shift'].max()>10
    assert scores.loc[:17,'Change-Point / Level Shift'].max()==0


def test_peer_fallback_same_month_only(demo):
    f=demo.copy();f['team']=f.representative
    med,z,pct,cohort=peer_signals(f,'total_sales',minimum=4)
    assert not cohort.str.contains('team').any()
    assert med.notna().all() and z.notna().all()
    small=f.iloc[:1]
    med,z,_,cohort=peer_signals(small,'total_sales',minimum=4)
    assert z.iloc[0]==0 and cohort.iloc[0]=='date_sparse'


def test_topk_and_ranking_metrics():
    scores=np.arange(100);truth=scores>=95
    result=metrics(truth,scores)
    assert result['Recall@5%']==1 and result['Precision@5%']==1
    assert result['Lift@5%']==20
    assert top_fraction_flags(np.ones(101),.05).sum()==6


def test_ensemble_arithmetic():
    values={'a':np.array([.1,.9]),'b':np.array([.3,.7])}
    np.testing.assert_allclose(ensemble_scores(values,{'a':3,'b':1},'weighted'),[.15,.85])
    np.testing.assert_allclose(ensemble_scores(values,{'a':3,'b':1},'maximum'),[.3,.9])
    with pytest.raises(ValueError):
        ensemble_scores(values,{'a':-1,'b':1},'weighted')


def test_business_rules_simulated_deviation(demo):
    f,_=engineer_commercial(demo)
    f.loc[0,'simulated_payout_delta_pct']=200
    scores,detail=business_rules(f)
    assert scores[0]>=.8
    assert set(['rule_name','flag','normalized_severity','explanation'])<=set(detail)


def test_fte_formula_and_scenarios():
    f=required_fte(300,100,2)
    assert f['required_fte']==3 and f['fte_gap']==1 and f['utilization']==1.5
    added=scenario(300,100,2,add_reps=1)
    assert added['remaining_gap']==0
    increase=scenario(300,100,2,demand_change=.2)
    assert increase['workload_after']==360 and increase['remaining_gap']==pytest.approx(1.6)
    with pytest.raises(ValueError):
        required_fte(100,0,2)


def test_forecasts_and_metrics_past_only():
    s=pd.Series(np.arange(1,14),index=pd.date_range('2017-01-01',periods=13,freq='MS'))
    assert forecast(s,'seasonal_naive',pd.Timestamp('2018-02-01'))==2
    assert forecast(s,'moving_average')==12
    assert forecast_metrics([10,20],[10,20])['WAPE']==0


def test_winsor_sensitivity_and_allocation_when_executed():
    p=ROOT/'artifacts/planning/hiring_need_by_business_unit.csv'
    if not p.exists():
        pytest.skip('Run extended benchmark for artifact integration checks')
    d=pd.read_csv(p)
    np.testing.assert_allclose(d.cleaning_difference,d.cleaned_required_fte-d.raw_required_fte,atol=1e-8)
    np.testing.assert_allclose(d.required_fte,d.forecast_workload/d.capacity_per_rep)
    allocation=pd.read_csv(ROOT/'artifacts/planning/fte_allocation.csv')
    np.testing.assert_allclose(allocation.groupby('representative').allocated_fte.sum(),1)


def test_executed_common_population_and_no_labels_in_clean():
    p=ROOT/'data/processed/benchmark_scores_long.csv'
    if not p.exists():
        pytest.skip('Run extended benchmark for artifact integration checks')
    scores=pd.read_csv(p);test=scores[scores.split=='test']
    assert test.groupby('model_name').observation_id.nunique().nunique()==1
    assert test.groupby('model_name').anomaly_flag.sum().nunique()==1
    clean=pd.read_csv(ROOT/'data/processed/analytical_dataset.csv')
    assert 'injected_anomaly_flag' not in clean
    manifest=json.loads((ROOT/'artifacts/models/extended/scoring_manifest.json').read_text())
    assert not {'anomaly_type','severity','injected_anomaly_flag'} & set(manifest['features'])


def test_stale_country_does_not_imply_zero_staffing(commercial_fixture,cfg,tmp_path):
    d=commercial_fixture.copy()
    stale=d[d.date<='2018-12-01'].copy()
    stale['country']='Missing in 2019'
    d=pd.concat([d,stale],ignore_index=True)
    results,_=run_planning(d,cfg,pd.Timestamp(cfg['train_end']),pd.Timestamp(cfg['validation_end']),tmp_path)
    absent=results[results.country=='Missing in 2019']
    assert not absent.planning_eligible.any()
    assert absent.fte_gap.isna().all() and absent.hiring_priority.isna().all()
    active=results[results.planning_eligible]
    assert active.allocated_current_fte.sum()==pytest.approx(5)
    scenarios=pd.read_csv(tmp_path/'artifacts/planning/hiring_scenarios.csv')
    assert 'Missing in 2019' not in set(scenarios.country)


def test_signed_log_distance_space_is_finite_and_monotone():
    f=pd.DataFrame({'x':[-100,-1,0,1,100]})
    p,X=fit_preprocessor(f,{'features':['x'],'scaler':'robust','signed_log1p':True,'clip_outliers':False})
    assert np.all(np.diff(X[:,0])>0)
    future=p.transform(pd.DataFrame({'x':[1e9,1e10]}))
    assert future[1,0]>future[0,0]>X[-1,0]
