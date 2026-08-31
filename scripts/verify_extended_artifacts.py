"""Cross-check executed artifacts, source integrity and model/queue consistency."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from field_rep_anomaly.extended_scoring import metrics


def main():
    root=Path(__file__).resolve().parents[1]
    meta=json.loads((root/'artifacts/reports/extended_run_metadata.json').read_text())
    assert hashlib.sha256((root/'data/raw/pharma-data.csv').read_bytes()).hexdigest()==meta['sha256']
    benchmark=pd.read_csv(root/'data/processed/controlled_benchmark_dataset.csv')
    clean=pd.read_csv(root/'data/processed/analytical_dataset.csv')
    assert len(benchmark)==len(clean)==meta['analytical_rows']
    assert 'injected_anomaly_flag' not in clean
    assert not benchmark.duplicated(['representative','product_class','date']).any()
    scores=pd.read_csv(root/'data/processed/benchmark_scores_long.csv')
    final=pd.read_csv(root/'artifacts/metrics/final_anomaly_model_benchmark.csv').set_index('model')
    label=benchmark.set_index('observation_id').injected_anomaly_flag
    for name,g in scores[scores.split=='test'].groupby('model_name'):
        assert len(g)==meta['test_rows'] and g.observation_id.nunique()==len(g)
        assert g.anomaly_flag.sum()==int(np.ceil(.05*len(g)))
        result=metrics(label.reindex(g.observation_id),g.raw_score.to_numpy())
        for metric,value in result.items():
            np.testing.assert_allclose(final.loc[name,metric],value,atol=1e-10)
    assert np.isfinite(scores[['raw_score','anomaly_score','anomaly_percentile']]).all().all()
    assert scores.anomaly_score.between(0,1).all()
    pre=joblib.load(root/'artifacts/models/extended/preprocessor.joblib')
    train=clean.date<=meta['train_end']
    np.testing.assert_allclose(pre.pipeline.named_steps['imputer'].statistics_,np.nanmedian(clean.loc[train,pre.feature_names],axis=0),atol=1e-12)
    for population in ['clean','benchmark']:
        for name in ['k-means','autoencoder','pca_reconstruction']:
            with np.load(root/f'artifacts/reports/{population}_{name}_all_feature_errors.npz') as a:
                assert a['contributions'].shape==(len(clean),meta['feature_count'])
                assert np.isfinite(a['contributions']).all()
                assert len(a['observation_ids'])==len(clean)
    planning=pd.read_csv(root/'artifacts/planning/hiring_need_by_business_unit.csv')
    stale=planning.loc[~planning.planning_eligible]
    assert stale.fte_gap.isna().all() and stale.hiring_priority.isna().all()
    current=planning.loc[planning.planning_eligible]
    assert np.isfinite(current[['required_fte','fte_gap','hiring_priority']]).all().all()
    np.testing.assert_allclose(current.required_fte,current.forecast_workload/current.capacity_per_rep)
    allocation=pd.read_csv(root/'artifacts/planning/fte_allocation.csv')
    np.testing.assert_allclose(allocation.groupby('representative').allocated_fte.sum(),1)
    assert np.isclose(current.allocated_current_fte.sum(),13)
    scenarios=pd.read_csv(root/'artifacts/planning/hiring_scenarios.csv')
    transferred=scenarios[scenarios.scenario.str.startswith('Reallocate')]
    assert len(transferred)==2
    assert abs((transferred.capacity_after-transferred.capacity_before).sum())<1e-9
    tests=ET.parse(root/'artifacts/reports/extended_tests.xml').getroot()[0].attrib
    assert int(tests['failures'])==0 and int(tests['errors'])==0 and int(tests['skipped'])==0
    plots=list((root/'artifacts/plots').glob('extended_*.png'))
    for p in plots:
        with Image.open(p) as im:im.verify()
    result={'artifact_checks':'passed','source_sha256':meta['sha256'],'tests':tests,
            'models':len(final),'test_rows_per_model':meta['test_rows'],
            'eligible_planning_units':len(current),'stale_units_excluded':len(stale),
            'plot_files_verified':len(plots),'runtime_seconds':meta['runtime_seconds']}
    (root/'artifacts/reports/extended_artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
