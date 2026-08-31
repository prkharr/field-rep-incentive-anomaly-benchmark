"""Executed real-data anomaly benchmark and independent planning extension."""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import platform
import shutil
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .commercial import load_commercial, build_population, add_demo_incentives, engineer_commercial, GRAIN
from .controlled_benchmark import inject_benchmark, TYPES
from .evaluation import top_fraction_flags
from .extended_scoring import PercentileCalibrator, metrics, selection_utility, stability, ensemble_scores
from .models.kmeans import KMeansClusteringModel
from .models.dbscan import DBSCANClusteringModel
from .models.isolation_forest import IsolationForestAnomaly
from .models.autoencoder import AutoencoderAnomaly
from .models.pca_reconstruction import PCAReconstruction
from .preprocessing import fit_preprocessor
from .temporal import temporal_scores, robust_peer, business_rules, TEMPORAL_NAMES
from .tuning import clustering_metrics
from .planning import run_planning

FAMILIES = {'K-Means':'clustering', 'DBSCAN':'density', 'Isolation Forest':'isolation',
            'Autoencoder':'reconstruction', 'PCA Reconstruction':'reconstruction',
            'Robust Peer Baseline':'peer', 'Business Rules':'rules',
            **{n:'temporal' for n in TEMPORAL_NAMES}}
EXPLAIN = {'K-Means':'Squared standardized centroid-distance contributions',
           'DBSCAN':'Noise/core assignment and training-neighbor distance',
           'Isolation Forest':'Training-median feature ablation for top review rows; peer/history deviations',
           'Autoencoder':'Per-feature squared reconstruction error',
           'PCA Reconstruction':'Per-feature squared reconstruction error',
           'Robust Peer Baseline':'Same-month cohort median/MAD/percentile with fallback',
           'Business Rules':'Named DEMO incentive/commercial rules',
           **{n:'Prior-only observed/expected residual and history length' for n in TEMPORAL_NAMES}}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x)), encoding='utf-8')


def write_csv(root, relative, frame):
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def model_raw(name, model, X):
    if name == 'K-Means':
        return model.distances(X)
    if name == 'DBSCAN':
        # Retain the existing density-distance/core model, but avoid ECDF saturation
        # when ranking new rows. Noise still sorts above every core/border row.
        d = model.neighbor_distances(X)
        return .75*(model.predict(X)==-1) + .25*d/(1+d)
    return model.raw_score(X)


def make_model(name, params, seed):
    if name == 'K-Means':
        return KMeansClusteringModel(random_state=seed, n_init=20, **params)
    if name == 'DBSCAN':
        return DBSCANClusteringModel(**params)
    if name == 'Isolation Forest':
        return IsolationForestAnomaly(random_state=seed, **params)
    if name == 'Autoencoder':
        return AutoencoderAnomaly(random_state=seed, **params)
    return PCAReconstruction(random_state=seed, **params)


def candidates(name, X, cfg):
    if name == 'K-Means':
        return [{'n_clusters':k} for k in range(2,13)]
    if name == 'DBSCAN':
        results = []
        for n in [5,8,12,16]:
            d = NearestNeighbors(n_neighbors=n).fit(X).kneighbors(X)[0][:,-1]
            results += [{'eps':float(np.quantile(d,q)), 'min_samples':n} for q in [.7,.82,.9,.96]]
        return results
    if name == 'Isolation Forest':
        params = cfg['isolation_forest']
        return [dict(zip(params, values)) for values in itertools.product(*params.values())]
    if name == 'Autoencoder':
        return [{'hidden_layer_sizes':tuple(a), 'max_iter':cfg['autoencoder']['max_iter'], 'alpha':cfg['autoencoder']['alpha']} for a in cfg['autoencoder']['architectures']]
    return [{'n_components':v} for v in cfg['pca_variance']]


def tune_models(train_X, validation_X, truth, cfg, validation_clean_X):
    fitted, selected_params, runtimes, trials, convergence = {},{},{},[],[]
    seed = cfg['seed']
    for name in ['K-Means','DBSCAN','Isolation Forest','Autoencoder','PCA Reconstruction']:
        start = time.perf_counter()
        best, best_fallback = None, None
        for params in candidates(name, train_X, cfg):
            tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = make_model(name,params,seed).fit(train_X)
            for w in caught:
                convergence.append({'model':name,'parameters':str(params),'warning':str(w.message)})
            raw = model_raw(name,model,validation_X)
            m = metrics(truth,raw,cfg['review_fraction'])
            eligible, utility = True, selection_utility(m)
            quality = {}
            if name in ['K-Means','DBSCAN']:
                labels = model.predict(validation_clean_X)
                train_labels = model.labels_
                counts = pd.Series(train_labels[train_labels>=0]).value_counts()/len(train_X)
                quality = clustering_metrics(validation_clean_X, labels, 1200, seed)
                noise = float(np.mean(labels<0))
                train_noise = float(np.mean(train_labels<0))
                eligible = len(counts)>=2 and counts.min()>=.01 and counts.max()<=.9 and train_noise<=.55
                sil = quality.get('silhouette_score',np.nan)
                utility = (float(sil) if np.isfinite(sil) else -1) * .6 + (1-float(counts.max()) if len(counts) else 0)*.3 - noise*.1
                quality.update(noise_pct=noise*100, train_noise_pct=train_noise*100,
                               largest_cluster_pct=float(counts.max()*100) if len(counts) else 0)
            trial = {'model':name,'parameters':json.dumps(params),'eligible':eligible,'selection_utility':utility,
                     'runtime_seconds':time.perf_counter()-tick,**m,**quality}
            trials.append(trial)
            candidate = (utility, model, params)
            if best_fallback is None or utility>best_fallback[0]:
                best_fallback = candidate
            if eligible and (best is None or utility>best[0]):
                best = candidate
        if best is None:
            best = best_fallback
            convergence.append({'model':name,'parameters':str(best[2]),'warning':'No configuration passed cluster-balance rules; retained best diagnostic model, not recommended for segmentation.'})
        fitted[name],selected_params[name],runtimes[name] = best[1],best[2],time.perf_counter()-start
        print(f'{name}: selected {best[2]} ({runtimes[name]:.1f}s including search)', flush=True)
    return fitted,selected_params,runtimes,pd.DataFrame(trials),convergence


def score_population(frame, X, fitted, cfg):
    raw = {name:model_raw(name,m,X) for name,m in fitted.items()}
    tick=time.perf_counter()
    temporal, detail = temporal_scores(frame,cfg['temporal'])
    family_time = (time.perf_counter()-tick)/len(TEMPORAL_NAMES)
    timing = {n:family_time for n in TEMPORAL_NAMES}
    raw.update({n:temporal[n].to_numpy() for n in TEMPORAL_NAMES})
    tick=time.perf_counter()
    raw['Robust Peer Baseline'], peer = robust_peer(frame,cfg['minimum_peer_count'])
    timing['Robust Peer Baseline']=time.perf_counter()-tick
    tick=time.perf_counter()
    raw['Business Rules'],rules = business_rules(frame)
    timing['Business Rules']=time.perf_counter()-tick
    return raw,detail,peer,rules,timing


def choose_ensemble(percentiles, val, truth, base_metrics, cfg):
    best_temporal=max(TEMPORAL_NAMES,key=lambda n:selection_utility(base_metrics[n]))
    reconstruction=max(['Autoencoder','PCA Reconstruction','K-Means'],key=lambda n:selection_utility(base_metrics[n]))
    weights={}
    for name, weight in cfg['ensemble']['weights'].items():
        weights[best_temporal if name=='temporal' else reconstruction if name=='reconstruction' else name]=weight
    # Remove near-identical non-rule rankings before ensemble search.
    dropped=[]
    for a,b in itertools.combinations(list(weights),2):
        if a in dropped or b in dropped:
            continue
        corr=stability(percentiles[a][val],percentiles[b][val])[0]
        if corr>cfg['ensemble']['correlation_limit']:
            loser=a if selection_utility(base_metrics[a])<selection_utility(base_metrics[b]) else b
            if loser!='Business Rules':
                dropped.append(loser)
    weights={n:w for n,w in weights.items() if n not in dropped}
    options=[]
    for kind in ['equal_percentile','rank_average','consensus','maximum','weighted']:
        variants=[weights]
        if kind=='weighted':
            variants += [{n:w*(2 if n==boost else 1) for n,w in weights.items()} for boost in weights]
        for w in variants:
            scores=ensemble_scores(percentiles,w,kind)
            m=metrics(truth,scores[val],cfg['review_fraction'])
            options.append({'kind':kind,'weights':json.dumps(w),'selection_utility':selection_utility(m),**m})
    comparison=pd.DataFrame(options).sort_values('selection_utility',ascending=False)
    best=comparison.iloc[0]
    return best['kind'],json.loads(best.weights),comparison,{'best_temporal':best_temporal,'reconstruction_component':reconstruction,'dropped_correlated':dropped}


def run(root, input_path, config_path):
    start=time.perf_counter()
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))['extended']
    train_end,validation_end=pd.Timestamp(cfg['train_end']),pd.Timestamp(cfg['validation_end'])
    # Preserve overlapping legacy evidence once, without deleting any existing files.
    for rel in ['data/processed/analytical_dataset.csv','data/processed/rep_risk_summary.csv','artifacts/metrics/model_selection_contributions.csv']:
        path=root/rel
        archive=root/'artifacts/legacy'/path.name
        if path.exists() and not archive.exists():
            archive.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,archive)
    d,audit=load_commercial(input_path)
    base,grain,rollups=build_population(d)
    demo=add_demo_incentives(base,cfg['demo_incentives'])
    clean,features=engineer_commercial(demo,cfg['minimum_peer_count'],cfg['demo_incentives'])
    train=(clean.date<=train_end).to_numpy()
    val=((clean.date>train_end)&(clean.date<=validation_end)).to_numpy()
    test=(clean.date>validation_end).to_numpy()
    if min(train.sum(),val.sum(),test.sum())==0:
        raise ValueError('Configured chronological split has an empty partition')
    clean['split']=np.select([train,val],['train','validation'],default='test')
    bench_base,injections=inject_benchmark(demo,cfg['injection_rate'],cfg['seed'],train_end,validation_end)
    bench,_=engineer_commercial(bench_base,cfg['minimum_peer_count'],cfg['demo_incentives'])
    bench['split']=clean['split']
    assert bench.observation_id.equals(clean.observation_id)
    np.testing.assert_allclose(clean.loc[train,features],bench.loc[train,features],equal_nan=True)
    # All-missing/constant TRAIN features cannot be meaningfully scaled or learned.
    usable=[c for c in features if clean.loc[train,c].nunique()>1]
    dropped=[c for c in features if c not in usable]
    # Keep DEMO payout deltas and adjustments despite constant clean TRAIN values.
    # Their scale is one monetary/percentage unit after robust scaling; this explicit
    # choice is audited and motivates comparison against a rule-free commercial family.
    for c in ['simulated_adjustment','simulated_payout_delta','simulated_payout_delta_pct']:
        if c in dropped:
            usable.append(c)
            dropped.remove(c)
    pre,Xtrain=fit_preprocessor(clean.loc[train],{'features':usable,'scaler':'robust','clip_outliers':cfg['preprocessing_clip'], 'signed_log1p':cfg['signed_log1p']})
    Xclean,Xbench=pre.transform(clean),pre.transform(bench)
    print(f'Real rows={len(d)}; analytical={len(clean)}; model features={len(usable)}; split={train.sum()}/{val.sum()}/{test.sum()}',flush=True)
    models,params,timing,trials,model_warnings=tune_models(Xtrain,Xbench[val],bench.loc[val,'injected_anomaly_flag'].to_numpy(),cfg,Xclean[val])
    raw,temporal,peer,rules,family_time=score_population(bench,Xbench,models,cfg)
    raw_clean,temporal_clean,peer_clean,rules_clean,_=score_population(clean,Xclean,models,cfg)
    timing.update(family_time)
    calibrators={n:PercentileCalibrator().fit(raw_clean[n][train]) for n in raw}
    percentiles={n:calibrators[n].transform(v) for n,v in raw.items()}
    clean_percentiles={n:calibrators[n].transform(v) for n,v in raw_clean.items()}
    validation_metrics={n:metrics(bench.loc[val,'injected_anomaly_flag'],s[val],cfg['review_fraction']) for n,s in raw.items()}
    corr=pd.DataFrame({n:s[val] for n,s in raw.items()}).corr(method='spearman')
    overlap=corr.copy()
    for a in raw:
        for b in raw:
            overlap.loc[a,b]=stability(raw[a][val],raw[b][val])[1]
    ensemble_kind,ensemble_weights,ensemble_table,choices=choose_ensemble(percentiles,val,bench.loc[val,'injected_anomaly_flag'],validation_metrics,cfg)
    ensemble_tick=time.perf_counter()
    raw['Best Ensemble']=ensemble_scores(percentiles,ensemble_weights,ensemble_kind)
    raw_clean['Best Ensemble']=ensemble_scores(clean_percentiles,ensemble_weights,ensemble_kind)
    timing['Best Ensemble']=time.perf_counter()-ensemble_tick+sum(timing[n] for n in ensemble_weights)
    params['Best Ensemble']={'kind':ensemble_kind,'weights':ensemble_weights}
    calibrators['Best Ensemble']=PercentileCalibrator().fit(raw_clean['Best Ensemble'][train])
    percentiles['Best Ensemble']=calibrators['Best Ensemble'].transform(raw['Best Ensemble'])
    clean_percentiles['Best Ensemble']=calibrators['Best Ensemble'].transform(raw_clean['Best Ensemble'])
    validation_metrics['Best Ensemble']=metrics(bench.loc[val,'injected_anomaly_flag'],raw['Best Ensemble'][val],cfg['review_fraction'])
    # Seed stability assessed on VALIDATION before final selection, never on test labels.
    stability_rows=[]
    alternate_raw=[]
    for seed in [cfg['seed']+1,cfg['seed']+2]:
        alternative={}
        for name in models:
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                repeat=make_model(name,params[name],seed).fit(Xtrain)
            for w in caught:
                model_warnings.append({'model':name,'parameters':f'stability seed={seed}','warning':str(w.message)})
            alternative[name]=model_raw(name,repeat,Xbench[val])
            sc,ov=stability(raw[name][val],alternative[name])
            stability_rows.append({'model':name,'seed':seed,'score_correlation':sc,'top5_queue_overlap':ov,
                                   'runtime_seconds':time.perf_counter()-tick,'method':'refit TRAIN with different seed; score same validation'})
        for n in raw:
            if n not in models and n!='Best Ensemble':
                alternative[n]=raw[n][val]
                stability_rows.append({'model':n,'seed':seed,'score_correlation':1.0,'top5_queue_overlap':1.0,'runtime_seconds':0,'method':'deterministic method; seed invariant, not perturbation robustness'})
        alt_pct={n:calibrators[n].transform(s) for n,s in alternative.items()}
        en=ensemble_scores(alt_pct,ensemble_weights,ensemble_kind)
        sc,ov=stability(raw['Best Ensemble'][val],en)
        stability_rows.append({'model':'Best Ensemble','seed':seed,'score_correlation':sc,'top5_queue_overlap':ov,'runtime_seconds':0,'method':'component seed refits; frozen calibration/weights'})
    stable=pd.DataFrame(stability_rows)
    stability_mean=stable.groupby('model').top5_queue_overlap.mean().to_dict()
    # 90% benchmark utility + 10% seed queue stability, all validation-only.
    utilities={n:.9*selection_utility(m)+.1*stability_mean[n] for n,m in validation_metrics.items()}
    best_single=max((n for n in utilities if n not in ['Best Ensemble','Business Rules']),key=utilities.get)
    em,sm=validation_metrics['Best Ensemble'],validation_metrics[best_single]
    material=(em['Recall@5%']>=sm['Recall@5%']+cfg['ensemble']['material_recall_gain']
              and em['PR_AUC']>=sm['PR_AUC']+cfg['ensemble']['material_pr_auc_gain']
              and stability_mean['Best Ensemble']>=stability_mean[best_single]-.05)
    selected='Best Ensemble' if material else best_single
    choice={**choices,'best_single':best_single,'recommended_model':selected,'ensemble_material_improvement':bool(material),
            'best_interpretable':max(['K-Means','Robust Peer Baseline','Business Rules'],key=utilities.get),
            'validation_ensemble_recall_gain':em['Recall@5%']-sm['Recall@5%'],
            'validation_ensemble_pr_auc_gain':em['PR_AUC']-sm['PR_AUC'],
            'selection_basis':'Validation ranking utility/stability; fixed material gains. Final TEST not used for selection.',
            'ensemble_kind':ensemble_kind,'ensemble_weights':ensemble_weights}
    # Freeze selection before computing any final-test model comparison.
    dump(root/'artifacts/reports/extended_model_selection.json',choice)
    corr=pd.DataFrame({n:s[val] for n,s in raw.items()}).corr(method='spearman')
    overlap=corr.copy()
    for a in raw:
        for b in raw:
            overlap.loc[a,b]=stability(raw[a][val],raw[b][val])[1]
    comparison=[]
    for name,scores in raw.items():
        comparison.append({'model':name,'model_family':FAMILIES.get(name,'ensemble'),
                           'parameters':json.dumps(params.get(name,cfg['temporal'] if name in TEMPORAL_NAMES else {})),
                           **metrics(bench.loc[test,'injected_anomaly_flag'],scores[test],cfg['review_fraction']),
                           'stability':stability_mean[name],'runtime_seconds':timing[name],
                           'explainability':EXPLAIN.get(name,'Weighted component signals'),
                           'operational_usefulness':'Prioritized review with human assessment',
                           'recommended_role':'Primary review ranking' if name==selected else ('Temporal specialist' if name==choices['best_temporal'] else 'Diagnostic comparator')})
    comparison=pd.DataFrame(comparison).sort_values('PR_AUC',ascending=False)
    print('Frozen validation choice:',selected,'; evaluating held-out test',flush=True)
    # Out-of-sample K-Means Euclidean-distance and training-inertia invariants.
    km=models['K-Means']; labels=km.predict(Xbench)
    manual=np.sqrt(((Xbench-km.cluster_centers_[labels])**2).sum(axis=1))
    sklearn_dist=km.model.transform(Xbench)[np.arange(len(Xbench)),labels]
    np.testing.assert_allclose(raw['K-Means'],manual,rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(manual,sklearn_dist,rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(np.sum(km.distances(Xtrain)**2),km.model.inertia_,rtol=1e-10)
    dump(root/'artifacts/reports/extended_kmeans_distance_validation.json',{
        'metric':'Euclidean L2 in TRAIN-fitted robust-scaled feature space',
        'all_assigned_centroids_nearest':bool(np.array_equal(labels,np.argmin(km.model.transform(Xbench),axis=1))),
        'manual_max_error':float(abs(manual-raw['K-Means']).max()),
        'sklearn_max_error':float(abs(manual-sklearn_dist).max()),
        'train_inertia_error':float(abs(np.sum(km.distances(Xtrain)**2)-km.model.inertia_))})
    clusters=[]
    for name in ['K-Means','DBSCAN']:
        lab=models[name].predict(Xclean[test])
        clusters.append({'model':name,**clustering_metrics(Xclean[test],lab,1200,cfg['seed']),
                         'noise_pct':float((lab<0).mean()*100),'clusters':len(set(lab)-{-1}),
                         'stability':stability_mean[name], 'population':'clean test; selection used validation only'})
        profiles=clean.loc[test, ['total_sales','total_quantity','distinct_customers','simulated_actual_payout']].copy()
        profiles['cluster']=lab
        write_csv(root,f'artifacts/metrics/extended_cluster_profiles_{name.lower().replace("-", "")}.csv',profiles.groupby('cluster').agg(['mean','count']).reset_index())
    cluster_table=pd.DataFrame(clusters)
    ctrial=trials[trials.model.isin(['K-Means','DBSCAN']) & trials.eligible]
    choice['best_segmentation']=ctrial.sort_values('selection_utility',ascending=False).iloc[0].model if len(ctrial) else 'Neither passed balance rules'
    dump(root/'artifacts/reports/extended_model_selection.json',choice)
    # Persistence includes the feature allowlist, TRAIN calibration and scoring policy.
    model_dir=root/'artifacts/models/extended'; model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(pre,model_dir/'preprocessor.joblib')
    for name,model in models.items():
        joblib.dump(model,model_dir/(name.lower().replace(' ','_')+'.joblib'))
    auto=models['Autoencoder'].model
    dump(model_dir/'autoencoder_training.json',{'iterations':auto.n_iter_,'max_iter':auto.max_iter,
         'loss_curve':auto.loss_curve_,'internal_train_validation_scores':auto.validation_scores_,
         'best_internal_train_validation_score':auto.best_validation_score_,
         'iteration_limit_reached':bool(auto.n_iter_>=auto.max_iter),
         'note':'Early stopping holdout is a random subset of TRAIN only; never chronological validation/test.'})
    dump(model_dir/'pca_training.json',{'components':models['PCA Reconstruction'].model.n_components_,
         'explained_variance_ratio':models['PCA Reconstruction'].model.explained_variance_ratio_.tolist()})
    joblib.dump(calibrators,model_dir/'calibrators.joblib')
    dump(model_dir/'scoring_manifest.json',{'features':usable,'config':cfg,'choice':choice,'train_end':str(train_end),
                                         'dbscan_note':'Nearest-core out-of-sample assignment; continuous noise-prioritized kNN distance, not refit DBSCAN',
                                         'calibration':'TRAIN ECDF plotting positions with bounded monotone tails',
                                         'threshold_note':'Raw TRAIN 95th-percentile exceedance plus separate exact 5% split review budget'})
    write_csv(root,'artifacts/models/extended/clean_history_reference.csv',clean[['observation_id']+GRAIN+['total_sales','total_quantity','distinct_customers','simulated_actual_payout']])
    model_sizes={n:(model_dir/(n.lower().replace(' ','_')+'.joblib')).stat().st_size for n in models}
    model_sizes['Best Ensemble']=sum(model_sizes.get(n,0) for n in ensemble_weights)
    comparison['model_size_bytes']=comparison.model.map(model_sizes).fillna(0).astype(int)
    # Sensitivity: fixed fitted models, no retuning/reselection; lower prevalence independent seed.
    sensitivity=[]
    low_audits=[]
    experiments=[('primary',bench,raw,cfg['seed'])]
    for sensitivity_seed in [cfg['seed']+101,cfg['seed']+102,cfg['seed']+103]:
        low_base,la=inject_benchmark(demo,cfg['sensitivity_rate'],sensitivity_seed,train_end,validation_end)
        low_audits.append(la)
        low,_=engineer_commercial(low_base,cfg['minimum_peer_count'],cfg['demo_incentives'])
        low_raw,_,_,_,_=score_population(low,pre.transform(low),models,cfg)
        low_pct={n:calibrators[n].transform(v) for n,v in low_raw.items()}
        low_raw['Best Ensemble']=ensemble_scores(low_pct,ensemble_weights,ensemble_kind)
        experiments.append(('low_prevalence',low,low_raw,sensitivity_seed))
    low_audit=pd.concat(low_audits,ignore_index=True)
    for population,frame,values,experiment_seed in experiments:
        for n,s in values.items():
            sensitivity.append({'experiment':population,'seed':experiment_seed,'model':n,'prevalence':float(frame.loc[test,'injected_anomaly_flag'].mean()),
                                **metrics(frame.loc[test,'injected_anomaly_flag'],s[test],cfg['review_fraction'])})
    type_metrics=[]
    for n,s in raw.items():
        top=top_fraction_flags(s[test],cfg['review_fraction'])
        for column,values in [('anomaly_type',[a[0] for a in TYPES]),('severity',['low','medium','high'])]:
            for value in values:
                mask=bench.loc[test,column].eq(value).to_numpy()
                type_metrics.append({'model':n,'grouping':column,'value':value,'support':int(mask.sum()),
                                     'captured_at_5pct':int((mask&top).sum()),
                                     'recall_at_5pct':float((mask&top).sum()/mask.sum()) if mask.any() else np.nan})
        families={'payout spike':['payout_spike'],'payout mismatch':['payout_mismatch','payout_reduction'],
                  'adjustment anomaly':['adjustment_anomaly'],'sales spike':['sales_spike'],
                  'quantity mismatch':['quantity_spike','quantity_mismatch','price_anomaly'],
                  'customer shift':['customer_burst','customer_collapse'],
                  'product-mix shift':['product_mix_shift'],'peer deviation':['peer_divergence'],
                  'temporal spike':['inactivity_burst','end_period_spike'],
                  'level shift':['upward_level_shift','downward_level_shift'],
                  'trend break':['upward_level_shift','downward_level_shift']}
        for category,members in families.items():
            mask=bench.loc[test,'anomaly_type'].isin(members).to_numpy()
            type_metrics.append({'model':n,'grouping':'business_family','value':category,'support':int(mask.sum()),
                                 'captured_at_5pct':int((mask&top).sum()),'recall_at_5pct':float((mask&top).sum()/mask.sum()) if mask.any() else np.nan})
    # Queues and consistent score interface for clean and benchmark populations.
    for population,frame,X,values,pct,detail,rule_detail,peer_detail in [
        ('benchmark',bench,Xbench,raw,percentiles,temporal,rules,peer),
        ('clean',clean,Xclean,raw_clean,clean_percentiles,temporal_clean,rules_clean,peer_clean)]:
        queue=frame.copy()
        score_rows=[]
        agreements=np.zeros(len(frame),int)
        for n,s in values.items():
            flag=np.zeros(len(frame),bool)
            for partition in [train,val,test]:
                flag[partition]=top_fraction_flags(s[partition],cfg['review_fraction'])
            threshold=float(np.quantile(values[n][train],1-cfg['review_fraction']))
            queue[n+' score']=pct[n]
            if n!='Best Ensemble':
                agreements+=flag
            score_rows.append(pd.DataFrame({'observation_id':frame.observation_id,'date':frame.date,'split':clean.split,
                                           'population':population,'model_name':n,'raw_score':s,
                                           'anomaly_score':pct[n],'anomaly_percentile':pct[n]*100,
                                           'threshold':threshold,'threshold_flag':s>threshold,'anomaly_flag':flag}))
        queue['model_agreement_count']=agreements
        queue['selected_score']=pct[selected]
        queue['selected_raw_score']=values[selected]
        queue['K-Means cluster']=models['K-Means'].predict(X)
        queue['DBSCAN cluster']=models['DBSCAN'].predict(X)
        queue['DBSCAN noise']=queue['DBSCAN cluster'].eq(-1)
        queue['temporal_score']=np.max([pct[n] for n in TEMPORAL_NAMES],axis=0)
        queue['review_status']='Not reviewed'
        queue['reviewer_comments']=''
        queue['population']=population
        driver_columns=['total_sales_peer_z','distinct_customers_peer_z','total_sales_history_deviation',
                        'simulated_actual_payout_history_deviation','simulated_payout_delta_pct','product_mix_change']
        deviations=frame[driver_columns].fillna(0).abs().to_numpy()
        top_indices=np.argsort(-deviations,axis=1)[:,:3]
        queue['top_drivers']=['; '.join(f'{driver_columns[j]}={frame.iloc[i][driver_columns[j]]:.2f}' for j in ix) for i,ix in enumerate(top_indices)]
        temporal_driver=detail.sort_values('score',ascending=False).drop_duplicates('observation_id').set_index('observation_id')
        queue['temporal_observed']=queue.observation_id.map(temporal_driver.observed)
        queue['temporal_expected']=queue.observation_id.map(temporal_driver.expected)
        queue['temporal_difference']=queue.temporal_observed-queue.temporal_expected
        queue=queue.sort_values('selected_score',ascending=False)
        flag_lookup=pd.concat(score_rows,ignore_index=True)[['observation_id','model_name','anomaly_flag']]
        detail=detail.merge(flag_lookup.rename(columns={'model_name':'model'}),on=['observation_id','model'],how='left',validate='many_to_one')
        write_csv(root,f'artifacts/reports/{population}_investigation_queue.csv',queue)
        write_csv(root,f'data/processed/{population}_scores_long.csv',pd.concat(score_rows,ignore_index=True))
        write_csv(root,f'data/processed/{population}_time_series_scores.csv',detail)
        write_csv(root,f'artifacts/reports/{population}_rule_signals.csv',rule_detail)
        write_csv(root,f'artifacts/reports/{population}_peer_explanations.csv',peer_detail)
        if population=='benchmark':
            write_csv(root,'artifacts/reports/anomaly_investigations.csv',queue)
            write_csv(root,'data/processed/scored_observations_all_models.csv',queue)
            write_csv(root,'data/processed/time_series_scores.csv',detail)
        else:
            rep_summary=queue.groupby(['representative','manager','team']).agg(periods=('observation_id','size'),
                mean_score=('selected_score','mean'),max_score=('selected_score','max')).reset_index()
            write_csv(root,'data/processed/rep_risk_summary.csv',rep_summary)
        for name in ['K-Means','Autoencoder','PCA Reconstruction']:
            contributions=models[name].feature_contributions(X) if name=='K-Means' else models[name].contributions(X)
            np.savez_compressed(root/f'artifacts/reports/{population}_{name.lower().replace(" ","_")}_all_feature_errors.npz',
                                contributions=contributions,features=np.asarray(usable,dtype=str),
                                observation_ids=frame.observation_id.to_numpy(dtype=str))
            top=np.argsort(-contributions,axis=1)[:,:3]
            explanations=[]
            for i,ix in enumerate(top):
                for j in ix:
                    explanations.append({'observation_id':frame.observation_id.iloc[i],'model':name,'feature':usable[j],'contribution':contributions[i,j]})
            write_csv(root,f'artifacts/reports/{population}_{name.lower().replace(" ","_")}_contributions.csv',pd.DataFrame(explanations))
        # Bounded feature ablation for the first 20 actual review candidates.
        review_idx=queue.head(20).index.to_numpy()
        ablation=models['Isolation Forest'].contributions(X[review_idx])
        ab_rows=[]
        for k,i in enumerate(review_idx):
            for j in np.argsort(-ablation[k])[:5]:
                ab_rows.append({'observation_id':frame.observation_id.iloc[i],'feature':usable[j],'score_reduction':ablation[k,j]})
        write_csv(root,f'artifacts/reports/{population}_isolation_forest_ablation.csv',pd.DataFrame(ab_rows))
    write_csv(root,'data/processed/analytical_dataset.csv',clean)
    write_csv(root,'data/processed/controlled_benchmark_dataset.csv',bench)
    write_csv(root,'data/processed/model_features.csv',clean[usable])
    for name,table in rollups.items():
        write_csv(root,f'data/processed/{name}.csv',table)
    contribution_rows=[]
    for n,m in validation_metrics.items():
        contribution_rows.append({'model':n,'validation_recall_component':.36*m['Recall@5%'],
                                  'validation_pr_auc_component':.36*m['PR_AUC'],'validation_f2_component':.18*m['F2'],
                                  'stability_component':.1*stability_mean[n],'selection_utility':utilities[n],
                                  'selected':n==selected})
    temporal_backtest=[]
    for (split,date),group in bench.loc[~train].groupby(['split','date']):
        for name in TEMPORAL_NAMES:
            temporal_backtest.append({'model':name,'split':split,'date':date,'observations':len(group),
                                      'injected_count':int(group.injected_anomaly_flag.sum()),
                                      **metrics(group.injected_anomaly_flag,raw[name][group.index],cfg['review_fraction'])})
    outputs={
        'final_anomaly_model_benchmark':comparison,
        'final_anomaly_model_benchmark_long':comparison.melt(id_vars=['model','model_family'],var_name='metric',value_name='value'),
        'ranking_metrics_all_models':comparison[['model']+[c for c in comparison if '@' in c]+['top_decile_capture']],
        'anomaly_type_metrics':pd.DataFrame(type_metrics),'model_stability':stable,
        'model_score_correlations':corr.rename_axis('model').reset_index(),
        'model_topk_overlap':overlap.rename_axis('model').reset_index(),
        'time_series_backtest':pd.DataFrame(temporal_backtest),
        'ensemble_comparison':ensemble_table,'model_selection_contributions':pd.DataFrame(contribution_rows),
        'validation_model_benchmark':pd.DataFrame([{'model':n,**m} for n,m in validation_metrics.items()]),
        'extended_clustering_benchmark':cluster_table,'hyperparameter_trials':trials,
        'prevalence_sensitivity':pd.DataFrame(sensitivity),'analytical_grain_comparison':grain,
    }
    for name,table in outputs.items():
        write_csv(root,f'artifacts/metrics/{name}.csv',table)
    write_csv(root,'artifacts/reports/controlled_injection_audit.csv',injections)
    write_csv(root,'artifacts/reports/low_prevalence_injection_audit.csv',low_audit)
    print('Anomaly benchmark and queues saved; running independent capacity backtests',flush=True)
    planning,forecast_metrics=run_planning(d,cfg,train_end,validation_end,root)
    audit.update(analytical_rows=len(clean),feature_count=len(usable),feature_names=usable,dropped_features=dropped,
                 analytical_grain='Rep x Product Class x Month',train_rows=int(train.sum()),validation_rows=int(val.sum()),test_rows=int(test.sum()),
                 train_end=str(train_end.date()),validation_end=str(validation_end.date()),
                 test_start=str(clean.loc[test,'date'].min().date()),test_end=str(clean.loc[test,'date'].max().date()),
                 seed=cfg['seed'],parameters=params,model_warnings=model_warnings,
                 python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','streamlit']},
                 primary_test_anomalies=int(bench.loc[test,'injected_anomaly_flag'].sum()),
                 primary_validation_anomalies=int(bench.loc[val,'injected_anomaly_flag'].sum()),skipped_models=[])
    make_plots(root,comparison,corr,temporal_clean,planning,raw,train,test)
    audit['runtime_seconds']=time.perf_counter()-start
    dump(root/'artifacts/reports/extended_run_metadata.json',audit)
    write_reports(root,audit,choice,comparison,planning,forecast_metrics)
    print(json.dumps({'runtime_seconds':audit['runtime_seconds'],'selected':selected,'test_rows':int(test.sum()),'planning_units':len(planning)}),flush=True)
    from .dashboard_data import build_all_dashboard_datasets
    dashboard_settings=yaml.safe_load(config_path.read_text(encoding='utf-8')).get('dashboard',{})
    build_all_dashboard_datasets(root,settings=dashboard_settings)
    return audit


def make_plots(root,comparison,corr,temporal,planning,raw,train,test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'artifacts/plots'; folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,6))
    comparison.set_index('model')[['Recall@5%','PR_AUC','F2']].plot.barh(ax=ax)
    ax.set_title('Held-out Jan–Apr 2019 controlled benchmark'); ax.set_xlabel('Metric (not real-world fraud accuracy)')
    fig.tight_layout();fig.savefig(folder/'extended_model_benchmark.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(corr)),corr.columns,rotation=90);ax.set_yticks(range(len(corr)),corr.index)
    ax.set_title('Validation score rank correlation');fig.colorbar(im,ax=ax);fig.tight_layout()
    fig.savefig(folder/'extended_score_correlations.png',dpi=140);plt.close(fig)
    example=temporal.query("model == 'Rolling Residual' and metric == 'total_sales'")
    example=example[(example.representative==example.representative.iloc[0])&(example.product_class==example.product_class.iloc[0])]
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(example.date,example.observed,label='Observed commercial sales');ax.plot(example.date,example.expected,label='Prior rolling median')
    ax.legend();ax.set_title('Clean commercial history; prior-only expectation');fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(folder/'extended_temporal_example.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));top=planning.head(10).copy();top.index=top.team+' / '+top.country+' / '+top.product_class
    top[['allocated_current_fte','required_fte']].plot.barh(ax=ax);ax.set_title('Modeled workload capacity: scenario estimates, not hiring decisions');fig.tight_layout()
    fig.savefig(folder/'extended_capacity_pressure.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,['Autoencoder','PCA Reconstruction']):
        ax.hist(np.log1p(raw[name][train]),bins=35,alpha=.6,label='Train');ax.hist(np.log1p(raw[name][test]),bins=25,alpha=.6,label='Test benchmark');ax.set_title(name);ax.set_xlabel('log(1 + reconstruction error)');ax.legend()
    fig.tight_layout();fig.savefig(folder/'extended_reconstruction_errors.png',dpi=140);plt.close(fig)


def markdown_table(df):
    # No optional tabulate dependency required for report generation.
    cols=list(df.columns)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/') for v in row)+' |')
    return '\n'.join(lines)


def write_reports(root,audit,choice,comparison,planning,forecasts):
    reports=root/'artifacts/reports'
    table=markdown_table(comparison[['model','Recall@5%','Lift@5%','PR_AUC','F2','stability','runtime_seconds']])
    summary=f'''# Executed extended commercial anomaly benchmark

Source: local user-provided `pharma-data.csv`, {audit['raw_rows']:,} rows × {audit['raw_columns']} columns.
After {audit['duplicates_removed']} exact duplicate removals and {audit['invalid_rows_removed']} invalid-row exclusions: {audit['clean_rows']:,} transactions.
Actual coverage: {audit['date_min']} to {audit['date_max']}, NOT a complete 2019 year.
Country coverage: {audit['country_date_coverage']}. Poland's absent 2019 records are NOT evidence of zero staffing/demand.
Grain: {audit['analytical_grain']}; {audit['analytical_rows']:,} observations, {audit['feature_count']} model features.
TRAIN: through {audit['train_end']} ({audit['train_rows']} rows); VALIDATION: through {audit['validation_end']} ({audit['validation_rows']} rows); TEST: {audit['test_start']}–{audit['test_end']} ({audit['test_rows']} rows).
Primary test contains {audit['primary_test_anomalies']} controlled labels; validation contains {audit['primary_validation_anomalies']}.

## Final TEST comparison (all models, identical population)

{table}

Classification flags and ranking metrics use exact top review budgets with deterministic ties. They are not calibrated probabilities of misconduct.
Runtime includes parameter search for fitted models; ensemble runtime includes its components. The entire run took {audit['runtime_seconds']:.1f}s; package versions, warnings, parameters and seed are in `extended_run_metadata.json`.

## Validation-selected recommendations

- Segmentation: {choice['best_segmentation']} (clean train/validation clustering quality, independent of anomaly test metrics).
- Best individual anomaly model: {choice['best_single']}.
- Best temporal specialist: {choice['best_temporal']}.
- Best interpretable comparator: {choice['best_interpretable']} (validation utility).
- Primary review architecture: {choice['recommended_model']}.
- Ensemble: {choice['ensemble_kind']} with {choice['ensemble_weights']}.
- Material ensemble improvement: {choice['ensemble_material_improvement']}; validation Recall@5% gain {choice['validation_ensemble_recall_gain']:.3f}, PR-AUC gain {choice['validation_ensemble_pr_auc_gain']:.3f}.
- Keep peer/history/rule explanations alongside the primary score, not as an automated adverse decision.

## Capacity scenarios

{markdown_table(planning.head(8)[['team','country','product_class','allocated_current_fte','required_fte','fte_gap','hiring_priority']])}

Current FTE summed across business units: {planning.allocated_current_fte.sum():.2f}; modeled required FTE: {planning.required_fte.sum():.2f}; sum of positive LOCAL gaps: {planning.additional_fte_need.sum():.2f}.
Local gaps are not a net hiring mandate: reallocation, partial FTE, forecast uncertainty, capacity assumptions and territory constraints matter.
Stale-source business units are explicitly ineligible, with no FTE gap or priority estimate.

## Important benchmark limitations

- PCA wins validation; Autoencoder may lead individual test metrics. The test set does not reselect the winner. Compare the numerical table rather than assuming a complex model is superior.
- The Autoencoder reached its bounded iteration limit; loss curves and early-stopping information are persisted. This is not a claim of fully converged optimization.
- Clean DEMO payout equals its deterministic expected formula. Reconstruction methods can detect deviations from this artificial relationship. This is not proof of performance on real payroll data.
- Some injected aggregate changes intentionally break quantity/price/payout relationships. They are controlled experiments, not a transaction-level fraud simulator.
- The cross-country coverage change introduces a structural historical break. Same-month peer scoring is retrospective; older mixed-country history is not fully comparable with Germany-only 2019.
- Lower prevalence runs use three independent seeds and random type subsets; only five positives per test run make these estimates very uncertain.
- Trend-break family recall here uses level shifts as a limited proxy; gradual trend-change detection needs longer independently labeled histories.
- K-Means uses Euclidean L2 after TRAIN median imputation, signed-log tail compression and RobustScaler. No test fitting or clipping. Its signed-log geometry is intentional and fully persisted.
'''
    (reports/'executed_extended_benchmark_summary.md').write_text(summary,encoding='utf-8')
    (reports/'model_selection_report.md').write_text(summary[summary.index('## Validation-selected'):]+'''

Models were frozen before test evaluation. Clustering selection uses unsupervised validation quality with train cluster-balance constraints. Other grids use validation ranking utility; final selection adds seed stability. Ensemble must improve validation recall by at least 0.03 AND PR-AUC by 0.02 without more than 0.05 stability loss. Equal percentile/rank averaging are identical here because model scores already represent TRAIN-reference percentile ranks.
These rules are transparent hackathon priorities, not evidence of production optimality. Synthetic labels inform validation only; final test metrics never choose a model or weights.
''',encoding='utf-8')
    (reports/'time_series_methodology.md').write_text('''# Temporal methodology

For each actual rep/product-class series, score the current observation against prior observations only. Calendar lags explicitly reindex monthly dates; missing months are not converted into zeros. Rolling residual uses prior six-period median and MAD (minimum 3 observations); EWMA uses alpha=0.3 updated AFTER scoring; seasonal residual looks up the exact month one year earlier. MAD scale has a 10%-of-baseline floor. Seasonal scores are zero until lag-12 is available and carry an availability flag, not fabricated history.

Level-shift signals use positive/negative CUSUM with drift=0.5 and require consecutive same-direction residuals. Separate trend signals compare earlier vs later portions of the PRIOR window. This is a lightweight screening detector, not a statistically calibrated change-point posterior. Earlier validation/test observations may update history for later months (rolling-origin operation); future observations never enter expectations. Same-month peer signals are retrospective after month close, not forecasts.

Poland appears only through 2018; Germany-only 2019 introduces a coverage break at the rep/product-class grain. This can create review-worthy residuals without employee behavior changing. Anomaly review must inspect the market view and coverage provenance first.

Temporal benchmark copies are injected BEFORE historical/peer feature derivation. A labeled event can therefore influence later expectations. Unlabeled follow-on flags are counted as false positives conservatively. Targets and contractual expected payouts remain fixed from the clean DEMO schedule; corrupted payouts do not rewrite the compensation plan.

Only monthly records exist. Quarter-end injections proxy end-of-period spikes; intra-month timing cannot be established. Two-month level shifts are short, and four test months provide limited temporal evidence. Support-zero anomaly types are reported as unavailable, never perfect detection.
''',encoding='utf-8')
    (reports/'hiring_need_methodology.md').write_text('''# Field-force capacity scenario methodology

Independent team × country × product-class × month planning, using real commercial measures. Backtest seasonal naive (lag-12, falling back to prior-three mean when unavailable), three-month mean, and alpha=.3 exponential smoothing. Select each metric's method by VALIDATION WAPE; report untouched TEST MAE/RMSE/WAPE/sMAPE/bias. Final forecast is one month beyond the actual data, May 2019—not a current 2026 hiring forecast.

Country coverage is incomplete: Poland stops in December 2018. Units lacking latest-month observations are marked ineligible with missing FTE/priority outputs, not assumed to have zero staffing. Only Germany supports May 2019 capacity scenarios. Test forecast errors likewise cover Germany only.

Normalize workload components using TRAIN per-rep-unit median loads; weights are configurable. Rep-unit loads sum to rep-month workload. Sustainable capacity is the training stable-period 60th percentile (absolute month-on-month workload growth <=50%), not the historical maximum. Missing actual working days, calls and vacancies make this a workload proxy.

Reps serve multiple units. Allocate each active rep's ONE FTE across their latest month's units in proportion to observed workload; allocations reconcile to the actual rep count. Distinct headcounts are shown but are NOT summed as available capacity. Required FTE=forecast workload/sustainable capacity. Gap=required FTE−allocated FTE. Capacity utilization=required FTE/allocated FTE.

The base case assumes all observed reps' capacity is available to the observed Germany scope. Missing Poland records do not establish that cross-country assignments ceased. This assumption can overstate available capacity; validate actual time/territory allocations before interpreting spare capacity.

Scenario bounds combine validation 80th-percentile absolute forecast error with training capacity 40th/80th quantiles; these are NOT confidence intervals. Hiring priority combines growth, utilization, gap, customer/geographic/product pressure; high forecast uncertainty discounts the score. Never use anomaly scores as hiring decisions. Forecast customer breadth and sales reflect observed demand, not total addressable opportunity.

Raw-vs-cleaned sensitivity clips workload to unit TRAIN 5th/95th percentiles with capacity held fixed; this can suppress real structural growth and is a sensitivity check, not truth. Scenarios are independent unit what-ifs. Reallocation explicitly debits a donor and credits a receiver by the same bounded fractional FTE, so headcount is conserved. Validate travel feasibility and staffing constraints before considering any action.
''',encoding='utf-8')
    (reports/'production_data_gaps.md').write_text('''# Production data requirements

Incentives: actual payouts; quotas/targets; compensation-plan rules; adjustments and approvals; call/activity records; territory assignments; leave/working days; approved exceptions; adjudicated review outcomes.

Capacity: HCP/customer opportunity; required call frequency; territory boundaries; travel times; vacancies; rep tenure/working capacity; hiring costs; ramp-up time; product-launch forecasts; market potential; access restrictions. Reconcile identifiers, ownership, currency, contracts, privacy/access controls and refresh frequency.

All incentive fields here are simulated; injected anomalies are benchmark labels, not employee misconduct. Raw Kaggle data may itself contain unusual or erroneous observations. Review signals require business investigation. Capacity outputs are historical scenarios, never automated employment decisions. Source license is not established from the ZIP; raw data is excluded from git and should not be redistributed without confirmation.
''',encoding='utf-8')
    # Reproduction refreshes only the marked executed-result block in the README.
    readme=root/'README.md'
    if readme.exists():
        text=readme.read_text(encoding='utf-8')
        begin,end='<!-- EXTENDED_RESULTS_START -->','<!-- EXTENDED_RESULTS_END -->'
        if begin in text and end in text:
            before=text.split(begin)[0]
            after=text.split(end,1)[1]
            readme.write_text(before+begin+'\n\n'+table+'\n\n'+end+after,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/raw/pharma-data.csv')
    parser.add_argument('--config',default='configs/config.yaml')
    parser.add_argument('--output-root',type=Path,help='Optional isolated output directory; leave existing executed artifacts untouched during verification.')
    args=parser.parse_args()
    root=args.output_root.resolve() if args.output_root else Path(__file__).resolve().parents[2]
    # Avoid BLAS oversubscription; reproducible and bounded on a small population.
    with threadpool_limits(limits=1):
        run(root,Path(args.input).resolve(),Path(args.config).resolve())


if __name__=='__main__':
    main()
