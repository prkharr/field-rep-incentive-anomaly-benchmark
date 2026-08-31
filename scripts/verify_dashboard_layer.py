"""Capture preservation evidence and compare an isolated full benchmark execution.

This verifier writes only its explicitly named evidence file under work/ or docs/.
It never modifies raw data or existing executed artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def input_fingerprints(root):
    paths=[root/'data/raw/pharma-data.csv']
    for folder in ['data/processed','artifacts']:
        paths += [p for p in (root/folder).rglob('*') if p.is_file()]
    return {str(p.relative_to(root)).replace('\\','/'):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(paths)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline',type=Path,default=Path('work/dashboard-layer/baseline.json'))
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--execution-root',type=Path)
    parser.add_argument('--report',type=Path,default=Path('work/dashboard-layer/verification.json'))
    args=parser.parse_args()
    root=args.root.resolve()
    current=input_fingerprints(root)
    if args.capture:
        if args.baseline.exists():
            raise FileExistsError('Preservation baseline already exists; do not replace it')
        args.baseline.parent.mkdir(parents=True,exist_ok=True)
        args.baseline.write_text(json.dumps(current,indent=2),encoding='utf-8')
        print(f'Preservation baseline captured: {len(current)} source/technical files')
        return
    before=json.loads(args.baseline.read_text(encoding='utf-8'))
    if current!=before:
        changed=[p for p in set(before)|set(current) if before.get(p)!=current.get(p)]
        raise AssertionError(f'Original source/technical artifacts changed: {changed}')
    report={'original_files_unchanged':True,'file_count':len(before)}
    if args.execution_root:
        execution=args.execution_root.resolve()
        benchmark='artifacts/metrics/final_anomaly_model_benchmark.csv'
        original=pd.read_csv(root/benchmark).sort_values('model').reset_index(drop=True)
        rerun=pd.read_csv(execution/benchmark).sort_values('model').reset_index(drop=True)
        numeric=[c for c in original.select_dtypes('number') if c not in ['runtime_seconds','model_size_bytes']]
        np.testing.assert_allclose(original[numeric],rerun[numeric],rtol=1e-9,atol=1e-12,equal_nan=True)
        pd.testing.assert_series_equal(original.model,rerun.model)
        for name in ['extended_model_selection.json']:
            left=json.loads((root/'artifacts/reports'/name).read_text(encoding='utf-8'))
            right=json.loads((execution/'artifacts/reports'/name).read_text(encoding='utf-8'))
            assert left==right,'Frozen model selection changed'
        capacity='artifacts/planning/hiring_need_by_business_unit.csv'
        pd.testing.assert_frame_equal(pd.read_csv(root/capacity),pd.read_csv(execution/capacity),check_exact=False,rtol=1e-10,atol=1e-12)
        meta=json.loads((execution/'artifacts/reports/extended_run_metadata.json').read_text(encoding='utf-8'))
        report.update(full_benchmark_executed=True,benchmark_metrics_unchanged=True,
                      capacity_outputs_unchanged=True,selection_unchanged=True,
                      benchmark_runtime_seconds=meta['runtime_seconds'],source_sha256=meta['sha256'],
                      isolated_dashboard_files=sorted(p.name for p in (execution/'data/dashboard').iterdir() if p.is_file()))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
