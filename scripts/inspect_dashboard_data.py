"""Document populated dashboard schemas and first three rows without fitting models."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


PREVIEW_FIELDS = {
    'dashboard_anomaly_review.csv':['review_rank','representative','product_class','month','pca_score_percentile','review_priority','model_agreement_summary'],
    'dashboard_rep_summary.csv':['representative','latest_month_available','high_priority_review_count','latest_pca_percentile','unique_customers_latest'],
    'dashboard_capacity_base.csv':['team','country','product_class','forecast_horizon','allocated_fte','required_fte','eligible_for_capacity_recommendation'],
    'dashboard_capacity_scenarios.csv':['team','country','product_class','scenario_name','allocated_fte','required_fte','fte_gap'],
    'dashboard_model_summary.csv':['model','role','recall_at_5pct','pr_auc','selected_for_primary_use'],
}


def display(value):
    if pd.isna(value):
        return 'null'
    if isinstance(value,float):
        return f'{value:.6g}'
    return str(value).replace('|',' / ').replace('\n',' ')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('docs/dashboard_data_layer.md'))
    parser.add_argument('--verification',type=Path,help='Optional JSON evidence from verify_dashboard_layer.py')
    parser.add_argument('--tests',type=Path,help='Optional pytest JUnit XML from this change')
    args=parser.parse_args()
    root=args.root.resolve()
    metadata=json.loads((root/'data/dashboard/dashboard_metadata.json').read_text(encoding='utf-8'))
    lines=['# Dashboard Data Layer — schema and populated previews','',
           'Generated directly from the exported manager CSVs. Each schema table includes the first three actual rows; `null` means unavailable, not zero. Displayed decimals are shortened here only; CSVs preserve their numeric precision.','',
           'The layer is read-only with respect to all existing raw, processed, model, metric and report artifacts. Only manager-facing exports and this documentation are produced.','',
           'Percentiles are 0–1. All incentive amounts retain the simulated_ prefix. Review priority is a deterministic presentation policy, not a probability. Month columns are ISO first-of-month dates.','']
    for name,fields in PREVIEW_FIELDS.items():
        frame=pd.read_csv(root/'data/dashboard'/name)
        schema=metadata['datasets'][name]
        if len(frame)!=schema['row_count'] or frame.empty:
            raise AssertionError(f'Invalid populated row count for {name}')
        if frame.duplicated(schema['grain']).any():
            raise AssertionError(f'Duplicate declared grain for {name}')
        lines += [f'## {name}','',f"Rows: **{len(frame):,}**. Columns: **{len(frame.columns)}**. Grain: **{' × '.join(schema['grain'])}**.",'',
                  '| Field | Export dtype | Null rows | Row 1 | Row 2 | Row 3 |',
                  '|---|---|---:|---|---|---|']
        for column in schema['columns']:
            field=column['name']
            values=frame[field].head(3).tolist()
            values += [None]*(3-len(values))
            lines.append('| '+ ' | '.join([field,column['dtype'],str(column['null_count'])]+[display(v) for v in values])+' |')
        lines += ['']
        print(f"{name}: {len(frame):,} rows × {len(frame.columns)} columns; grain={' × '.join(schema['grain'])}")
        print(frame[fields].head(3).to_string(index=False))
    lines += ['## Provenance and definitions','',
              f"Source SHA-256: `{metadata['source_csv_sha256']}`.",'',
              f"Source rows: {metadata['source_row_count']:,}; analytical rows: {metadata['modeling_row_count']:,}; seed: {metadata['seed']}.",'',
              metadata['known_poland_coverage_limitation'],'']
    for key,value in metadata['definitions'].items():
        lines += [f'- **{key}:** {value}']
    lines += ['', 'Capacity workload components are independently forecast counts, not additive shares of the composite index. Workload forecast bounds invert persisted FTE bounds using saved training capacity quantiles. Validation/test WAPE is pooled by selected method; country coverage differs by split. Net-zero reallocation uses signed FTE changes and conserved donor/receiver pairing.','',
              'The metadata git commit identifies HEAD at export time; git_worktree_dirty discloses uncommitted implementation changes. It does not claim to identify the later commit that includes the generated files.','']
    if args.verification:
        verified=json.loads(args.verification.read_text(encoding='utf-8'))
        lines += ['## Executed verification','',
                  f"Preserved original source/technical files: {verified['file_count']}; unchanged: {verified['original_files_unchanged']}.",'',
                  f"Full benchmark executed in an isolated directory: {verified.get('full_benchmark_executed')}; runtime: {verified.get('benchmark_runtime_seconds'):.3f} seconds.",'',
                  f"Benchmark metrics unchanged: {verified.get('benchmark_metrics_unchanged')}; model selection unchanged: {verified.get('selection_unchanged')}; capacity outputs unchanged: {verified.get('capacity_outputs_unchanged')}.",'']
    if args.tests:
        tests=ET.parse(args.tests).getroot()[0].attrib
        lines += [f"Tests: {tests['tests']}; failures: {tests['failures']}; errors: {tests['errors']}; skipped: {tests['skipped']}.",'']
    output=args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines),encoding='utf-8')
    print(f'Full schemas and first-three-row previews: {output}')


if __name__=='__main__':
    main()
