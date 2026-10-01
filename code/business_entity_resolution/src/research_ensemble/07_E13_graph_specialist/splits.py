#!/opt/conda/bin/python3
"""E13 split report (light): paired delta vs v2b by the Codex lab's evaluation splits (accuracy_lab_20260925/data/entities.parquet eval_split:
tune = non-locked val with s1_idx % 2 == 0 (used by the Codex lab to choose the 'graph' configuration), report = the other non-locked half, locked = 30k),
for each evaluated tag, at val and at test density. Uses pv2b.evalx.paired (same bootstrap as the shared evaluator). -> logs/splits.json"""
import os, sys, json
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS']: os.environ[k] = '3'
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2b')
import polars as pl
from pv2b.evalx import paired
HERE = os.path.dirname(os.path.abspath(__file__)); RES = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/results'
LAB = '/workspace/saumilya/amazon-ml/work/matching/accuracy_lab_20260925'
M = pl.read_parquet(f'{LAB}/data/entities.parquet', columns=['s1_idx', 'eval_split', 'country']).with_columns(pl.col('s1_idx').cast(pl.Int32))
BV = pl.read_parquet('/workspace/saumilya/amazon-ml/work/matching/prod_v2b/output/val_abc_rob_cv2_all_p2/val_row_scores.parquet', columns=['s1_idx', 'f'])
BD = pl.read_parquet('/workspace/saumilya/amazon-ml/work/matching/density_val/out/row_scores_v2b.parquet').filter(
    (pl.col('policy') == 'dens:R10c_m0.0') & (pl.col('set') == 'Q')).select('s1_idx', 'f')
out = {}
for tag in ['E13_cv2', 'E13_cv2_frozen', 'E13_hybrid_codexsingle_cv2dec', 'E13_codex_graph_sel']:
    o = {}
    for kind, base, suf in [('val', BV, '_val_rows.parquet'), ('density', BD, '_dens_rows.parquet')]:
        p = f'{RES}/eval_{tag}{suf}'
        if not os.path.exists(p): continue
        R = pl.read_parquet(p, columns=['s1_idx', 'f']).with_columns(pl.col('s1_idx').cast(pl.Int32)).join(M, on='s1_idx', how='left')
        o[kind] = {}
        for sp in ['tune', 'report', 'locked']:
            Rs = R.filter(pl.col('eval_split') == sp).select('s1_idx', 'f')
            o[kind][sp] = dict(macro=float(Rs['f'].mean()), base=float(base.join(Rs.select('s1_idx'), on='s1_idx', how='semi')['f'].mean()), **paired(Rs, base))
        for c in ['US', 'India']:
            Rs = R.filter(pl.col('country') == c).select('s1_idx', 'f')
            o[kind][c] = dict(macro=float(Rs['f'].mean()), **paired(Rs, base))
    out[tag] = o
    print(tag, {k: {s: (round(v['diff'], 6), [round(x, 6) for x in v['ci95']]) for s, v in d.items()} for k, d in o.items()}, flush=True)
json.dump(out, open(f'{HERE}/logs/splits.json', 'w'), indent=1)
