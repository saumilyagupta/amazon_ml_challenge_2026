#!/opt/conda/bin/python3
"""E13: clean 2-fold refit of the Codex lab's 'graph' band specialist (accuracy_lab_20260925/experiment.py --model graph).
Fold f trains on SAMPLE band rows with fold == f and ~es, early-stops on fold == f & es (v2b cv2 convention: model f -> OOF for fold 1-f).
Residual LightGBM: init_score = logit(clip(p2, 1e-6, 1-1e-6)); hyper-parameters identical to the Codex model.
Writes models/graph_cv2_fold{f}.txt, data/raw_fold{f}.parquet (raw residual score for ALL band_graph rows),
data/raw_dens_fold{f}.parquet (raw residual score for the density band rows), logs/fit_fold{f}.json.
usage: fit_cv2.py --fold 0|1 [--threads 6] [--band BAND.parquet] [--dens DENS_BAND.parquet|none] [--features F.json] [--tag graph_cv2] [--outdir DIR]
Reuse on a NEW base model (v2c): build its band tables with e13_block.build_band (sample OOF + val rows with meta label/grp/fold/es; density rows),
then run this with --band/--dens/--outdir pointing at them (defaults = the Codex lab's v2b band tables used for E13)."""
import os, sys, json, time, argparse
LAB = '/workspace/saumilya/amazon-ml/work/matching/accuracy_lab_20260925'   # read-only
ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, required=True); ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--band', default=f'{LAB}/data/band_graph.parquet'); ap.add_argument('--dens', default=f'{LAB}/data/density_band.parquet')
ap.add_argument('--features', default=f'{LAB}/data/features_graph.json'); ap.add_argument('--tag', default='graph_cv2')
ap.add_argument('--outdir', default=os.path.dirname(os.path.abspath(__file__)))
A = ap.parse_args()
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS']: os.environ[k] = str(A.threads)
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import resource
import numpy as np, polars as pl, lightgbm as lgb
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
HERE = A.outdir
T0 = time.time()
def log(*a): print(f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s]', *a, flush=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
F = json.load(open(A.features))
if A.features == f'{LAB}/data/features_graph.json': assert F == json.load(open(f'{LAB}/logs/fit_graph.json'))['features'] and len(F) == 310
f = A.fold
B = pl.read_parquet(A.band, columns=['s1_idx', 'cand_idx', 'label', 'grp', 'fold', 'es'] + F)
log('band_graph', B.shape, 'RSS GB', round(rss(), 1))
assert B['p2'].min() > 0.001 and B['p2'].max() < 0.999
X = np.nan_to_num(B.select(F).to_numpy().astype(np.float32), nan=-1., posinf=1e6, neginf=-1e6)
y = B['label'].to_numpy().astype(np.float32)
samp = (B['grp'] == 'sample').to_numpy(); fo = B['fold'].to_numpy(); es = B['es'].to_numpy()
tr = samp & (fo == f) & ~es; ev = samp & (fo == f) & es
p0 = B['p2'].to_numpy().astype(np.float64); z0 = logit(np.clip(p0, 1e-6, 1 - 1e-6))
log(f'fold {f}: train rows {int(tr.sum())} (S1 {B.filter(pl.Series(tr))["s1_idx"].n_unique()}, pos {y[tr].mean():.4f}); '
    f'es rows {int(ev.sum())} (S1 {B.filter(pl.Series(ev))["s1_idx"].n_unique()}); other-fold sample rows {int((samp & (fo != f)).sum())}; val rows {int((~samp).sum())}')
PARAMS = dict(objective='binary', metric='binary_logloss', learning_rate=0.035, num_leaves=15, min_data_in_leaf=150, lambda_l2=10,
              feature_fraction=0.9, num_threads=A.threads, verbosity=-1, seed=20260925)
ds = lgb.Dataset(X[tr], label=y[tr], feature_name=F, init_score=z0[tr])
de = lgb.Dataset(X[ev], label=y[ev], reference=ds, init_score=z0[ev])
t = time.time(); evr = {}
m = lgb.train(PARAMS, ds, num_boost_round=1400, valid_sets=[de], valid_names=['es'],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.record_evaluation(evr), lgb.log_evaluation(100)])
fit_s = time.time() - t
os.makedirs(f'{HERE}/models', exist_ok=True); os.makedirs(f'{HERE}/data', exist_ok=True); os.makedirs(f'{HERE}/logs', exist_ok=True)
m.save_model(f'{HERE}/models/{A.tag}_fold{f}.txt')
log(f'fit done {fit_s:.0f}s best_iter {m.best_iteration}', 'RSS GB', round(rss(), 1))
raw = m.predict(X, raw_score=True, num_threads=A.threads)   # best_iteration is used by default after early stopping
B.select('s1_idx', 'cand_idx').with_columns(pl.Series('raw', raw.astype(np.float32))).write_parquet(f'{HERE}/data/raw_fold{f}.parquet')
p = expit(z0 + raw)
oth = samp & (fo != f)
stats = dict(fold=f, fit_seconds=fit_s, best_iteration=int(m.best_iteration), rounds_run=len(evr['es']['binary_logloss']),
             train_rows=int(tr.sum()), es_rows=int(ev.sum()), oof_rows=int(oth.sum()), train_pos_rate=float(y[tr].mean()),
             es_logloss=float(log_loss(y[ev], p[ev])), es_base_logloss_p2=float(log_loss(y[ev], p0[ev])),
             es_auc=float(roc_auc_score(y[ev], p[ev])), es_base_auc_p2=float(roc_auc_score(y[ev], p0[ev])),
             oof_logloss=float(log_loss(y[oth], p[oth])), oof_base_logloss_p2=float(log_loss(y[oth], p0[oth])),
             oof_auc=float(roc_auc_score(y[oth], p[oth])), oof_base_auc_p2=float(roc_auc_score(y[oth], p0[oth])),
             oof_ap=float(average_precision_score(y[oth], p[oth])), oof_base_ap_p2=float(average_precision_score(y[oth], p0[oth])),
             params=PARAMS, num_boost_round_cap=1400, early_stopping=100,
             importance_gain_top35=sorted(zip(F, map(float, m.feature_importance(importance_type='gain'))), key=lambda x: -x[1])[:35])
del X, B
if A.dens == 'none':
    stats['seconds_total'] = time.time() - T0; stats['peak_rss_gb'] = rss(); json.dump(stats, open(f'{HERE}/logs/fit_fold{f}.json', 'w'), indent=1); log('DONE (no density)'); sys.exit(0)
# density band (features recomputed at test density by the Codex lab's prepare_density.py)
D = pl.read_parquet(A.dens, columns=['s1_idx', 'cand_idx'] + F)
Xd = np.nan_to_num(D.select(F).to_numpy().astype(np.float32), nan=-1., posinf=1e6, neginf=-1e6)
rawd = m.predict(Xd, raw_score=True, num_threads=A.threads)
D.select('s1_idx', 'cand_idx').with_columns(pl.Series('raw', rawd.astype(np.float32))).write_parquet(f'{HERE}/data/raw_dens_fold{f}.parquet')
stats['density_rows'] = D.height; stats['seconds_total'] = time.time() - T0; stats['peak_rss_gb'] = rss()
json.dump(stats, open(f'{HERE}/logs/fit_fold{f}.json', 'w'), indent=1)
log('DONE', {k: v for k, v in stats.items() if k not in ('importance_gain_top35', 'params')})
