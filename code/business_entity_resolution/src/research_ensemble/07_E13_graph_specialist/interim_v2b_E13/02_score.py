#!/opt/conda/bin/python3
"""B1 step 2: assemble the test band table (band_base + band_s1 parts + band_g chunks = the 310 E13 FEATURES) and score it:
  * MAIN (coordinator 15:15, 'hybrid'): p_full = e13_block.score_band(B, model='full') = expit(logit(clip(p2, 1e-6, 1-1e-6)) + raw_full) with
    exp/E13_codex_refit/models/graph_full_codex.txt (sha256-identical to accuracy_lab_20260925/models/graph.txt, the Codex single model);
    decided later with the R10c decoder REFIT on E13's 2-fold OOF (results/eval_E13_cv2_R10c_m0.txt).
  * VARIANT: p_cv2 = e13_block.score_band(B) = expit(logit(clip(p2)) + mean(raw_fold0, raw_fold1)) with models/graph_cv2_fold{0,1}.txt.
  * verification: on s1_idx % 10 == 0, p_full must equal accuracy_lab_20260925/data/test_sample_prob.parquet (Codex test_sample.py, same model)
    to max abs diff < 1e-6; g_* recomputed in ONE call on that 1/10 subset (as test_sample.py did) must equal the chunked g_* columns.
  Scores are computed in row chunks (bounded RAM) with the exact e13_block matrix (_matrix = nan_to_num(float32, nan=-1, +-inf -> +-1e6)).
Writes data/p3_test.parquet (s1_idx, cand_idx, p3 = MAIN), data/p3_cv2_test.parquet (p3 = VARIANT), data/raw_band_scores.parquet, logs/02_score.json.
usage: 02_score.py [--which full|cv2] [--rows 1000000] [--no-verify-g]"""
import b1common as C
import os, sys, glob, json, time, argparse, gc, hashlib
ap = argparse.ArgumentParser(); ap.add_argument('--which', default='full', choices=['full', 'cv2']); ap.add_argument('--rows', type=int, default=1000000)
ap.add_argument('--no-verify-g', action='store_true'); A = ap.parse_args()
import numpy as np, polars as pl, lightgbm as lgb
from scipy.special import expit, logit
import e13_block as E
log = C.log; K2 = ['s1_idx', 'cand_idx']; t0 = time.time()
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
FEAT = E.FEATURES; assert json.load(open(f'{C.LAB}/logs/fit_graph.json'))['features'] == FEAT
RES = dict(mode=C.MODE, which=A.which, full_model=E.FULL_MODEL_FILE, full_model_sha256=sha(E.FULL_MODEL_FILE), codex_graph_sha256=sha(f'{C.LAB}/models/graph.txt'),
           cv2_models={f: sha(f) for f in E.MODEL_FILES})
RES['full_model_is_codex_graph'] = RES['full_model_sha256'] == RES['codex_graph_sha256']; assert RES['full_model_is_codex_graph']
C.gate('assemble')
B = pl.read_parquet(f'{C.DD}/band_base.parquet')
X = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f'{C.DD}/band_s1/part-*.parquet'))]); assert X.height == B.height, (X.height, B.height)
G = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f'{C.DD}/band_g/chunk-*.parquet'))]); assert G.height == B.height, (G.height, B.height)
assert X.select(pl.struct(K2).n_unique()).item() == X.height and G.select(pl.struct(K2).n_unique()).item() == G.height
nb = B.height; B = B.join(X, on=K2, how='inner').join(G, on=K2, how='inner'); del X, G; gc.collect(); assert B.height == nb
B = B.sort(K2); miss = [c for c in FEAT if c not in B.columns]; assert not miss, miss
RES['band'] = dict(rows=B.height, s1=B['s1_idx'].n_unique(), by_country=dict(B.group_by('country').len().sort('country').iter_rows()),
                   nan_or_null_cells=int(sum(B[c].null_count() + (B[c].is_nan().sum() if B[c].dtype in (pl.Float32, pl.Float64) else 0) for c in FEAT)))
log('band assembled', RES['band'], f'RSS {C.rss_gb():.1f} GB')
z0 = logit(np.clip(B['p2'].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6))
fraw = f'{C.DD}/raw_band_scores.parquet'   # raw margins, saved first so a later failure does not need a re-score
if os.path.exists(fraw):
    Rw = B.select(K2).join(pl.read_parquet(fraw), on=K2, how='left'); assert Rw['raw_full'].null_count() == 0
    raw_cv = np.column_stack([Rw['raw_f0'].to_numpy(), Rw['raw_f1'].to_numpy()]); raw_full = Rw['raw_full'].to_numpy(); log('reused raw scores', fraw); del Rw
else:
    bcv = [lgb.Booster(model_file=f) for f in E.MODEL_FILES]; bfull = lgb.Booster(model_file=E.FULL_MODEL_FILE)
    raw_cv = np.zeros((B.height, 2)); raw_full = np.zeros(B.height)
    for s in range(0, B.height, A.rows):
        C.gate(f'score rows {s}')
        Xm = E._matrix(B.slice(s, A.rows))
        for j, b in enumerate(bcv): raw_cv[s:s + len(Xm), j] = b.predict(Xm, raw_score=True, num_threads=C.TH)
        raw_full[s:s + len(Xm)] = bfull.predict(Xm, raw_score=True, num_threads=C.TH)
        log('scored rows', s + len(Xm), f'RSS {C.rss_gb():.1f} GB'); del Xm
    B.select(K2).with_columns(pl.Series('raw_f0', raw_cv[:, 0]), pl.Series('raw_f1', raw_cv[:, 1]), pl.Series('raw_full', raw_full)).write_parquet(fraw); log('raw scores saved', fraw)
S = B.select(*K2, 'country', 'p2').with_columns(pl.Series('p_full', expit(z0 + raw_full)), pl.Series('p_cv2', expit(z0 + raw_cv.mean(axis=1))),
                                                pl.Series('p_cv2_f0', expit(z0 + raw_cv[:, 0])), pl.Series('p_cv2_f1', expit(z0 + raw_cv[:, 1])))
# (1) the full model on every 10th S1 vs the Codex lab's own test-sample scoring (same model, their feature build)
ref = pl.read_parquet(f'{C.LAB}/data/test_sample_prob.parquet'); mine = S.filter(pl.col('s1_idx') % 10 == 0)
J = ref.join(mine.select(*K2, 'p_full'), on=K2, how='full', coalesce=True); d = (J['prob'] - J['p_full']).abs()
RES['verify_codex_sample'] = dict(ref_rows=ref.height, mine_rows=mine.height, only_ref=int(J['p_full'].null_count()), only_mine=int(J['prob'].null_count()),
                                  max_abs_diff=float(d.max()), n_diff_gt_1e9=int((d > 1e-9).sum()), n_diff_gt_1e6=int((d > 1e-6).sum()))
log('VERIFY full model (Codex graph) on s1%10==0 vs test_sample_prob', RES['verify_codex_sample'])
# (2) chunk invariance of g_*: one make_features call on the whole 1/10 subset (as test_sample.py) == the chunked columns
if not A.no_verify_g:
    t = time.time(); Ps = pl.scan_parquet(C.PREDS).filter(pl.col('s1_idx') % 10 == 0).select('s1_idx', 'cand_idx', 'p2').collect()
    R, R1 = E.load_records('test'); kb = B.filter(pl.col('s1_idx') % 10 == 0).select(K2)
    G10 = E.make_features(kb, Ps, R, R1); del Ps, R, R1
    Jg = G10.join(B.select(K2 + E.G_FEATURES), on=K2, how='inner', suffix='_chunked'); assert Jg.height == kb.height == G10.height
    RES['verify_g_one_call_vs_chunked'] = dict(rows=Jg.height, max_abs_diff={c: float((Jg[c].cast(pl.Float64) - Jg[c + '_chunked'].cast(pl.Float64)).abs().max()) for c in E.G_FEATURES},
                                               rows_any_diff=int(Jg.select(pl.any_horizontal([(pl.col(c) != pl.col(c + '_chunked')) for c in E.G_FEATURES])).to_series().sum()), seconds=round(time.time() - t, 1))
    log('VERIFY g_* one call (1/10 S1) vs chunked', RES['verify_g_one_call_vs_chunked']); del G10, Jg
f = S['p_full'].to_numpy(); c = S['p_cv2'].to_numpy(); p2 = S['p2'].to_numpy(); lg = lambda x: logit(np.clip(x, 1e-9, 1 - 1e-9))
RES['compare'] = dict(corr_logit_full_cv2=float(np.corrcoef(lg(f), lg(c))[0, 1]), mean_abs_full_cv2=float(np.abs(f - c).mean()),
                      mean_abs_full_p2=float(np.abs(f - p2).mean()), mean_abs_cv2_p2=float(np.abs(c - p2).mean()),
                      cross05_up_full=int(((p2 < 0.5) & (f >= 0.5)).sum()), cross05_down_full=int(((p2 >= 0.5) & (f < 0.5)).sum()),
                      cross05_up_cv2=int(((p2 < 0.5) & (c >= 0.5)).sum()), cross05_down_cv2=int(((p2 >= 0.5) & (c < 0.5)).sum()),
                      mean_p2=float(p2.mean()), mean_full=float(f.mean()), mean_cv2=float(c.mean()),
                      by_country={r['country']: r for r in S.group_by('country').agg(pl.len().alias('n'), pl.col('p2').mean().alias('p2'), pl.col('p_full').mean().alias('full'), pl.col('p_cv2').mean().alias('cv2'),
                                  ((pl.col('p2') < 0.5) & (pl.col('p_full') >= 0.5)).sum().alias('up05_full'), ((pl.col('p2') >= 0.5) & (pl.col('p_full') < 0.5)).sum().alias('down05_full')).to_dicts()})
log('compare', RES['compare'])
main, var = ('p_full', 'p_cv2') if A.which == 'full' else ('p_cv2', 'p_full')
S.select(*K2, pl.col(main).alias('p3')).write_parquet(f'{C.DD}/p3_test.parquet')
S.select(*K2, pl.col('p_cv2').alias('p3')).write_parquet(f'{C.DD}/p3_cv2_test.parquet')
S.select(*K2, 'p_full', 'p_cv2', 'p_cv2_f0', 'p_cv2_f1').write_parquet(f'{C.DD}/p3_all_variants_test.parquet')
RES['p3_main'] = main; RES['seconds'] = time.time() - t0; RES['peak_rss_gb'] = C.peak_rss_gb()
json.dump(RES, open(f'{C.LD}/02_score.json', 'w'), indent=1, default=str); log('DONE main p3 =', main, f'{time.time()-t0:.0f}s', 'peak RSS', round(C.peak_rss_gb(), 1))
