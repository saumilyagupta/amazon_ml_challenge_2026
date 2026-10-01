"""E13 block for the v2c integrator: the Codex accuracy lab's 'graph' band specialist, refit 2-fold by v2b's S1 fold.

Specialist = residual LightGBM on the uncertain band 0.001 < p2 < 0.999: prob = expit(logit(clip(p2, 1e-6, 1-1e-6)) + raw),
raw = mean of the two fold models' raw scores (val/test/density) or the OTHER fold's raw score (sample rows, out-of-fold).
310 features (models/features.json) = 267 v2b stage-1 `abc_rob` features + p1 + 12 stage-2 list/record features + 6 sibling features
+ p2 + 5 per-S1 p2 aggregates (p2_sum, p2_n50, p2_n90, p2_max, p2_rank; computed over ALL candidates of the S1, see `p2_aggregates`)
+ 18 g_* features (`make_features`: full-name S1 frequency in 3 views; similarities to the S1's p2>=0.99 siblings, incl. same-source).

IMPORTANT: the g_* features and the p2 aggregates are functions of the BASE model's p2 (siblings = candidates with p2 >= 0.99), and the
specialist was trained on v2b's OOF p2. On a NEW base model (e.g. v2c) the features must be recomputed from that model's p2 and the two
fold models refit on that model's sample OOF band (see RESULTS.md); scoring a v2c band with these v2b-trained models is not validated.

Band-table recipe (exactly how the training/eval tables were built by accuracy_lab_20260925/prepare.py + sibling_features.py):
  P  = the base model's full prediction table (all candidate pairs of the S1s; s1_idx, cand_idx, p1, stage-2 + sibling features, p2)
  P  = p2_aggregates(P)                                     # over ALL rows of each S1, BEFORE the band filter
  B  = P.filter(band_expr()).join(<267 stage-1 abc_rob columns from the feats parts>, on=['s1_idx','cand_idx'])
  B  = B.join(make_features(B, P, R, R1), on=['s1_idx','cand_idx'])   # R/R1 = prod_v1/data/rec_{split}_{s23,s1}.parquet views
  prob = score_band(B)                                      # replace p2 by prob on the band rows, keep p2 elsewhere, then refit R10c
"""
import os, sys, json
import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = '/workspace/saumilya/amazon-ml/work/matching/accuracy_lab_20260925'   # read-only; only sibling_features.compute_features is imported
FEATURES = json.load(open(os.path.join(HERE, 'models', 'features.json')))
MODEL_FILES = [os.path.join(HERE, 'models', f'graph_cv2_fold{f}.txt') for f in (0, 1)]
FULL_MODEL_FILE = os.path.join(HERE, 'models', 'graph_full_codex.txt')   # the Codex single model (all sample band rows), sha256 = its manifest's
_FULL = None
BAND = (0.001, 0.999)
G_FEATURES = [c for c in FEATURES if c.startswith('g_')]
REC_COLS = ['erow', 'country', 'src', 'name_n', 'name_ns', 'name_core', 'name_raw']
_BOOSTERS = None


def band_expr(pcol='p2'):
    """polars expression selecting the specialist's band 0.001 < p < 0.999."""
    return (pl.col(pcol) > BAND[0]) & (pl.col(pcol) < BAND[1])


def _boosters():
    global _BOOSTERS
    if _BOOSTERS is None:
        import lightgbm as lgb
        _BOOSTERS = [lgb.Booster(model_file=f) for f in MODEL_FILES]
    return _BOOSTERS


def _matrix(band_df):
    miss = [c for c in FEATURES if c not in band_df.columns]
    if miss:
        raise KeyError(f'band_df lacks {len(miss)} of the 310 features, e.g. {miss[:5]}')
    X = band_df.select(FEATURES).to_numpy().astype(np.float32)
    return np.nan_to_num(X, nan=-1., posinf=1e6, neginf=-1e6)   # exactly as experiment.py


def _full():
    global _FULL
    if _FULL is None:
        import lightgbm as lgb
        _FULL = lgb.Booster(model_file=FULL_MODEL_FILE)
    return _FULL


def raw_scores(band_df, threads=4):
    """(n, 2) raw residual scores of the fold-0 and fold-1 models (without the logit(p2) offset)."""
    X = _matrix(band_df)
    return np.column_stack([b.predict(X, raw_score=True, num_threads=threads) for b in _boosters()])


def score_band(band_df, oof_fold=None, threads=4, model='cv2'):
    """Specialist probability for band rows (np.float64 array, same row order as band_df).
    band_df: polars DataFrame with the 310 FEATURES (p2 included; rows should satisfy 0.001 < p2 < 0.999).
    Default (val / test / density): expit(logit(clip(p2)) + mean(raw_fold0, raw_fold1)).
    oof_fold: optional array/Series of v2b S1 folds (0/1) or -1 for non-sample rows. Sample rows of fold k get the fold-(1-k) model
    only (out-of-fold, model f was trained on sample fold f), rows with -1 get the mean. Use this only for v2b's own sample S1.
    model='full': the full-sample Codex model (models/graph_full_codex.txt) for VAL / TEST / DENSITY rows only (in-sample on the sample,
    never feed it to a decoder refit). Measured 'hybrid' = full-model val/density scores + R10c refit on the cv2 OOF sample scores:
    val +0.000492 [+0.000381, +0.000604], density +0.000597 [+0.000472, +0.000716] (vs cv2 mean-of-folds +0.000381 / +0.000363)."""
    from scipy.special import expit, logit
    if model == 'full':
        if oof_fold is not None:
            raise ValueError('model=full has no out-of-fold scores (trained on all sample band rows)')
        raw = _full().predict(_matrix(band_df), raw_score=True, num_threads=threads)
        z0 = logit(np.clip(band_df['p2'].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6))
        return expit(z0 + raw)
    R = raw_scores(band_df, threads)
    raw = R.mean(axis=1)
    if oof_fold is not None:
        fo = np.asarray(oof_fold)
        raw = np.where(fo == 0, R[:, 1], np.where(fo == 1, R[:, 0], raw))
    z0 = logit(np.clip(band_df['p2'].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6))
    return expit(z0 + raw)


def p2_aggregates(P, pcol='p2'):
    """The 5 per-S1 aggregates of the base model's p2 used as features (prepare.py / prepare_density.py), computed over ALL
    candidate rows of each S1 (call on the full prediction table BEFORE filtering to the band). Note: p2_rank is an 'ordinal'
    rank, so ties are broken by row order (keep the table in its natural (s1_id, cand_id) order as in v2b's parts)."""
    p = pl.col(pcol)
    return P.with_columns(p.sum().over('s1_idx').alias('p2_sum'), (p > 0.5).sum().over('s1_idx').alias('p2_n50'),
                          (p > 0.9).sum().over('s1_idx').alias('p2_n90'), p.max().over('s1_idx').alias('p2_max'),
                          p.rank('ordinal', descending=True).over('s1_idx').alias('p2_rank'))


def make_features(B, P, R, R1, pcol='p2'):
    """18 g_* features for the band pairs of ANY prediction table (val, density, test), via the Codex lab's
    sibling_features.compute_features (imported read-only, no bytecode written into the lab directory).
    B : band pairs, at least (s1_idx, cand_idx) as Int32 (the rows to featurise, e.g. P.filter(band_expr())).
    P : the SAME base model's scores for ALL candidate pairs of those S1 (s1_idx, cand_idx, <pcol>); siblings = rows with p >= 0.99
        (top 8 per S1, the candidate itself excluded). Labels are never used.
    R : candidate record view, prod_v1/data/rec_{split}_s23.parquet with REC_COLS (erow = cand_idx); DataFrame or LazyFrame.
    R1: S1 record view, prod_v1/data/rec_{split}_s1.parquet with REC_COLS = the frequency universe for g_s1freq_* (all S1 records of
        the split: train = 2.2M train S1 (how band_graph was built), test = the 1.73M test S1 (transductive, label-free input statistic),
        density universe = the kept S1 only, as in prepare_density.py).
    Returns a DataFrame (s1_idx, cand_idx, *G_FEATURES) with exactly B's rows in B's order (missing sibling evidence = -1)."""
    sys.dont_write_bytecode = True
    if LAB not in sys.path:
        sys.path.insert(0, LAB)
    from sibling_features import compute_features   # module import sets *_NUM_THREADS=3 env vars (no effect once numpy/polars are loaded)
    keys = B.select(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
    Pp = P.select(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32), pl.col(pcol).alias('p2'))
    Pp = Pp.join(keys.select('s1_idx').unique(), on='s1_idx', how='semi')
    # R is only used to look up the band candidates' and the p>=0.99 siblings' names/sources/country -> restrict it to those erows
    # (identical output, much less memory on the 10M-row candidate views); R1 (frequency universe) is used in full.
    need = pl.concat([keys.select('cand_idx'), Pp.filter(pl.col('p2') >= 0.99).select('cand_idx')]).unique().select(pl.col('cand_idx').cast(pl.UInt32).alias('erow'))
    R = (R.lazy() if isinstance(R, pl.DataFrame) else R).select(REC_COLS).with_columns(pl.col('erow').cast(pl.UInt32)).join(need.lazy(), on='erow', how='semi').collect()
    R1 = (R1.collect() if isinstance(R1, pl.LazyFrame) else R1).select(REC_COLS)
    G = compute_features(keys, Pp, R, R1)
    out = keys.with_row_index('_r').join(G, on=['s1_idx', 'cand_idx'], how='left').sort('_r').drop('_r')
    assert out.height == keys.height and out.select(G_FEATURES).null_count().sum_horizontal().item() == 0
    return out.select('s1_idx', 'cand_idx', *G_FEATURES)


AGG_FEATURES = ['p2_sum', 'p2_n50', 'p2_n90', 'p2_max', 'p2_rank']


def build_band(P, feat_parts, R, R1, pcol='p2'):
    """Band table with the 310 FEATURES, built exactly like accuracy_lab_20260925/prepare.py (+ prepare_density.py, sibling_features.py):
    P          : the base model's FULL prediction table for the S1s (all candidate pairs, natural row order) with p1, the stage-2 and sibling
                 features and <pcol> (v2b format: prod_v2b/data/preds_abc_rob_cv2_all.parquet, density_val/universe/preds_v2b_dens.parquet,
                 or the test pair parts' columns).
    feat_parts : parquet paths holding (s1_idx, cand_idx) + the 267 stage-1 abc_rob columns (prod_v2b/feats/{train,test}_NNN.parquet,
                 density_val/universe/feats_dens/part_*.parquet); each part is scanned in streaming mode and semi-joined to the band keys.
    R, R1      : record views (load_records). Returns the band rows (0.001 < p < 0.999) with meta columns of P + all FEATURES, sorted by (s1_idx, cand_idx)."""
    if pcol != 'p2':
        P = P.with_columns(pl.col(pcol).alias('p2'))
    P = p2_aggregates(P, 'p2')
    B = P.filter(band_expr('p2'))
    s1_cols = [c for c in FEATURES if c not in B.columns and c not in G_FEATURES]
    keys = B.select('s1_idx', 'cand_idx').lazy()
    parts = [pl.scan_parquet(f).select(['s1_idx', 'cand_idx'] + s1_cols).join(keys, on=['s1_idx', 'cand_idx'], how='semi').collect(engine='streaming')
             for f in feat_parts]
    X = pl.concat(parts)
    B = B.join(X, on=['s1_idx', 'cand_idx'], how='inner')
    B = B.join(make_features(B, P, R, R1, 'p2'), on=['s1_idx', 'cand_idx'], how='left').sort('s1_idx', 'cand_idx')
    miss = [c for c in FEATURES if c not in B.columns]
    assert not miss, miss
    return B


def load_records(split, lazy_r=True):
    """(R, R1) record views for split 'train' or 'test' with the columns make_features needs. R is returned as a LazyFrame scan by
    default (make_features filters it to the needed erows before collecting); R1 is read in full (it is the frequency universe)."""
    base = '/workspace/saumilya/amazon-ml/work/matching/prod_v1/data'
    R = pl.scan_parquet(f'{base}/rec_{split}_s23.parquet').select(REC_COLS)
    return (R if lazy_r else R.collect(), pl.read_parquet(f'{base}/rec_{split}_s1.parquet', columns=REC_COLS))
