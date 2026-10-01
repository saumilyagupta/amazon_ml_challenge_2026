"""LightGBM helpers + stage-2 (per-S1 list / per-record competition) features."""
import time
import numpy as np, polars as pl
from .chan import recside_stage2

S2_SIDE = ['p1_rank_s1', 'p1_max_s1', 'p1_second_s1', 'p1_gap_next', 'p1_gap_top', 'p1_n05_s1', 'p1_n08_s1', 'p1_sum_s1']
S2_REC = ['p1_max_other_rec', 'p1_rel_other', 'p1_rank_rec', 'p1_n05_other_rec']
S2FEATS = ['p1'] + S2_SIDE + S2_REC


def train_lgb(Xtr, ytr, Xva, yva, feats, params, rounds=400, es=30, cat=(), log=print, wtr=None, wva=None):
    import lightgbm as lgb
    t = time.time(); P = dict(params); cf = [c for c in cat if c in feats]
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=list(feats), categorical_feature=cf, free_raw_data=True, params={'max_bin': P.get('max_bin', 255)})
    dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr, feature_name=list(feats), categorical_feature=cf)
    ev = {}
    m = lgb.train(P, dtr, rounds, valid_sets=[dva], valid_names=['es'], callbacks=[lgb.early_stopping(es, verbose=False), lgb.record_evaluation(ev), lgb.log_evaluation(50)])
    log(f'  trained {len(ytr)} rows, best_iter {m.best_iteration}, es logloss {ev["es"]["binary_logloss"][m.best_iteration - 1]:.5f}, {time.time()-t:.0f}s')
    return m


def s1_side_features(P: pl.DataFrame) -> pl.DataFrame:
    """P: s1_idx, cand_idx, p1 over each S1's FULL candidate list -> + S2_SIDE."""
    P = P.sort('s1_idx', 'p1', descending=[False, True])
    P = P.with_columns(
        pl.col('p1').rank('ordinal', descending=True).over('s1_idx').cast(pl.Int16).alias('p1_rank_s1'),
        pl.col('p1').max().over('s1_idx').alias('p1_max_s1'),
        pl.col('p1').top_k(2).min().over('s1_idx').alias('_sec'), pl.len().over('s1_idx').alias('_n'),
        (pl.col('p1') - pl.col('p1').shift(-1).over('s1_idx').fill_null(0.0)).alias('p1_gap_next'),
        (pl.col('p1') > 0.5).sum().over('s1_idx').cast(pl.Int16).alias('p1_n05_s1'),
        (pl.col('p1') > 0.8).sum().over('s1_idx').cast(pl.Int16).alias('p1_n08_s1'),
        pl.col('p1').sum().over('s1_idx').alias('p1_sum_s1'))
    return P.with_columns(pl.when(pl.col('_n') > 1).then(pl.col('_sec')).otherwise(0.0).alias('p1_second_s1'),
                          (pl.col('p1_max_s1') - pl.col('p1')).alias('p1_gap_top')).drop('_sec', '_n')


def stage2_features_dense(P: pl.DataFrame) -> pl.DataFrame:
    """v1 formulation: P is the FULL dense table of all S1; record-side competition within P itself."""
    P = s1_side_features(P)
    P = P.with_columns(pl.col('p1').rank('ordinal', descending=True).over('cand_idx').cast(pl.Int16).alias('p1_rank_rec'),
                       pl.col('p1').max().over('cand_idx').alias('_m1'), pl.col('p1').top_k(2).min().over('cand_idx').alias('_m2'),
                       pl.len().over('cand_idx').alias('_nr'), (pl.col('p1') > 0.5).sum().over('cand_idx').alias('_c5'))
    P = P.with_columns(pl.when(pl.col('_nr') == 1).then(0.0).when(pl.col('p1_rank_rec') == 1).then(pl.col('_m2')).otherwise(pl.col('_m1')).alias('p1_max_other_rec'),
                       (pl.col('_c5') - (pl.col('p1') > 0.5).cast(pl.UInt32)).cast(pl.Int16).alias('p1_n05_other_rec'))
    return P.with_columns((pl.col('p1') - pl.col('p1_max_other_rec')).alias('p1_rel_other')).drop('_m1', '_m2', '_nr', '_c5')


def stage2_features(P: pl.DataFrame, comp: pl.DataFrame = None) -> pl.DataFrame:
    """P: s1_idx, cand_idx, p1 (+ passthrough). comp: competitor pairs (s1_idx, cand_idx, q = stage-1 prob) over the dense table of ALL S1;
    None = P is itself the dense table of all S1 (v1)."""
    base = P.select('s1_idx', 'cand_idx', 'p1')
    if comp is None:
        S = stage2_features_dense(base)
    else:
        S = s1_side_features(base).join(recside_stage2(base, comp, 'p1', 'p1').select('s1_idx', 'cand_idx', *S2_REC), on=['s1_idx', 'cand_idx'], how='left')
    return P.join(S.drop('p1'), on=['s1_idx', 'cand_idx'], how='left')
