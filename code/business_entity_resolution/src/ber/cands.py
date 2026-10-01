"""Candidate tables + dense / competition ("table") features.

dense_candidates(): the production dense channel = forward ranks 1..kf UNION reverse rank 1 + reverse ranks 2-5 within `delta` of the
record's top-1 (built for ALL S1 of the split = the "dense table"). table_features() accepts any frame with s1_idx, cand_idx
(+ passthrough ch_* columns) and adds the dense score and the competition features; it must be called on the FULL table of the
split (dense table of all S1 UNION the pairs to featurise) so that ranks / claimant counts never depend on which S1 are evaluated."""
import numpy as np, polars as pl


def load_forward(W, enc, split, cols=('s1_idx', 'cand_idx', 'rank', 'score')):
    return pl.read_parquet(W.forward(enc, split), columns=list(cols))


def load_reverse(W, enc, split):
    return pl.read_parquet(W.reverse(enc, split), columns=['cand_idx', 's1_idx', 'rank', 'score'])


def ids_to_idx(W, df, split):
    i1 = pl.read_parquet(W.records(split, 's1'), columns=['entity_id']).with_row_index('s1_idx').rename({'entity_id': 's1_id'})
    i2 = pl.read_parquet(W.records(split, 's23'), columns=['entity_id']).with_row_index('cand_idx').rename({'entity_id': 'cand_id'})
    out = df.join(i1, on='s1_id', how='left').join(i2, on='cand_id', how='left')
    assert out['s1_idx'].null_count() == 0 and out['cand_idx'].null_count() == 0, 'unknown ids'
    return out.with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))


def dense_candidates(F, R, kf=20, delta=0.02):
    f = F.filter(pl.col('rank') <= kf).select('s1_idx', 'cand_idx', pl.lit(True).alias('ch_fwd'))
    r = R.with_columns(pl.col('score').max().over('cand_idx').alias('_t')) \
         .filter((pl.col('rank') == 1) | (pl.col('score') >= pl.col('_t') - delta)).select('s1_idx', 'cand_idx', pl.lit(True).alias('ch_rev'))
    U = f.join(r, on=['s1_idx', 'cand_idx'], how='full', coalesce=True).with_columns(pl.col('ch_fwd').fill_null(False), pl.col('ch_rev').fill_null(False))
    return U.with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32)).sort('s1_idx', 'cand_idx')


def pair_cosine(E1, E23, ia, ib, chunk=2_000_000):
    ia = np.asarray(ia, np.int64); ib = np.asarray(ib, np.int64); out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        a = E1[ia[s:s + chunk]].astype(np.float32); b = E23[ib[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = np.einsum('ij,ij->i', a, b)
    return out


def table_features(U, F, R, W=None, enc=None, split=None, gate=0.02):
    """U: s1_idx, cand_idx (+ passthrough cols). F/R: forward / reverse lists of the dense encoder. Pairs from other channels that are
    in neither list get their exact cosine from the stored embeddings (W, enc, split needed)."""
    s1s = F.filter(pl.col('rank') <= 2).group_by('s1_idx').agg(pl.col('score').filter(pl.col('rank') == 1).first().alias('s1_top1'),
                                                               pl.col('score').filter(pl.col('rank') == 2).first().alias('s1_top2'))
    rs = R.group_by('cand_idx').agg(pl.col('score').filter(pl.col('rank') == 1).first().alias('rev_top1'),
                                    pl.col('score').filter(pl.col('rank') == 2).first().alias('rev_top2'))
    rs = rs.join(R.join(rs.select('cand_idx', 'rev_top1'), on='cand_idx').filter(pl.col('score') >= pl.col('rev_top1') - gate)
                 .group_by('cand_idx').agg(pl.len().cast(pl.Int8).alias('n_claim_gate')), on='cand_idx', how='left')
    T = U.join(F.select('s1_idx', 'cand_idx', pl.col('rank').cast(pl.Int16).alias('fwd_rank'), pl.col('score').alias('_fs')), on=['s1_idx', 'cand_idx'], how='left')
    T = T.join(R.select('s1_idx', 'cand_idx', pl.col('rank').cast(pl.Int8).alias('rev_rank'), pl.col('score').alias('_rs')), on=['s1_idx', 'cand_idx'], how='left')
    T = T.with_columns(pl.coalesce('_fs', '_rs').alias('cos')).drop('_fs', '_rs')
    if T['cos'].null_count():
        E1 = np.load(W.emb(enc, split, 's1'), mmap_mode='r'); E23 = np.load(W.emb(enc, split, 's23'), mmap_mode='r')
        m = T['cos'].is_null().to_numpy(); c = T['cos'].to_numpy().copy()
        c[m] = pair_cosine(E1, E23, T['s1_idx'].to_numpy()[m], T['cand_idx'].to_numpy()[m])
        T = T.with_columns(pl.Series('cos', c, pl.Float32))
    T = T.join(s1s, on='s1_idx', how='left').join(rs, on='cand_idx', how='left')
    T = T.with_columns(
        pl.col('fwd_rank').fill_null(51), pl.col('rev_rank').fill_null(6), pl.col('n_claim_gate').fill_null(0),
        (pl.col('cos') - pl.col('s1_top1')).alias('rel_s1'), (pl.col('s1_top1') - pl.col('s1_top2')).alias('s1_gap12'),
        (pl.col('cos') - pl.col('rev_top1')).alias('rel_rec'), (pl.col('rev_top1') - pl.col('rev_top2')).alias('rev_gap12'),
        pl.col('cos').rank('ordinal', descending=True).over('s1_idx').cast(pl.Int16).alias('rank_s1'),
        pl.len().over('s1_idx').cast(pl.Int16).alias('n_cands_s1'),
        pl.col('cos').rank('ordinal', descending=True).over('cand_idx').cast(pl.Int16).alias('rank_rec_tab'),
        pl.len().over('cand_idx').cast(pl.Int16).alias('n_s1_rec_tab'),
        (pl.col('cos') - pl.col('cos').max().over('cand_idx')).alias('rel_rec_tab'))
    ch = [c for c in T.columns if c.startswith('ch_')]
    T = T.with_columns(pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in ch]).alias('n_channels') if ch else pl.lit(1).alias('n_channels'))
    return T.drop('s1_top2', 'rev_top2')
