"""Channel-evidence features from the blocking union (37 u_* features, + 7 for the fine-tuned member) and record-side competition
features that stay consistent between train and test.

Record-side competition (how many / how strongly other S1 claim the same record) is always defined over the DENSE TABLE of ALL S1
of the split (fwd@20 + gated reverse, which exists identically for train and test) plus the pair itself. Lexical channels are only
run for the query S1 (on train a sample), so counting lexical claimants would be inconsistent between train and test."""
import numpy as np, polars as pl

FLAGS = ['in_fwd', 'in_rev', 'in_c1', 'in_c1r', 'in_c2r', 'in_c3r', 'in_c3', 'in_c2']
RANKED = {'fr': ['fs'], 'rr': ['rs', 'rd'], 'c1': ['c1s', 'c1nk'], 'c2': ['c2s', 'c2nk'], 'c3': ['c3s'],
          'c1r': ['c1rs', 'c1rnk', 'c1rg'], 'c2r': ['c2rs', 'c2rnk', 'c2rg'], 'c3r': ['c3rs', 'c3rg']}
RAW_COLS = FLAGS + ['n_member_channels', 'fused_rank', 'fused_hd_rank'] + [c for r, d in RANKED.items() for c in [r] + d]
CHAN_FEATS = ['u_' + c for c in RAW_COLS] + ['u_n_s1', 'u_n_lexrev_s1']
FT_FLAGS = ['in_fwd_ft', 'in_rev_ft']
FT_RANKED = {'fr_ft': ['fs_ft'], 'rr_ft': ['rs_ft', 'rd_ft']}
FT_RAW_COLS = FT_FLAGS + [c for r, d in FT_RANKED.items() for c in [r] + d]
CHAN_FEATS_FT = ['u_' + c for c in FT_RAW_COLS]
FT_PASSTHROUGH = ['ft_seen', 'ft_seen_s1']


def channel_features(U: pl.DataFrame) -> pl.DataFrame:
    """Absent channel (rank 0) -> rank AND its score/nk/gap are null (NaN for LightGBM), never imputed. Membership flags stay 0/1."""
    ex = [pl.col(f).cast(pl.Float32).alias('u_' + f) for f in FLAGS]
    ex += [pl.col(c).cast(pl.Float32).alias('u_' + c) for c in ('n_member_channels', 'fused_rank', 'fused_hd_rank')]
    for r, deps in RANKED.items():
        pres = pl.col(r) > 0
        ex.append(pl.when(pres).then(pl.col(r).cast(pl.Float32)).otherwise(None).alias('u_' + r))
        ex += [pl.when(pres).then(pl.col(d).cast(pl.Float32)).otherwise(None).alias('u_' + d) for d in deps]
    ex += [pl.len().over('s1_idx').cast(pl.Float32).alias('u_n_s1'),
           (pl.col('in_c1r') | pl.col('in_c2r') | pl.col('in_c3r')).cast(pl.Int32).sum().over('s1_idx').cast(pl.Float32).alias('u_n_lexrev_s1')]
    if all(c in U.columns for c in FT_RAW_COLS):
        ex += [pl.col(f).cast(pl.Float32).alias('u_' + f) for f in FT_FLAGS]
        for r, deps in FT_RANKED.items():
            pres = pl.col(r) > 0
            ex.append(pl.when(pres).then(pl.col(r).cast(pl.Float32)).otherwise(None).alias('u_' + r))
            ex += [pl.when(pres).then(pl.col(d).cast(pl.Float32)).otherwise(None).alias('u_' + d) for d in deps]
        ex += [pl.col(c) for c in FT_PASSTHROUGH if c in U.columns]
    return U.select(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32), *ex)


def _keys(cand, v):
    return np.sort(cand.astype(np.float64) + (1.0 - v.astype(np.float64)) * 0.5 * (1 - 1e-7))


def _count_greater(keys_sorted, cand, val):
    q = cand.astype(np.float64) + (1.0 - val.astype(np.float64)) * 0.5 * (1 - 1e-7)
    return np.searchsorted(keys_sorted, q, 'left') - np.searchsorted(keys_sorted, cand.astype(np.float64), 'left')


def recside_table(T: pl.DataFrame, dense: pl.DataFrame) -> pl.DataFrame:
    """Override rank_rec_tab / n_s1_rec_tab / rel_rec_tab of T with the version computed over dense(ALL S1) + the pair itself."""
    d = dense.select('s1_idx', 'cand_idx', pl.col('rank_rec_tab').alias('_rk'), pl.col('n_s1_rec_tab').alias('_n'), pl.col('rel_rec_tab').alias('_rel'))
    T = T.join(d, on=['s1_idx', 'cand_idx'], how='left')
    agg = dense.group_by('cand_idx').agg(pl.len().cast(pl.Int32).alias('_nd'), pl.col('cos').max().alias('_mx'))
    T = T.join(agg, on='cand_idx', how='left').with_columns(pl.col('_nd').fill_null(0))
    keys = _keys(dense['cand_idx'].to_numpy(), dense['cos'].to_numpy())
    nd = T['_rk'].is_null().to_numpy(); cand = T['cand_idx'].to_numpy(); cos = T['cos'].to_numpy()
    rk = np.zeros(T.height, np.float32); rk[nd] = 1 + _count_greater(keys, cand[nd], cos[nd])
    T = T.with_columns(pl.Series('_rk2', rk))
    T = T.with_columns(
        pl.when(pl.col('_rk').is_not_null()).then(pl.col('_rk').cast(pl.Float32)).otherwise(pl.col('_rk2')).cast(pl.Int16).alias('rank_rec_tab'),
        pl.when(pl.col('_n').is_not_null()).then(pl.col('_n').cast(pl.Int32)).otherwise(pl.col('_nd') + 1).cast(pl.Int16).alias('n_s1_rec_tab'),
        pl.when(pl.col('_rel').is_not_null()).then(pl.col('_rel'))
          .otherwise(pl.when(pl.col('_mx').is_null()).then(0.0).otherwise(pl.min_horizontal(pl.lit(0.0), pl.col('cos') - pl.col('_mx')))).cast(pl.Float32).alias('rel_rec_tab'))
    return T.drop('_rk', '_n', '_rel', '_nd', '_mx', '_rk2')


def recside_stage2(P: pl.DataFrame, comp: pl.DataFrame, pcol='p1', prefix='p1') -> pl.DataFrame:
    """Record-side competition for pairs P (s1_idx, cand_idx, pcol) against competitor pairs comp (s1_idx, cand_idx, q) of ALL S1:
    {prefix}_max_other_rec, _rel_other, _rank_rec, _n05_other_rec (other = any other S1 claiming the record; max_other = 0 if none)."""
    c = comp.select('s1_idx', 'cand_idx', 'q')
    agg = c.sort('cand_idx', 'q', descending=[False, True]).group_by('cand_idx', maintain_order=True).agg(
        pl.col('q').first().alias('_m1'), pl.col('s1_idx').first().alias('_a1'), pl.col('q').slice(1, 1).first().alias('_m2'),
        (pl.col('q') > 0.5).sum().cast(pl.Int32).alias('_c5'))
    P = P.join(agg, on='cand_idx', how='left').join(c.rename({'q': '_self'}), on=['s1_idx', 'cand_idx'], how='left')
    keys = _keys(c['cand_idx'].to_numpy(), c['q'].to_numpy())
    cand = P['cand_idx'].to_numpy(); p = P[pcol].to_numpy()
    gt = _count_greater(keys, cand, p)
    selfgt = (P['_self'].fill_null(-1.0).to_numpy() > p).astype(np.int64)
    P = P.with_columns(pl.Series('_gt', (gt - selfgt).astype(np.int32)))
    P = P.with_columns(
        pl.when(pl.col('_m1').is_null()).then(0.0)
          .when(pl.col('_a1') == pl.col('s1_idx')).then(pl.col('_m2').fill_null(0.0)).otherwise(pl.col('_m1')).cast(pl.Float32).alias(f'{prefix}_max_other_rec'),
        (pl.col('_c5').fill_null(0) - (pl.col('_self').fill_null(0.0) > 0.5).cast(pl.Int32)).cast(pl.Int16).alias(f'{prefix}_n05_other_rec'),
        (pl.col('_gt') + 1).cast(pl.Int16).alias(f'{prefix}_rank_rec'))
    P = P.with_columns((pl.col(pcol) - pl.col(f'{prefix}_max_other_rec')).cast(pl.Float32).alias(f'{prefix}_rel_other'))
    return P.drop('_m1', '_a1', '_m2', '_c5', '_self', '_gt')
