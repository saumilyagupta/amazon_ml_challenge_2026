"""Labels and per-S1 truth counts (train split)."""
import polars as pl
from .cands import ids_to_idx


def gt_idx(W, split='train'):
    """labelled pairs as (s1_idx, cand_idx, label=1)."""
    G = pl.read_parquet(W.gt_pairs(split))
    return ids_to_idx(W, G, split).select('s1_idx', 'cand_idx', pl.lit(1, pl.Int8).alias('label'))


def truth_counts(W, P, G, grp_frame, grps, split='train'):
    """per-S1 frame for the S1 in `grps`: s1_idx, entity_id, country, grp, m (all truths in the GT), r (truths retained in P)."""
    ids = pl.read_parquet(W.records(split, 's1'), columns=['entity_id', 'country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    M = ids.join(grp_frame.select('s1_idx', 'grp'), on='s1_idx').filter(pl.col('grp').is_in(grps))
    m = G.group_by('s1_idx').agg(pl.len().cast(pl.Int32).alias('m'))
    r = P.join(M.select('s1_idx'), on='s1_idx', how='semi').group_by('s1_idx').agg(pl.col('label').cast(pl.Int32).sum().alias('r'))
    return M.join(m, on='s1_idx', how='left').join(r, on='s1_idx', how='left').with_columns(pl.col('m').fill_null(0), pl.col('r').fill_null(0))
