"""Blocking union (REC20) with per-channel provenance, in the schema used by the matcher (union v1 / union v2 of the research runs).

Members: dense fwd@20 (encoder = dense_member) | dense reverse gated (rank 1, or rank<=5 with top1-gap < gate) | C1@20 | C1r rank 1 |
         C2r rank 1 | C3r (rank 1, or rank<=5 with gap < 0.05) | C3@10 | C2@10.
Every member pair gets ALL available channel evidence by left joins (rank 0 = absent): zero-shot dense fr/fs (forward rank/cos),
rr/rs/rd (reverse rank/cos/gap to the record's top-1), lexical c1.., c3rg; with the fine-tuned encoder also fr_ft.. rd_ft.
in_fwd / in_rev always keep the ZERO-SHOT definitions (fwd<=20, reverse gated 0.02) as evidence flags; in_fwd_ft / in_rev_ft are the
fine-tuned ones. fused = max over member channels of 1/rank + 0.01*sum(1/rank) + 1e-4*max(dense cos of the member encoder);
fused_hd damps the reverse channels of hub S1s (x min(1, 20/#members of that channel)); ranks within S1 break ties by cand_idx."""
import os, time
import numpy as np, polars as pl

LEX = {'c1': ('nk',), 'c2': ('nk',), 'c3': (), 'c1r': ('nk', 'gap'), 'c2r': ('nk', 'gap'), 'c3r': ('gap',)}
V1COLS = ['s1_id', 'cand_id', 'in_fwd', 'in_rev', 'in_c1', 'in_c1r', 'in_c2r', 'in_c3r', 'in_c3', 'in_c2', 'fr', 'fs', 'rr', 'rs', 'rd', 'c1', 'c1s', 'c1nk', 'c2', 'c2s', 'c2nk',
          'c3', 'c3s', 'c1r', 'c1rs', 'c1rnk', 'c1rg', 'c2r', 'c2rs', 'c2rnk', 'c2rg', 'c3r', 'c3rs', 'c3rg', 'cand_idx', 's1_idx', 'n_member_channels', 'fused', 'fused_hd', 'fused_rank', 'fused_hd_rank']
FTCOLS = ['in_fwd_ft', 'in_rev_ft', 'fr_ft', 'fs_ft', 'rr_ft', 'rs_ft', 'rd_ft', 'ft_seen', 'ft_seen_s1']


def build(W, split, name, qmask, dense_member='zs', gate=0.02, fwd_k=20, c1_k=20, c3_k=10, c2_k=10, c3r_delta=0.05, zs_gate=0.02, ft_seen_pairs=None, log=print):
    out = W.union(name, split)
    if os.path.exists(out): log('skip union (exists)', out); return
    t0 = time.time()
    n23 = pl.read_parquet(W.records(split, 's23'), columns=['entity_id']).height
    assert n23 < 16777216
    KEY = (pl.col('s1_idx').cast(pl.Int64) * 16777216 + pl.col('cand_idx').cast(pl.Int64)).alias('key')
    qidx = pl.DataFrame({'s1_idx': np.where(qmask)[0].astype(np.int32)})
    def dense(enc, sfx):
        F = pl.scan_parquet(W.forward(enc, split)).select('s1_idx', 'cand_idx', pl.col('rank').cast(pl.Int16).alias('fr' + sfx), pl.col('score').alias('fs' + sfx))
        F = F.join(qidx.lazy(), on='s1_idx', how='semi').with_columns(KEY).drop('s1_idx', 'cand_idx').collect()
        R = (pl.scan_parquet(W.reverse(enc, split)).select('cand_idx', 's1_idx', pl.col('rank').cast(pl.Int16).alias('rr' + sfx), pl.col('score').alias('rs' + sfx))
             .with_columns((pl.col('rs' + sfx).max().over('cand_idx') - pl.col('rs' + sfx)).alias('rd' + sfx)))
        R = R.join(qidx.lazy(), on='s1_idx', how='semi').with_columns(KEY).drop('s1_idx', 'cand_idx').collect()
        return F, R
    Fz, Rz = dense('zs', ''); log('zero-shot dense evidence', Fz.height, Rz.height)
    ft = dense_member == 'ft' or os.path.exists(W.forward('ft', split))
    if ft:
        Ff, Rf = dense('ft', '_ft'); log('fine-tuned dense evidence', Ff.height, Rf.height)
    def ch(nm, extra):
        cols = [pl.col('rank').cast(pl.Int16).alias(nm), pl.col('score').alias(nm + 's')]
        if 'nk' in extra: cols.append(pl.col('nk').alias(nm + 'nk'))
        if 'gap' in extra: cols.append(pl.col('gap').alias(nm + 'g'))
        return pl.scan_parquet(W.lexical(nm, split)).select('s1_idx', 'cand_idx', *cols).join(qidx.lazy(), on='s1_idx', how='semi').with_columns(KEY).drop('s1_idx', 'cand_idx').collect()
    C = {k: ch(k, v) for k, v in LEX.items()}
    dsfx = '_ft' if dense_member == 'ft' else ''
    Fd, Rd = (Ff, Rf) if dense_member == 'ft' else (Fz, Rz)
    MEM = {'fwd': (Fd, pl.col('fr' + dsfx) <= fwd_k), 'rev': (Rd, (pl.col('rr' + dsfx) == 1) | ((pl.col('rr' + dsfx) <= 5) & (pl.col('rd' + dsfx) < gate))),
           'c1': (C['c1'], pl.col('c1') <= c1_k), 'c1r': (C['c1r'], pl.col('c1r') == 1), 'c2r': (C['c2r'], pl.col('c2r') == 1),
           'c3r': (C['c3r'], (pl.col('c3r') == 1) | ((pl.col('c3r') <= 5) & (pl.col('c3rg') < c3r_delta))), 'c3': (C['c3'], pl.col('c3') <= c3_k), 'c2': (C['c2'], pl.col('c2') <= c2_k)}
    keys = pl.concat([t.filter(f).select('key') for t, f in MEM.values()]).unique()
    U = keys.join(Fz, on='key', how='left').join(Rz, on='key', how='left')
    if ft: U = U.join(Ff, on='key', how='left').join(Rf, on='key', how='left')
    for k, d in C.items(): U = U.join(d, on='key', how='left')
    log('member keys', keys.height, 'evidence joined', f'{time.time() - t0:.0f}s')
    def le(c, k): return pl.col(c).is_not_null() & (pl.col(c) <= k)
    def eq1(c): return pl.col(c).is_not_null() & (pl.col(c) == 1)
    U = U.with_columns(le('fr', 20).alias('in_fwd'), (eq1('rr') | (le('rr', 5) & (pl.col('rd') < zs_gate))).alias('in_rev'),
                       le('c1', c1_k).alias('in_c1'), eq1('c1r').alias('in_c1r'), eq1('c2r').alias('in_c2r'),
                       (eq1('c3r') | (le('c3r', 5) & (pl.col('c3rg') < c3r_delta))).alias('in_c3r'), le('c3', c3_k).alias('in_c3'), le('c2', c2_k).alias('in_c2'))
    if ft:
        U = U.with_columns(le('fr_ft', fwd_k).alias('in_fwd_ft'), (eq1('rr_ft') | (le('rr_ft', 5) & (pl.col('rd_ft') < gate))).alias('in_rev_ft'))
    U = U.with_columns((pl.col('key') // 16777216).cast(pl.Int32).alias('s1_idx'), (pl.col('key') % 16777216).cast(pl.Int32).alias('cand_idx'))
    if ft:
        if ft_seen_pairs is not None and split == 'train':
            P = ft_seen_pairs.with_columns(KEY)
            U = U.with_columns(pl.col('key').is_in(P['key'].implode()).alias('ft_seen'), pl.col('s1_idx').is_in(P['s1_idx'].unique().implode()).alias('ft_seen_s1'))
        else:
            U = U.with_columns(pl.lit(False).alias('ft_seen'), pl.lit(False).alias('ft_seen_s1'))
    pairs = [('fr' + dsfx, 'in_fwd' + dsfx), ('rr' + dsfx, 'in_rev' + dsfx), ('c1', 'in_c1'), ('c1r', 'in_c1r'), ('c2r', 'in_c2r'), ('c3r', 'in_c3r'), ('c2', 'in_c2'), ('c3', 'in_c3')]
    flags = [f for _, f in pairs]
    def rrk(c, flag): return pl.when(pl.col(flag) & pl.col(c).is_not_null()).then(1.0 / pl.col(c).cast(pl.Float64)).otherwise(0.0)
    rrs = [rrk(c, f) for c, f in pairs]
    cos = 1e-4 * pl.max_horizontal(pl.col('fs' + dsfx).fill_null(0), pl.col('rs' + dsfx).fill_null(0))
    U = U.with_columns(pl.sum_horizontal([pl.col(f).cast(pl.UInt8) for f in flags]).cast(pl.UInt8).alias('n_member_channels'),
                       (pl.max_horizontal(rrs) + 0.01 * pl.sum_horizontal(rrs) + cos).cast(pl.Float32).alias('fused'))
    REVF = {'rr' + dsfx, 'c1r', 'c2r', 'c3r'}
    hd = [rrk(c, f) * (pl.min_horizontal(pl.lit(1.0), 20.0 / pl.col(f).cast(pl.Float64).sum().over('s1_idx').clip(1, None)) if c in REVF else 1.0) for c, f in pairs]
    U = U.with_columns((pl.max_horizontal(hd) + 0.01 * pl.sum_horizontal(hd) + cos).cast(pl.Float32).alias('fused_hd'))
    U = U.sort(['s1_idx', 'fused_hd', 'cand_idx'], descending=[False, True, False]).with_columns(pl.int_range(1, pl.len() + 1).over('s1_idx').cast(pl.Int32).alias('fused_hd_rank'))
    U = U.sort(['s1_idx', 'fused', 'cand_idx'], descending=[False, True, False]).with_columns(pl.int_range(1, pl.len() + 1).over('s1_idx').cast(pl.Int32).alias('fused_rank'))
    RKC = ['fr', 'rr', 'c1', 'c1r', 'c2', 'c2r', 'c3', 'c3r'] + (['fr_ft', 'rr_ft'] if ft else [])
    U = U.with_columns([pl.col(c).fill_null(0).cast(pl.Int16) for c in RKC])
    ids1 = pl.read_parquet(W.records(split, 's1'), columns=['entity_id']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32)).rename({'entity_id': 's1_id'})
    ids2 = pl.read_parquet(W.records(split, 's23'), columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32)).rename({'entity_id': 'cand_id'})
    U = U.join(ids1, on='s1_idx', how='left').join(ids2, on='cand_idx', how='left')
    U = U.select(V1COLS + (FTCOLS if ft else [])).sort(['s1_idx', 'fused_rank'])
    U.write_parquet(out + '.tmp'); os.rename(out + '.tmp', out)
    n = U.group_by('s1_idx').len()['len']
    log(f'wrote {out}: {U.height} rows, {n.len()} S1, mean/p95/max cands {n.mean():.2f}/{n.quantile(0.95)}/{n.max()}',
        {f: int(U[f].sum()) for f in flags}, f'{time.time() - t0:.0f}s')
