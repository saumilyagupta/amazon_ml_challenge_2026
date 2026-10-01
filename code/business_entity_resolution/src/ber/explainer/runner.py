"""Chunked multiprocessing runner (fork, <= 16 workers) for tens of millions of pairs.

run_parallel(pairs, records, n_workers=16, chunk_size=20000) -> polars.DataFrame (input row order)
run_to_parquet(pairs, records, out_dir, part_rows=2_000_000)   -> writes out_dir/part_XXX.parquet (restartable)
pairs: polars DataFrame with s1_id, cand_id (other columns are ignored; join them back on the row order / ids).
records: polars DataFrame from explainer.io.load_records(split) (entity_id, name, addr, country, src, name_en, addr_en).
The parent joins the texts (polars, fast); workers compute views once per record within their chunk and all pair features.
"""
import os
import time

import numpy as np
import polars as pl

from .features import features_from_texts, ALL_FEATURES, FEATURES
from .parallel import MAX_WORKERS
from .wordlists import all_tables

_TCOLS = ['name', 'addr', 'country', 'src', 'name_en', 'addr_en']


def _join_texts(P, records):
    r = records
    if 'src' not in r.columns:
        r = r.with_columns(pl.col('entity_id').str.slice(1, 1).cast(pl.Int8).alias('src'))
    for c in ('name_en', 'addr_en'):
        if c not in r.columns:
            r = r.with_columns(pl.lit(None, pl.Utf8).alias(c))
    r = r.select(['entity_id'] + _TCOLS)
    ra = r.rename({c: 'a_' + c for c in _TCOLS}).rename({'entity_id': 's1_id'})
    rb = r.rename({c: 'b_' + c for c in _TCOLS}).rename({'entity_id': 'cand_id'})
    out = P.join(ra, on='s1_id', how='left').join(rb, on='cand_id', how='left').sort('_row')
    miss = out.filter(pl.col('a_country').is_null() | pl.col('b_country').is_null()).height
    if miss:
        raise ValueError(f'{miss} pairs reference ids missing from records')
    return out


def _worker(task):
    t0 = time.time()
    a_ids, b_ids, a_rows, b_rows, vec = task
    a_rec = dict(zip(a_ids, a_rows))
    b_rec = dict(zip(b_ids, b_rows))
    out = features_from_texts(a_ids, a_rec, b_ids, b_rec, vec=vec)
    return out, len(a_ids), time.time() - t0


def _tasks(J, chunk_size, vec=True):
    a_cols = ['a_' + c for c in _TCOLS]
    b_cols = ['b_' + c for c in _TCOLS]
    for lo in range(0, J.height, chunk_size):
        c = J.slice(lo, chunk_size)
        yield (c['s1_id'].to_list(), c['cand_id'].to_list(), c.select(a_cols).rows(), c.select(b_cols).rows(), vec)


def run_parallel(pairs, records, n_workers=16, chunk_size=20000, stats=None, verbose=False, vec=True):
    """vec=False skips the two char-TF-IDF columns (n3_core_tfidf, a13_addr_tfidf; ~20% faster)."""
    import multiprocessing as mp
    all_tables()  # load tables in the parent so forked workers share them
    P = pairs.select('s1_id', 'cand_id').with_row_index('_row')
    J = _join_texts(P, records)
    n = max(1, min(n_workers, MAX_WORKERS))
    t0 = time.time()
    outs, busy, npairs = [], 0.0, 0
    if n == 1:
        it = map(_worker, _tasks(J, chunk_size, vec))
        pool = None
    else:
        pool = mp.get_context('fork').Pool(n)
        it = pool.imap(_worker, _tasks(J, chunk_size, vec), chunksize=1)
    try:
        for out, k, dt in it:
            outs.append(out)
            busy += dt
            npairs += k
            if verbose:
                print(f'  {npairs}/{J.height} pairs  {npairs / (time.time() - t0):.0f} pairs/s wall', flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    wall = time.time() - t0
    cols = {'s1_id': J['s1_id'], 'cand_id': J['cand_id']}
    for f in (ALL_FEATURES if vec else FEATURES):
        cols[f] = np.concatenate([o[f] for o in outs]) if outs else np.zeros(0, np.float32)
    if stats is not None:
        stats.update(pairs=npairs, wall_s=wall, pairs_per_s_wall=npairs / max(wall, 1e-9),
                     pairs_per_s_per_worker=npairs / max(busy, 1e-9), workers=n)
    return pl.DataFrame(cols)


def run_to_parquet(pairs, records, out_dir, part_rows=2_000_000, n_workers=16, chunk_size=20000, log=print, vec=True):
    """Feature a very large pair table in parts; existing parts are skipped (restartable)."""
    os.makedirs(out_dir, exist_ok=True)
    nparts = (pairs.height + part_rows - 1) // part_rows
    for k in range(nparts):
        out = f'{out_dir}/part_{k:04d}.parquet'
        if os.path.exists(out):
            continue
        st = {}
        X = run_parallel(pairs.slice(k * part_rows, part_rows), records, n_workers, chunk_size, stats=st, vec=vec)
        X.write_parquet(out + '.tmp')
        os.replace(out + '.tmp', out)
        log(f'part {k + 1}/{nparts}: {X.height} pairs, {st["pairs_per_s_wall"]:.0f} pairs/s wall, '
            f'{st["pairs_per_s_per_worker"]:.0f} pairs/s/worker')
