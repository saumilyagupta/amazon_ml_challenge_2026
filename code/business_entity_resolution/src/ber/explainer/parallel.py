"""Tiny fork-based process pool helper (<= 16 workers, per the CPU protocol)."""
import multiprocessing as mp

MAX_WORKERS = 16


def pmap(func, items, n_workers=16, chunksize=1, ordered=True):
    n = max(1, min(n_workers, MAX_WORKERS))
    if n == 1:
        return [func(x) for x in items]
    ctx = mp.get_context('fork')
    with ctx.Pool(n) as pool:
        it = pool.imap(func, items, chunksize=chunksize) if ordered else pool.imap_unordered(func, items, chunksize=chunksize)
        return list(it)
