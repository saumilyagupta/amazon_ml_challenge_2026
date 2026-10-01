"""Matcher v3, change D: the France-safe 14-feature subset of the team's reverse-engineering branch features (branch subset Q14).

The branch (github saumilyagupta/KL-convergence, feat/features-reverse-engineering @ b14e566: scripts/explain_pairs.py, scripts/features.py;
research port work/features/branch_port/src) computes 54 per-pair features with its own noise explainer. On top of our explainer library
a 16-feature subset Q carried the whole significant gain (stage 1 +0.00054 [+0.00041, +0.00067] on 220,730 validation S1); v3 uses Q minus
bp_x_n_ops_addr / bp_x_loose (Q14): those two count address operations with the branch's US/India address lists, which lack French street
types and departments (France-unsafe). Columns are stored with the prefix 'bp_' (the branch reuses v1 feature names).

    ctx = load_ctx(W)                          # learned filler lists (resources/branch_lists/coverage.json) + per-country IDF (cached)
    Q = run_parallel(pairs, rec, ctx, 8)       # pairs: s1_id, cand_id; rec: ber.explainer.load_records(W, split) -> frame of Q14 columns

Learned inputs (label use: train-split pairs only):
  coverage.json  name / address filler lists learned by explain_pairs.learn() (filler-min 15) on one half of a 200k stratified sample of
                 TRAIN-SPLIT ground-truth pairs (tools/learn_branch_lists.py re-creates it; validation S1 excluded)
  IDF            explain_pairs-folded token IDF over S1 names / addresses per country (features.build_idf, unsupervised): countries present in
                 the TRAIN S1 file (US, India) from the train S1, other countries (France) from the TEST S1 (transductive, label-free).
                 The same tables are used for train and test features. Cached as WORK/records/branch_idf.pkl (~95 s to build)."""
import json, os, pickle, time
import multiprocessing as mp
import numpy as np, polars as pl
from ..paths import RES
from . import features as F, fastpatch

Q16 = ['bp_n_shared_idf', 'bp_n_added_all_filler', 'bp_n_added_max_idf', 'bp_n_len_ratio', 'bp_n_left_count', 'bp_n_left_max_idf',
       'bp_a_num_rel', 'bp_a_num_offset', 'bp_a_street_words_sim', 'bp_a_street_words_jacc', 'bp_x_legal_form', 'bp_x_n_ops_name',
       'bp_x_n_ops_addr', 'bp_x_loose', 'bp_x_word_dropped', 'bp_x_token_corrupted']
FRANCE_UNSAFE = ['bp_x_n_ops_addr', 'bp_x_loose']
Q14 = [q for q in Q16 if q not in FRANCE_UNSAFE]
COVERAGE = os.path.join(RES, 'branch_lists', 'coverage.json')
PROBE = ('Acme Widgets Inc', '1 Main St, Springfield, IL', 'ACME WIDGETS', '1 MAIN ST, SPRINGFIELD, IL', 'S2', 'US')


def build_idf(W, log=print, need=()):
    """Branch recipe: features.build_idf over S1 names / addresses per country; train S1 for the countries of the train file, test S1 otherwise.
    need: countries that must have a table (e.g. the countries of the split being featurised); a cache without them is rebuilt (this happens
    only if prepare.py had not yet written the other split when the cache was created)."""
    path = W.p('records', 'branch_idf.pkl')
    if os.path.exists(path):
        idf = pickle.load(open(path, 'rb'))
        if all(('name', c) in idf for c in need):
            return idf
        log('branch idf cache lacks', sorted(c for c in need if ('name', c) not in idf), '-> rebuilt')
    names, addrs, seen = {}, {}, set()
    for split in ('train', 'test'):
        if not os.path.exists(W.records(split, 's1')):
            continue
        r = pl.read_parquet(W.records(split, 's1'), columns=['business_name', 'business_address', 'country'])
        for c in sorted(set(r['country'].unique().to_list()) - seen):
            g = r.filter(pl.col('country') == c)
            names[c], addrs[c] = g['business_name'].to_list(), g['business_address'].to_list(); seen.add(c)
            log(f'branch idf docs {c}: {g.height} {split} S1')
    t = time.time(); idf = F.build_idf(names, addrs); log(f'branch build_idf {time.time() - t:.0f}s', {k: len(v[0]) for k, v in idf.items()})
    pickle.dump(idf, open(path + '.tmp', 'wb')); os.replace(path + '.tmp', path)
    return idf


def load_ctx(W, log=print, coverage=COVERAGE, need=()):
    cov = json.load(open(coverage))
    return {'name_filler': set(cov['name_filler']), 'addr_filler': set(cov['addr_filler']), 'idf': build_idf(W, log, need)}


_CTX = None; _FEATS = None; _KEEP = None


def feature_names(ctx):
    return list(F.pair_features(*PROBE, ctx).keys())


def _init(ctx, feats, keep):
    global _CTX, _FEATS, _KEEP
    _CTX, _FEATS, _KEEP = ctx, feats, keep
    fastpatch.apply(); fastpatch.register(ctx['name_filler'], ctx['addr_filler'])   # memoisation, values identical to the unpatched code


def _work(rows):
    """rows: list of (a_name, a_addr, b_name, b_addr, src, country) -> float32 [n, len(keep)]."""
    out = np.empty((len(rows), len(_KEEP)), np.float32); pos = [_FEATS.index(k) for k in _KEEP]
    for i, (an, aa, bn, ba, src, c) in enumerate(rows):
        v = list(F.pair_features(an, aa, bn, ba, src, c, _CTX).values())
        out[i] = [v[j] for j in pos]
    return out


def attach_text(P, rec):
    """P: s1_id, cand_id -> + a_name, a_addr, country (S1 side; raw text) and b_name, b_addr, src ('S2'/'S3')."""
    R1 = rec.filter(pl.col('src') == 1).select(pl.col('entity_id').alias('s1_id'), pl.col('name').alias('a_name'), pl.col('addr').alias('a_addr'), 'country')
    R2 = rec.filter(pl.col('src') != 1).select(pl.col('entity_id').alias('cand_id'), pl.col('name').alias('b_name'), pl.col('addr').alias('b_addr'),
                                               ('S' + pl.col('src').cast(pl.Utf8)).alias('src'))
    Q = P.select('s1_id', 'cand_id').join(R1, on='s1_id', how='left', maintain_order='left').join(R2, on='cand_id', how='left', maintain_order='left')
    for c in ('a_name', 'a_addr', 'country', 'b_name', 'b_addr', 'src'):
        assert Q[c].null_count() == 0, f'{c}: {Q[c].null_count()} unmatched ids'
    return Q


def run_parallel(P, rec, ctx, n_workers=8, chunk=20000, keep=None):
    """P: s1_id, cand_id (any order) -> polars frame of the `keep` columns (default Q14, 'bp_' prefix) in the input row order."""
    keep = [k[3:] for k in (keep or Q14)]; feats = feature_names(ctx)
    Q = attach_text(P, rec)
    rows = list(zip(Q['a_name'].to_list(), Q['a_addr'].to_list(), Q['b_name'].to_list(), Q['b_addr'].to_list(), Q['src'].to_list(), Q['country'].to_list()))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    if n_workers <= 1 or len(chunks) <= 1:
        _init(ctx, feats, keep); X = np.concatenate([_work(c) for c in chunks]) if chunks else np.zeros((0, len(keep)), np.float32)
    else:
        with mp.get_context('fork').Pool(n_workers, initializer=_init, initargs=(ctx, feats, keep)) as pool:
            X = np.concatenate(list(pool.imap(_work, chunks, chunksize=1)))
    return pl.DataFrame({'bp_' + k: X[:, j] for j, k in enumerate(keep)})
