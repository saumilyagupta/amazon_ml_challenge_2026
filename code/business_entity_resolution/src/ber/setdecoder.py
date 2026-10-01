"""Learned set decoders (copied from the research decision library work/research/postproc/decide_lib.py; logic unchanged).

Input U: polars frame with s1, qid (candidate record), src ('S2'/'S3'), p (stage-2 probability) and qr = exclusivity rank of the pair for
its record. Matcher v2b uses the ARGMAX-VS-BEST-OTHER exclusivity: qr = 1 iff p >= p_other + margin (margin 0), where p_other is the best
OTHER S1's competitor probability for the same record over the dense table of ALL S1 of the split (ber.decide.add_comp), else qr = 2.
R10c (final v2b policy): for every S1 the surviving (qr = 1, p > 0.02) candidates are sorted by p; a LightGBM regressor predicts the
realised F0.5 of each prefix k = 0..10 from S1-level aggregates + prefix statistics; the prefix with the highest prediction is output.
It is trained on the OUT-OF-FOLD scores of the training-sample S1 (fit_set_decoder) and applied unchanged to validation / test.
R7g (count gate): singleton classifier + Poisson count model (fit_s1_models / decide); kept for completeness."""
import numpy as np, polars as pl, lightgbm as lgb

FE = ['pmax', 'psum', 'n50', 'n20', 'n80', 'ncand_best', 'p2nd', 'n50_s2', 'n50_s3', 'pmax_all', 'ncand_all']
KMAX = 10


def f05(pred, true):
    """pred/true are sets. Singleton rule: true empty -> 1.0 if pred empty else 0.0 (same as tools/score.py)."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred); r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def add_qrank(U): return U.with_columns(pl.col('p').rank('ordinal', descending=True).over('qid').alias('qr'))


def s1_features(U):
    """U must already have 'qr' computed over the FULL universe; restrict to the S1s of interest afterwards."""
    Aq = U.filter(pl.col('qr') == 1)
    f = Aq.group_by('s1').agg([pl.col('p').max().alias('pmax'), pl.col('p').sum().alias('psum'), (pl.col('p') > 0.5).sum().alias('n50'),
                               (pl.col('p') > 0.2).sum().alias('n20'), (pl.col('p') > 0.8).sum().alias('n80'), pl.len().alias('ncand_best'),
                               pl.col('p').top_k(2).min().alias('p2nd'),
                               ((pl.col('src') == 'S2') & (pl.col('p') > 0.5)).sum().alias('n50_s2'), ((pl.col('src') == 'S3') & (pl.col('p') > 0.5)).sum().alias('n50_s3')])
    g = U.group_by('s1').agg([pl.col('p').max().alias('pmax_all'), pl.len().alias('ncand_all')])
    return f.join(g, on='s1', how='full', coalesce=True).fill_null(0)


def fit_s1_models(F, n_true, threads=8):
    """F: s1_features on OUT-OF-FOLD pair scores of training S1s; n_true: array of true match counts (0 = singleton)."""
    X = F.select(FE).to_numpy()
    ms = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=31, verbose=-1, num_threads=threads), lgb.Dataset(X, (n_true == 0).astype(int)), 300)
    mc = lgb.train(dict(objective='poisson', learning_rate=0.05, num_leaves=31, verbose=-1, num_threads=threads), lgb.Dataset(X, n_true), 300)
    return ms, mc


def decide(U, ms, mc, T=0.55, SLACK=-0.25, TAU=0.6, s1_subset=None):
    """R7g. Returns dict s1 -> set(qid). S1s absent from the output must be written as empty rows."""
    if 'qr' not in U.columns: U = add_qrank(U)
    if s1_subset is not None: U = U.filter(pl.col('s1').is_in(list(s1_subset)))
    F = s1_features(U); X = F.select(FE).to_numpy()
    P0 = dict(zip(F['s1'].to_list(), ms.predict(X))); C = dict(zip(F['s1'].to_list(), mc.predict(X)))
    out = {}
    W = U.filter(pl.col('qr') == 1).sort(['s1', 'p'], descending=[False, True])
    for s1, qids, ps in W.group_by('s1', maintain_order=True).agg('qid', 'p').iter_rows():
        if P0[s1] > TAU: out[s1] = set(); continue
        k = max(int(np.round(C[s1] + SLACK)), 0)
        out[s1] = set(q for q, p in zip(qids[:k], ps[:k]) if p > T)
    return out


# ---------------- R10c: learned set decoder ----------------
# deliberately no q-side margin feature: it needs the competing S1 rows, which the OOF training table lacks; p (stage 2) already contains it.
def prefix_table(U, gt=None):
    """U: s1,qid,src,p,qr (qr over the full universe). Returns X (float32), y (realised F0.5 per prefix or None), keys [(s1,k)], lists {s1:[qid sorted by p]}."""
    F = s1_features(U); Fd = {r[0]: r[1:] for r in F.select(['s1'] + FE).iter_rows()}
    U = U.with_columns((pl.col('src') == 'S3').cast(pl.Int8).alias('is3'))
    W = U.filter((pl.col('qr') == 1) & (pl.col('p') > 0.02)).sort(['s1', 'p'], descending=[False, True]).group_by('s1', maintain_order=True).agg('qid', 'p', 'is3')
    X = []; y = []; key = []; lists = {}
    for s1, qids, ps, is3 in W.iter_rows():
        lists[s1] = qids; ps = np.array(ps); n = len(ps); cs = np.concatenate([[0], np.cumsum(ps)])
        c3 = np.concatenate([[0], np.cumsum(is3)])
        for k in range(0, min(n, KMAX) + 1):
            pk = ps[k - 1] if k > 0 else 1.0; pn = ps[k] if k < n else 0.0
            X.append(list(Fd[s1]) + [k, pk, pn, cs[k], cs[n] - cs[k], pk - pn, n, c3[k], k - c3[k], cs[k] / max(k, 1)]); key.append((s1, k))
            if gt is not None: y.append(f05(set(qids[:k]), gt[s1]))
    return np.array(X, np.float32), (np.array(y, np.float32) if gt is not None else None), key, lists


def fit_set_decoder(U_oof, gt, threads=8):
    """U_oof: OUT-OF-FOLD scored universe of training S1s (qr computed over the full universe); gt: s1 -> set of true ids."""
    X, y, _, _ = prefix_table(U_oof, gt)
    return lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, verbose=-1, num_threads=threads), lgb.Dataset(X, y), 600)


def decide_set(U, model):
    """Returns s1 -> set(qid); S1s with no surviving candidate are absent (= empty prediction)."""
    if 'qr' not in U.columns: U = add_qrank(U)
    X, _, key, lists = prefix_table(U); pr = model.predict(X); best = {}
    for (s1, k), v in zip(key, pr):
        if s1 not in best or v > best[s1][1]: best[s1] = (k, v)
    return {s1: set(lists[s1][:k]) for s1, (k, v) in best.items()}


# ---------------- glue for the package's (s1_idx, cand_idx) frames ----------------
def universe(Pc, margin=0.0):
    """Pc: s1_idx, cand_idx, src ('S2'/'S3'), p, p_other (ber.decide.add_comp). -> decide_lib universe with argmax-vs-best-other qr."""
    return Pc.select(pl.col('s1_idx').alias('s1'), pl.col('cand_idx').alias('qid'), 'src', 'p',
                     pl.when(pl.col('p') >= pl.col('p_other') + margin).then(1).otherwise(2).alias('qr'))


def to_frame(dec):
    rows = [(s, q) for s, qs in dec.items() for q in qs]
    return pl.DataFrame(rows, schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32}, orient='row')


def truth_sets(M, P):
    """M: s1_idx, m (all truths of the S1, incl. those outside the candidates); P: s1_idx, cand_idx, label. -> s1 -> set of true ids
    (missing truths as negative placeholders, so that F0.5 counts them)."""
    tr = P.filter(pl.col('label') == 1).group_by('s1_idx').agg(pl.col('cand_idx'))
    d = {s: set(c) for s, c in tr.iter_rows()}
    return {s: d.get(s, set()) | {-(k + 1) for k in range(m - len(d.get(s, set())))} for s, m in M.select('s1_idx', 'm').iter_rows()}


# ---------------- matcher v3: own-probability competition + strict one-owner (research work/matching/prod_v3/07_decide.py, 08_test.py) ----------------
def own_competition(P, comp_other=None, pcol='p2'):
    """competitor table (s1_idx, cand_idx, q) for the exclusivity rule with the pipeline's OWN stage-2 probabilities: every pair this variant
    scores carries its own p2; pairs it does not score keep the other table's q (train: v1's p2 over the dense table of all train S1 for the
    ~72% of train S1 outside the union file). Test: comp_other=None -> own pairs only (every test S1 is scored)."""
    own = P.select('s1_idx', 'cand_idx', pl.col(pcol).alias('q'))
    if comp_other is None:
        return own
    return pl.concat([comp_other.select('s1_idx', 'cand_idx', 'q').join(own, on=['s1_idx', 'cand_idx'], how='anti'), own])


def one_owner(sel):
    """sel: s1_idx, cand_idx, p (selected pairs). Every S2/S3 record keeps only the selecting S1 with the highest p (ties: lowest s1_idx)."""
    return sel.sort(['cand_idx', 'p', 's1_idx'], descending=[False, True, False]).group_by('cand_idx', maintain_order=True).head(1)


def multi_owner(sel):
    g = sel.group_by('cand_idx').agg(pl.len().alias('n'))
    return dict(records=int((g['n'] > 1).sum()), pairs=int(g.filter(pl.col('n') > 1)['n'].sum()))
