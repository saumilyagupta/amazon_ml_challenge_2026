"""Lexical blocking channels (CPU, polars joins / sparse_dot_topn; no Python loops over pairs).

C1  : compound inverted-index keys on the transliterated name view + address (re-implementation of the baseline's compound keys with
      rarest-first token selection). Per record: 4 name skeleton tokens, 2 numbers, 6 address words (rarest by pool df).
      keys s (skel) | ss (unordered skel pair) | sd (skel x number) | dw (number x word) | sw (skel x 4 rarest words) | c (whole-name skel).
C2  : address-only keys dw | dd (number pair) | ww (word pair of the 6 rarest) | w (single rare word).
      Keys with pool df > 200 are dropped; idf = ln(N_country_pool / df); score = sum of idf over shared keys; top-50 per S1.
C1r / C2r : REVERSE direction (every S2/S3 record queries an index of ALL S1 of its country, S1-side df <= 20, top-3, gap to its top-1).
C3  : name-only char_wb 3-gram TF-IDF over the sub-pool of records with an EMPTY / null address (top-10 per S1).
C3r : reverse of C3 (each empty-address record -> top-5 S1 by the same TF-IDF fit on all S1 names).
All channels are country-scoped (loop over the countries present; nothing is hard-coded, France works out of the box).
Outputs lexical/{ch}_{split}.parquet: s1_idx, cand_idx, rank, score [, nk] [, gap]."""
import os, time
import numpy as np, polars as pl
from .lexnorm import skel_word, NAME_STOP, ADDR_STOP, ABBR

TYPES = dict(s=1, ss=2, sd=3, dw=4, sw=5, c=6, dd=7, ww=8, w=9)


def _h(e): return e.hash(seed=17)


def _tokens(D, idc, ncol):
    nt = (D.select(idc, pl.col(ncol).str.split(' ').alias('tok')).explode('tok')
          .with_columns(pl.int_range(pl.len()).over(idc).alias('pos'))
          .filter(pl.col('tok').str.len_chars() >= 2, ~pl.col('tok').is_in(list(NAME_STOP))))
    nums = (D.select(idc, pl.col('addr_c').str.extract_all(r'\d+').alias('num')).explode('num').drop_nulls('num')
            .with_columns(pl.col('num').str.strip_chars_start('0')).with_columns(pl.when(pl.col('num') == '').then(pl.lit('0')).otherwise(pl.col('num')).alias('num'))
            .unique([idc, 'num'], maintain_order=True).with_columns(pl.int_range(pl.len()).over(idc).alias('pos')).filter(pl.col('pos') < 2))
    words = (D.select(idc, pl.col('addr_c').str.extract_all(r'[a-z]+').alias('w')).explode('w').drop_nulls('w')
             .with_columns(pl.col('w').replace(ABBR)).filter(pl.col('w').str.len_chars() >= 2, ~pl.col('w').is_in(list(ADDR_STOP)))
             .unique([idc, 'w'], maintain_order=True))
    return nt, nums, words


def _select_rare(T, idc, col, dfmap, k):
    T = T.join(dfmap, on=col, how='left').with_columns(pl.col('df').fill_null(0))
    return (T.sort([idc, 'df', 'hh']).with_columns(pl.int_range(pl.len()).over(idc).alias('r')).filter(pl.col('r') < k).select(idc, col, 'hh', 'r'))


def _pair_keys(A, B, idc, typ, unordered=False):
    j = A.select(idc, pl.col('hh').alias('a'), pl.col('r').alias('ra') if unordered else pl.lit(0).alias('ra')).join(
        B.select(idc, pl.col('hh').alias('b'), pl.col('r').alias('rb') if unordered else pl.lit(1).alias('rb')), on=idc)
    if unordered:
        j = j.filter(pl.col('ra') < pl.col('rb')).with_columns(pl.min_horizontal('a', 'b').alias('a2'), pl.max_horizontal('a', 'b').alias('b2')).drop('a', 'b').rename({'a2': 'a', 'b2': 'b'})
    return j.select(idc, pl.struct(pl.lit(TYPES[typ], pl.UInt8).alias('t'), 'a', 'b').hash(seed=5).alias('key'))


def _single_keys(A, idc, typ):
    return A.select(idc, pl.struct(pl.lit(TYPES[typ], pl.UInt8).alias('t'), pl.col('hh').alias('a'), pl.lit(0, pl.UInt64).alias('b')).hash(seed=5).alias('key'))


def _build_keys(D, idc, dfs, dfw, channel):
    nums, words = D['_nums'], D['_words']
    W = _select_rare(words, idc, 'w', dfw, 6)
    Nn = nums.select(idc, 'num', pl.col('pos').alias('r'), _h(pl.col('num')).alias('hh'))
    if channel == 'c1':
        S = _select_rare(D['_sk'], idc, 'sk', dfs, 4); W4 = W.filter(pl.col('r') < 4)
        out = [_single_keys(S, idc, 's'), _pair_keys(S, S, idc, 'ss', True), _pair_keys(S, Nn, idc, 'sd'), _pair_keys(Nn, W, idc, 'dw'),
               _pair_keys(S, W4, idc, 'sw'), _single_keys(D['_c'], idc, 'c')]
    else:
        out = [_pair_keys(Nn, W, idc, 'dw'), _pair_keys(Nn, Nn, idc, 'dd', True), _pair_keys(W, W, idc, 'ww', True), _single_keys(W, idc, 'w')]
    return pl.concat(out).unique()


def _prep(D, idc, ncol, skm, cmap):
    nt, nums, words = _tokens(D, idc, ncol)
    sk = (nt.join(skm, on='tok', how='left').filter(pl.col('sk').str.len_chars() >= 2).unique([idc, 'sk'], maintain_order=True)
          .select(idc, 'sk', _h(pl.col('sk')).alias('hh')))
    words = words.with_columns(_h(pl.col('w')).alias('hh'))
    c = (nt.sort([idc, 'pos']).group_by(idc).agg(pl.col('tok').str.join('').alias('cs')).join(cmap, on='cs', how='left')
         .filter(pl.col('csk').str.len_chars() >= 3).select(idc, _h(pl.col('csk')).alias('hh')))
    return {'_nums': nums, '_words': words, '_sk': sk, '_c': c, '_nt': nt}


def _skel_maps(ntA, ntB, ida, idb):
    toks = pl.concat([ntA['tok'], ntB['tok']]).unique()
    skm = pl.DataFrame({'tok': toks, 'sk': [skel_word(x) for x in toks.to_list()]})
    cs = pl.concat([ntA.sort([ida, 'pos']).group_by(ida).agg(pl.col('tok').str.join('').alias('cs'))['cs'],
                    ntB.sort([idb, 'pos']).group_by(idb).agg(pl.col('tok').str.join('').alias('cs'))['cs']]).unique()
    cmap = pl.DataFrame({'cs': cs, 'csk': [skel_word(x) for x in cs.to_list()]})
    return skm, cmap


def run_c12(W, split, qmask, channels=('c1', 'c2'), K=50, maxdf=200, shard=60000, log=print):
    """forward C1 / C2 for the query S1 (qmask: bool array over S1 rows). Pool = all S2/S3 of the split."""
    todo = [ch for ch in channels if not os.path.exists(W.lexical(ch, split))]
    if not todo: log('skip c1/c2 (exist)'); return
    ncol = 'name_c'
    P = pl.read_parquet(W.norm(split, 'pool'), columns=['country', ncol, 'addr_c']).with_row_index('rid')
    S1 = pl.read_parquet(W.norm(split, 's1'), columns=['country', ncol, 'addr_c']).with_row_index('s1_idx')
    Q = S1.filter(pl.Series(qmask)).rename({'s1_idx': 'qid'})
    RES = {ch: [] for ch in todo}; t0 = time.time()
    for cty in sorted(Q['country'].unique().to_list()):
        Pc = P.filter(pl.col('country') == cty).select('rid', ncol, 'addr_c'); Qc = Q.filter(pl.col('country') == cty).select('qid', ncol, 'addr_c')
        N = Pc.height
        if N == 0: continue
        ntP, _, _ = _tokens(Pc.select('rid', ncol, pl.lit('').alias('addr_c')), 'rid', ncol); ntQ, _, _ = _tokens(Qc.select('qid', ncol, pl.lit('').alias('addr_c')), 'qid', ncol)
        skm, cmap = _skel_maps(ntP, ntQ, 'rid', 'qid'); del ntP, ntQ
        DP = _prep(Pc, 'rid', ncol, skm, cmap); DQ = _prep(Qc, 'qid', ncol, skm, cmap)
        dfs = DP['_sk'].group_by('sk').agg(pl.len().alias('df')); dfw = DP['_words'].group_by('w').agg(pl.len().alias('df'))
        for ch in todo:
            kp = _build_keys(DP, 'rid', dfs, dfw, ch); kq = _build_keys(DQ, 'qid', dfs, dfw, ch)
            post = kp.group_by('key').agg(pl.len().alias('df')).filter(pl.col('df') <= maxdf) \
                     .with_columns((np.log(N) - pl.col('df').cast(pl.Float64).log()).cast(pl.Float32).alias('idf'))
            kq = kq.join(post.select('key'), on='key', how='semi')
            kp = kp.join(kq.select('key').unique(), on='key', how='semi').join(post.select('key', 'idf'), on='key')
            kq = kq.sort('qid'); qids = Qc['qid'].to_numpy()
            for i in range(0, len(qids), shard):
                lo, hi = qids[i], qids[min(i + shard, len(qids)) - 1]
                j = kq.filter(pl.col('qid').is_between(lo, hi)).join(kp, on='key')
                g = j.group_by('qid', 'rid').agg(pl.col('idf').sum().alias('score'), pl.len().cast(pl.UInt8).alias('nk'))
                g = g.with_columns(pl.struct('qid', 'rid').hash(seed=3).alias('tb')).sort(['qid', 'score', 'nk', 'tb'], descending=[False, True, True, False])
                g = g.with_columns((pl.int_range(pl.len()).over('qid') + 1).alias('rank')).filter(pl.col('rank') <= K)
                RES[ch].append(g.select(pl.col('qid').cast(pl.Int32).alias('s1_idx'), pl.col('rid').cast(pl.Int32).alias('cand_idx'), pl.col('rank').cast(pl.UInt8), 'score', 'nk'))
            log(f'{split} {cty} {ch} forward done {time.time() - t0:.0f}s')
    for ch in todo:
        R = pl.concat(RES[ch]).sort(['s1_idx', 'rank']); R.write_parquet(W.lexical(ch, split)); log('wrote', ch, R.height)


def run_c12r(W, split, qmask, channels=('c1', 'c2'), K=3, maxdf=20, shard=400000, log=print):
    """reverse C1r / C2r: every pool record -> top-K S1 among ALL S1 of its country; keep pairs whose S1 is a query S1."""
    todo = [ch for ch in channels if not os.path.exists(W.lexical(ch + 'r', split))]
    if not todo: log('skip c1r/c2r (exist)'); return
    ncol = 'name_c'
    P = pl.read_parquet(W.norm(split, 'pool'), columns=['country', ncol, 'addr_c']).with_row_index('rid')
    S1 = pl.read_parquet(W.norm(split, 's1'), columns=['country', ncol, 'addr_c']).with_row_index('sid')
    RES = {ch: [] for ch in todo}; t0 = time.time()
    for cty in sorted(S1['country'].unique().to_list()):
        Pc = P.filter(pl.col('country') == cty).select('rid', ncol, 'addr_c'); Sc = S1.filter(pl.col('country') == cty).select('sid', ncol, 'addr_c')
        if Pc.height == 0: continue
        ntP, _, _ = _tokens(Pc.select('rid', ncol, pl.lit('').alias('addr_c')), 'rid', ncol); ntS, _, _ = _tokens(Sc.select('sid', ncol, pl.lit('').alias('addr_c')), 'sid', ncol)
        skm, cmap = _skel_maps(ntP, ntS, 'rid', 'sid'); del ntP, ntS
        DP = _prep(Pc, 'rid', ncol, skm, cmap); DS = _prep(Sc, 'sid', ncol, skm, cmap)
        dfs = DP['_sk'].group_by('sk').agg(pl.len().alias('df')); dfw = DP['_words'].group_by('w').agg(pl.len().alias('df'))
        NS = Sc.height
        for ch in todo:
            ks = _build_keys(DS, 'sid', dfs, dfw, ch); kp = _build_keys(DP, 'rid', dfs, dfw, ch)
            post = ks.group_by('key').agg(pl.len().alias('df')).filter(pl.col('df') <= maxdf).with_columns((np.log(NS) - pl.col('df').cast(pl.Float64).log()).cast(pl.Float32).alias('idf'))
            ks = ks.join(post.select('key', 'idf'), on='key'); kp = kp.join(post.select('key'), on='key', how='semi').sort('rid')
            rids = Pc['rid'].to_numpy()
            for i in range(0, len(rids), shard):
                lo, hi = rids[i], rids[min(i + shard, len(rids)) - 1]
                j = kp.filter(pl.col('rid').is_between(lo, hi)).join(ks, on='key')
                g = j.group_by('rid', 'sid').agg(pl.col('idf').sum().alias('score'), pl.len().cast(pl.UInt8).alias('nk'))
                g = g.with_columns(pl.struct('rid', 'sid').hash(seed=3).alias('tb')).sort(['rid', 'score', 'nk', 'tb'], descending=[False, True, True, False])
                g = g.with_columns((pl.int_range(pl.len()).over('rid') + 1).alias('rank'), pl.col('score').first().over('rid').alias('top1')).filter(pl.col('rank') <= K)
                RES[ch].append(g.select(pl.col('sid').cast(pl.Int32).alias('s1_idx'), pl.col('rid').cast(pl.Int32).alias('cand_idx'), pl.col('rank').cast(pl.UInt8), 'score',
                                        (pl.col('top1') - pl.col('score')).alias('gap'), 'nk'))
            log(f'{split} {cty} {ch}r reverse done {time.time() - t0:.0f}s')
    q = np.where(qmask)[0].astype(np.int32)
    for ch in todo:
        R = pl.concat(RES[ch]).filter(pl.col('s1_idx').is_in(pl.Series(q).implode())).sort(['s1_idx', 'cand_idx'])
        R.write_parquet(W.lexical(ch + 'r', split)); log('wrote', ch + 'r', R.height)


def _txt(D, ncol='name_c'):
    STOP = list(NAME_STOP)
    t = (D.select('i', pl.col(ncol).str.split(' ').alias('t')).explode('t')
         .filter(pl.col('t').str.len_chars() >= 2, ~pl.col('t').is_in(STOP)).group_by('i', maintain_order=True).agg(pl.col('t').str.join(' ')))
    return D.select('i').join(t, on='i', how='left').with_columns(pl.col('t').fill_null(''))['t'].to_list()


def run_c3(W, split, qmask, K=10, max_df=0.05, threads=16, log=print):
    """forward name-only channel over the empty-address sub-pool."""
    if os.path.exists(W.lexical('c3', split)): log('skip c3 (exists)'); return
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    P = pl.read_parquet(W.norm(split, 'pool'), columns=['country', 'name_c', 'addr_empty']).with_row_index('rid').filter(pl.col('addr_empty'))
    Q = pl.read_parquet(W.norm(split, 's1'), columns=['country', 'name_c']).with_row_index('s1_idx').filter(pl.Series(qmask))
    out = []
    for cty in sorted(Q['country'].unique().to_list()):
        p = P.filter(pl.col('country') == cty).with_row_index('i'); q = Q.filter(pl.col('country') == cty).with_row_index('i')
        if p.height == 0: continue
        v = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1, max_df=max_df if p.height * max_df >= 1 else 1.0, dtype=np.float32, lowercase=False)
        Xp = v.fit_transform(_txt(p)); Xq = v.transform(_txt(q))
        R = sp_matmul_topn(Xq, Xp.T.tocsr(), top_n=K, threshold=1e-6, sort=True, n_threads=threads).tocsr()
        cnt = np.diff(R.indptr); qi = np.repeat(np.arange(q.height), cnt)
        rank = (np.arange(R.nnz) - np.repeat(R.indptr[:-1], cnt) + 1).astype(np.uint8)
        out.append(pl.DataFrame({'s1_idx': q['s1_idx'].to_numpy()[qi].astype(np.int32), 'cand_idx': p['rid'].to_numpy()[R.indices].astype(np.int32),
                                 'rank': rank, 'score': R.data.astype(np.float32)}))
        log(f'{split} {cty} c3 sub-pool {p.height} queries {q.height} pairs {R.nnz}')
    R = pl.concat(out).sort(['s1_idx', 'rank']) if out else pl.DataFrame(schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32, 'rank': pl.UInt8, 'score': pl.Float32})
    R.write_parquet(W.lexical('c3', split)); log('wrote c3', R.height)


def run_c3r(W, split, qmask, K=5, max_df=0.05, threads=16, log=print):
    """reverse name-only channel: each empty-address record -> top-K S1 among ALL S1 of its country."""
    if os.path.exists(W.lexical('c3r', split)): log('skip c3r (exists)'); return
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    P = pl.read_parquet(W.norm(split, 'pool'), columns=['country', 'name_c', 'addr_empty']).with_row_index('rid').filter(pl.col('addr_empty'))
    S1 = pl.read_parquet(W.norm(split, 's1'), columns=['country', 'name_c']).with_row_index('s1_idx').with_columns(pl.Series('isq', qmask))
    out = []
    for cty in sorted(S1['country'].unique().to_list()):
        s = S1.filter(pl.col('country') == cty).with_row_index('i'); p = P.filter(pl.col('country') == cty).with_row_index('i')
        if p.height == 0: continue
        v = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1, max_df=max_df if s.height * max_df >= 1 else 1.0, dtype=np.float32, lowercase=False)
        Xs = v.fit_transform(_txt(s)); Xp = v.transform(_txt(p))
        R = sp_matmul_topn(Xp, Xs.T.tocsr(), top_n=K, threshold=1e-6, sort=True, n_threads=threads).tocsr()
        cnt = np.diff(R.indptr); pi = np.repeat(np.arange(p.height), cnt)
        rank = (np.arange(R.nnz) - np.repeat(R.indptr[:-1], cnt) + 1).astype(np.uint8)
        top1 = np.repeat(np.where(cnt > 0, R.data[np.minimum(R.indptr[:-1], max(R.nnz - 1, 0))], 0), cnt)
        d = pl.DataFrame({'s1_idx': s['s1_idx'].to_numpy()[R.indices].astype(np.int32), 'isq': s['isq'].to_numpy()[R.indices],
                          'cand_idx': p['rid'].to_numpy()[pi].astype(np.int32), 'rank': rank, 'score': R.data.astype(np.float32),
                          'gap': (top1 - R.data).astype(np.float32)}).filter(pl.col('isq')).drop('isq')
        out.append(d); log(f'{split} {cty} c3r queries {p.height} pairs kept {d.height}')
    R = pl.concat(out).sort(['s1_idx', 'rank']) if out else pl.DataFrame(schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32, 'rank': pl.UInt8, 'score': pl.Float32, 'gap': pl.Float32})
    R.write_parquet(W.lexical('c3r', split)); log('wrote c3r', R.height)
