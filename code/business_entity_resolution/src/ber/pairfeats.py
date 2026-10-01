"""Pair features for (S1, candidate) pairs: 89 stage-1 features (baseline-28 + playbook additions on the romanised and raw views).
Vectorised: rapidfuzz process.cpdist (C++ threads), scipy sparse row ops, numpy; 45-70k pairs/s at 16 threads.
    fb = FeatureBuilder(W, 'train'); X = fb.pair_features(T)   # T: frame with s1_idx, cand_idx and table_features() columns
"""
import os, time
import numpy as np, polars as pl, scipy.sparse as sp
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler

STR_COLS = ['name_n', 'name_core', 'name_ns', 'name_skel', 'addr_n', 'name_raw', 'addr_raw']
REC_NUM = ['ntok_name', 'ntok_addr', 'name_nonascii', 'addr_nonascii', 'script', 'f_url', 'f_handle', 'f_formerly', 'f_urltail', 'addr_empty',
           'addr_null', 'addr_nodigit', 'addr_noletter', 'name_noletter', 'has_unit', 'fake', 'name_freq_s1', 'name_freq_pool', 'addr_freq_s1',
           'addr_freq_pool', 'n_nums', 'num_first']
TABLE_FEATS = ['cos', 'fwd_rank', 'rev_rank', 's1_top1', 'rel_s1', 's1_gap12', 'rev_top1', 'rel_rec', 'rev_gap12', 'n_claim_gate',
               'rank_s1', 'n_cands_s1', 'rank_rec_tab', 'n_s1_rec_tab', 'rel_rec_tab', 'n_channels', 'ch_fwd', 'ch_rev']
BASELINE28 = ['n_ratio', 'n_tsort', 'n_tset', 'n_partial', 'n_jw', 'ns_ratio', 'sk_ratio', 'sk_tset',
              'a_tset', 'a_tsort', 'a_ptset', 'num_inter', 'num_jac', 'num_cnt1', 'num_cnt2', 'addr_empty1', 'addr_empty2',
              'ntok_name1', 'ntok_name2', 'name_nonascii1', 'name_nonascii2',
              'cos', 'n_channels', 'rank_s1', 'rel_s1', 'rev_rank', 'rel_rec', 'n_claim_gate']
PLAYBOOK = ['n_ratio_raw', 'n_tset_raw', 'n_partial_raw', 'a_tset_raw', 'ns_partial', 'core_eq', 'ns_eq', 'a_ratio',
            'n_inter', 'n_wjac', 'n_cov1', 'n_cov2', 'n_miss_max', 'n_extra_max', 'n_miss_cnt', 'n_extra_cnt',
            'a_inter', 'a_wjac', 'a_cov1', 'a_cov2', 'a_miss_max', 'a_extra_max',
            'num_first_eq', 'num_first_off1', 'num_first_ldiff', 'num_first_both', 'num_miss1', 'num_extra2', 'has_unit1', 'has_unit2',
            'addr_null2', 'addr_nodigit2', 'addr_noletter2', 'name_noletter2', 'script2', 'f_url2', 'f_handle2', 'f_formerly2', 'f_urltail2',
            'fake2', 'name_freq_s1_1', 'name_freq_pool2', 'name_freq_s1_2', 'addr_freq_s1_1', 'addr_freq_pool2', 'addr_nonascii2', 'ntok_addr1',
            'ntok_addr2', 'is_s3', 'fwd_rank', 's1_top1', 's1_gap12', 'rev_top1', 'rev_gap12', 'n_cands_s1', 'rank_rec_tab', 'n_s1_rec_tab',
            'rel_rec_tab', 'ch_fwd', 'ch_rev', 'ch_both']
ALL_FEATS = BASELINE28 + [f for f in PLAYBOOK if f not in BASELINE28]      # 89
CATEGORICAL = ['script2']


def _csr(lists: pl.Series, ncol):
    lens = lists.list.len().to_numpy().astype(np.int64)
    ind = lists.explode().drop_nulls().to_numpy().astype(np.int32)
    indptr = np.zeros(len(lens) + 1, np.int64); np.cumsum(lens, out=indptr[1:])
    M = sp.csr_matrix((np.ones(len(ind), np.float32), ind, indptr), shape=(len(lens), ncol))
    M.sum_duplicates(); M.data[:] = 1.0
    return M


class FeatureBuilder:
    def __init__(self, W, split, threads=16, log=print):
        t0 = time.time(); self.split = split; self.log = log; self.NT = threads
        R1 = pl.read_parquet(W.rec(split, 's1')); R2 = pl.read_parquet(W.rec(split, 's23'))
        self.S = {c: (R1[c].to_numpy(), R2[c].to_numpy()) for c in STR_COLS}
        self.N = {c: (R1[c].cast(pl.Float32).to_numpy(), R2[c].cast(pl.Float32).to_numpy()) for c in REC_NUM if c != 'num_first'}
        self.N['num_first'] = (R1['num_first'].cast(pl.Int64).to_numpy(), R2['num_first'].cast(pl.Int64).to_numpy())
        self.src2 = R2['src'].to_numpy()
        self.idf = {}; self.M = {}
        for nm in ('name', 'addr'):
            w = np.load(W.idf(split, nm)); self.idf[nm] = w
            self.M[nm] = (_csr(R1[f'{nm}_tids'], len(w)), _csr(R2[f'{nm}_tids'], len(w)))
        n1 = R1['nums'].list.unique(); n2 = R2['nums'].list.unique()
        allv = np.unique(np.concatenate([n1.explode().drop_nulls().to_numpy(), n2.explode().drop_nulls().to_numpy()]))
        def num_csr(s):
            lens = s.list.len().to_numpy().astype(np.int64); vals = s.explode().drop_nulls().to_numpy()
            ind = np.searchsorted(allv, vals).astype(np.int32)
            indptr = np.zeros(len(lens) + 1, np.int64); np.cumsum(lens, out=indptr[1:])
            M = sp.csr_matrix((np.ones(len(ind), np.float32), ind, indptr), shape=(len(lens), max(len(allv), 1))); M.sum_duplicates(); M.data[:] = 1.0
            return M
        self.M['num'] = (num_csr(n1), num_csr(n2))
        self.sumw = {nm: (self.M[nm][0] @ self.idf[nm], self.M[nm][1] @ self.idf[nm]) for nm in ('name', 'addr')}
        self.cnt = {nm: (np.diff(self.M[nm][0].indptr).astype(np.float32), np.diff(self.M[nm][1].indptr).astype(np.float32)) for nm in ('name', 'addr', 'num')}
        del R1, R2
        log(f'[FeatureBuilder {split}] ready in {time.time()-t0:.0f}s')

    def _cp(self, col, ia, ib, scorer):
        return process.cpdist(self.S[col][0][ia], self.S[col][1][ib], scorer=scorer, workers=self.NT, dtype=np.float32)

    def _sparse(self, nm, ia, ib, chunk=1_000_000):
        A, B = self.M[nm]; w = self.idf.get(nm)
        n = len(ia); inter = np.empty(n, np.float32); out = {}
        if w is not None:
            sh = np.empty(n, np.float32); mm = np.empty(n, np.float32); em = np.empty(n, np.float32); D = sp.diags(w)
        for s in range(0, n, chunk):
            Aa = A[ia[s:s + chunk]]; Bb = B[ib[s:s + chunk]]
            I = Aa.multiply(Bb).tocsr()
            inter[s:s + chunk] = np.asarray(I.sum(axis=1)).ravel()
            if w is not None:
                sh[s:s + chunk] = I @ w
                mm[s:s + chunk] = ((Aa - I) @ D).max(axis=1).toarray().ravel()
                em[s:s + chunk] = ((Bb - I) @ D).max(axis=1).toarray().ravel()
        out['inter'] = inter
        if w is not None:
            s1w = self.sumw[nm][0][ia]; s2w = self.sumw[nm][1][ib]
            out['wjac'] = np.where(s1w + s2w - sh > 0, sh / np.maximum(s1w + s2w - sh, 1e-6), -1).astype(np.float32)
            out['cov1'] = np.where(s1w > 0, sh / np.maximum(s1w, 1e-6), -1).astype(np.float32)
            out['cov2'] = np.where(s2w > 0, sh / np.maximum(s2w, 1e-6), -1).astype(np.float32)
            out['miss_max'] = mm; out['extra_max'] = em
        return out

    def pair_features(self, T: pl.DataFrame, timing=None):
        t0 = time.time(); tm = {} if timing is None else timing
        ia = T['s1_idx'].to_numpy().astype(np.int64); ib = T['cand_idx'].to_numpy().astype(np.int64)
        F = {}
        F['n_ratio'] = self._cp('name_n', ia, ib, fuzz.ratio); F['n_tsort'] = self._cp('name_n', ia, ib, fuzz.token_sort_ratio)
        F['n_tset'] = self._cp('name_n', ia, ib, fuzz.token_set_ratio); F['n_partial'] = self._cp('name_n', ia, ib, fuzz.partial_ratio)
        F['n_jw'] = self._cp('name_n', ia, ib, JaroWinkler.normalized_similarity)
        F['ns_ratio'] = self._cp('name_ns', ia, ib, fuzz.ratio); F['ns_partial'] = self._cp('name_ns', ia, ib, fuzz.partial_ratio)
        F['sk_ratio'] = self._cp('name_skel', ia, ib, fuzz.ratio); F['sk_tset'] = self._cp('name_skel', ia, ib, fuzz.token_set_ratio)
        F['a_tset'] = self._cp('addr_n', ia, ib, fuzz.token_set_ratio); F['a_tsort'] = self._cp('addr_n', ia, ib, fuzz.token_sort_ratio)
        F['a_ptset'] = self._cp('addr_n', ia, ib, fuzz.partial_token_set_ratio); F['a_ratio'] = self._cp('addr_n', ia, ib, fuzz.ratio)
        F['n_ratio_raw'] = self._cp('name_raw', ia, ib, fuzz.ratio); F['n_tset_raw'] = self._cp('name_raw', ia, ib, fuzz.token_set_ratio)
        F['n_partial_raw'] = self._cp('name_raw', ia, ib, fuzz.partial_ratio); F['a_tset_raw'] = self._cp('addr_raw', ia, ib, fuzz.token_set_ratio)
        tm['rapidfuzz'] = tm.get('rapidfuzz', 0) + time.time() - t0; t1 = time.time()
        F['core_eq'] = (self.S['name_core'][0][ia] == self.S['name_core'][1][ib]).astype(np.float32)
        F['ns_eq'] = (self.S['name_ns'][0][ia] == self.S['name_ns'][1][ib]).astype(np.float32)
        for nm, p in (('name', 'n'), ('addr', 'a')):
            o = self._sparse(nm, ia, ib)
            for k, v in o.items(): F[f'{p}_{k}'] = v
        F['n_miss_cnt'] = self.cnt['name'][0][ia] - F['n_inter']; F['n_extra_cnt'] = self.cnt['name'][1][ib] - F['n_inter']
        o = self._sparse('num', ia, ib)
        c1 = self.cnt['num'][0][ia]; c2 = self.cnt['num'][1][ib]
        F['num_inter'] = o['inter']; F['num_cnt1'] = c1; F['num_cnt2'] = c2
        F['num_jac'] = np.where((c1 > 0) & (c2 > 0), o['inter'] / np.maximum(c1 + c2 - o['inter'], 1), -1).astype(np.float32)
        F['num_miss1'] = c1 - o['inter']; F['num_extra2'] = c2 - o['inter']
        f1 = self.N['num_first'][0][ia]; f2 = self.N['num_first'][1][ib]; both = (f1 >= 0) & (f2 >= 0)
        F['num_first_both'] = both.astype(np.float32)
        F['num_first_eq'] = np.where(both, (f1 == f2), -1).astype(np.float32)
        F['num_first_off1'] = np.where(both, np.abs(f1 - f2) == 1, -1).astype(np.float32)
        F['num_first_ldiff'] = np.where(both, np.log1p(np.abs(f1.astype(np.float64) - f2)), -1).astype(np.float32)
        tm['sparse'] = tm.get('sparse', 0) + time.time() - t1
        g1 = lambda c: self.N[c][0][ia]; g2 = lambda c: self.N[c][1][ib]
        F['addr_empty1'] = g1('addr_empty'); F['addr_empty2'] = g2('addr_empty')
        F['ntok_name1'] = g1('ntok_name'); F['ntok_name2'] = g2('ntok_name'); F['ntok_addr1'] = g1('ntok_addr'); F['ntok_addr2'] = g2('ntok_addr')
        F['name_nonascii1'] = g1('name_nonascii'); F['name_nonascii2'] = g2('name_nonascii'); F['addr_nonascii2'] = g2('addr_nonascii')
        F['has_unit1'] = g1('has_unit'); F['has_unit2'] = g2('has_unit')
        for c in ('addr_null', 'addr_nodigit', 'addr_noletter', 'name_noletter', 'script', 'f_url', 'f_handle', 'f_formerly', 'f_urltail', 'fake'):
            F[c + '2'] = g2(c)
        F['name_freq_s1_1'] = np.log1p(g1('name_freq_s1')); F['name_freq_pool2'] = np.log1p(g2('name_freq_pool'))
        F['name_freq_s1_2'] = np.log1p(g2('name_freq_s1')); F['addr_freq_s1_1'] = np.log1p(g1('addr_freq_s1')); F['addr_freq_pool2'] = np.log1p(g2('addr_freq_pool'))
        F['is_s3'] = (self.src2[ib] == 3).astype(np.float32)
        out = T.select('s1_idx', 'cand_idx', *[c for c in TABLE_FEATS if c in T.columns])
        out = out.with_columns((pl.col('ch_fwd') & pl.col('ch_rev')).alias('ch_both') if 'ch_fwd' in out.columns else pl.lit(False).alias('ch_both'))
        out = out.with_columns([pl.col(c).cast(pl.Float32) for c in out.columns if c not in ('s1_idx', 'cand_idx')])
        out = out.with_columns([pl.Series(k, np.asarray(v, np.float32)) for k, v in F.items()])
        tm['total'] = tm.get('total', 0) + time.time() - t0
        return out
