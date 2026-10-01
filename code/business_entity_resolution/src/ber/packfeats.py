"""Recompute the v1 pair features that depend on the name core / canonical address / IDF, on francepack record views.
Same formulas as the v1 FeatureBuilder.pair_features (ber.pairfeats; copied), restricted to the AFFECTED list (43), + 3 new pack features.
Package copy of work/research/france/featureshift/packfeats.py (logic unchanged; the record-table prefix is now a required argument).
    pf = PackFeatures('test', prefix=W.p('records', 'pack_test_assfcl-'), idf='global')   # record tables from ber.packrecords.build
    F = pf.pair_features(s1_idx, cand_idx)                       # dict name -> float32 array
With Cfg.off() record views and idf='global' the output equals the v1 features bit for bit (research check: 2,000 val + 2,000 test pairs).
Matcher v3 uses idf='global' and only the 43 AFFECTED features (the 3 NEW_FEATS would be constant on the US/India training rows)."""
import os, time
import numpy as np, polars as pl, scipy.sparse as sp
from rapidfuzz import process, fuzz
HERE = os.path.dirname(os.path.abspath(__file__))
NT = int(os.environ.get('PV1_THREADS', os.environ.get('OMP_NUM_THREADS', '8')))   # rapidfuzz workers (results do not depend on it)

NAME_CORE_FEATS = ['core_eq', 'ns_eq', 'ns_ratio', 'ns_partial', 'sk_ratio', 'sk_tset', 'n_inter', 'n_wjac', 'n_cov1', 'n_cov2', 'n_miss_max',
                   'n_extra_max', 'n_miss_cnt', 'n_extra_cnt', 'name_freq_s1_1', 'name_freq_pool2', 'name_freq_s1_2']
ADDR_FEATS = ['a_tset', 'a_tsort', 'a_ptset', 'a_ratio', 'a_inter', 'a_wjac', 'a_cov1', 'a_cov2', 'a_miss_max', 'a_extra_max', 'ntok_addr1',
              'ntok_addr2', 'addr_freq_s1_1', 'addr_freq_pool2', 'addr_empty1', 'addr_empty2']
NUM_FEATS = ['num_inter', 'num_cnt1', 'num_cnt2', 'num_jac', 'num_miss1', 'num_extra2', 'num_first_both', 'num_first_eq', 'num_first_off1', 'num_first_ldiff']
AFFECTED = NAME_CORE_FEATS + ADDR_FEATS + NUM_FEATS
NEW_FEATS = ['pk_adm_agree', 'pk_sfx_agree', 'pk_skey_eq']
STR_COLS = ['name_core', 'name_ns', 'name_skel', 'addr_n', 'adm', 'hn_sfx', 'street_key']
REC_NUM = ['ntok_addr', 'addr_empty', 'name_freq_s1', 'name_freq_pool', 'addr_freq_s1', 'addr_freq_pool']


def _csr(lists: pl.Series, ncol):
    lens = lists.list.len().to_numpy().astype(np.int64)
    ind = lists.explode().drop_nulls().to_numpy().astype(np.int32)
    indptr = np.zeros(len(lens) + 1, np.int64); np.cumsum(lens, out=indptr[1:])
    M = sp.csr_matrix((np.ones(len(ind), np.float32), ind, indptr), shape=(len(lens), ncol))
    M.sum_duplicates(); M.data[:] = 1.0
    return M


class PackFeatures:
    def __init__(self, split, tag='assfcl-', idf='global', prefix=None, log=print):
        t0 = time.time(); assert prefix, 'prefix of the pack record tables (ber.packrecords.build) is required'
        R1 = pl.read_parquet(f'{prefix}_s1.parquet'); R2 = pl.read_parquet(f'{prefix}_s23.parquet')
        self.S = {c: (R1[c].to_numpy(), R2[c].to_numpy()) for c in STR_COLS}
        self.N = {c: (R1[c].cast(pl.Float32).to_numpy(), R2[c].cast(pl.Float32).to_numpy()) for c in REC_NUM}
        self.N['num_first'] = (R1['num_first'].cast(pl.Int64).to_numpy(), R2['num_first'].cast(pl.Int64).to_numpy())
        self.idf, self.M = {}, {}
        for nm in ('name', 'addr'):
            w = np.load(f'{prefix}_idf_{nm}_{idf}.npy'); self.idf[nm] = w
            self.M[nm] = (_csr(R1[f'{nm}_tids'], len(w)), _csr(R2[f'{nm}_tids'], len(w)))
        n1 = R1['nums'].list.unique(); n2 = R2['nums'].list.unique()
        allv = np.unique(np.concatenate([n1.explode().drop_nulls().to_numpy(), n2.explode().drop_nulls().to_numpy()]))
        def num_csr(s):
            lens = s.list.len().to_numpy().astype(np.int64); vals = s.explode().drop_nulls().to_numpy()
            ind = np.searchsorted(allv, vals).astype(np.int32)
            indptr = np.zeros(len(lens) + 1, np.int64); np.cumsum(lens, out=indptr[1:])
            M = sp.csr_matrix((np.ones(len(ind), np.float32), ind, indptr), shape=(len(lens), len(allv))); M.sum_duplicates(); M.data[:] = 1.0
            return M
        self.M['num'] = (num_csr(n1), num_csr(n2))
        self.sumw = {nm: (self.M[nm][0] @ self.idf[nm], self.M[nm][1] @ self.idf[nm]) for nm in ('name', 'addr')}
        self.cnt = {nm: (np.diff(self.M[nm][0].indptr).astype(np.float32), np.diff(self.M[nm][1].indptr).astype(np.float32)) for nm in ('name', 'addr', 'num')}
        del R1, R2
        log(f'[PackFeatures {split} {tag} idf={idf}] ready in {time.time()-t0:.0f}s')

    def _cp(self, col, ia, ib, scorer):
        return process.cpdist(self.S[col][0][ia], self.S[col][1][ib], scorer=scorer, workers=NT, dtype=np.float32)

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

    def pair_features(self, ia, ib, new=True):
        ia = np.asarray(ia, np.int64); ib = np.asarray(ib, np.int64); F = {}
        F['ns_ratio'] = self._cp('name_ns', ia, ib, fuzz.ratio); F['ns_partial'] = self._cp('name_ns', ia, ib, fuzz.partial_ratio)
        F['sk_ratio'] = self._cp('name_skel', ia, ib, fuzz.ratio); F['sk_tset'] = self._cp('name_skel', ia, ib, fuzz.token_set_ratio)
        F['a_tset'] = self._cp('addr_n', ia, ib, fuzz.token_set_ratio); F['a_tsort'] = self._cp('addr_n', ia, ib, fuzz.token_sort_ratio)
        F['a_ptset'] = self._cp('addr_n', ia, ib, fuzz.partial_token_set_ratio); F['a_ratio'] = self._cp('addr_n', ia, ib, fuzz.ratio)
        F['core_eq'] = (self.S['name_core'][0][ia] == self.S['name_core'][1][ib]).astype(np.float32)
        F['ns_eq'] = (self.S['name_ns'][0][ia] == self.S['name_ns'][1][ib]).astype(np.float32)
        for nm, p in (('name', 'n'), ('addr', 'a')):
            for k, v in self._sparse(nm, ia, ib).items():
                F[f'{p}_{k}'] = v
        F['n_miss_cnt'] = self.cnt['name'][0][ia] - F['n_inter']; F['n_extra_cnt'] = self.cnt['name'][1][ib] - F['n_inter']
        o = self._sparse('num', ia, ib); c1 = self.cnt['num'][0][ia]; c2 = self.cnt['num'][1][ib]
        F['num_inter'] = o['inter']; F['num_cnt1'] = c1; F['num_cnt2'] = c2
        F['num_jac'] = np.where((c1 > 0) & (c2 > 0), o['inter'] / np.maximum(c1 + c2 - o['inter'], 1), -1).astype(np.float32)
        F['num_miss1'] = c1 - o['inter']; F['num_extra2'] = c2 - o['inter']
        f1 = self.N['num_first'][0][ia]; f2 = self.N['num_first'][1][ib]; both = (f1 >= 0) & (f2 >= 0)
        F['num_first_both'] = both.astype(np.float32)
        F['num_first_eq'] = np.where(both, (f1 == f2), -1).astype(np.float32)
        F['num_first_off1'] = np.where(both, np.abs(f1 - f2) == 1, -1).astype(np.float32)
        F['num_first_ldiff'] = np.where(both, np.log1p(np.abs(f1.astype(np.float64) - f2)), -1).astype(np.float32)
        g1 = lambda c: self.N[c][0][ia]; g2 = lambda c: self.N[c][1][ib]
        F['addr_empty1'] = g1('addr_empty'); F['addr_empty2'] = g2('addr_empty')
        F['ntok_addr1'] = g1('ntok_addr'); F['ntok_addr2'] = g2('ntok_addr')
        F['name_freq_s1_1'] = np.log1p(g1('name_freq_s1')); F['name_freq_pool2'] = np.log1p(g2('name_freq_pool'))
        F['name_freq_s1_2'] = np.log1p(g2('name_freq_s1')); F['addr_freq_s1_1'] = np.log1p(g1('addr_freq_s1')); F['addr_freq_pool2'] = np.log1p(g2('addr_freq_pool'))
        if new:
            def tri(col):
                a = self.S[col][0][ia]; b = self.S[col][1][ib]
                return np.where((a == '') | (b == ''), -1, (a == b)).astype(np.float32)
            F['pk_adm_agree'] = tri('adm'); F['pk_sfx_agree'] = tri('hn_sfx'); F['pk_skey_eq'] = tri('street_key')
        return {k: np.asarray(v, np.float32) for k, v in F.items()}
