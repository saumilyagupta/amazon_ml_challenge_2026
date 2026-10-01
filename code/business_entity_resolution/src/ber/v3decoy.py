"""Matcher v3, change A: offset-conditioned decoy features (15), replacing v2b's dec_num_shift / dec_flag (v2b's dec_extra is kept).
Research original: work/matching/prod_v3/pv3/decoy.py (+ 02b_patch_dec.py); pair logic copied unchanged, record sources = package work dir.

The data generator plants DECOYS: a copy of an S1 name with ONE extra word, at a house number S1 + k with k in the generator's offset
set D = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21} (the mirror offset -k is ~100x rarer). The explainer word lists are NOT changed (several decoy
words are also true-copy filler at offset 0); instead the decoy signal is given to the model conditioned on the house-number offset.

Record level (row order = s1_idx / cand_idx): numeric parts of the FIRST compound house-number token of the romanised address addr_en
(PO Box / PMB numbers removed first; zero-stripped, letters ignored), and the matcher view's name_core token multiset (ber.records).
Pair level:
  dec_off      signed offset of the first integer, record minus S1 (clipped to +-999; -999 = a side has no number)
  dec_off_all  same, but for multi-part numbers with the same part count and EXACTLY ONE differing numeric part the offset of that part
               (E-16-28 vs E-16-49 -> 21; 8-3-231/A/32 vs 8-3-231/A/35 -> 3); otherwise = dec_off
  dec_inD      dec_off_all in D; dec_absinD  |dec_off_all| in D   (-1 = a side has no number)
  dec_plus1    record core = S1 core + exactly one word (multiset)
  dec_add_decoy   dec_plus1 and the added word is in the COUNTRY decoy list (resources/decoy_lists.json)
  dec_add_filler  dec_plus1 and the added word is in the country's true-copy filler list (= the explainer's learned filler list)
  dec_plus1_inD, dec_add_decoy_inD, dec_add_filler_inD   the three above AND k in D
  dec_expl_inD       explainer x1_name_explained AND dec_plus1 AND k in D ('explained, but at a decoy offset')
  dec_filleradd_inD  explainer op_filler_add AND k in D
  dec_add_ctag / dec_add_ctag_inD   dec_plus1 and the added word is a country tag (france / india / usa ...) [AND k in D]
  dec_legal_inD      explainer op_legal_add | op_legal_restyle AND k in D (India legal switches are 14% true -> a learned feature, never a rule)

Word lists (resources/decoy_lists.json, label-free): France decoy words = the 6 words mined from the UNLABELLED TEST inputs (transductive);
US (24) / India (8) = words added to S1 names at offsets k in D with n(k in D) >= 500, n(k in D) / (n(k < 0) + 1) >= 20 and
n(k in D) > n(k = 0) on the train INPUTS (no labels); true-copy filler = the explainer's learned filler lists. D is generator-level."""
import json, os
from collections import Counter
import numpy as np, polars as pl
from .paths import RES

DEC_FEATS = ['dec_off', 'dec_off_all', 'dec_inD', 'dec_absinD', 'dec_plus1', 'dec_add_decoy', 'dec_add_filler',
             'dec_plus1_inD', 'dec_add_decoy_inD', 'dec_add_filler_inD', 'dec_expl_inD', 'dec_filleradd_inD',
             'dec_add_ctag', 'dec_add_ctag_inD', 'dec_legal_inD']
EXPL_INPUTS = ['x1_name_explained', 'op_filler_add', 'op_legal_add', 'op_legal_restyle']   # explainer columns the interactions use
DOFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
CTAG = {'france', 'india', 'usa', 'us', 'america', 'american', 'francaise', 'francais', 'indian', 'frankreich', 'inde'}
NP = 6
_BOX = r'\b(p\s*\.?\s*o\s*\.?\s*box|pmb|box)\s*#?\s*\d+'
_TOK = r'((?:[a-z]{1,3}\s*[-/]\s*)?\d+[a-z]?(?:\s*[-/]\s*[a-z]{0,3}\d*[a-z]{0,3})*)'
LISTS_PATH = os.path.join(RES, 'decoy_lists.json')
_L = {}


def lists():
    """{'decoy': {country: set}, 'filler': {country: set}} from resources/decoy_lists.json (countries without a list: empty sets)."""
    if not _L:
        d = json.load(open(LISTS_PATH))
        _L['decoy'] = {c: set(v['decoy_list']) for c, v in d.items() if isinstance(v, dict)}
        _L['filler'] = {c: set(v['true_copy_filler']) for c, v in d.items() if isinstance(v, dict)}
    return _L


def num_parts(addr: pl.Series):
    """(N, NP) int64 array of the numeric parts of the first compound number token (-1 = absent), + part count."""
    s = addr.fill_null('').str.to_lowercase().str.replace_all(_BOX, ' ')
    d = pl.DataFrame({'s': s}).with_columns(pl.col('s').str.extract(_TOK, 1).fill_null('').alias('tok'))
    d = d.with_columns(pl.col('tok').str.extract_all(r'\d+').list.eval(pl.element().str.slice(-9, 9).cast(pl.Int64)).alias('parts'))
    n = d['parts'].list.len().to_numpy().astype(np.int16)
    P = np.full((d.height, NP), -1, np.int64)
    for i in range(NP):
        P[:, i] = d['parts'].list.get(i, null_on_oob=True).fill_null(-1).to_numpy()
    return P, np.minimum(n, NP).astype(np.int16)


def build_records(W, split):
    """cached WORK/records/dec_{split}.npz (P1/n1 for S1 rows, P2/n2 for S2/S3 rows) + name_core token strings from rec_{split}_{s1,s23}."""
    out = W.p('records', f'dec_{split}.npz'); res = {}
    if os.path.exists(out):
        z = np.load(out); res.update({k: z[k] for k in ('P1', 'n1', 'P2', 'n2')})
    else:
        for part, k in (('s1', 1), ('s23', 2)):
            res[f'P{k}'], res[f'n{k}'] = num_parts(pl.read_parquet(W.records(split, part), columns=['addr_en'])['addr_en'])
        np.savez(out + '.tmp.npz', **{k: res[k] for k in ('P1', 'n1', 'P2', 'n2')}); os.replace(out + '.tmp.npz', out)
    for part, k in (('s1', 1), ('s23', 2)):
        core = pl.read_parquet(W.rec(split, part), columns=['name_core'])['name_core'].to_numpy().astype(object)
        assert len(core) == len(res[f'n{k}']), f'{part}: name_core rows != record rows'
        res[f'core{k}'] = core
    return res


def pair_features(rec, ia, ib, country, x1_name_explained=None, op_filler_add=None, op_legal_add=None, op_legal_restyle=None):
    """ia / ib: s1_idx / cand_idx arrays; country: per-pair S1 country (numpy object); the explainer flags of the pair. -> dict name -> array."""
    L = lists(); DECOY, FILLER = L['decoy'], L['filler']
    P1 = rec['P1'][ia]; P2 = rec['P2'][ib]; n1 = rec['n1'][ia]; n2 = rec['n2'][ib]
    both = (n1 > 0) & (n2 > 0)
    off = np.where(both, np.clip(P2[:, 0] - P1[:, 0], -999, 999), -999).astype(np.float32)
    m = np.minimum(n1, n2)
    pos = np.arange(NP)[None, :]
    valid = pos < m[:, None]
    diff = (P1 != P2) & valid
    nd = diff.sum(1)
    one = both & (n1 == n2) & (nd == 1)
    idx = np.where(one, diff.argmax(1), 0)
    off_all = np.where(one, np.clip(P2[np.arange(len(ia)), idx] - P1[np.arange(len(ia)), idx], -999, 999), off).astype(np.float32)
    D = np.array(DOFF)
    inD = np.isin(off_all, D).astype(np.float32)
    absinD = np.isin(np.abs(off_all), D).astype(np.float32)
    inD[~both] = -1; absinD[~both] = -1
    # name: record core = S1 core + exactly one word
    c1 = rec['core1'][ia]; c2 = rec['core2'][ib]
    plus1 = np.zeros(len(ia), np.float32); addw = np.empty(len(ia), object); addw[:] = ''
    for i in range(len(ia)):
        a = c1[i]; b = c2[i]
        if not a or not b or len(b) <= len(a): continue
        ta = a.split(); tb = b.split()
        if len(tb) != len(ta) + 1: continue
        d = Counter(tb); d.subtract(Counter(ta))     # multiset difference
        if any(v < 0 for v in d.values()): continue
        ex = [w for w, v in d.items() for _ in range(v)]
        if len(ex) == 1:
            plus1[i] = 1; addw[i] = ex[0]
    add_dec = np.zeros(len(ia), np.float32); add_fil = np.zeros(len(ia), np.float32)
    for c in sorted(set(DECOY) | set(FILLER)):
        sel = (country == c) & (plus1 == 1)
        if sel.any():
            w = addw[sel]
            add_dec[sel] = np.fromiter((x in DECOY.get(c, ()) for x in w), np.float32, count=len(w))
            add_fil[sel] = np.fromiter((x in FILLER.get(c, ()) for x in w), np.float32, count=len(w))
    pos = (inD == 1)
    flag = lambda v: (np.asarray(v) == 1) if v is not None else np.zeros(len(ia), bool)
    xe, fa, la, lr = flag(x1_name_explained), flag(op_filler_add), flag(op_legal_add), flag(op_legal_restyle)
    ct = np.fromiter((x in CTAG for x in addw), np.float32, count=len(ia)) * (plus1 == 1)
    F = {'dec_off': off, 'dec_off_all': off_all, 'dec_inD': inD, 'dec_absinD': absinD, 'dec_plus1': plus1, 'dec_add_decoy': add_dec, 'dec_add_filler': add_fil,
         'dec_plus1_inD': (pos & (plus1 == 1)), 'dec_add_decoy_inD': (pos & (add_dec == 1)), 'dec_add_filler_inD': (pos & (add_fil == 1)),
         'dec_expl_inD': (pos & (plus1 == 1) & xe), 'dec_filleradd_inD': (pos & fa),
         'dec_add_ctag': ct, 'dec_add_ctag_inD': (pos & (ct == 1)), 'dec_legal_inD': (pos & (la | lr))}
    return {k: np.asarray(v, np.float32) for k, v in F.items()}


def add_features(X: pl.DataFrame, rec) -> pl.DataFrame:
    """X: a feature part with s1_idx, cand_idx, country and the explainer columns EXPL_INPUTS -> X + DEC_FEATS (float32)."""
    F = pair_features(rec, X['s1_idx'].to_numpy().astype(np.int64), X['cand_idx'].to_numpy().astype(np.int64), X['country'].to_numpy().astype(object),
                      *[X[c].to_numpy() for c in EXPL_INPUTS])
    return X.with_columns([pl.Series(k, F[k]).cast(pl.Float32) for k in DEC_FEATS])
