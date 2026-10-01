"""Matcher v2b feature additions (copied from the research code work/matching/prod_v2b/pv2b/newfeats.py and 02_features.py; logic and
constants unchanged, only the record sources are the package work dir).

(b) the 127 explainer features (ber.explainer) are computed per feature part in features.py.
(c) record / pair level (C_FEATS, 14):
  * multi-part house number: first compound number token of the address (PO Box / PMB numbers removed first), split into up to 8 parts
    ('E-16-28' -> e|16|28, '8-3-231/A/32/A1' -> 8|3|231|a|32|a1), numeric parts zero-stripped, hashed (0 = absent); ALL-PARTS comparison
    hn_both, hn_np1, hn_np2, hn_eq_all, hn_first_eq, hn_pos_diff, hn_head_eq_tail_diff (ghost copy E-16-28 vs E-16-49), hn_s1_in_c;
    -1 when a side has no number token.
  * decoy: dec_extra (# decoy-vocabulary tokens the candidate core name adds to an S1 core name it fully contains, only when the IDF token
    counts say 0 missing / 1-2 extra), dec_num_shift (1 <= |primary number delta| <= 9; -1 if a side has none), dec_flag (both).
  * France-safe admin conflict adm_conflict from the explainer's a11_admin_agree (department->region aliases): 1 disagree, 0 agree, NaN missing.
  * provenance prov (0 dense-ft & zero-shot dense, 1 dense-ft only, 2 zero-shot dense only, 3 lexical only; categorical), hub flag u_hub200.
Stage-2 level (after p1): sibling corroboration SIB_FEATS (sibling_features)."""
import os, numpy as np, polars as pl
from rapidfuzz import process, fuzz

NP = 8
DECOY = {'group', 'holdings', 'partners', 'enterprises', 'industries', 'overseas', 'ventures', 'exports', 'infratech', 'south', 'valley', 'eastgate',
         'uptown', 'summit', 'coastal', 'west', 'east', 'midtown', 'harbor', 'north', 'southside', 'westgate', 'greater', 'lakeside', 'metro',
         'riverside', 'downtown', 'highland', 'northside', 'central', 'public',
         # French equivalents (France unseen in train; harmless elsewhere)
         'groupe', 'holding', 'partenaires', 'entreprises', 'nord', 'sud', 'est', 'ouest', 'centre', 'grand'}
_BOX = r'\b(p\s*\.?\s*o\s*\.?\s*box|pmb|box)\s*#?\s*\d+'
_TOK = r'((?:[a-z]{1,3}\s*[-/]\s*)?\d+[a-z]?(?:\s*[-/]\s*[a-z]{0,3}\d*[a-z]{0,3})*)'
HN_FEATS = ['hn_both', 'hn_np1', 'hn_np2', 'hn_eq_all', 'hn_first_eq', 'hn_pos_diff', 'hn_head_eq_tail_diff', 'hn_s1_in_c']
DEC_FEATS = ['dec_extra', 'dec_num_shift', 'dec_flag']
MISC_FEATS = ['adm_conflict', 'prov', 'u_hub200']
C_FEATS = HN_FEATS + DEC_FEATS + MISC_FEATS
SIB_FEATS = ['sib_n', 'sib_num_share', 'sib_hn_eq', 'sib_addr_max', 'sib_name_max', 'sib_addr_best_p']
CONF, KSIB = 0.5, 6


def hn_frame(addr: pl.Series):
    s = addr.fill_null('').str.to_lowercase().str.replace_all(_BOX, ' ')
    d = pl.DataFrame({'s': s}).with_columns(pl.col('s').str.extract(_TOK, 1).fill_null('').alias('tok'))
    d = d.with_columns(pl.col('tok').str.extract_all(r'[a-z]*\d+[a-z]*|[a-z]+').alias('parts'))
    d = d.with_columns(pl.col('parts').list.eval(pl.when(pl.element().str.contains(r'^\d+$')).then(pl.element().str.replace(r'^0+(\d)', '$1'))
                                                  .otherwise(pl.element())).alias('parts'))
    d = d.with_columns(pl.col('parts').list.len().cast(pl.Int8).alias('np'))
    H = np.zeros((d.height, NP), np.uint64)
    for i in range(NP):
        h = d.select(pl.col('parts').list.get(i, null_on_oob=True).hash(seed=12345).alias('h'), pl.col('parts').list.len().alias('n'))
        v = h['h'].to_numpy().astype(np.uint64); v[(h['n'].to_numpy() <= i)] = 0; H[:, i] = v
    return H, np.minimum(d['np'].to_numpy(), 127).astype(np.int8)


def build_records(W, split):
    """multi-part house numbers per record (S1 rows, S2/S3 rows in the prepared record order), cached in WORK/records/hn_{split}.npz."""
    out = W.p('records', f'hn_{split}.npz')
    if os.path.exists(out):
        z = np.load(out); return z['H1'], z['n1'], z['H2'], z['n2']
    res = []
    for part in ('s1', 's23'):
        res += list(hn_frame(pl.read_parquet(W.records(split, part), columns=['addr_en'])['addr_en']))
    np.savez(out + '.tmp.npz', H1=res[0], n1=res[1], H2=res[2], n2=res[3]); os.replace(out + '.tmp.npz', out)
    return res


def pair_hn(H1, n1, H2, n2, ia, ib):
    a = H1[ia]; b = H2[ib]; na = n1[ia].astype(np.int16); nb = n2[ib].astype(np.int16)
    both = (na > 0) & (nb > 0)
    m = np.minimum(np.minimum(na, nb), NP)
    pos = np.arange(NP)[None, :]
    diff = (a != b) & (pos < m[:, None])
    pos_diff = diff.sum(1)
    eq_all = both & (na == nb) & (pos_diff == 0)
    first_eq = both & (a[:, 0] == b[:, 0])
    L = np.clip(na, 1, NP) - 1          # all positions before the last equal, last differs (same part count >= 2)
    head_eq = ((~diff) | (pos >= L[:, None])).all(1)
    last_diff = a[np.arange(len(a)), L] != b[np.arange(len(b)), np.clip(nb, 1, NP) - 1]
    htd = both & (na == nb) & (na >= 2) & head_eq & last_diff
    inb = ((a[:, :, None] == b[:, None, :]) & (b[:, None, :] != 0)).any(2) & (pos < np.minimum(na, NP)[:, None])
    s1c = inb.sum(1) / np.maximum(np.minimum(na, NP), 1)
    f = lambda x: np.where(both, x, -1).astype(np.float32)
    return {'hn_both': both.astype(np.float32), 'hn_np1': na.astype(np.float32), 'hn_np2': nb.astype(np.float32), 'hn_eq_all': f(eq_all),
            'hn_first_eq': f(first_eq), 'hn_pos_diff': f(pos_diff), 'hn_head_eq_tail_diff': f(htd), 'hn_s1_in_c': f(s1c)}


def pair_decoy(core1, core2, nf1, nf2, ia, ib, n_miss, n_extra):
    """core1/core2: name_core string arrays (ber.records views); nf1/nf2: num_first int arrays (-1 missing)."""
    n = len(ia); dec = np.zeros(n, np.float32)
    idx = np.nonzero((n_miss == 0) & (n_extra >= 1) & (n_extra <= 2))[0]
    for i in idx:
        t1 = set(core1[ia[i]].split()); t2 = set(core2[ib[i]].split())
        if t1 and t1 <= t2:
            dec[i] = len((t2 - t1) & DECOY)
    f1 = nf1[ia]; f2 = nf2[ib]; both = (f1 >= 0) & (f2 >= 0); d = np.abs(f1 - f2)
    shift = np.where(both, (d >= 1) & (d <= 9), -1).astype(np.float32)
    return {'dec_extra': dec, 'dec_num_shift': shift, 'dec_flag': ((dec > 0) & (shift == 1)).astype(np.float32)}


def c_features(X: pl.DataFrame, hn, fb) -> pl.DataFrame:
    """X: a feature part with s1_idx, cand_idx, n_miss_cnt, n_extra_cnt, a11_admin_agree, in_dft, in_dense, u_n_s1.
    hn = build_records(W, split); fb = the part's ber.pairfeats.FeatureBuilder (record views). Returns X + C_FEATS (research order)."""
    H1, n1, H2, n2 = hn; ia = X['s1_idx'].to_numpy(); ib = X['cand_idx'].to_numpy()
    newf = pair_hn(H1, n1, H2, n2, ia, ib)
    core1, core2 = fb.S['name_core']; nf1, nf2 = fb.N['num_first']
    newf.update(pair_decoy(core1, core2, nf1, nf2, ia, ib, X['n_miss_cnt'].to_numpy(), X['n_extra_cnt'].to_numpy()))
    a11 = X['a11_admin_agree'].to_numpy()
    newf['adm_conflict'] = np.where(a11 < 0, np.nan, (a11 == 0)).astype(np.float32)
    dft = X['in_dft'].to_numpy(); zs = X['in_dense'].to_numpy()
    newf['prov'] = np.where(dft & zs, 0, np.where(dft, 1, np.where(zs, 2, 3))).astype(np.float32)
    newf['u_hub200'] = (X['u_n_s1'].to_numpy() > 200).astype(np.float32)
    return X.with_columns([pl.Series(k, newf[k]) for k in C_FEATS])


# ---------------- stage-2 level: sibling corroboration ----------------
def sibling_records(W, split):
    """S2/S3 record views needed by sibling_features (ber.records views)."""
    R = pl.read_parquet(W.rec(split, 's23'), columns=['nums', 'num_first', 'addr_n', 'name_n'])
    return sibling_records_from(R)


def sibling_records_from(R: pl.DataFrame):
    nums = R.select(pl.int_range(pl.len(), dtype=pl.Int32).alias('idx'), pl.col('nums').list.unique().alias('v')).explode('v').drop_nulls()
    return {'num_first': R['num_first'].to_numpy(), 'addr_n': R['addr_n'].to_numpy().astype(object), 'name_n': R['name_n'].to_numpy().astype(object),
            'nums_df': nums.with_columns(pl.col('v').cast(pl.Int64))}


def sibling_features(P: pl.DataFrame, rec, threads=8, chunk_s1=60000) -> pl.DataFrame:
    """P: s1_idx, cand_idx, p1 over the FULL candidate lists of the S1s. Confident siblings = the other top-KSIB candidates of the same S1
    with p1 >= CONF. Returns s1_idx, cand_idx + SIB_FEATS:
      sib_n            # confident siblings (excluding the pair itself)
      sib_num_share    # confident siblings sharing >= 1 address number with this record
      sib_hn_eq        # confident siblings with the same primary house number (both present)
      sib_addr_max     max token_set_ratio(address, sibling address)  (-1 if no sibling)
      sib_name_max     max token_set_ratio(name, sibling name)        (-1 if no sibling)
      sib_addr_best_p  p1 of the sibling with the best address match (-1 if none)"""
    out = []
    s1u = P['s1_idx'].unique().sort().to_numpy()
    for s in range(0, len(s1u), chunk_s1):
        lo, hi = s1u[s], s1u[min(s + chunk_s1, len(s1u)) - 1]
        Q = P.filter(pl.col('s1_idx').is_between(lo, hi)).select('s1_idx', 'cand_idx', 'p1')
        C = Q.filter(pl.col('p1') >= CONF).sort('s1_idx', 'p1', descending=[False, True]).group_by('s1_idx', maintain_order=True).head(KSIB) \
             .select('s1_idx', pl.col('cand_idx').alias('sib'), pl.col('p1').alias('sp'))
        X = Q.select('s1_idx', 'cand_idx').join(C, on='s1_idx', how='inner').filter(pl.col('sib') != pl.col('cand_idx'))
        a = X['cand_idx'].to_numpy(); b = X['sib'].to_numpy()
        nf = rec['num_first']; ha = nf[a]; hb = nf[b]
        hn = ((ha >= 0) & (ha == hb)).astype(np.int8)
        ad = process.cpdist(rec['addr_n'][a], rec['addr_n'][b], scorer=fuzz.token_set_ratio, workers=threads, dtype=np.float32)
        nm = process.cpdist(rec['name_n'][a], rec['name_n'][b], scorer=fuzz.token_set_ratio, workers=threads, dtype=np.float32)
        ad[(rec['addr_n'][a] == '') | (rec['addr_n'][b] == '')] = -1
        X = X.with_columns(pl.Series('hn', hn), pl.Series('ad', ad), pl.Series('nm', nm))
        num = rec['nums_df']          # number sharing via exploded number sets
        xa = X.select(pl.int_range(pl.len()).alias('r'), 'cand_idx', 'sib')
        ea = xa.join(num, left_on='cand_idx', right_on='idx', how='inner').select('r', 'sib', 'v')
        sh = ea.join(num, left_on=['sib', 'v'], right_on=['idx', 'v'], how='semi').select('r').unique()
        share = np.zeros(X.height, np.int8); share[sh['r'].to_numpy()] = 1
        X = X.with_columns(pl.Series('ns', share))
        G = X.sort('ad', descending=True).group_by('s1_idx', 'cand_idx').agg(
            pl.len().cast(pl.Float32).alias('sib_n'), pl.col('ns').sum().cast(pl.Float32).alias('sib_num_share'), pl.col('hn').sum().cast(pl.Float32).alias('sib_hn_eq'),
            pl.col('ad').max().alias('sib_addr_max'), pl.col('nm').max().alias('sib_name_max'), pl.col('sp').first().alias('sib_addr_best_p'))
        R = Q.select('s1_idx', 'cand_idx').join(G, on=['s1_idx', 'cand_idx'], how='left').with_columns(
            pl.col('sib_n').fill_null(0.0), pl.col('sib_num_share').fill_null(0.0), pl.col('sib_hn_eq').fill_null(0.0),
            pl.col('sib_addr_max').fill_null(-1.0), pl.col('sib_name_max').fill_null(-1.0), pl.col('sib_addr_best_p').fill_null(-1.0))
        out.append(R)
    return pl.concat(out)
