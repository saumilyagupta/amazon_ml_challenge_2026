"""Learn the explainer's per-country word lists (spec section 6) from the challenge files -> <out>/ (the shipped lists are in
resources/explainer_wordlists/, produced by the research copy of this script: work/features/explainer/scripts/01_wordlists.py).

usage: /opt/conda/bin/python3 tools/learn_wordlists.py --work WORK --gt <dataset>/train/train_ground_truth.tsv [--out DIR] [--workers 16]
       (WORK = a work dir after prepare.py for BOTH splits; ~10-20 min with 16 workers; <out> defaults to WORK/explainer_wordlists;
        use the lists with ber.explainer.set_cache(<out>) or copy them over resources/explainer_wordlists/)

What is learned, and from which files (no external data):
* US / India pair-based lists (6.1 legal forms, 6.2 name filler, 6.3 address generator words, 6.5 street abbreviations) from 300k labelled
  (S1, S2/S3) pairs per country sampled from the TRAIN ground truth restricted to TRAIN-SPLIT S1 (the 220,730 validation S1 of
  resources/splits/val_s1_ids.txt are excluded, so validation stays clean): tokens the generator inserts / drops / moves between the S1
  name/address and its copies, with frequency and purity thresholds (ber.explainer.wordlists.learn_from_pairs).
* France (unseen in train) from UNLABELLED TEST ANCHOR PAIRS: France S1 with a unique street key (house number + folded street words),
  joined to the S2/S3 records carrying the same key, kept when the name-core token_set_ratio >= 95 (325,942 anchor pairs; 94.7% of such
  S1 have a record at the key); the same pair statistics then give the French legal forms (ei/eurl/sa/sarl/sas/sasu/sci), filler words
  (fils, services, developpement, associes, groupe, compagnie ...) and street abbreviations (av, bd, r, imp, rte, ch, crs ...).
  Transductive but label-free (only the test INPUT files are read).
* Label-free, all files of the country (train + test): 6.4 junk names (single tokens absent from every S1 name of the country, seen >= k
  times at distinct addresses), 6.6 admin aliases (co-occurrence, >= 90% purity; none beyond the seed tables were found), 6.7 IDF tables
  (3M-record sample per country; France all records), S1 name-core frequencies, char 3-5-gram IDF (400k-record sample).
"""
import os, sys, time, json, random, argparse
os.environ.setdefault('OMP_NUM_THREADS', '8'); os.environ.setdefault('MKL_NUM_THREADS', '8'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('POLARS_MAX_THREADS', '16')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--gt', required=True); ap.add_argument('--out', default=None)
ap.add_argument('--val-ids', default=None, help='S1 ids excluded from the pair-based lists (default resources/splits/val_s1_ids.txt)')
ap.add_argument('--workers', type=int, default=16)
ARGS = ap.parse_args()
from collections import Counter, defaultdict
import numpy as np, polars as pl
from rapidfuzz import fuzz
from ber.paths import Work, RES
from ber.explainer.io import load_records as _load_records, read_tsv
from ber.explainer import wordlists as WL
from ber.explainer.views import name_view, addr_view, RE_TOK, fold
from ber.explainer.parallel import pmap
WK = Work(ARGS.work)
OUT = ARGS.out or WK.p('explainer_wordlists')
os.makedirs(OUT, exist_ok=True); WL.set_cache(OUT)
load_records = lambda split: _load_records(WK, split)
_VAL = set(l.strip() for l in open(ARGS.val_ids or os.path.join(RES, 'splits', 'val_s1_ids.txt')) if l.strip())

T0 = time.time()
log = lambda *a: print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)
NW = max(1, min(16, ARGS.workers))
REPORT = {}
G = {}  # globals shared with forked workers


def chunks(lst, n):
    k = max(1, (len(lst) + n - 1) // n)
    return [lst[i:i + k] for i in range(0, len(lst), k)]


# ============================================================ stage 1: pair statistics
def w_pairs(rows):
    TT = G['T']
    views = []
    street = []
    for (n1, a1, n2, a2, src2, c, ne2, ae2) in rows:
        T = TT[c]
        na, nb = name_view(n1, 1, T), name_view(n2, src2, T, ne2)
        aa, ab = addr_view(a1, T), addr_view(a2, T, ae2)
        views.append((na, nb, aa, ab))
        if aa.house is not None and ab.house is not None and aa.house.val == ab.house.val:
            street.append((RE_TOK.findall(aa.comps[aa.house.comp]), RE_TOK.findall(ab.comps[ab.house.comp])))
    s1_final = Counter()
    st = WL.learn_from_pairs(views, s1_final)
    return st, street


def merge_stats(parts):
    tot = None
    streets = []
    for st, sp in parts:
        streets.extend(sp)
        if tot is None:
            tot = st
            continue
        tot['npairs'] += st['npairs']
        for k in ('ins_name', 'seen_name', 'ins_addr', 'seen_addr', 'moved'):
            tot[k].update(st[k])
    return tot, streets


def s1_final_counts(names, country):
    T = G['T'][country]
    c = Counter()
    for n in names:
        v = name_view(n, 1, T)
        if v.sides[0].tokens:
            c[v.sides[0].tokens[-1]] += 1
    return c


def w_final(args):
    names, country = args
    return s1_final_counts(names, country)


def w_fr_views(rows):
    T = G['T']['France']
    out = []
    for eid, n, a, src in rows:
        av = addr_view(a, T)
        nv = name_view(n, src, T)
        out.append((eid, src, av.street_key or '', ' '.join(nv.sides[0].core)))
    return out


seedT = {c: WL.Tables(c, {}, use_learned_filler=False) for c in WL.COUNTRIES}
G['T'] = seedT
tr = load_records('train')
te = load_records('test')
log('records loaded', tr.height, te.height)
gt = read_tsv(ARGS.gt).rename({'source1_entity_id': 's1_id', 'matched_entity_ids': 'cand_id'})
gt = gt.filter(~pl.col('s1_id').is_in(list(_VAL)))          # = train_split_ground_truth.tsv of the research run (validation S1 excluded)
gt = gt.with_columns(pl.col('cand_id').str.split(',')).explode('cand_id').filter(pl.col('cand_id') != '')
recA = tr.filter(pl.col('src') == 1).select(pl.col('entity_id').alias('s1_id'), pl.col('name').alias('n1'), pl.col('addr').alias('a1'), 'country')
recB = tr.filter(pl.col('src') != 1).select(pl.col('entity_id').alias('cand_id'), pl.col('name').alias('n2'), pl.col('addr').alias('a2'),
                                            pl.col('src').alias('src2'), pl.col('name_en').alias('ne2'), pl.col('addr_en').alias('ae2'))
P = gt.join(recA, on='s1_id').join(recB, on='cand_id')
learned = {c: {} for c in WL.COUNTRIES}
street_all = {}
import pickle
S1P = f'{OUT}/stage1_stats.pkl'
if os.path.exists(S1P):
    G['stats'], street_all, REPORT['france_anchors'] = pickle.load(open(S1P, 'rb'))
    log('stage-1 stats loaded from cache')
for c in ([] if os.path.exists(S1P) else ['US', 'India']):
    pc = P.filter(pl.col('country') == c); pc = pc.sample(n=min(300_000, pc.height), seed=0)   # min(): only binds on small subsets (smoke test)
    rows = pc.select('n1', 'a1', 'n2', 'a2', 'src2', 'country', 'ne2', 'ae2').rows()
    st, streets = merge_stats(pmap(w_pairs, chunks(rows, NW * 4), NW))
    s1n = tr.filter((pl.col('src') == 1) & (pl.col('country') == c))['name'].to_list()
    fin = Counter()
    for x in pmap(w_final, [(ch, c) for ch in chunks(s1n, NW * 2)], NW):
        fin.update(x)
    st['s1_final'] = fin
    G.setdefault('stats', {})[c] = st
    street_all[c] = streets
    log(c, 'pairs', st['npairs'], 'street-aligned', len(streets))

# ---- France anchors: same house number + same folded street + name-core tsr >= 95
if not os.path.exists(S1P):
    fr = te.filter(pl.col('country') == 'France').select('entity_id', 'name', 'addr', 'src')
    fv = [x for part in pmap(w_fr_views, chunks(fr.rows(), NW * 4), NW) for x in part]
    fvd = pl.DataFrame(fv, schema={'entity_id': pl.Utf8, 'src': pl.Int64, 'skey': pl.Utf8, 'core': pl.Utf8}, orient='row')   # typed: also valid when a subset has no France
    s1k = fvd.filter((pl.col('src') == 1) & (pl.col('skey') != ''))
    uniq = s1k.group_by('skey').len().filter(pl.col('len') == 1).select('skey')
    s1k = s1k.join(uniq, on='skey')
    cand = fvd.filter((pl.col('src') != 1) & (pl.col('skey') != ''))
    anc = s1k.join(cand, on='skey', suffix='_b')
    anc = anc.with_columns(pl.struct('core', 'core_b').map_elements(lambda r: fuzz.token_set_ratio(r['core'], r['core_b']), return_dtype=pl.Float64).alias('tsr'))
    n_s1_unique = s1k.height
    cov = anc.select('entity_id').n_unique()
    anc95 = anc.filter(pl.col('tsr') >= 95)
    REPORT['france_anchors'] = dict(fr_s1=int((fvd['src'] == 1).sum()), s1_with_unique_key=n_s1_unique, s1_unique_key_with_s23_at_key=cov,
                                    share=round(cov / max(n_s1_unique, 1), 4), keyed_pairs=anc.height, anchor_pairs_tsr95=anc95.height)
    log('France anchors', REPORT['france_anchors'])
    anc95.select(pl.col('entity_id').alias('s1_id'), pl.col('entity_id_b').alias('cand_id'), 'tsr').write_parquet(f'{OUT}/France_anchor_pairs.parquet')
    frd = te.filter(pl.col('country') == 'France')
    A = frd.filter(pl.col('src') == 1).select(pl.col('entity_id').alias('s1_id'), pl.col('name').alias('n1'), pl.col('addr').alias('a1'), 'country')
    B = frd.filter(pl.col('src') != 1).select(pl.col('entity_id').alias('cand_id'), pl.col('name').alias('n2'), pl.col('addr').alias('a2'),
                                               pl.col('src').alias('src2'), pl.col('name_en').alias('ne2'), pl.col('addr_en').alias('ae2'))
    PA = anc95.select(pl.col('entity_id').alias('s1_id'), pl.col('entity_id_b').alias('cand_id')).join(A, on='s1_id').join(B, on='cand_id')
    rows = PA.select('n1', 'a1', 'n2', 'a2', 'src2', 'country', 'ne2', 'ae2').rows()
    st, streets = merge_stats(pmap(w_pairs, chunks(rows, NW * 4), NW))
    s1n = frd.filter(pl.col('src') == 1)['name'].to_list()
    fin = Counter()
    for x in pmap(w_final, [(ch, 'France') for ch in chunks(s1n, NW * 2)], NW):
        fin.update(x)
    st['s1_final'] = fin
    G['stats']['France'] = st
    street_all['France'] = streets
    log('France anchor stats pairs', st['npairs'], 'street-aligned', len(streets))
    pickle.dump((G['stats'], street_all, REPORT['france_anchors']), open(S1P, 'wb'))

# ---- select lists (thresholds relative to the number of pairs)
for c in WL.COUNTRIES:
    st = G['stats'][c]
    npairs = st['npairs']
    mi_n = max(30, int(0.0001 * npairs))
    mi_a = max(30, int(0.0015 * npairs))
    legal, filler, afill = WL.select_lists(st, min_ins_name=mi_n, min_rate_name=0.1, min_ins_addr=mi_a, min_rate_addr=0.5,
                                          min_moved=max(10, int(0.0003 * npairs)))
    sabbr, sstats = WL.learn_street_abbr(street_all[c], min_count=max(10, int(0.0001 * npairs)), min_share=0.6)
    from ber.explainer import tables as TB
    from ber.explainer.views import hg as _hg
    _lh = {_hg(x) for x in TB.LEGAL_CANON} | {_hg(x) for x in TB.CONNECTIVES}
    fl = [t for t, _, _ in filler if _hg(t) not in _lh and t not in ('nee', 'aka', 'dba', 'fka', 'shree', 'shri', 'sri', 'smt', 'the')]
    legal = [t for t in legal if t not in fl and t not in TB.SEED_FILLER.get(c, set())]
    cities = set(TB.IN_CITY_ALIASES) | set(TB.IN_CITY_ALIASES.values())
    af = [t for t, _, _ in afill if t not in cities]
    learned[c].update(legal=legal, filler=fl, addr_filler=af, street_abbr=sabbr)
    log(c, 'top inserted name tokens (tok, n_inserted, rate)', [(t, n, round(n / max(st['seen_name'][t], 1), 3)) for t, n in st['ins_name'].most_common(45)])
    REPORT[c] = dict(npairs=npairs, legal=legal, filler=filler, addr_filler=afill, street_abbr=sstats,
                     top_final_s1=st['s1_final'].most_common(40), top_moved=st['moved'].most_common(30),
                     top_ins_name=[(t, n, round(n / max(st['seen_name'][t], 1), 3)) for t, n in st['ins_name'].most_common(60)],
                     top_ins_addr=[(t, n, round(n / max(st['seen_addr'][t], 1), 3)) for t, n in st['ins_addr'].most_common(60)])
    log(c, 'legal', legal)
    log(c, 'filler', [t for t, _, _ in filler])
    log(c, 'addr_filler', [t for t, _, _ in afill])
    log(c, 'street_abbr', sabbr)

# ============================================================ stage 2: record passes (junk, admin, IDF, S1 frequencies)
G['T'] = {c: WL.Tables(c, learned[c]) for c in WL.COUNTRIES}


def w_s1(rows):
    TT = G['T']
    core, vocab, place = defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    for n, a, c in rows:
        T = TT[c]
        nv = name_view(n, 1, T)
        s = nv.sides[0]
        core[c][s.core_key] += 1
        vocab[c].update(s.tokens)
        av = addr_view(a, T)
        if len(av.admin) == 1:
            code = next(iter(av.admin))
            for ci, cc in enumerate(av.comps):
                if not av.comp_is_admin[ci] and not any(ch.isdigit() for ch in cc):
                    place[c][(WL._akey(cc), code)] += 1
    return core, vocab, place


def w_cooc(rows):
    TT = G['T']
    pm, s1comps = G['placemap'], G['s1comps']
    co, tot, wadm = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for a, c, ae in rows:
        T = TT[c]
        av = addr_view(a, T, ae)
        keys = [WL._akey(cc) for ci, cc in enumerate(av.comps) if not av.comp_is_admin[ci] and not any(ch.isdigit() for ch in cc)]
        known = [pm[c][k] for k in keys if k in pm[c]]
        for k in keys:
            if k in s1comps[c] or len(k) < 3:
                continue
            tot[c][k] += 1
            if av.admin:
                wadm[c][k] += 1
            for code in known:
                co[c][(k, code)] += 1
    return co, tot, wadm


def w_junk(rows):
    TT = G['T']
    out = []
    for n, a, c, src in rows:
        T = TT[c]
        nv = name_view(n, src, T)
        for si, s in enumerate(nv.sides):
            if len(s.content) == 1 and s.web is None:
                t = s.content[0]
                if t.isalpha() and 5 <= len(t) <= 14 and t not in G['vocab'][c]:
                    out.append((c, t, fold(a)))
    return out


def w_idf(rows):
    TT = G['T']
    dn, da, nn = defaultdict(Counter), defaultdict(Counter), Counter()
    for n, a, c, src, ne, ae in rows:
        T = TT[c]
        nv = name_view(n, src, T, ne)
        dn[c].update(set(nv.sides[0].content))
        av = addr_view(a, T, ae)
        da[c].update(av.tokset)
        nn[c] += 1
    return dn, da, nn


def w_ctext(rows):
    TT = G['T']
    out = []
    for n, a, c, src, ne, ae in rows:
        T = TT[c]
        nv = name_view(n, src, T, ne)
        av = addr_view(a, T, ae)
        out.append((c, ' '.join(nv.sides[0].core) if not nv.empty else '', ' '.join(av.tokens)))
    return out


both = pl.concat([tr, te])
# --- S1 pass: core frequencies, S1 vocabulary, place -> admin map
s1 = both.filter(pl.col('src') == 1).select('name', 'addr', 'country').rows()
core, vocab, place = defaultdict(Counter), defaultdict(set), defaultdict(Counter)
for cc, vv, pp in pmap(w_s1, chunks(s1, NW * 4), NW):
    for c in cc:
        core[c].update(cc[c]); vocab[c] |= vv[c]; place[c].update(pp[c])
log('S1 pass done', {c: (len(core[c]), len(vocab[c]), len(place[c])) for c in core})
placemap, s1comps = {}, {}
for c in WL.COUNTRIES:
    agg = defaultdict(Counter)
    for (k, code), n in place[c].items():
        agg[k][code] += n
    s1comps[c] = set(agg)
    pm = {}
    for k, cnt in agg.items():
        code, n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if tot >= 3 and n / tot >= 0.95:
            pm[k] = code
    placemap[c] = pm
G['placemap'], G['s1comps'], G['vocab'] = placemap, s1comps, vocab
# --- admin aliases by co-occurrence (S2/S3 sample, 1M per country)
admin_alias = {}
for c in WL.COUNTRIES:
    smp = both.filter((pl.col('src') != 1) & (pl.col('country') == c))
    smp = smp.sample(n=min(1_000_000, smp.height), seed=1).select('addr', 'country', 'addr_en').rows()
    co, tot, wadm = defaultdict(Counter), Counter(), Counter()
    for cc, tt, ww in pmap(w_cooc, chunks(smp, NW * 4), NW):
        co[c].update(cc[c]); tot.update(tt[c]); wadm.update(ww[c])
    agg = defaultdict(Counter)
    for (k, code), n in co[c].items():
        agg[k][code] += n
    al = {}
    seed = WL.seed_admin(c)
    places_per = defaultdict(set)
    # validation of the seed aliases with the same co-occurrence statistic (e.g. Gironde -> Nouvelle-Aquitaine purity)
    sv = {}
    for k, code in seed.items():
        cnt = agg.get(k)
        if cnt and tot.get(k, 0) >= 30 and len(k) > 2:
            top, n = cnt.most_common(1)[0]
            sv[k] = dict(seed_code=code, cooc_top=top, purity=round(n / sum(cnt.values()), 3), n=tot[k], with_admin=round(wadm[k] / tot[k], 3))
    REPORT.setdefault(c, {})['admin_seed_validation'] = sv
    for k, n_tot in tot.items():
        if n_tot < 30 or k in seed:
            continue
        if any(w in G['T'][c].street for w in k.split()):  # admin names never contain street types ('r lyderic')
            continue
        cnt = agg.get(k)
        if not cnt:
            continue
        code, n = cnt.most_common(1)[0]
        # a real alias REPLACES the admin component: rarely co-occurs with a seed admin comp
        if n >= 20 and n / sum(cnt.values()) >= 0.9 and n / n_tot >= 0.5 and wadm[k] / n_tot < 0.2:
            al[k] = (code, n_tot, round(n / sum(cnt.values()), 3), round(wadm[k] / n_tot, 3))
    admin_alias[c] = al
    learned[c]['admin_alias'] = {k: v[0] for k, v in al.items()}
    REPORT.setdefault(c, {})['admin_alias'] = sorted(al.items(), key=lambda x: -x[1][1])[:80]
    log(c, 'admin aliases learned', len(al), sorted(al.items(), key=lambda x: -x[1][1])[:25])
# --- junk names (6.4): single-token names absent from all S1 names of the country, >= K distinct addresses
K = 5
cand = both.filter((pl.col('src') != 1) & (pl.col('name').str.strip_chars().str.count_matches(r'\s+') <= 6)
                   & ~pl.col('name').str.contains(r'[\x{0900}-\x{0DFF}]')).select('name', 'addr', 'country', 'src').rows()  # junk names are Latin
jr = [x for part in pmap(w_junk, chunks(cand, NW * 4), NW) for x in part]
jd = pl.DataFrame(jr, schema=['country', 'tok', 'addr'], orient='row')
js = jd.group_by('country', 'tok').agg(pl.col('addr').n_unique().alias('n_addr'), pl.len().alias('n'))
for c in WL.COUNTRIES:
    jc = js.filter((pl.col('country') == c) & (pl.col('n_addr') >= K)).sort('n', descending=True)
    learned[c]['junk'] = jc['tok'].to_list()
    REPORT.setdefault(c, {})['junk'] = dict(n=jc.height, records=int(jc['n'].sum()), top=jc.head(30).rows())
    log(c, 'junk names', jc.height, 'records', int(jc['n'].sum()), jc.head(12).rows())
# --- IDF (6.7): sample of up to 3M records per country (S1+S2+S3)
G['T'] = {c: WL.Tables(c, learned[c]) for c in WL.COUNTRIES}
big = {c: {} for c in WL.COUNTRIES}
for c in WL.COUNTRIES:
    sc = both.filter(pl.col('country') == c)
    smp = sc.sample(n=min(3_000_000, sc.height), seed=2).select('name', 'addr', 'country', 'src', 'name_en', 'addr_en').rows()
    dn, da, N = Counter(), Counter(), 0
    for a, b, n in pmap(w_idf, chunks(smp, NW * 4), NW):
        dn.update(a[c]); da.update(b[c]); N += n[c]
    big[c]['idf_name'] = {t: round(float(np.log(N / d)), 3) for t, d in dn.items() if d >= 2}
    big[c]['idf_addr'] = {t: round(float(np.log(N / d)), 3) for t, d in da.items() if d >= 2}
    big[c]['idf_name_default'] = round(float(np.log(N)), 3)
    big[c]['idf_addr_default'] = round(float(np.log(N)), 3)
    cf = core[c]
    counts = np.array(sorted(cf.values()), dtype=np.int64)
    # percentile over S1 RECORDS: share of S1 records whose core frequency <= x
    vals, mult = np.unique(counts, return_counts=True)
    rec = vals * mult
    big[c]['core_freq'] = {k: v for k, v in cf.items() if v >= 2}
    big[c]['core_freq_pct'] = [[0] + vals.tolist(), [0.0] + (np.cumsum(rec) / rec.sum()).round(5).tolist()]
    big[c]['s1_vocab'] = sorted(vocab[c])
    log(c, 'IDF N', N, 'name vocab', len(big[c]['idf_name']), 'addr vocab', len(big[c]['idf_addr']), 'core keys>=2', len(big[c]['core_freq']))
    # char 3-5gram IDF for N3/A13 (400k-record sample)
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=2 ** 20, alternate_sign=False, norm=None, dtype=np.float32, binary=True)
    tx = [x for part in pmap(w_ctext, chunks(smp[:400_000], NW * 4), NW) for x in part]
    for j, kind in ((1, 'name'), (2, 'addr')):
        X = hv.transform([t[j] for t in tx])
        df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
        idf = (np.log((X.shape[0] + 1) / (df + 1)) + 1).astype(np.float32)
        np.save(f'{OUT}/{c}_cidf_{kind}.npy', idf)
for c in WL.COUNTRIES:
    WL.save_learned(c, learned[c], big[c])
json.dump(REPORT, open(f'{OUT}/learn_report.json', 'w'), indent=1, ensure_ascii=False, default=str)
log('DONE')
