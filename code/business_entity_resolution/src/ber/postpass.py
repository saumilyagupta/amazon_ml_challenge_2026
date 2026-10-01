"""Matcher v3 post-pass (config block 'postpass', run by predict.py after the decision): label-free decoy VETOES, removal only (no re-scoring,
no re-decision). Research original: work/matching/v3_postpass (src/pp.py, src/test_eval_v3.py, src/write_variant.py, src/vx/common.py) on the
pattern tables of work/matching/v2b_postpass2/src/patterns.py and the neighbour mining of work/research/france/forensics/scripts/
{01_views, 08_decoys, 08b_decoys_test_usin}.py. Every pattern table is DERIVED here from the split's own records and scored pairs; only the
small decoy word lists are shipped (resources/postpass_decoy_words.json, re-derivable with learn_words()).

The data generator plants decoys of S1 entities: same street, house number = S1 house + k with k in D = {1,2,3,4,5,7,9,11,13,21}, and a name
that is the S1 name + one decoy word, + a country tag, or with a switched / added legal form. v3's offset-conditioned features reject most of
them; the post-pass removes a selected pair (S1, record) iff it matches one of these patterns (rules applied in this order):
  sixword       record name content = S1 content + exactly one word of the country's decoy word list, same street, k in D (all countries)
  FRcountry     France: legal-free name tokens = S1 tokens + one country tag ('france'), same street, k in D, and all number parts equal except
                the house number h1 -> h1 + k ('allnum')
  FRlegal       France: same legal-free token multiset, canonical legal-form multiset differs, the record carries a legal form, k in D, allnum
  USlegal_no12  US: as FRlegal with the English legal canon, k in D minus {1, 2} (US train has many true copies with a mistyped number there)
  one-owner     each S2/S3 record keeps only its highest-p selecting S1 (a no-op after v3's own-competition decoder)
Offset 0 (true copies) is never touched; the mirror offsets -k are reported as the label-free purity control (report['mirror']).

Label use: none. Record views = explainer name / address views (ber.explainer with the shipped word lists): house number, sorted street tokens,
name content tokens; the country / legal rules read the raw name / address (ber.postpass_rules). Decoy words: added to S1 content on the same
street at k > 0 with n(k > 0) >= 500 and n(k > 0) / (n(k < 0) + 1) >= 20 among the 120 most frequent added words; US / India from the TRAIN
inputs, France from the unlabelled TEST inputs (transductive). D is generator-level (the same in all countries).
Caches: WORK/records/pp_views_{split}_{s1,s23}.parquet (views of every record; ~5 min with 8 workers on the 11.7M test records, measured 300 s)."""
import collections, json, os, time
import multiprocessing as mp
import polars as pl
from .paths import RES
from . import postpass_rules as R

D = list(R.D); D_NO12 = [k for k in D if k > 2]; K2 = ['s1_idx', 'cand_idx']
RULES = ('sixword', 'FRcountry', 'FRlegal', 'USlegal_no12')
OFF_MINE = list(range(-25, 26))
WORDS_PATH = os.path.join(RES, 'postpass_decoy_words.json')
VIEW_COLS = ['entity_id', 'country', 'src', 'house', 'street_toks', 'content', 'name', 'addr']
_TB = {}


# ---------------------------------------------------------------- record views (research forensics/scripts/01_views.py, the columns used here)
def _view_rows(rows):
    """rows: (name, addr, country, src, name_en, addr_en) -> [(house, street_toks, content)] (needs _TB = ber.explainer.all_tables())."""
    from .explainer.views import name_view, addr_view
    out = []
    for name, addr, country, src, name_en, addr_en in rows:
        T = _TB[country]
        a = addr_view(addr, T, addr_en); n = name_view(name, src, T, name_en); h = a.house
        out.append((None if h is None else int(min(h.val, 2 ** 62)), ' '.join(sorted(a.street_toks)), ' '.join(n.main.content)))
    return out


def record_views(W, split, threads=8, log=print):
    """-> (V1, V2): views of the S1 / S2+S3 records of the split in prepared row order (row == s1_idx / cand_idx); columns VIEW_COLS."""
    out = []
    for part in ('s1', 's23'):
        path = W.p('records', f'pp_views_{split}_{part}.parquet')
        if os.path.exists(path):
            V = pl.read_parquet(path)
            miss = [c for c in VIEW_COLS if c not in V.columns]
            if miss:   # a cache written without the id / source columns: take them from the prepared records (same row order)
                R0 = pl.read_parquet(W.records(split, part), columns=['entity_id', 'country', 'src', 'business_name', 'business_address'])
                R0 = R0.select('entity_id', 'country', pl.col('src').cast(pl.Int8), pl.col('business_name').alias('name'), pl.col('business_address').alias('addr'))
                assert R0.height == V.height, f'{path}: {V.height} rows vs {R0.height} records'
                V = pl.concat([V, R0.select([c for c in miss])], how='horizontal')
            out.append(V.select(VIEW_COLS)); continue
        if not _TB:
            from .explainer import all_tables
            _TB.update(all_tables())
        R0 = pl.read_parquet(W.records(split, part), columns=['entity_id', 'business_name', 'business_address', 'country', 'src', 'name_en', 'addr_en'])
        rows = R0.select('business_name', 'business_address', 'country', 'src', 'name_en', 'addr_en').rows()
        chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]; t = time.time(); res = []
        with mp.get_context('fork').Pool(max(1, threads)) as pool:
            for r in pool.imap(_view_rows, chunks):
                res.extend(r)
        V = pl.DataFrame(res, schema={'house': pl.Int64, 'street_toks': pl.Utf8, 'content': pl.Utf8}, orient='row')
        V = pl.concat([R0.select('entity_id', 'country', pl.col('src').cast(pl.Int8)), V,
                       R0.select(pl.col('business_name').alias('name'), pl.col('business_address').alias('addr'))], how='horizontal').select(VIEW_COLS)
        V.write_parquet(path + '.tmp'); os.replace(path + '.tmp', path); log(f'post-pass record views {split}/{part}: {V.height} records in {time.time() - t:.0f}s')
        out.append(V); del R0, rows, res
    return out[0], out[1]


# ---------------------------------------------------------------- neighbour mining (research forensics/scripts/08_decoys.py, logic unchanged)
def neighbours(V):
    """V: record views of ONE country (all S1 + S2/S3 records). -> s1_id, cand_id, k, rel in {superset+1, superset+2, equal}, added."""
    V = V.filter(pl.col('house').is_not_null() & (pl.col('street_toks') != '') & (pl.col('house') < 10 ** 9)).with_columns(pl.col('street_toks').hash().alias('sh'))
    s1 = V.filter(pl.col('src') == 1).select(pl.col('entity_id').alias('s1_id'), 'sh', pl.col('house').alias('h1'), pl.col('content').alias('ca'))
    rc = V.filter(pl.col('src') != 1).select(pl.col('entity_id').alias('cand_id'), 'sh', pl.col('house').alias('h2'), pl.col('content').alias('cb'))
    # memory-safe join: S1 exploded over offsets, keyed by (street hash, house+k, S1's min content token); records exploded over their content
    # tokens (superset / equal relations always share that token)
    s1 = s1.with_row_index('i1'); rc = rc.with_row_index('i2')
    ex = s1.filter(pl.col('ca') != '').select('i1', 'sh', 'h1', pl.col('ca').str.split(' ').list.min().hash().alias('th')) \
           .with_columns(pl.lit(OFF_MINE, dtype=pl.List(pl.Int16)).alias('k')).explode('k').with_columns((pl.col('h1') + pl.col('k')).alias('h2')).drop('h1')
    rx = rc.filter(pl.col('cb') != '').select('i2', 'sh', 'h2', pl.col('cb').str.split(' ').alias('t')).explode('t').with_columns(pl.col('t').hash().alias('th')).drop('t').unique()
    P = ex.join(rx, on=['sh', 'h2', 'th']).select('i1', 'i2', 'k'); del ex, rx
    P = P.join(s1.select('i1', 's1_id', 'ca'), on='i1').join(rc.select('i2', 'cand_id', 'cb'), on='i2').drop('i1', 'i2')
    rel, add = [], []
    for a, b in zip(P['ca'].to_list(), P['cb'].to_list()):
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            rel.append('empty'); add.append(None); continue
        if sa == sb: rel.append('equal'); add.append(None)
        elif sa < sb:
            d = sb - sa
            rel.append('superset+1' if len(d) == 1 else 'superset+2'); add.append(' '.join(sorted(d)))
        elif sb < sa: rel.append('subset'); add.append(None)
        else: rel.append('other'); add.append(None)
    P = P.with_columns(pl.Series('rel', rel), pl.Series('added', add, dtype=pl.Utf8))
    return P.filter(pl.col('rel').is_in(['superset+1', 'superset+2', 'equal'])).select('s1_id', 'cand_id', 'k', 'rel', 'added')


def added_words(N, top=120):
    """[(word, n at k > 0, n at k < 0, n at k = 0)] for superset+1 pairs, the `top` most frequent by n(k > 0) + n(k = 0) (research decoys.json)."""
    sp = N.filter(pl.col('rel') == 'superset+1')
    wpos = collections.Counter(sp.filter(pl.col('k') > 0)['added'].to_list())
    wneg = collections.Counter(sp.filter(pl.col('k') < 0)['added'].to_list())
    w0 = collections.Counter(sp.filter(pl.col('k') == 0)['added'].to_list())
    words = sorted(set(wpos) | set(w0), key=lambda w: -(wpos[w] + w0[w]))
    return [(w, wpos[w], wneg[w], w0[w]) for w in words[:top]]


def decoy_words_from(aw, min_pos=500, ratio=20):
    return sorted(w for w, npos, nneg, n0 in aw if npos >= min_pos and npos / (nneg + 1) >= ratio)


def decoy_words(path=WORDS_PATH):
    """shipped lists {country: [words]} (resources/postpass_decoy_words.json)."""
    d = json.load(open(path))
    return {c: v['words'] for c, v in d.items() if isinstance(v, dict) and 'words' in v}


def learn_words(W, threads=8, log=print):
    """re-derive the decoy word lists: countries of the TRAIN records from train, the other countries (France) from the TEST records."""
    out = {}
    for split in ('train', 'test'):
        V = pl.concat(record_views(W, split, threads, log))
        for c in sorted(set(V['country'].unique().to_list()) - set(out)):
            out[c] = decoy_words_from(added_words(neighbours(V.filter(pl.col('country') == c)))); log(f'decoy words {c} ({split}): {len(out[c])}')
    return out


# ---------------------------------------------------------------- 'country' / 'legal' patterns (research v2b_postpass2/src/patterns.py, logic unchanged)
def _classify(args):
    rows, country = args
    return R.classify_rows(rows, country)


def _patterns(V1c, V2c, cands, country, threads=8):
    offs = sorted(set(D) | {-k for k in D} | {0})
    ok = lambda V: V.filter(pl.col('house').is_not_null() & pl.col('street_toks').is_not_null() & (pl.col('street_toks') != '') & (pl.col('house') < 10 ** 9)) \
                    .with_columns(pl.col('street_toks').hash().alias('sh'))
    s1 = ok(V1c).select('s1_idx', pl.col('house').alias('h1'), pl.col('sh').alias('sh1'), pl.col('name').alias('n1'), pl.col('addr').alias('a1'))
    s2 = ok(V2c).select('cand_idx', pl.col('house').alias('h2'), pl.col('sh').alias('sh2'), pl.col('name').alias('n2'), pl.col('addr').alias('a2'))
    J = cands.select(K2).join(s1, on='s1_idx').join(s2, on='cand_idx').filter(pl.col('sh1') == pl.col('sh2')).with_columns((pl.col('h2') - pl.col('h1')).alias('k')).filter(pl.col('k').is_in(offs))
    rows = list(zip(J['n1'].to_list(), J['n2'].to_list(), J['a1'].to_list(), J['a2'].to_list(), J['h1'].to_list(), J['k'].to_list()))
    ch = [(rows[i:i + 50000], country) for i in range(0, len(rows), 50000)]
    if threads > 1 and len(ch) > 1:
        with mp.get_context('fork').Pool(threads) as pool:
            out = [x for part in pool.map(_classify, ch) for x in part]
    else:
        out = [x for c in ch for x in _classify(c)]
    J = J.with_columns(pl.Series('kind', [o[0] for o in out], pl.Utf8), pl.Series('extra', [o[1] for o in out], pl.Utf8), pl.Series('allnum', [o[2] for o in out], pl.Boolean)) \
         .filter(pl.col('kind').is_not_null())
    return J.select(K2 + [pl.col('k').cast(pl.Int32), 'kind', 'extra', 'allnum'])


def pattern_pairs(cands, V1, V2, words, threads=8, log=print):
    """cands: the scored pairs (s1_idx, cand_idx) of the split; V1 / V2: record_views(). -> one table of pattern pairs among the candidates:
    s1_idx, cand_idx, country, k, kind ('sixword' | 'country' | 'legal'), extra (added word / legal switch), allnum (null for sixword)."""
    V1 = V1.with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32)); V2 = V2.with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
    C = cands.select(K2).join(V1.select('s1_idx', 'country'), on='s1_idx'); out = []
    offs = D + [-k for k in D]
    for c in sorted(C['country'].unique().to_list()):
        t = time.time(); V1c = V1.filter(pl.col('country') == c); V2c = V2.filter(pl.col('country') == c); Cc = C.filter(pl.col('country') == c)
        N = neighbours(pl.concat([V1c.drop('s1_idx'), V2c.drop('cand_idx')]))
        a = N.filter((pl.col('rel') == 'superset+1') & pl.col('k').is_in(offs) & pl.col('added').is_in(words.get(c, [])))
        a = a.join(V1c.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id').join(V2c.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id') \
             .select(K2 + [pl.col('k').cast(pl.Int32), pl.lit('sixword').alias('kind'), pl.col('added').alias('extra'), pl.lit(None, pl.Boolean).alias('allnum')]) \
             .unique(K2).join(Cc.select(K2), on=K2, how='semi')
        p = _patterns(V1c, V2c, Cc, c, threads)
        out.append(pl.concat([a, p]).with_columns(pl.lit(c).alias('country')).select(K2 + ['country', 'k', 'kind', 'extra', 'allnum']))
        log(f'post-pass patterns {c}: neighbour pairs {N.height}, sixword {a.height}, country/legal {p.height} ({time.time() - t:.0f}s)')
    return pl.concat(out)


# ---------------------------------------------------------------- vetoes (research v3_postpass/src/pp.py rule_sets + test_eval_v3.py sequential part)
def rule_sets(T):
    """-> {rule: (veto pairs at +k, mirror pairs at -k)}"""
    Rs = {}
    b = T.filter(pl.col('kind') == 'sixword'); Rs['sixword'] = (b.filter(pl.col('k').is_in(D)), b.filter(pl.col('k').is_in([-k for k in D])))
    for kind, nm in (('country', 'FRcountry'), ('legal', 'FRlegal')):
        b = T.filter((pl.col('kind') == kind) & (pl.col('country') == 'France') & pl.col('allnum'))
        Rs[nm] = (b.filter(pl.col('k').is_in(D)), b.filter(pl.col('k').is_in([-k for k in D])))
    b = T.filter((pl.col('kind') == 'legal') & (pl.col('country') == 'US') & pl.col('allnum'))
    Rs['USlegal_no12'] = (b.filter(pl.col('k').is_in(D_NO12)), b.filter(pl.col('k').is_in([-k for k in D_NO12])))
    return Rs


def apply(sel, T, rules=RULES, one_owner=True, s1country=None, log=print):
    """sel: selected pairs s1_idx, cand_idx, p. Sequential removal in the order of `rules`, then strict one-owner. -> (final, report)"""
    Rs = rule_sets(T); cur = sel.select(K2 + ['p']); rep = {'rules': list(rules), 'one_owner': bool(one_owner), 'steps': {}}
    bc = (lambda df: dict(df.join(s1country, on='s1_idx').group_by('country').len().sort('country').rows())) if s1country is not None else (lambda df: {})
    for nm in rules:
        Vp = Rs[nm][0].select(K2); rem = cur.join(Vp, on=K2, how='semi'); cur = cur.join(Vp, on=K2, how='anti')
        mir = sel.join(Rs[nm][1].select(K2), on=K2, how='semi').height
        rep['steps'][nm] = dict(removed=rem.height, by_country=bc(rem), pattern_pairs_plus_k=Rs[nm][0].height, pattern_pairs_minus_k=Rs[nm][1].height,
                                selected_minus_k_mirror=mir)
        log('post-pass veto', nm, rep['steps'][nm])
    if one_owner:
        n0 = cur.height; cur = cur.sort(['cand_idx', 'p', 's1_idx'], descending=[False, True, False]).group_by('cand_idx', maintain_order=True).head(1)
        rep['steps']['one_owner'] = dict(removed=n0 - cur.height)
    rem = sel.join(cur, on=K2, how='anti')
    rep.update(matches_before=sel.height, matches_after=cur.height, removed=rem.height, removed_by_country=bc(rem))
    return cur, rep


def run(W, split, cands, sel, PPC=None, threads=8, log=print):
    """predict.py hook: cands = every scored pair of the split, sel = the decision (s1_idx, cand_idx, p) -> {'selected', 'report'}."""
    PPC = PPC or {}; t = time.time()
    rules = [r for r in PPC.get('rules', RULES) if r != 'one_owner']; one_owner = bool(PPC.get('one_owner', True))
    words = decoy_words(); V1, V2 = record_views(W, split, threads, log)
    T = pattern_pairs(cands, V1, V2, words, threads, log)
    s1c = V1.select('country').with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32)); del V1, V2
    final, rep = apply(sel, T, rules, one_owner, s1c, log)
    assert final.join(sel, on=K2, how='anti').height == 0, 'the post-pass may only remove pairs'
    rep.update(decoy_words={c: len(v) for c, v in words.items()}, pattern_pairs={f'{a}|{b}': n for a, b, n in T.group_by('country', 'kind').len().sort('country', 'kind').rows()},
               seconds=round(time.time() - t))
    return {'selected': final, 'report': rep}
