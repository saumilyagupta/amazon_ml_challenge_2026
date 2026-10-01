"""B1 copies (verbatim logic, paths rebound) of the helpers the post-pass and the gate need:
  ids / read_sub / test_stats / france_proxies / decoy_words / decoy_pairs / DOFF  <- v2a_experiments/src/vx/common.py (read-only source)
  one_owner                                        <- v2a_experiments/src/v1_one_owner.py + combo.py (keep per record the S1 with the highest p; ties -> lowest s1_idx)
  extra_proxies (P1s / N1 / N2 / SWAP_GEN / SWAP_ABBR) <- v2b_postpass2/src/test_eval.py; strict-anchor filler-free / filler-added split <- prod_v3/10_gate.py
  decide_fast (vectorised R10c, verified == decide_lib.decide_set by the Codex lab) <- accuracy_lab_20260925/set_decode.py, with the qr column passed in
Nothing is imported from those directories."""
import b1common as C
import polars as pl, numpy as np

EB, FR = C.EB, C.FR


def ids(split, kind):
    if kind == 's1':
        return pl.read_parquet(f'{EB}/ids/{split}_s1.parquet', columns=['entity_id', 'country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    return pl.read_parquet(f'{EB}/ids/{split}_s23.parquet', columns=['entity_id', 'country']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))


def read_sub(path, split='test'):
    """submission TSV -> polars (s1_idx, cand_idx, country) pairs."""
    d = pl.read_csv(path, separator='\t', quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True)
    col = d.columns[1]
    d = d.with_columns(pl.col(col).fill_null('').str.split(',')).explode(col).filter(pl.col(col) != '')
    i1 = ids(split, 's1'); i2 = ids(split, 's23')
    out = d.rename({d.columns[0]: 'entity_id', col: 'cid'}).join(i1, on='entity_id').join(i2.select(pl.col('entity_id').alias('cid'), 'cand_idx'), on='cid')
    assert out.height == d.height, (out.height, d.height)
    return out.select('s1_idx', 'cand_idx', 'country')


def test_stats(pred):
    """per-country matches/S1, empty share, multi-assigned records (one-owner violations)."""
    i1 = ids('test', 's1')
    k = i1.join(pred.group_by('s1_idx').agg(pl.len().alias('k')), on='s1_idx', how='left').with_columns(pl.col('k').fill_null(0))
    out = {}
    mo = pred.join(i1.select('s1_idx', 'country'), on='s1_idx', how='left', suffix='_s') if 'country' not in pred.columns else pred
    mult = mo.group_by('cand_idx').agg(pl.len().alias('n'), pl.col('country').first())
    for c in ['US', 'India', 'France', 'all']:
        kc = k if c == 'all' else k.filter(pl.col('country') == c)
        mc = mult if c == 'all' else mult.filter(pl.col('country') == c)
        out[c] = dict(n_s1=kc.height, matches=int(kc['k'].sum()), matches_per_s1=round(float(kc['k'].mean()), 4), empty_share=round(float((kc['k'] == 0).mean()), 5),
                      multi_assigned_records=int((mc['n'] > 1).sum()), multi_pairs=int(mc.filter(pl.col('n') > 1)['n'].sum()))
    return out


_FRP = {}
def france_proxies(pred):
    """label-free France proxies (vx.common.france_proxies): strict anchors (tsr >= 95) acceptance, op slices on the anchor_ops sample,
    mined decoy pairs (decoy vs copy_filler) acceptance, France one-owner violations."""
    if not _FRP:
        i1 = ids('test', 's1'); i2 = ids('test', 's23')
        def idx(df):
            return df.join(i1.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id').join(i2.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id')
        A = pl.read_parquet(f'{FR}/forensics/data/anchors_keyed_France.parquet', columns=['s1_id', 'cand_id', 'tsr'])
        _FRP['strict'] = idx(A.filter(pl.col('tsr') >= 95)).select('s1_idx', 'cand_idx')
        O = pl.read_parquet(f'{FR}/forensics/data/anchor_ops_France.parquet', columns=['s1_id', 'cand_id', 'tsr', 'op_filler_add', 'op_amp', 'op_truncate', 'op_word_drop', 'op_country_tag', 'op_acronym'])
        _FRP['ops'] = idx(O.filter(pl.col('tsr') >= 95))
        Dp = pl.read_parquet(f'{FR}/literature/fr_decoy_pairs.parquet', columns=['s1_id', 'cand_id', 'kind'])
        _FRP['decoy'] = idx(Dp)
    sel = pred.select('s1_idx', 'cand_idx').unique().with_columns(pl.lit(True).alias('sel'))
    acc = lambda df: df.join(sel, on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('sel').fill_null(False))
    out = {}
    s = acc(_FRP['strict']); out['strict_anchor_n'] = s.height; out['strict_anchor_accept'] = round(float(s['sel'].mean()), 5)
    o = acc(_FRP['ops']); out['ops_sample_strict_accept'] = round(float(o['sel'].mean()), 5)
    for op in ['op_filler_add', 'op_amp', 'op_truncate', 'op_word_drop', 'op_country_tag', 'op_acronym']:
        q = o.filter(pl.col(op) > 0); out[f'{op}_accept'] = (q.height, round(float(q['sel'].mean()), 4) if q.height else None)
    q = o.filter(pl.col('op_filler_add') == 0); out['strict_anchor_filler_free_accept'] = (q.height, round(float(q['sel'].mean()), 5))
    q = o.filter(pl.col('op_filler_add') > 0); out['strict_anchor_filler_added_accept'] = (q.height, round(float(q['sel'].mean()), 5))
    d = acc(_FRP['decoy'])
    for kd in ['decoy', 'copy_filler']:
        q = d.filter(pl.col('kind') == kd); out[f'{kd}_accept'] = (q.height, round(float(q['sel'].mean()), 5), int(q['sel'].sum()))
    i1 = ids('test', 's1').select('s1_idx', 'country')
    fr = pred.select('s1_idx', 'cand_idx').join(i1, on='s1_idx').filter(pl.col('country') == 'France')
    mult = fr.group_by('cand_idx').agg(pl.len().alias('n'))
    out['france_one_owner_violating_records'] = int((mult['n'] > 1).sum())
    return out


_PLX = {}
def extra_proxies(pred):
    """pseudo-label proxies per country (v2b_postpass2 test_eval.extra_proxies / prod_v3 10_gate): P1s acceptance, N1 / N2 negative acceptance,
    SWAP_GEN / SWAP_ABBR acceptance (n, share, count)."""
    k2 = ['s1_idx', 'cand_idx']
    if not _PLX:
        _PLX['PL'] = pl.read_parquet(f'{FR}/pseudolabels/out/pseudolabels_test.parquet', columns=['s1_idx', 'cand_idx', 'country', 'cls', 'p1s'])
        _PLX['SW'] = pl.read_parquet(f'{FR}/pseudolabels/out/swap_pairs_test.parquet', columns=['s1_idx', 'cand_idx', 'country', 'swap'])
    s = pred.select(k2).unique().with_columns(pl.lit(True).alias('sel')); out = {}
    for c in ('France', 'US', 'India'):
        q = _PLX['PL'].filter(pl.col('country') == c).join(s, on=k2, how='left').with_columns(pl.col('sel').fill_null(False))
        out[c] = {'P1s_accept': round(float(q.filter(pl.col('p1s'))['sel'].mean()), 5),
                  'N1_accept': (q.filter(pl.col('cls') == 'N1').height, round(float(q.filter(pl.col('cls') == 'N1')['sel'].mean()), 6), int(q.filter(pl.col('cls') == 'N1')['sel'].sum())),
                  'N2_accept': (q.filter(pl.col('cls') == 'N2').height, round(float(q.filter(pl.col('cls') == 'N2')['sel'].mean()), 6), int(q.filter(pl.col('cls') == 'N2')['sel'].sum()))}
        w = _PLX['SW'].filter(pl.col('country') == c).join(s, on=k2, how='left').with_columns(pl.col('sel').fill_null(False))
        for sw in ('SWAP_GEN', 'SWAP_ABBR'):
            z = w.filter(pl.col('swap') == sw); out[c][f'{sw}_accept'] = (z.height, round(float(z['sel'].mean()), 5), int(z['sel'].sum()))
    return out


def decoy_words():
    """label-free per-country decoy vocabulary (forensics/out/decoys.json added_words = [word, n at k in D, n at k<0, n at k=0]):
    >= 500 records at decoy offsets and +k/-k ratio >= 20. France list mined from the unlabelled test France files (transductive), US/India from train inputs."""
    import json
    d = json.load(open(f'{FR}/forensics/out/decoys.json'))
    return {c: sorted(w for w, npos, nneg, n0 in d[c]['added_words'] if npos >= 500 and npos / (nneg + 1) >= 20) for c in ('US', 'India', 'France')}


DOFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]


def decoy_pairs(split):
    """neighbour-decoy pattern pairs (forensics/data/neighbours_*): ADD = record content = S1 content + exactly one country decoy word, same street
    tokens, house number = S1 house + k, k in D; EQ = identical content name at k in D minus {1, 2}. Returns s1_idx, cand_idx, kind, k, country."""
    W = decoy_words(); fs = {'train': [('US', 'neighbours_US'), ('India', 'neighbours_India')],
                             'test': [('US', 'neighbours_test_US'), ('India', 'neighbours_test_India'), ('France', 'neighbours_France')]}[split]
    i1 = ids(split, 's1'); i2 = ids(split, 's23'); out = []
    for c, f in fs:
        n = pl.read_parquet(f'{FR}/forensics/data/{f}.parquet', columns=['k', 's1_id', 'cand_id', 'rel', 'added'])
        a = n.filter((pl.col('rel') == 'superset+1') & pl.col('k').is_in(DOFF) & pl.col('added').is_in(W[c])).with_columns(pl.lit('ADD').alias('kind'))
        e = n.filter((pl.col('rel') == 'equal') & pl.col('k').is_in([x for x in DOFF if x > 2])).with_columns(pl.lit('EQ').alias('kind'))
        x = pl.concat([a, e]).join(i1.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id').join(i2.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id')
        out.append(x.select('s1_idx', 'cand_idx', 'kind', pl.col('k').cast(pl.Int16), pl.lit(c).alias('country')))
    return pl.concat(out).unique(['s1_idx', 'cand_idx'])


def one_owner(sel):
    """sel: s1_idx, cand_idx, p (selected pairs). Keep, per S2/S3 record, only the S1 with the highest p (ties -> lowest s1_idx)."""
    return sel.sort(['cand_idx', 'p', 's1_idx'], descending=[False, True, False]).group_by('cand_idx', maintain_order=True).head(1)


def decide_fast(U, model, threads=4):
    """Vectorised R10c (accuracy_lab_20260925/set_decode.decide_fast) on U = (s1, qid, src, p, qr) exactly as decide_lib.decide_set consumes it."""
    import decide_lib as DL
    U = U.select('s1', 'qid', 'src', pl.col('p').cast(pl.Float64), pl.col('qr'))
    F = DL.s1_features(U)
    W = U.filter((pl.col('qr') == 1) & (pl.col('p') > 0.02)).sort(['s1', 'p'], descending=[False, True])
    W = W.with_columns(pl.col('p').cum_count().over('s1').alias('k'), pl.col('p').cum_sum().over('s1').alias('cs'), (pl.col('src') == 'S3').cast(pl.Int32).cum_sum().over('s1').alias('c3'),
                       pl.col('p').shift(-1).over('s1').fill_null(0).alias('pn'), pl.len().over('s1').alias('n'), pl.col('p').sum().over('s1').alias('total'))
    Z = W.group_by('s1', maintain_order=True).first().select('s1', pl.lit(0).cast(pl.UInt32).alias('k'), pl.lit(1.).alias('pk'), pl.col('p').alias('pn'), pl.lit(0.).alias('cs'), 'total', 'n', pl.lit(0).cast(pl.Int32).alias('c3'))
    T = W.filter(pl.col('k') <= 10).select('s1', 'k', pl.col('p').alias('pk'), 'pn', 'cs', 'total', 'n', 'c3')
    T = pl.concat([Z, T], how='vertical_relaxed').join(F, on='s1').with_columns((pl.col('total') - pl.col('cs')).alias('tail'), (pl.col('pk') - pl.col('pn')).alias('gap'),
                                                                                (pl.col('k').cast(pl.Int32) - pl.col('c3')).alias('c2'), (pl.col('cs') / pl.col('k').clip(lower_bound=1)).alias('avg')).sort('s1', 'k')
    cols = DL.FE + ['k', 'pk', 'pn', 'cs', 'tail', 'gap', 'n', 'c3', 'c2', 'avg']
    X = T.select(cols).to_numpy().astype(np.float32)
    T = T.with_columns(pl.Series('u', model.predict(X, num_threads=threads)))
    K = T.sort(['s1', 'u', 'k'], descending=[False, True, False]).group_by('s1', maintain_order=True).first().select('s1', pl.col('k').alias('kb'))
    return W.join(K, on='s1').filter(pl.col('k') <= pl.col('kb')).select(pl.col('s1').alias('s1_idx'), pl.col('qid').alias('cand_idx'))
