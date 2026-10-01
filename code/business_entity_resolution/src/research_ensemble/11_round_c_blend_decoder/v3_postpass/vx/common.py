"""Shared helpers for the v2a_experiments scripts (paths, env caps, id maps, val truth, scoring, submission I/O, France proxies).
Every other project directory is READ-ONLY and only imported through sys.path."""
import os, sys, time, json


def envcap(n=8):
    n = int(os.environ.get("VX_THREADS", n))
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[k] = str(n)
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'; os.environ.setdefault('POLARS_MAX_THREADS', str(n))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['WANDB_MODE'] = 'disabled'; os.environ['WANDB_DISABLED'] = 'true'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ.setdefault('PV1_THREADS', str(n))


W = '/workspace/saumilya/amazon-ml/work'
X = f'{W}/matching/v3_postpass'; XA = f'{W}/matching/v2a_experiments'; XP2 = f'{W}/matching/v2b_postpass2'; V3 = f'{W}/matching/prod_v3'
XD, XL, XO = f'{X}/data', f'{X}/logs', f'{X}/output'
V1 = f'{W}/matching/prod_v1'; V2A = f'{W}/matching/prod_v2a'; V2B = f'{W}/matching/prod_v2b'
EB = f'{W}/blocking/embedding/full'; SPL = f'{W}/splits'; SR = '/workspace/saumilya/amazon-ml/student_resource'
FR = f'{W}/research/france'; DV = f'{W}/matching/density_val'
for p in (V1, V2A, V2B, f'{W}/common', f'{W}/research/postproc'):
    if p not in sys.path: sys.path.insert(0, p)


def logger(name=None):
    T0 = time.time()
    fh = open(f'{XL}/{name}.log', 'a') if name else None
    def log(*a):
        s = f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True)
        if fh: fh.write(s + '\n'); fh.flush()
    return log


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def ids(split, kind):
    import polars as pl
    if kind == 's1':
        return pl.read_parquet(f'{EB}/ids/{split}_s1.parquet', columns=['entity_id', 'country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    return pl.read_parquet(f'{EB}/ids/{split}_s23.parquet', columns=['entity_id', 'country']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))


def val_truth():
    """220,730 val S1: s1_idx, entity_id, country, m (#truths), r (#truths in the union-v1 candidates), ncand, is_locked, half (hash)."""
    import polars as pl, zlib
    from score import load_id_lists
    i1 = ids('train', 's1')
    g = load_id_lists(f'{SPL}/val_ground_truth.tsv')
    M = i1.join(pl.DataFrame({'entity_id': list(g), 'm': [len(v) for v in g.values()]}), on='entity_id', how='inner')
    P = pl.read_parquet(f'{V2A}/output/val_predictions.parquet', columns=['s1_idx', 'label'])
    r = P.group_by('s1_idx').agg(pl.col('label').cast(pl.Int32).sum().alias('r'), pl.len().alias('ncand'))
    M = M.join(r, on='s1_idx', how='left').with_columns(pl.col('r').fill_null(0), pl.col('ncand').fill_null(0))
    locked = set(open(f'{V1}/data/locked_val_s1_ids.txt').read().split())
    M = M.with_columns(pl.col('entity_id').is_in(pl.Series(sorted(locked)).implode()).alias('is_locked'),
                       pl.Series('half', [zlib.crc32(('vx_half:' + s).encode()) % 2 for s in M['entity_id'].to_list()]).cast(pl.Int8))
    assert M.height == 220730
    return M


def row_scores(pred, M):
    from pv1.decide import row_scores as rs
    return rs(pred.select('s1_idx', 'label'), M)


def slices(pred, M):
    import polars as pl
    R = row_scores(pred, M)
    f = lambda x: float(x['f'].mean()) if x.height else None
    out = dict(all=f(R), US=f(R.filter(pl.col('country') == 'US')), India=f(R.filter(pl.col('country') == 'India')))
    if 'is_locked' in R.columns: out['locked30k'] = f(R.filter(pl.col('is_locked')))
    if 'half' in R.columns: out['tune_half0'] = f(R.filter(pl.col('half') == 0)); out['report_half1'] = f(R.filter(pl.col('half') == 1))
    out['singleton'] = f(R.filter(pl.col('m') == 0)); out['m1'] = f(R.filter(pl.col('m') == 1))
    out['matches_per_s1'] = float(R['k'].mean()); out['empty_share'] = float((R['k'] == 0).mean())
    out['micro_precision'] = float(R['t'].sum() / max(R['k'].sum(), 1))
    return out, R


def paired(Ra, Rb, n_boot=1000, seed=0):
    import numpy as np
    J = Ra.select('s1_idx', 'f').join(Rb.select('s1_idx', 'f'), on='s1_idx', suffix='_b'); d = (J['f'] - J['f_b']).to_numpy()
    rng = np.random.default_rng(seed); bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return dict(n=len(d), diff=float(d.mean()), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()))


def efs_fixed(Pv, lam, q0, sids, L=12, NS=256, seed=0, chunk=20000):
    """pv1.decide.efs_choose with an EXPLICIT, sorted S1 list `sids` (numpy int array) so that two probability versions of the same S1 set
    get identical Monte-Carlo draws per S1 (S1 without eligible pairs are included as all-zero rows and come out empty).
    Pv: s1_idx, cand_idx, p, elig (+ anything). Returns the selected pairs."""
    import numpy as np, polars as pl
    rng = np.random.default_rng(seed)
    sids = np.sort(np.asarray(sids, dtype=np.int64))
    E = Pv.filter(pl.col('elig')).sort('s1_idx', 'p', descending=[False, True]).group_by('s1_idx', maintain_order=True).head(L)
    E = E.with_columns(pl.int_range(pl.len()).over('s1_idx').alias('j'))
    pos = np.searchsorted(sids, E['s1_idx'].to_numpy())
    assert (sids[np.minimum(pos, len(sids) - 1)] == E['s1_idx'].to_numpy()).all(), 'eligible S1 outside sids'
    Pm = np.zeros((len(sids), L), np.float32); Pm[pos, E['j'].to_numpy()] = E['p'].to_numpy()
    kbest = np.zeros(len(sids), np.int32)
    for s in range(0, len(sids), chunk):
        p = Pm[s:s + chunk]
        Y = rng.random((NS,) + p.shape, dtype=np.float32) < p[None]
        Mc = Y.sum(2)
        miss = np.where(Mc > 0, rng.poisson(lam, Mc.shape), (rng.random(Mc.shape) < q0).astype(np.int64))
        Mt = Mc + miss; T = np.cumsum(Y, 2)
        k = np.arange(1, L + 1)[None, None, :]
        F = np.where(T > 0, 5 * T / (4 * k + Mt[..., None]), 0.0).mean(0)
        F0 = (Mt == 0).mean(0)
        kbest[s:s + chunk] = np.concatenate([F0[:, None], F], 1).argmax(1)
    kdf = pl.DataFrame({'s1_idx': sids.astype(np.int32), 'kb': kbest})
    return E.join(kdf, on='s1_idx').filter(pl.col('j') < pl.col('kb')).drop('j', 'kb')


def read_sub(path, split='test'):
    """submission TSV -> polars (s1_idx, cand_idx) pairs (+ s1 country)."""
    import polars as pl
    d = pl.read_csv(path, separator='\t', quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True)
    col = d.columns[1]
    d = d.with_columns(pl.col(col).fill_null('').str.split(',')).explode(col).filter(pl.col(col) != '')
    i1 = ids(split, 's1'); i2 = ids(split, 's23')
    out = d.rename({d.columns[0]: 'entity_id', col: 'cid'}).join(i1, on='entity_id').join(i2.select(pl.col('entity_id').alias('cid'), 'cand_idx'), on='cid')
    assert out.height == d.height, (out.height, d.height)
    return out.select('s1_idx', 'cand_idx', 'country')


def write_sub(pred, cand, outdir, split='test'):
    """pred / cand: (s1_idx, cand_idx) frames. Writes matching_results.tsv and candidate_pairs.tsv exactly like prod_v2a/06_test.py."""
    import polars as pl
    os.makedirs(outdir, exist_ok=True)
    i1 = ids(split, 's1'); i2 = ids(split, 's23').select('cand_idx', 'entity_id')
    def write(df, path, col):
        g = df.join(i2, on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').unique().sort().str.join(',').alias('ids'))
        g = i1.join(g, on='s1_idx', how='left').with_columns(pl.col('ids').fill_null('')).sort('s1_idx')
        g.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('ids').alias(col)).write_csv(path, separator='\t', quote_style='never')
        return g
    gm = write(pred.select('s1_idx', 'cand_idx'), f'{outdir}/matching_results.tsv', 'matched_entity_ids')
    if cand is not None:
        write(cand.select('s1_idx', 'cand_idx'), f'{outdir}/candidate_pairs.tsv', 'candidate_entity_ids')
    return gm


def test_stats(pred):
    """per-country matches/S1, empty share, multi-assigned records (one-owner violations)."""
    import polars as pl
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
    """label-free France proxies on a test prediction (s1_idx, cand_idx):
    strict anchors (unique street key S1 x record at the key, name-core tsr >= 95; forensics/data/anchors_keyed_France) acceptance,
    op slices on the 400k anchor_ops_France sample (filler_add / amp / truncate), mined decoy pairs (literature/fr_decoy_pairs: decoy vs
    copy_filler) acceptance, France one-owner violations."""
    import polars as pl
    if not _FRP:
        i1 = ids('test', 's1'); i2 = ids('test', 's23')
        def idx(df):
            return df.join(i1.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id').join(i2.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id')
        A = pl.read_parquet(f'{FR}/forensics/data/anchors_keyed_France.parquet', columns=['s1_id', 'cand_id', 'tsr'])
        _FRP['strict'] = idx(A.filter(pl.col('tsr') >= 95)).select('s1_idx', 'cand_idx')
        _FRP['keyed'] = idx(A).select('s1_idx', 'cand_idx', 'tsr')
        O = pl.read_parquet(f'{FR}/forensics/data/anchor_ops_France.parquet', columns=['s1_id', 'cand_id', 'tsr', 'op_filler_add', 'op_amp', 'op_truncate', 'op_word_drop', 'op_country_tag', 'op_acronym'])
        _FRP['ops'] = idx(O.filter(pl.col('tsr') >= 95))
        Dp = pl.read_parquet(f'{FR}/literature/fr_decoy_pairs.parquet', columns=['s1_id', 'cand_id', 'kind'])
        _FRP['decoy'] = idx(Dp)
    sel = pred.select('s1_idx', 'cand_idx').unique().with_columns(pl.lit(True).alias('sel'))
    def acc(df):
        x = df.join(sel, on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('sel').fill_null(False))
        return x
    out = {}
    s = acc(_FRP['strict']); out['strict_anchor_n'] = s.height; out['strict_anchor_accept'] = round(float(s['sel'].mean()), 5)
    o = acc(_FRP['ops']); out['ops_sample_strict_accept'] = round(float(o['sel'].mean()), 5)
    for op in ['op_filler_add', 'op_amp', 'op_truncate', 'op_word_drop', 'op_country_tag', 'op_acronym']:
        q = o.filter(pl.col(op) > 0); out[f'{op}_accept'] = (q.height, round(float(q['sel'].mean()), 4) if q.height else None)
    d = acc(_FRP['decoy'])
    for kd in ['decoy', 'copy_filler']:
        q = d.filter(pl.col('kind') == kd); out[f'{kd}_accept'] = (q.height, round(float(q['sel'].mean()), 5), int(q['sel'].sum()))
    i1 = ids('test', 's1').select('s1_idx', 'country')
    fr = pred.select('s1_idx', 'cand_idx').join(i1, on='s1_idx').filter(pl.col('country') == 'France')
    mult = fr.group_by('cand_idx').agg(pl.len().alias('n'))
    out['france_one_owner_violating_records'] = int((mult['n'] > 1).sum())
    return out


_R10 = {}
def r10c(P, comp, split, margin=0.0, model=None):
    """v2b's final decision: argmax-vs-best-other exclusivity (p >= v1 p2 of the best OTHER S1 of the record + margin) + the saved R10c
    learned prefix set-decoder (prod_v2b/models/abc_rob_cv2_all_p2_R10c_m0.0.txt). P: s1_idx, cand_idx, p (+ label) over the FULL union
    lists of the S1 to decide; comp: v1 dense pairs (s1_idx, cand_idx, q). Deterministic, per-S1 -> exact on any S1 subset."""
    import polars as pl, lightgbm as lgb
    import decide_lib as DL
    from pv2a.decide import add_comp
    if model is None:
        if 'm' not in _R10: _R10['m'] = lgb.Booster(model_file=f'{V2B}/models/abc_rob_cv2_all_p2_R10c_m0.0.txt')
        model = _R10['m']
    k = f'src_{split}'
    if k not in _R10: _R10[k] = ids(split, 's23').select('cand_idx', pl.col('entity_id').str.slice(0, 2).alias('src'))
    if P.height == 0:
        return pl.DataFrame(schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32, **({'label': pl.Int8} if 'label' in P.columns else {})})
    Q = add_comp(P.select('s1_idx', 'cand_idx', 'p'), comp).join(_R10[k], on='cand_idx', how='left')
    U = Q.select(pl.col('s1_idx').alias('s1'), pl.col('cand_idx').alias('qid'), 'src', 'p', pl.when(pl.col('p') >= pl.col('p_other') + margin).then(1).otherwise(2).alias('qr'))
    dec = DL.decide_set(U, model)
    out = pl.DataFrame([(s, q) for s, qs in dec.items() for q in qs], schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32}, orient='row')
    if 'label' in P.columns: out = out.join(P.select('s1_idx', 'cand_idx', 'label'), on=['s1_idx', 'cand_idx'], how='left')
    return out


def decoy_words():
    """label-free per-country decoy vocabulary from the forensics neighbour mining (out/decoys.json added_words = [word, n at house offsets
    k in D, n at k<0, n at k=0]): words with >= 500 records at decoy offsets and a positive/negative offset ratio >= 20."""
    import json
    d = json.load(open(f'{FR}/forensics/out/decoys.json'))
    return {c: sorted(w for w, npos, nneg, n0 in d[c]['added_words'] if npos >= 500 and npos / (nneg + 1) >= 20) for c in ('US', 'India', 'France')}


DOFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]


def decoy_pairs(split):
    """neighbour-decoy pattern pairs (label-free, forensics/data/neighbours_*): ADD = record content = S1 content + exactly one country decoy word,
    same street tokens, house number = S1 house + k, k in D; EQ = identical content name at k in D minus {1, 2}. Returns s1_idx, cand_idx, kind, k, country."""
    import polars as pl
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
