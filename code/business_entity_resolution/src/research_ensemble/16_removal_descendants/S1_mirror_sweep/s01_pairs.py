"""S1_mirror_sweep step 1: label-free +-k offset pair families for ALL name relations (US/India val+test, France test).

Pair = (S1, S2/S3 record) on the same street tokens (forensics views, same key as research/france/forensics/08b) with
house = S1 house + k, k in [-25, 25] \\ {0}, sharing >= 1 content token (content = legal/tag-free folded name tokens),
plus an acronym join (record content = initials of the S1 content).
Relation (a partition, one family per pair):
  CTAG   content equal, tags (country tag) differ
  LEGADD / LEGSW / LEGDROP   content equal, legal form added / switched / dropped
  EQ     content, legal and tags equal (identical name)
  ADDW / ADDF / ADDO   S1 + one word: decoy-list word / true-copy filler word / other word
  ADD2   S1 + >= 2 words
  SUB    record content a strict subset (word dropped)
  SWDW / SWFIL / SWOTH / TYPO   one word replaced (same #words): by a decoy word / by a filler word / other word / by a typo (lev <= 2)
  ACR    record = acronym of the S1 content
  NEAR   other relation with token-sort ratio >= 85; rest dropped
Offset bucket: D12 = |k| in {1,2}; D3 = |k| in D\\{1,2}; ND = |k| not in D (control: generator emits no decoys there).
Outputs out/pairs_{val,test}.parquet with #8 acceptance (+ val labels, + O3 US_LEG removal flag on test).
"""
import os, sys, json, time
os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import polars as pl
import numpy as np
from rapidfuzz.process import cpdist
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

W = '/workspace/saumilya/amazon-ml/work'
FO = f'{W}/research/france/forensics'
OUT = f'{W}/winning_strategy_20260926/agents/S1_mirror_sweep/out'
IT = f'{W}/matching/iterate_20260926'
EB = f'{W}/blocking/embedding/full'
D = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
OFF = [k for k in range(-25, 26) if k != 0]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


dj = json.load(open(f'{FO}/out/decoys.json'))
DW = {c: set(w for w, npos, nneg, n0 in dj[c]['added_words'] if npos >= 500 and npos / (nneg + 1) >= 20) for c in ('US', 'India', 'France')}
FILLER = {'US': {'center', 'centre', 'services', 'service', 'partners', 'enterprises', 'trust', 'council', 'society', 'federation', 'solutions', 'associates', 'company'},
          'India': {'center', 'centre', 'services', 'service', 'partners', 'shree', 'sri', 'shri', 'trust', 'society', 'enterprises', 'solutions', 'associates'},
          'France': {'fils', 'services', 'developpement', 'associes', 'groupe', 'compagnie', 'cie', 'et'}}
COLS = ['entity_id', 'country', 'src', 'house', 'street_toks', 'content', 'legal', 'tags']


def build(V, c, s1_keep=None):
    V = V.filter(pl.col('house').is_not_null() & (pl.col('street_toks') != '') & (pl.col('house') < 10**9) & (pl.col('content') != '')) \
         .with_columns(pl.col('street_toks').hash().alias('sh'))
    s1 = V.filter(pl.col('src') == 1)
    if s1_keep is not None:
        s1 = s1.filter(pl.col('entity_id').is_in(s1_keep))
    s1 = s1.select(pl.col('entity_id').alias('s1_id'), 'sh', pl.col('house').alias('h1'), pl.col('content').alias('ca'),
                   pl.col('legal').alias('la'), pl.col('tags').alias('ta')).with_row_index('i1')
    rc = V.filter(pl.col('src') != 1).select(pl.col('entity_id').alias('cand_id'), 'sh', pl.col('house').alias('h2'), pl.col('content').alias('cb'),
                                           pl.col('legal').alias('lb'), pl.col('tags').alias('tb')).with_row_index('i2')
    offs = pl.lit(OFF, dtype=pl.List(pl.Int16))
    ex = s1.select('i1', 'sh', 'h1', pl.col('ca').str.split(' ').list.unique().list.head(4).alias('t')).explode('t') \
           .with_columns(pl.col('t').hash().alias('th')).drop('t').with_columns(offs.alias('k')).explode('k') \
           .with_columns((pl.col('h1') + pl.col('k')).alias('h2')).drop('h1')
    rx = rc.select('i2', 'sh', 'h2', pl.col('cb').str.split(' ').alias('t')).explode('t').with_columns(pl.col('t').hash().alias('th')).drop('t').unique()
    P = ex.join(rx, on=['sh', 'h2', 'th']).select('i1', 'i2', 'k').unique(['i1', 'i2'])
    del ex, rx
    # acronym join: record content (single token) == initials of S1 content (>= 2 tokens)
    ea = s1.select('i1', 'sh', 'h1', pl.col('ca').str.split(' ').alias('t')).filter(pl.col('t').list.len() >= 2) \
           .with_columns(pl.col('t').list.eval(pl.element().str.slice(0, 1)).list.join('').alias('acr')).drop('t') \
           .with_columns(offs.alias('k')).explode('k').with_columns((pl.col('h1') + pl.col('k')).alias('h2')).drop('h1')
    ra = rc.filter(~pl.col('cb').str.contains(' ')).select('i2', 'sh', 'h2', pl.col('cb').alias('acr'))
    PA = ea.join(ra, on=['sh', 'h2', 'acr']).select('i1', 'i2', 'k').unique(['i1', 'i2']).with_columns(pl.lit(True).alias('is_acr'))
    del ea, ra
    P = pl.concat([P.with_columns(pl.lit(False).alias('is_acr')), PA]).unique(['i1', 'i2'], keep='first')
    log(c, 'raw pairs', P.height, 'acr', PA.height)
    P = P.join(s1.select('i1', 's1_id', 'ca', 'la', 'ta'), on='i1').join(rc.select('i2', 'cand_id', 'cb', 'lb', 'tb'), on='i2').drop('i1', 'i2')
    A = pl.col('ca').str.split(' ').list.unique(); B = pl.col('cb').str.split(' ').list.unique()
    P = P.with_columns(A.list.len().alias('na'), B.list.len().alias('nb'), A.list.set_intersection(B).list.len().alias('ni'),
                       B.list.set_difference(A).list.first().alias('wnew'), A.list.set_difference(B).list.first().alias('wold'))
    na, nb, ni = pl.col('na'), pl.col('nb'), pl.col('ni')
    rel = (pl.when(pl.col('is_acr')).then(pl.lit('ACR'))
           .when((ni == na) & (na == nb)).then(pl.lit('EQALL'))
           .when((ni == na) & (nb == na + 1)).then(pl.lit('ADD1'))
           .when((ni == na) & (nb >= na + 2)).then(pl.lit('ADD2'))
           .when((ni == nb) & (nb < na)).then(pl.lit('SUB'))
           .when((na == nb) & (ni == na - 1) & (na >= 2)).then(pl.lit('SW1'))
           .otherwise(pl.lit('OTH')))
    P = P.with_columns(rel.alias('rel'))
    # typo test for SW1, near test for OTH (rapidfuzz, vectorised)
    sw = P['rel'] == 'SW1'
    lev = np.full(P.height, 99, dtype=np.int32)
    idx = np.flatnonzero(sw.to_numpy())
    if len(idx):
        lev[idx] = cpdist(P['wold'].gather(idx).fill_null('').to_list(), P['wnew'].gather(idx).fill_null('').to_list(), scorer=Levenshtein.distance, workers=4)
    near = np.zeros(P.height, dtype=np.float32)
    idx = np.flatnonzero((P['rel'] == 'OTH').to_numpy())
    if len(idx):
        near[idx] = cpdist(P['ca'].gather(idx).to_list(), P['cb'].gather(idx).to_list(), scorer=fuzz.token_sort_ratio, workers=4)
    P = P.with_columns(pl.Series('lev', lev), pl.Series('near', near))
    dw, fil = list(DW[c]), list(FILLER[c])
    fam = (pl.when(pl.col('rel') == 'EQALL').then(
               pl.when(pl.col('ta') != pl.col('tb')).then(pl.lit('CTAG'))
                 .when((pl.col('la') == '') & (pl.col('lb') != '')).then(pl.lit('LEGADD'))
                 .when((pl.col('la') != '') & (pl.col('lb') == '')).then(pl.lit('LEGDROP'))
                 .when(pl.col('la') != pl.col('lb')).then(pl.lit('LEGSW')).otherwise(pl.lit('EQ')))
           .when(pl.col('rel') == 'ADD1').then(
               pl.when(pl.col('wnew').is_in(dw)).then(pl.lit('ADDW')).when(pl.col('wnew').is_in(fil)).then(pl.lit('ADDF')).otherwise(pl.lit('ADDO')))
           .when(pl.col('rel') == 'SW1').then(
               pl.when(pl.col('lev') <= 2).then(pl.lit('TYPO')).when(pl.col('wnew').is_in(dw)).then(pl.lit('SWDW'))
                 .when(pl.col('wnew').is_in(fil)).then(pl.lit('SWFIL')).otherwise(pl.lit('SWOTH')))
           .when(pl.col('rel') == 'OTH').then(pl.when(pl.col('near') >= 85).then(pl.lit('NEAR')).otherwise(None))
           .otherwise(pl.col('rel')))
    ak = pl.col('k').abs()
    P = P.with_columns(fam.alias('fam')).filter(pl.col('fam').is_not_null()).with_columns(
        pl.when(ak.is_in([1, 2])).then(pl.lit('D12')).when(ak.is_in(D)).then(pl.lit('D3')).otherwise(pl.lit('ND')).alias('bucket'),
        pl.when(pl.col('k') > 0).then(pl.lit('+')).otherwise(pl.lit('-')).alias('side'),
        pl.lit(c).alias('country'))
    log(c, 'classified', P.height, dict(P['fam'].value_counts().sort('fam').rows()))
    return P.select('s1_id', 'cand_id', 'country', 'k', 'bucket', 'side', 'fam', 'wnew', 'wold')


def ids(split, kind):
    f = f'{EB}/ids/{split}_{"s1" if kind == "s1" else "s23"}.parquet'
    col = 's1_idx' if kind == 's1' else 'cand_idx'
    return pl.read_parquet(f, columns=['entity_id']).with_row_index(col).with_columns(pl.col(col).cast(pl.Int32))


split = sys.argv[1]
if split == 'test':
    V = pl.read_parquet(f'{FO}/data/views_test_usin.parquet', columns=COLS)
    parts = [build(V.filter(pl.col('country') == c), c) for c in ('US', 'India')]
    del V
    VF = pl.read_parquet(f'{FO}/data/views_test_fr.parquet', columns=COLS)
    parts.append(build(VF, 'France')); del VF
    P = pl.concat(parts)
    i1, i2 = ids('test', 's1'), ids('test', 's23')
    P = P.join(i1.rename({'entity_id': 's1_id'}), on='s1_id').join(i2.rename({'entity_id': 'cand_id'}), on='cand_id')
    sel = pl.read_parquet(f'{IT}/submissions/nc_specialist_legal_fr/selected.parquet').with_columns(pl.lit(1, dtype=pl.Int8).alias('acc'))
    o3 = pl.read_parquet(f'{W}/winning_strategy_20260926/agents/O3_metric_transfer/out/p3_remove_US_LEG.parquet', columns=['s1_idx', 'cand_idx']) \
           .with_columns(pl.lit(1, dtype=pl.Int8).alias('o3rm'))
    P = P.join(sel, on=['s1_idx', 'cand_idx'], how='left').join(o3, on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('acc').fill_null(0), pl.col('o3rm').fill_null(0))
    log('test o3 removed covered', int(P['o3rm'].sum()), 'of', o3.height)
else:
    val_ids = [l.strip() for l in open(f'{W}/splits/val_s1_ids.txt') if l.strip()]
    V = pl.read_parquet(f'{FO}/data/views_train.parquet', columns=COLS)
    P = pl.concat([build(V.filter(pl.col('country') == c), c, s1_keep=val_ids) for c in ('US', 'India')])
    del V
    i1, i2 = ids('train', 's1'), ids('train', 's23')
    P = P.join(i1.rename({'entity_id': 's1_id'}), on='s1_id').join(i2.rename({'entity_id': 'cand_id'}), on='cand_id')
    sel = pl.read_parquet(f'{IT}/empty_address/val_nc_robust_selected.parquet').with_columns(pl.lit(1, dtype=pl.Int8).alias('acc'))
    gt = pl.read_parquet(f'{W}/internal_eval/data/val_gt_pairs.parquet').rename({'gt': 'label'})
    P = P.join(sel, on=['s1_idx', 'cand_idx'], how='left').join(gt, on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('acc').fill_null(0), pl.col('label').fill_null(0))
P.write_parquet(f'{OUT}/pairs_{split}.parquet')
log(split, 'wrote', P.height, P.group_by('country').agg(pl.len(), pl.col('acc').sum()).rows())
