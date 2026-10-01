"""Shared post-pass definitions for v3_postpass (label-free veto pattern sets, identical rules to v2a_experiments V3dALL_V1 and
v2b_postpass2 V3dALL_V1_FRcl_USlegal_no12). Pattern tables are read-only inputs:
  - six-word ADD decoys: forensics neighbours_* (rel == 'superset+1', added word in the country decoy list = vx.common.decoy_words()),
    same explainer street tokens, house offset k (record - S1); the veto uses k in D, the mirror control k in -D.
  - 'country' / 'legal' kinds: v2b_postpass2/data/pat_<split>_<country>.parquet (rules.py; computed over the union-v2 candidates, which
    are identical for v2b and v3 on val, sample and test -- verified in check_inputs.py); 'allnum' = every number part agrees except h1 -> h1+k."""
import polars as pl
from vx.common import W, FR, XP2, ids, decoy_words
import rules
D = list(rules.D); D_NO12 = [k for k in D if k > 2]; K2 = ['s1_idx', 'cand_idx']


def add_pairs(split):
    """six-word/ADD decoy pattern pairs at k in +D and -D (mirror). -> s1_idx, cand_idx, k, added, country"""
    Wd = decoy_words()
    fs = {'train': [('US', 'neighbours_US'), ('India', 'neighbours_India')],
          'test': [('US', 'neighbours_test_US'), ('India', 'neighbours_test_India'), ('France', 'neighbours_France')]}[split]
    i1 = ids(split, 's1'); i2 = ids(split, 's23'); out = []
    offs = D + [-k for k in D]
    for c, f in fs:
        n = pl.read_parquet(f'{FR}/forensics/data/{f}.parquet', columns=['k', 's1_id', 'cand_id', 'rel', 'added'])
        a = n.filter((pl.col('rel') == 'superset+1') & pl.col('k').is_in(offs) & pl.col('added').is_in(Wd[c]))
        a = a.join(i1.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id').join(i2.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id')
        out.append(a.select('s1_idx', 'cand_idx', pl.col('k').cast(pl.Int32), 'added', pl.lit(c).alias('country')))
    return pl.concat(out).unique(K2)


def pat(split, countries):
    return pl.concat([pl.read_parquet(f'{XP2}/data/pat_{split}_{c}.parquet', columns=['s1_idx', 'cand_idx', 'k', 'kind', 'extra', 'allnum'])
                      .with_columns(pl.lit(c).alias('country')) for c in countries])


def side(col='k'):
    return pl.when(pl.col(col) > 0).then(pl.lit('+k')).when(pl.col(col) < 0).then(pl.lit('-k')).otherwise(pl.lit('0')).alias('side')


def rule_sets(ADD, PT):
    """the vetoes as (veto pair set at +k, mirror pair set at -k) per rule name."""
    R = {}
    R['sixword'] = (ADD.filter(pl.col('k').is_in(D)), ADD.filter(pl.col('k').is_in([-k for k in D])))
    for kind, nm in (('country', 'FRcountry'), ('legal', 'FRlegal')):
        b = PT.filter((pl.col('kind') == kind) & (pl.col('country') == 'France') & pl.col('allnum'))
        R[nm] = (b.filter(pl.col('k').is_in(D)), b.filter(pl.col('k').is_in([-k for k in D])))
    b = PT.filter((pl.col('kind') == 'legal') & (pl.col('country') == 'US') & pl.col('allnum'))
    R['USlegal_no12'] = (b.filter(pl.col('k').is_in(D_NO12)), b.filter(pl.col('k').is_in([-k for k in D_NO12])))
    return R
