#!/opt/conda/bin/python3
"""Label-free pair classes on TEST (and the same classes on VAL with real labels, for calibration).

Decoy families (house offset k = record house - S1 house on the same street tokens; the generator only emits decoys at k in +D):
  ADDW  record name = S1 content + one word from the country's learned decoy list (forensics neighbours, rel superset+1)
  ADDO  record name = S1 content + one other word (superset+1, word not in the decoy list)
  SUP2  S1 content + two words (superset+2)
  EQ3   identical content name, |k| in D minus {1,2}
  EQ12  identical content name, |k| in {1,2} (true +1/+2 number typos exist, so +/-k is NOT symmetric here)
  CTAG  legal-free name = S1 + a country tag, all other number parts equal (v2b_postpass2 pattern tables)
  LEG3 / LEG12  identical legal-free name, legal form switched/added, all other number parts equal, |k| in D3 / {1,2}
  side '+' = k in +D (decoys + true copies), '-' = k in -D (mirror: true copies only), '0' = k == 0 (same number)
Pseudo-label classes (research/france/pseudolabels): P1/P1s/P2 anchors, K0/KN keyed, N1/N1b/N2 structural negatives, SWAP_* one-word swaps.
France strict anchors (forensics anchors_keyed_France, tsr >= 95) with op flags from the 400k anchor_ops sample.
Outputs: data/pc_{test,val}_fam.parquet, data/pc_{test,val}_pl.parquet, data/pc_test_frstrict.parquet."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ie_common import *
envcap(8)
import polars as pl
log = logger('03_pairclass')
FO = f'{FR}/forensics'


def decoy_words():
    d = json.load(open(f'{FO}/out/decoys.json'))
    return {c: sorted(w for w, npos, nneg, n0 in d[c]['added_words'] if npos >= 500 and npos / (nneg + 1) >= 20) for c in ('US', 'India', 'France')}


DW = decoy_words(); log('decoy words', DW)
OFF = D + [-k for k in D]


def side_expr():
    return pl.when(pl.col('k') > 0).then(pl.lit('+')).when(pl.col('k') < 0).then(pl.lit('-')).otherwise(pl.lit('0')).alias('side')


def neighbour_fams(split, i1, i2, s1_keep=None):
    fs = {'test': [('US', 'neighbours_test_US'), ('India', 'neighbours_test_India'), ('France', 'neighbours_France')],
          'val': [('US', 'neighbours_US'), ('India', 'neighbours_India')]}[split]
    out = []
    for c, f in fs:
        cols = ['k', 's1_id', 'cand_id', 'rel', 'added'] + (['lab'] if split == 'val' else [])
        n = pl.read_parquet(f'{FO}/data/{f}.parquet', columns=cols).filter(pl.col('k').is_in(OFF + [0]))
        if s1_keep is not None:
            n = n.filter(pl.col('s1_id').is_in(s1_keep))
        n = n.with_columns(
            pl.when((pl.col('rel') == 'superset+1') & pl.col('added').is_in(DW[c])).then(pl.lit('ADDW'))
              .when(pl.col('rel') == 'superset+1').then(pl.lit('ADDO'))
              .when(pl.col('rel') == 'superset+2').then(pl.lit('SUP2'))
              .when((pl.col('rel') == 'equal') & pl.col('k').abs().is_in([1, 2])).then(pl.lit('EQ12'))
              .when(pl.col('rel') == 'equal').then(pl.lit('EQ3')).otherwise(None).alias('fam'))
        n = n.filter(pl.col('fam').is_not_null())
        n = n.join(i1.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id').join(i2.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id')
        sel = ['s1_idx', 'cand_idx', pl.lit(c).alias('country'), 'fam', pl.col('k').cast(pl.Int16), 'added'] + ([(pl.col('lab') == 'true').cast(pl.Int8).alias('label')] if split == 'val' else [])
        out.append(n.select(sel))
        log(split, c, f, n.height)
    return pl.concat(out)


def pattern_fams(split, countries, s1_keep_idx=None):
    out = []
    for c in countries:
        f = f'{M}/v2b_postpass2/data/pat_{"test" if split == "test" else "train"}_{c}.parquet'
        cols = ['s1_idx', 'cand_idx', 'k', 'kind', 'allnum'] + (['label'] if split == 'val' else [])
        p = pl.read_parquet(f, columns=cols).filter(pl.col('allnum') & pl.col('k').is_in(OFF + [0]))
        if s1_keep_idx is not None:
            p = p.join(s1_keep_idx, on='s1_idx', how='semi')
        p = p.with_columns(pl.when(pl.col('kind') == 'country').then(pl.lit('CTAG'))
                           .when(pl.col('k').abs().is_in([1, 2])).then(pl.lit('LEG12')).otherwise(pl.lit('LEG3')).alias('fam'))
        sel = ['s1_idx', 'cand_idx', pl.lit(c).alias('country'), 'fam', pl.col('k').cast(pl.Int16), pl.lit(None, dtype=pl.String).alias('added')]
        if split == 'val': sel.append(pl.col('label').cast(pl.Int8))
        out.append(p.select(sel)); log(split, 'pattern', c, p.height)
    return pl.concat(out)


# ---------------- TEST
i1t, i2t = ids('test', 's1'), ids('test', 's23')
fam_t = pl.concat([neighbour_fams('test', i1t, i2t), pattern_fams('test', ['US', 'India', 'France'])]).with_columns(side_expr()).unique(['s1_idx', 'cand_idx', 'fam'])
fam_t.write_parquet(f'{DATA}/pc_test_fam.parquet')
log('test fam table', fam_t.height, fam_t.group_by('country', 'fam', 'side').len().sort('country', 'fam', 'side').to_dicts())
PLt = pl.read_parquet(f'{FR}/pseudolabels/out/pseudolabels_test.parquet', columns=['s1_idx', 'cand_idx', 'country', 'cls', 'p1s'])
SW = pl.read_parquet(f'{FR}/pseudolabels/out/swap_pairs_test.parquet', columns=['s1_idx', 'cand_idx', 'swap'])
PLt = PLt.join(SW, on=K2, how='left').with_columns(pl.col('cls').cast(pl.Categorical), pl.col('country').cast(pl.Categorical))
PLt.write_parquet(f'{DATA}/pc_test_pl.parquet'); log('test pseudo-label table', PLt.height)
del PLt
A = pl.read_parquet(f'{FO}/data/anchors_keyed_France.parquet', columns=['s1_id', 'cand_id', 'tsr']).filter(pl.col('tsr') >= 95)
O = pl.read_parquet(f'{FO}/data/anchor_ops_France.parquet', columns=['s1_id', 'cand_id', 'op_filler_add', 'op_amp', 'op_truncate', 'op_word_drop', 'op_country_tag', 'op_acronym'])
A = A.join(O, on=['s1_id', 'cand_id'], how='left').with_columns(pl.col('op_filler_add').is_not_null().alias('in_ops_sample'))
A = A.join(i1t.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id').join(i2t.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id').drop('s1_id', 'cand_id')
A.write_parquet(f'{DATA}/pc_test_frstrict.parquet'); log('France strict anchors', A.height, 'in ops sample', int(A['in_ops_sample'].sum()))

# ---------------- VAL (labels)
val_ids = [l.strip() for l in open(f'{SPL}/val_s1_ids.txt') if l.strip()]
i1v, i2v = ids('train', 's1'), ids('train', 's23')
keep_idx = i1v.filter(pl.col('entity_id').is_in(val_ids)).select('s1_idx')
fam_v = pl.concat([neighbour_fams('val', i1v, i2v, s1_keep=val_ids), pattern_fams('val', ['US', 'India'], s1_keep_idx=keep_idx)]).with_columns(side_expr()).unique(['s1_idx', 'cand_idx', 'fam'])
# neighbours 'lab' vs official val GT: re-derive labels from the GT to be safe
sys.path.insert(0, f'{W}/common')
from score import load_id_lists
g = load_id_lists(f'{SPL}/val_ground_truth.tsv')
GT = pl.DataFrame({'s1': list(g), 'c': [sorted(v) for v in g.values()]}).explode('c').drop_nulls() \
       .join(i1v.select(pl.col('entity_id').alias('s1'), 's1_idx'), on='s1').join(i2v.select(pl.col('entity_id').alias('c'), 'cand_idx'), on='c').select(K2).with_columns(pl.lit(1, dtype=pl.Int8).alias('gt'))
GT.write_parquet(f'{DATA}/val_gt_pairs.parquet')
fam_v = fam_v.join(GT, on=K2, how='left').with_columns(pl.col('gt').fill_null(0))
agree = fam_v.filter(pl.col('label').is_not_null()).select((pl.col('label') == pl.col('gt')).mean()).item()
log('val fam label vs GT agreement', agree)
fam_v = fam_v.drop('label').rename({'gt': 'label'})
fam_v.write_parquet(f'{DATA}/pc_val_fam.parquet')
log('val fam table', fam_v.height, fam_v.group_by('country', 'fam', 'side').agg(pl.len(), pl.col('label').mean()).sort('country', 'fam', 'side').to_dicts())
PLv = pl.read_parquet(f'{FR}/pseudolabels/out/pseudolabels_val.parquet', columns=['s1_idx', 'cand_idx', 'country', 'cls', 'p1s', 'label'])
PLv.with_columns(pl.col('cls').cast(pl.Categorical), pl.col('country').cast(pl.Categorical)).write_parquet(f'{DATA}/pc_val_pl.parquet'); log('val pseudo-label table', PLv.height)
log('DONE')
