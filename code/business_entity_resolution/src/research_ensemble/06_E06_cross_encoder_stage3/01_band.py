# E06 step 1: band tables (CPU, light: 4 threads).
#  - train file (prod_v2b/output/pairs/train, OOF p2 on the sample, fold-avg p2 on val): WIDE band 0.005 < p2 < 0.995 for ALL rows
#    (sample = CE training band; val = extra scoring rows for band-width variants); flag bs = 0.02 < p2 < 0.99 (the stacking band).
#  - density universe (density_val/universe/preds_v2b_dens.parquet + feats_dens slice flags): same wide band.
# Output: data/band_train.parquet, data/band_dens.parquet, logs/01_band.json
import os, sys, json, glob, time
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2b')
from pv2b.common import envcap; envcap(4)
import numpy as np, polars as pl
from pv2b.newfeats import SIB_FEATS
from pv1.model import S2FEATS
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'
PT = '/workspace/saumilya/amazon-ml/work/matching/prod_v2b/output/pairs/train'
DU = '/workspace/saumilya/amazon-ml/work/matching/density_val/universe'
LO, HI, SLO, SHI = 0.005, 0.995, 0.02, 0.99
FLAGS = ['addr_empty2', 'name_freq_s1_1', 'n11_junk_name', 'f_url2', 'n10_alias', 'script2', 'x1_name_explained', 'x1_both_explained']
META = ['s1_idx', 'cand_idx', 's1_id', 'cand_id', 'label', 'grp', 'fold', 'es', 'country', 'in_dft']
COLS = META + list(dict.fromkeys(['p1'] + list(S2FEATS) + SIB_FEATS + FLAGS + ['p2']))
T0 = time.time(); log = lambda *a: print(f'[+{time.time()-T0:.0f}s]', *a, flush=True)
parts = sorted(glob.glob(f'{PT}/part-*.parquet')); log('parts', len(parts))
out, stats = [], []
for p in parts:
    d = pl.read_parquet(p, columns=COLS)
    stats.append(d.group_by('grp', 'fold', 'es').agg(pl.len().alias('n'), ((pl.col('p2') > LO) & (pl.col('p2') < HI)).sum().alias('n_wide'),
                 ((pl.col('p2') > SLO) & (pl.col('p2') < SHI)).sum().alias('n_band')))
    out.append(d.filter((pl.col('p2') > LO) & (pl.col('p2') < HI)))
    log(os.path.basename(p), d.height, out[-1].height)
B = pl.concat(out).with_columns(((pl.col('p2') > SLO) & (pl.col('p2') < SHI)).alias('bs'))
S = pl.concat(stats).group_by('grp', 'fold', 'es').agg(pl.col('n', 'n_wide', 'n_band').sum()).sort('grp', 'fold', 'es')
B.write_parquet(f'{E}/data/band_train.parquet'); log('band_train', B.height)
R = {'counts_all_rows': S.to_dicts()}
def summ(df):
    return df.group_by('grp', 'fold', 'es').agg(pl.len().alias('n'), pl.col('label').cast(pl.Float64).mean().alias('pos_rate')).sort('grp', 'fold', 'es').to_dicts()
R['wide_band'] = summ(B); R['scoring_band'] = summ(B.filter(pl.col('bs')))
R['wide_band_tot'] = {g: B.filter(pl.col('grp') == g).height for g in ['sample', 'val', 'locked']}
R['scoring_band_tot'] = dict(sample=B.filter(pl.col('bs') & (pl.col('grp') == 'sample')).height, val=B.filter(pl.col('bs') & (pl.col('grp') != 'sample')).height)
log(json.dumps(R['wide_band_tot']), json.dumps(R['scoring_band_tot']))
# density band
D = pl.read_parquet(f'{DU}/preds_v2b_dens.parquet').filter((pl.col('p2') > LO) & (pl.col('p2') < HI))
fl = []
for p in sorted(glob.glob(f'{DU}/feats_dens/part_*.parquet')):
    fl.append(pl.read_parquet(p, columns=['s1_idx', 'cand_idx'] + FLAGS).join(D.select('s1_idx', 'cand_idx'), on=['s1_idx', 'cand_idx'], how='semi'))
F = pl.concat(fl).unique(['s1_idx', 'cand_idx'])
n0 = D.height; D = D.join(F, on=['s1_idx', 'cand_idx'], how='left'); assert D.height == n0
assert D['addr_empty2'].null_count() == 0, D['addr_empty2'].null_count()
D = D.with_columns(((pl.col('p2') > SLO) & (pl.col('p2') < SHI)).alias('bs'))
D.write_parquet(f'{E}/data/band_dens.parquet')
R['dens_band'] = dict(wide=D.height, scoring=int(D['bs'].sum()), pos_rate_scoring=float(D.filter(pl.col('bs'))['label'].cast(pl.Float64).mean()))
log('dens', R['dens_band'])
json.dump(R, open(f'{E}/logs/01_band.json', 'w'), indent=1, default=str); log('done')
