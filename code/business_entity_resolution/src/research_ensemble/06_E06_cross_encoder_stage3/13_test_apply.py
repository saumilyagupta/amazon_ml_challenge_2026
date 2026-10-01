# E06 step 13 (CPU, light: 4 threads; LightGBM predict only): apply the E06 stage 3 to v2b's TEST band 0.02 < p2 < 0.99 and write the hand-off file
# for the builder: build/p3_test_E06.parquet (s1_idx, cand_idx, p3; band rows only, p3 = p2 outside the band implied) + build/p3_test_E06_SUCCESS + build/p3_test_E06.json.
# Inputs: B1's test band tables (build/interim_v2b_E13/data/band_base.parquet, band_s1/part-*.parquet, band_g/chunk-*.parquet), CE logits = mean of the two
# fold CEs (scores/ce_xw_testrest_f{0,1} for US/India, scores/ce_xw_fr_f{0,1} for France), stage-3 fold models models/s3_xw_<variant>_fold{0,1}.txt.
# usage: python3 13_test_apply.py --variant p3b|p3bg
import os, sys, json, time, glob
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2b')
from pv2b.common import envcap; envcap(4)
import numpy as np, polars as pl, lightgbm as lgb
from scipy.special import expit, logit
from pv2b.newfeats import SIB_FEATS
from pv1.model import S2FEATS
V = sys.argv[sys.argv.index('--variant') + 1]
OUTN = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'p3_test_E06'
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'; BD = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/build'; B1 = f'{BD}/interim_v2b_E13/data'
T0 = time.time(); log = lambda *a: print(f'[+{time.time()-T0:.0f}s]', *a, flush=True)
K = ['s1_idx', 'cand_idx']
FLAGS = ['addr_empty2', 'name_freq_s1_1', 'n11_junk_name', 'f_url2', 'n10_alias', 'script2', 'x1_name_explained', 'x1_both_explained']
AGG = ['p2_sum', 'p2_n50', 'p2_n90', 'p2_max', 'p2_rank']
G = ['g_n99', 'g_name_n_max', 'g_name_n_same_src', 'g_name_n_exact_n', 'g_name_raw_max', 'g_name_raw_same_src', 'g_name_raw_exact_n', 'g_name_core_max', 'g_name_core_same_src',
     'g_name_core_exact_n', 'g_name_ns_max', 'g_name_ns_same_src', 'g_name_ns_exact_n', 'g_name_weighted', 'g_name_near_n', 'g_s1freq_name_raw', 'g_s1freq_name_n', 'g_s1freq_name_ns']
FEAT = ['lp2', 'ce_logit'] + list(S2FEATS) + SIB_FEATS + FLAGS + ['in_dft'] + (G + AGG if V == 'p3bg' else [])
inb = (pl.col('p2') > 0.02) & (pl.col('p2') < 0.99)
B = pl.read_parquet(f'{B1}/band_base.parquet').filter(inb); n = B.height
F = pl.concat([pl.read_parquet(p, columns=[*K, *FLAGS]) for p in sorted(glob.glob(f'{B1}/band_s1/part-*.parquet'))]).join(B.select(K), on=K, how='semi')
B = B.join(F, on=K, how='left')
if V == 'p3bg':
    Gt = pl.concat([pl.read_parquet(p, columns=[*K, *G]) for p in sorted(glob.glob(f'{B1}/band_g/chunk-*.parquet'))]).join(B.select(K), on=K, how='semi')
    B = B.join(Gt, on=K, how='left'); assert B['g_n99'].null_count() == 0, B['g_n99'].null_count()
rd = lambda nm: pl.read_parquet(f'{E}/scores/{nm}.parquet')
ce = pl.concat([rd('ce_xw_testrest_f0').join(rd('ce_xw_testrest_f1'), on=K, suffix='_1'), rd('ce_xw_fr_f0').join(rd('ce_xw_fr_f1'), on=K, suffix='_1')])
ce = ce.select(*K, ((pl.col('ce_logit') + pl.col('ce_logit_1')) / 2).alias('ce_logit'))
B = B.join(ce, on=K, how='left')
assert B.height == n and B['ce_logit'].null_count() == 0 and B['addr_empty2'].null_count() == 0, (B.height, n, B['ce_logit'].null_count())
B = B.with_columns(pl.Series('lp2', logit(np.clip(B['p2'].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6))), pl.col('in_dft').cast(pl.Float32))
X = np.nan_to_num(B.select(FEAT).to_numpy().astype(np.float32), nan=-1.0, posinf=1e6, neginf=-1e6)
models = [f'{E}/models/s3_xw_{V}_fold{f}.txt' for f in (0, 1)]
raw = 0.5 * sum(lgb.Booster(model_file=m).predict(X, raw_score=True) for m in models)
B = B.with_columns(pl.Series('p3', expit(B['lp2'].to_numpy() + raw).astype(np.float32)))
out = B.select('s1_idx', 'cand_idx', 'p3')
tmp = f'{BD}/{OUTN}.tmp.parquet'; out.write_parquet(tmp); os.replace(tmp, f'{BD}/{OUTN}.parquet')
st = B.group_by('country').agg(pl.len().alias('band_pairs'), pl.col('p2').mean().alias('mean_p2'), pl.col('p3').mean().alias('mean_p3'),
                               (pl.col('p3') - pl.col('p2')).abs().mean().alias('mean_abs_shift'), ((pl.col('p2') < 0.5) & (pl.col('p3') >= 0.5)).sum().alias('cross_up_0.5'),
                               ((pl.col('p2') >= 0.5) & (pl.col('p3') < 0.5)).sum().alias('cross_down_0.5'), pl.col('ce_logit').mean().alias('mean_ce_logit'),
                               (pl.col('addr_empty2') > 0).mean().alias('empty_addr_share')).sort('country')
tim = {nm: json.load(open(f'{E}/scores/{nm}_timing.json')) for nm in ['ce_xw_fr', 'ce_xw_testrest']}
J = dict(variant=V, band='0.02 < p2 < 0.99 of v2b test p2 (B1 band_base)', rows=out.height, p3_eq_p2_outside_band=True,
         ce_models=[f'{E}/models/ce_xw_f{f}/ep1' for f in (0, 1)], ce_input="'<name> | <addr>'.lower() of records_test, pair <s> A </s></s> B </s>, max_len 128, logit = mean of the two fold CEs",
         stage3_models=models, stage3_features=FEAT, stage3_formula='p3 = expit(logit(clip(p2,1e-6,1-1e-6)) + mean(raw_fold0, raw_fold1)), NaN -> -1',
         per_country=st.to_dicts(), gpu_scoring=tim, apply_seconds=round(time.time() - T0),
         val=json.load(open(f'/workspace/saumilya/amazon-ml/work/matching/prod_v2c/results/eval_E06_ce_band_xw_{V}.json'))['val']['paired_vs_v2b'] if os.path.exists(f'/workspace/saumilya/amazon-ml/work/matching/prod_v2c/results/eval_E06_ce_band_xw_{V}.json') else None)
json.dump(J, open(f'{BD}/{OUTN}.json', 'w'), indent=1, default=str)
open(f'{BD}/{OUTN}_SUCCESS', 'w').write(f'{V} {out.height} rows {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}\n')
log('wrote', out.height, 'rows; per country', st.to_dicts())
