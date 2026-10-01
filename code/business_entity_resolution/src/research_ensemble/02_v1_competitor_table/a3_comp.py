"""A3 step 0: v1 competition p_other + src for all 83,761,275 test pairs, computed exactly as prod_v2c/build/interim_v2b_E13/src/03_decide.py
(add_comp(P[s1,cand,p=v2b p2], v1_competitors('test','p2'))). Imports the B1 chain modules read-only; B1_HOME re-targeted to vr2_stress_submission/build."""
import os, sys, time
os.environ['B1_HOME'] = '/workspace/saumilya/amazon-ml/work/matching/vr2_stress_submission/build'; os.environ['B1_THREADS'] = '8'
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/build/interim_v2b_E13/src')
import b1common as C
import polars as pl, gc
import pv2b.common  # noqa
from pv2a.stage2 import v1_competitors
from pv2a.decide import add_comp
K2 = ['s1_idx', 'cand_idx']; t0 = time.time()
OUT = '/workspace/saumilya/amazon-ml/work/matching/vr2_stress_submission/data/test_comp.parquet'
i2 = pl.read_parquet(f'{C.EB}/ids/test_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
P = pl.read_parquet(C.PREDS, columns=['s1_idx', 'cand_idx', 'country', 'in_dft', 'p2']); assert P.height == C.N_PAIRS
comp = v1_competitors('test', 'p2'); C.log('v1 competitors', comp.height)
Pc = add_comp(P.select('s1_idx', 'cand_idx', pl.col('p2').alias('p')), comp).select(*K2, 'p_other'); del comp; gc.collect()
P = P.join(Pc, on=K2, how='left').join(i2.select('cand_idx', pl.col('entity_id').str.slice(0, 2).alias('src')), on='cand_idx', how='left'); del Pc
assert P['p_other'].null_count() == 0 and P['src'].null_count() == 0
P.write_parquet(OUT); C.log('wrote', OUT, P.height, f'{time.time()-t0:.0f}s peak RSS {C.peak_rss_gb():.1f} GB')
