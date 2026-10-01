import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
import polars as pl, json, time
from pathlib import Path
W=Path('/workspace/saumilya/amazon-ml/work'); O=W/'matching/iterate_20260926/empty_address'; V=W/'matching/variance_research_round2_20260926/data'; C=W/'internal_eval/agents/B_archetypes/cache'; K=['s1_idx','cand_idx']
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def views(split):
 s=pl.read_parquet(C/f'views_{split}_s1.parquet').select('s1_idx','country','core_key','content','legal','tags','n_core_shared').rename({'core_key':'s_core','content':'s_content','legal':'s_legal','tags':'s_tags'})
 r=pl.scan_parquet(C/f'views_{split}_rec.parquet').filter(pl.col('a_empty')).select('cand_idx','core_key','content','legal','tags').collect().rename({'core_key':'r_core','content':'r_content','legal':'r_legal','tags':'r_tags'})
 log('views',split,s.height,r.height)
 return s,r
def prep(kind):
 split='test' if kind=='test' else 'train'; s,r=views(split)
 if kind=='val':path=V/'val_decoder.parquet';sel=pl.read_parquet(V/'seed_consensus_selected.parquet');rows=pl.read_parquet(V/'seed_consensus_rows.parquet')
 elif kind=='density':path=V/'density_decoder.parquet';sel=pl.read_parquet(V/'density_seed_consensus_selected.parquet');rows=pl.read_parquet(V/'density_seed_consensus_rows.parquet')
 else:path=W/'matching/ensemble_v1/data/test_v3anchor.parquet';sel=pl.read_parquet(W/'internal_eval/data/sel/g7_vr2.parquet');rows=None
 d=pl.scan_parquet(path).join(r.lazy(),on='cand_idx',how='inner').collect()
 d=d.join(s,on='s1_idx').with_columns((pl.col('s_core')==pl.col('r_core')).alias('same_core'),((pl.col('s_content')==pl.col('r_content'))&(pl.col('s_legal')==pl.col('r_legal'))&(pl.col('s_tags')==pl.col('r_tags'))).alias('same_name'))
 log(kind,'candidate empty',d.height,'same core',d['same_core'].sum())
 d=d.join(sel.with_columns(pl.lit(True).alias('selected')),on=K,how='left').with_columns(pl.col('selected').fill_null(False))
 owners=sel.select('cand_idx',pl.col('s1_idx').alias('owner'))
 d=d.join(owners,on='cand_idx',how='left').with_columns(pl.col('owner').is_not_null().alias('owned'),(pl.col('owner').is_not_null()&(pl.col('owner')!=pl.col('s1_idx'))).alias('other_owned'))
 # Distinct competing exact-core query count among all queries, including outside evaluated subset, is in n_core_shared.
 if kind != 'test':
  gt=pl.read_parquet(W/'matching/variance_study_20260926/val_truth.parquet').select(K).with_columns(pl.lit(True).alias('y'))
  d=d.join(gt,on=K,how='left').with_columns(pl.col('y').fill_null(False)).join(rows.select('s1_idx','eval_split','k','t','m','f'),on='s1_idx',how='left')
 else:d=d.join(sel.group_by('s1_idx').len('k'),on='s1_idx',how='left').with_columns(pl.col('k').fill_null(0))
 d.write_parquet(O/f'{kind}_empty.parquet');log('saved',kind,d.height)
 if kind=='val':
  t=d.filter(pl.col('eval_split')=='tune');stats=t.group_by('country','same_core','same_name','selected','other_owned').agg(pl.len().alias('n'),pl.col('y').mean().alias('true_rate'),pl.col('p').mean().alias('mean_p')).sort('country','n',descending=[False,True]).to_dicts();(O/'discovery_population.json').write_text(json.dumps(stats,indent=2));log(stats)
if __name__=='__main__':
 import sys
 for k in sys.argv[1:]:prep(k)
