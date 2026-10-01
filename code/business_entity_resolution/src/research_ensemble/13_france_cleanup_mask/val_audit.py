import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[k]='8'
from pathlib import Path
import polars as pl,json
ROOT=Path('/workspace/saumilya/amazon-ml/work');OUT=ROOT/'matching/iterate_20260926/france';K=['s1_idx','cand_idx']
F=pl.read_parquet(ROOT/'internal_eval/data/pc_val_fam.parquet'); GT=pl.read_parquet(ROOT/'internal_eval/data/val_gt_pairs.parquet').select(K).with_columns(pl.lit(True).alias('truth'))
src={'vr2':ROOT/'matching/variance_research_round2_20260926/data/seed_consensus_selected.parquet','nc':ROOT/'matching/v2shash/round_c/data/ordinary_blend_e0.25_selected.parquet'}
r={}
for base,p in src.items():
 S=pl.read_parquet(p).select(K);FS=F.join(S,on=K,how='semi').join(GT,on=K,how='left').with_columns(pl.col('truth').fill_null(False));res=[]
 for country in ['US','India']:
  seen={s:pl.DataFrame(schema={'s1_idx':pl.Int32,'cand_idx':pl.Int32}) for s in ['+','-']}
  for fam in ['CTAG','LEG3','LEG12','EQ12','EQ3']:
   row={'country':country,'family':fam}
   for side in ['+','-']:
    C=FS.filter((pl.col('country')==country)&(pl.col('fam')==fam)&(pl.col('side')==side)).select(K+['truth']).unique(K)
    new=C.join(seen[side],on=K,how='anti');seen[side]=pl.concat([seen[side],C.select(K)]).unique()
    row.update({f'selected_{side}':C.height,f'true_{side}':int(C['truth'].sum()),f'selected_disjoint_{side}':new.height,f'true_disjoint_{side}':int(new['truth'].sum())})
   res.append(row)
 r[base]=res
print(json.dumps(r,indent=2));(OUT/'out/val_mirror_audit.json').write_text(json.dumps(r,indent=2))
