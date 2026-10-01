import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[k]='8'
import json
from pathlib import Path
import polars as pl
ROOT=Path('/workspace/saumilya/amazon-ml/work'); OUT=ROOT/'matching/iterate_20260926/france'; K=['s1_idx','cand_idx']
F=pl.read_parquet(ROOT/'internal_eval/data/pc_test_fam.parquet').filter(pl.col('country')=='France')
SOURCES={'vr2':ROOT/'internal_eval/data/sel/g7_vr2.parquet','nc':ROOT/'matching/v2shash/round_c/test_release/data/test_selected.parquet'}
RULES=['CTAG','LEG3','LEG12','EQ12']; m=3.46; MISS=1-5*(m-1)/(5*(m-1)+1); FP=1-5*m/(5*m+4); NT=1732544
result={}
for name,p in SOURCES.items():
 S=pl.read_parquet(p).select(K); FS=F.join(S,on=K,how='semi')
 A=pl.read_parquet(ROOT/'internal_eval/data/pc_test_frstrict.parquet').select(K).join(S,on=K,how='semi').select('s1_idx').unique().with_columns(pl.lit(True).alias('anch'))
 FS=FS.join(A,on='s1_idx',how='left').with_columns(pl.col('anch').fill_null(False))
 counts=[]; seen={s:pl.DataFrame(schema={'s1_idx':pl.Int32,'cand_idx':pl.Int32}) for s in ['+','-']}
 for fam in RULES+['EQ3']:
  row={'family':fam}
  for side in ['+','-']:
   cur=FS.filter((pl.col('fam')==fam)&(pl.col('side')==side)).select(K).unique()
   new=cur.join(seen[side],on=K,how='anti'); row[f'n_{side}_raw']=cur.height;row[f'n_{side}_disjoint']=new.height
   seen[side]=pl.concat([seen[side],cur]).unique()
  p1=row['n_+_disjoint'];mn=row['n_-_disjoint'];t=min(p1,mn)
  row['mirror_dLB']=(FP*(p1-t)-MISS*t)/NT;counts.append(row)
 U=FS.filter(pl.col('fam').is_in(RULES)).select(K+['side']).unique()
 p1=U.filter(pl.col('side')=='+').height;mn=U.filter(pl.col('side')=='-').height
 subset=[]
 # Diagnose EQ3 remaining after ppB, using symmetric guard subsets.
 clean=FS.filter(pl.col('fam')=='EQ3').join(U.select(K),on=K,how='anti')
 for anch in [None,True,False]:
  C=clean if anch is None else clean.filter(pl.col('anch')==anch)
  for kk in [None,3,4,5,7,9,11,13,21]:
   Ck=C if kk is None else C.filter(pl.col('k').abs()==kk)
   np=Ck.filter(pl.col('side')=='+').height;nm=Ck.filter(pl.col('side')=='-').height;t=min(np,nm)
   subset.append({'anchor':anch,'abs_offset':kk,'plus':np,'minus':nm,'mirror_dLB':(FP*(np-t)-MISS*t)/NT})
 result[name]={'constants':{'fp':FP,'miss':MISS},'sequential_controls':counts,'union_plus':p1,'union_minus':mn,'union_mirror_dLB':(FP*(p1-mn)-MISS*mn)/NT,'disjoint_mirror_dLB':sum(r['mirror_dLB'] for r in counts[:4]),'EQ3_diagnostic':subset}
 # Save reusable input-only ppB and a conservative union without EQ12.
 for n,fams in [('ppB_FR',RULES),('ppLegal_FR',RULES[:3])]:
  mask=F.filter(pl.col('fam').is_in(fams)&(pl.col('side')=='+')).select(K).unique()
  if name=='vr2':mask.write_parquet(OUT/f'data/mask_{n}.parquet')
  rem=mask.join(S,on=K,how='semi');rem.write_parquet(OUT/f'data/{name}_{n}_removed.parquet')
  result[name][n]={'removed':rem.height,'queries':rem['s1_idx'].n_unique()}
print(json.dumps(result,indent=2));(OUT/'out/audit.json').write_text(json.dumps(result,indent=2))
