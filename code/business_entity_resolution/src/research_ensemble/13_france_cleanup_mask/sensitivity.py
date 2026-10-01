import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[k]='8'
import json
from pathlib import Path
import polars as pl
ROOT=Path('/workspace/saumilya/amazon-ml/work');OUT=ROOT/'matching/iterate_20260926/france';K=['s1_idx','cand_idx'];F=pl.read_parquet(ROOT/'internal_eval/data/pc_test_fam.parquet').filter((pl.col('country')=='France')&pl.col('fam').is_in(['CTAG','LEG3','LEG12']))
src={'vr2':ROOT/'internal_eval/data/sel/g7_vr2.parquet','nc':ROOT/'matching/v2shash/round_c/test_release/data/test_selected.parquet'}
r={};fp=.187793427230047;miss=.07518796992481203
for base,path in src.items():
 S=pl.read_parquet(path);num=S.group_by('s1_idx').len('nsel');D=F.join(S,on=K,how='semi').join(num,on='s1_idx');P=D.filter(pl.col('side')=='+');M=D.filter(pl.col('side')=='-')
 dropped=P.select(K).unique().group_by('s1_idx').len('nremove').join(num,on='s1_idx');empt=dropped.filter(pl.col('nremove')==pl.col('nsel'))
 E=P.join(empt.select('s1_idx'),on='s1_idx',how='semi');P1=P.filter(pl.col('nsel')==1);M1=M.filter(pl.col('nsel')==1)
 raw=[]
 for mult in [0.5,1,2,4,8,12,16]:
  true=min(P.height,mult*M.height);raw.append({'true_count_mirror_multiplier':mult,'expected_true':true,'expected_dLB':((P.height-true)*fp-true*miss)/1732544})
 r[base]={'plus_selected':P.height,'minus_selected':M.height,'break_even_true_fraction':fp/(fp+miss),'break_even_mirror_multiplier':(P.height*fp/(fp+miss))/M.height,'sensitivity':raw,'empty_change_queries':empt.height,'emptied_query_pairs_removed':E.height,'empty_change_by_family':E.group_by('fam').len().to_dicts(),'selected_singletons_plus':P1.height,'selected_singletons_minus':M1.height,'nsel_controls':D.group_by('side','nsel').len().sort('side','nsel').to_dicts()}
print(json.dumps(r,indent=2));(OUT/'out/sensitivity.json').write_text(json.dumps(r,indent=2))
