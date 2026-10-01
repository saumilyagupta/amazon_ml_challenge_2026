import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
os.environ['OMP_WAIT_POLICY']='PASSIVE'
import polars as pl,numpy as np,json,time,lightgbm as lgb
from scipy.special import logit,expit
from pathlib import Path
O=Path(__file__).resolve().parent;W=O.parents[2];VR=W/'matching/variance_research_round2_20260926';C=W/'internal_eval/agents/B_archetypes/cache';K=['s1_idx','cand_idx']
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def extra(b,split='train',density=False):
 s=pl.read_parquet(C/f'views_{split}_s1.parquet').select('s1_idx','country','core_key','content','legal','tags','n_core_shared')
 if density:
  ids=pl.read_parquet(W/'blocking/embedding/full/ids/train_s1.parquet',columns=['entity_id']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
  drop=pl.read_csv(W/'matching/density_val/universe/dropped_s1_ids.txt',has_header=False,new_columns=['entity_id'])
  keep=ids.join(drop,on='entity_id',how='anti').select('s1_idx')
  s=s.join(keep,on='s1_idx',how='semi').drop('n_core_shared')
  freqcore=s.group_by('country','core_key').len('n_core_shared');s=s.join(freqcore,on=['country','core_key'])
 r=pl.scan_parquet(C/f'views_{split}_rec.parquet').filter(pl.col('a_empty')).select('cand_idx','core_key','content','legal','tags').collect()
 freq=s.group_by('country','core_key','legal','tags').len('full_form_s1_count')
 r=r.rename({'core_key':'r_core','content':'r_content','legal':'r_legal','tags':'r_tags'})
 b=b.join(s.rename({'core_key':'s_core','content':'s_content','legal':'s_legal','tags':'s_tags'}),on='s1_idx').join(r,on='cand_idx')
 b=b.join(freq.rename({'core_key':'r_core','legal':'r_legal','tags':'r_tags'}),on=['country','r_core','r_legal','r_tags'],how='left')
 b=b.with_columns(pl.col('full_form_s1_count').fill_null(0),
  (pl.col('country')=='India').cast(pl.Int8).alias('india'),
  (pl.col('s_core')==pl.col('r_core')).cast(pl.Int8).alias('exact_core'),
  (pl.col('s_content')==pl.col('r_content')).cast(pl.Int8).alias('ordered_core'),
  (pl.col('s_legal')==pl.col('r_legal')).cast(pl.Int8).alias('legal_equal'),
  (pl.col('s_tags')==pl.col('r_tags')).cast(pl.Int8).alias('tags_equal'),
  (pl.col('s_legal')=='').cast(pl.Int8).alias('s_legal_empty'),
  (pl.col('r_legal')=='').cast(pl.Int8).alias('r_legal_empty'),
  (pl.col('s_tags')=='').cast(pl.Int8).alias('s_tags_empty'),
  (pl.col('r_tags')=='').cast(pl.Int8).alias('r_tags_empty'))
 for j in ['s','r']:
  for term in ['INC','LLC','LTD','PVT','CORP','LLP']:
   b=b.with_columns(pl.col(f'{j}_legal').str.contains(term,literal=True).cast(pl.Int8).alias(f'{j}_{term}'))
 return b
if __name__=='__main__':
 import sys
 kind=sys.argv[1]
 if kind=='fit':
  b=pl.scan_parquet(VR/'data/anchor_band.parquet').filter((pl.col('addr_empty2')>0)&(pl.col('p2')>.03)).collect();b=extra(b)
  feats=json.loads((VR/'features.json').read_text())+['n_core_shared','full_form_s1_count','india','exact_core','ordered_core','legal_equal','tags_equal','s_legal_empty','r_legal_empty','s_tags_empty','r_tags_empty']+[f'{j}_{term}' for j in ['s','r'] for term in ['INC','LLC','LTD','PVT','CORP','LLP']]
  (O/'specialist_features.json').write_text(json.dumps(feats,indent=2));log('data',b.shape)
  x=np.nan_to_num(b.select(feats).to_numpy().astype(np.float32),nan=-1,posinf=1e6,neginf=-1e6);y=b['label'].to_numpy();z=b['anchor_z'].to_numpy();tr=((b['grp']=='sample')&~b['sample_es']).to_numpy();es=((b['grp']=='sample')&b['sample_es']).to_numpy();out=b.select(K+['grp','p2']);infos=[]
  for seed in [61]:
   train=lgb.Dataset(x[tr],label=y[tr],init_score=z[tr],feature_name=feats);early=lgb.Dataset(x[es],label=y[es],init_score=z[es],reference=train)
   m=lgb.train(dict(objective='binary',metric='binary_logloss',learning_rate=.025,num_leaves=31,min_data_in_leaf=120,lambda_l2=30,feature_fraction=.85,bagging_fraction=.85,bagging_freq=1,num_threads=4,verbosity=-1,seed=seed),train,num_boost_round=800,valid_sets=[early],callbacks=[lgb.early_stopping(70,verbose=False)])
   m.save_model(str(O/f'specialist_{seed}.txt'));delta=m.predict(x,raw_score=True,num_threads=4);out=out.with_columns(pl.Series(f'delta_{seed}',delta));info=dict(seed=seed,best_iteration=m.best_iteration,early_logloss=m.best_score['valid_0']['binary_logloss'],fit_pairs=int(tr.sum()),early_pairs=int(es.sum()),top_features=sorted(zip(feats,m.feature_importance('gain').tolist()),key=lambda q:-q[1])[:30]);infos.append(info);log(info)
  out.filter(pl.col('grp')!='sample').write_parquet(O/'specialist_val_scores.parquet');(O/'specialist_fit.json').write_text(json.dumps(infos,indent=2))
 else:
  path=VR/'data/density_band.parquet' if kind=='density' else W/'matching/vr2_release_20260926/data/test_anchor_band.parquet'
  b=pl.scan_parquet(path).filter((pl.col('addr_empty2')>0)&(pl.col('p2')>.03));b=b.filter(pl.col('country').is_in(['US','India'])) if kind=='test' else b;b=b.collect();b=extra(b,'test' if kind=='test' else 'train',density=(kind=='density'));feats=json.loads((O/'specialist_features.json').read_text());x=np.nan_to_num(b.select(feats).to_numpy().astype(np.float32),nan=-1,posinf=1e6,neginf=-1e6);out=b.select(K+['p2'])
  for seed in [61]:
   m=lgb.Booster(model_file=str(O/f'specialist_{seed}.txt'));assert m.feature_name()==feats;out=out.with_columns(pl.Series(f'delta_{seed}',m.predict(x,raw_score=True,num_threads=4)))
  out.write_parquet(O/f'specialist_{kind}_scores.parquet');log('inference',kind,out.height)
