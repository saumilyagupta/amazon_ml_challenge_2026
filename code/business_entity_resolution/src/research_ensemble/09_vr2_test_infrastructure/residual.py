"""Anchor-trained residual with newly recomputed sibling/query evidence."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import sys,json,time,zlib
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.special import logit,expit
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];M=W/'matching';E=M/'ensemble_v1';K=['s1_idx','cand_idx']
sys.path.insert(0,str(M/'prod_v2c/exp/E13_codex_refit'))
import e13_block as G
BASE=['p1','p1_max_other_rec','p1_rel_other','p1_rank_rec','sib_n','sib_hn_eq','sib_addr_max','sib_name_max','n_ratio_raw','n_tset_raw','n_partial_raw','n_ratio','n_tsort','n_wjac','n_cov1','n_cov2','n_miss_cnt','n_extra_cnt','core_eq','ns_eq','a_tset','a_tsort','a_ratio','a_wjac','a_cov1','a_cov2','num_first_eq','num_first_off1','num_first_ldiff','num_first_both','num_inter','num_jac','num_cnt1','num_cnt2','num_miss1','num_extra2','has_unit1','has_unit2','addr_empty1','addr_empty2','addr_null2','addr_nodigit2','ntok_addr1','ntok_addr2','ntok_name1','ntok_name2','name_freq_s1_1','name_freq_pool2','name_freq_s1_2','addr_freq_s1_1','addr_freq_pool2','fake2','script2','is_s3','cos','n_channels','n1_core_equal','n4_soft_ratio_b','n5_left_cnt','n6_drop_cnt','n8_acronym','n10_alias','n13_script_mismatch','a1_num_bp','a2_num_exact','a3_num_rel','a6_street_sim','a7_street_key_eq','a10_loc_sim','a12_addr_missing_b','dec_extra','dec_num_shift','dec_flag']
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def save(n,o):(HERE/n).write_text(json.dumps(o,indent=2))
def prepare():
 B=pl.read_parquet(M/'accuracy_lab_20260925/data/band_graph.parquet',columns=K+['grp','label','eval_split']+BASE)
 P=pl.read_parquet(E/'data/preds_v3anchor.parquet',columns=K+['p']).rename({'p':'p2'})
 B=B.join(G.p2_aggregates(P),on=K).filter(pl.col('p2').is_between(.001,.999,closed='none'))
 log('band',B.shape)
 R,R1=G.load_records('train')
 features=G.make_features(B,P,R,R1);B=B.join(features,on=K)
 grouping=R1.select(pl.col('erow').cast(pl.Int32).alias('s1_idx'),'name_n','country')
 keys=[c+':'+n for c,n in grouping.select('country','name_n').iter_rows()]
 grouping=grouping.select('s1_idx').with_columns(pl.Series('sample_es',[zlib.crc32(('round2es:'+x).encode())%10==0 for x in keys]))
 B=B.join(grouping,on='s1_idx')
 mem=pl.scan_parquet(E/'data/members_val.parquet').select(K+['m_v2b','m_v3','m_e06','m_e13']).join(B.select(K).lazy(),on=K,how='semi').collect()
 B=B.join(mem,on=K)
 # Existing v3 decoy features are static/pairwise inputs, computed on these same candidates.
 files=sorted((M/'prod_v3/feats').glob('train_[0-9]*.parquet'))
 dec=[c for c in pl.read_parquet_schema(files[0]) if c.startswith('dec_') and c not in B.columns]
 parts=[pl.scan_parquet(f).select(K+dec).join(B.select(K).lazy(),on=K,how='semi').collect() for f in files]
 B=B.join(pl.concat(parts),on=K)
 zs=np.column_stack([logit(np.clip(B[c].to_numpy().astype(float),1e-6,1-1e-6)) for c in ['m_v2b','m_v3','m_e06','m_e13']])
 for j,n in enumerate(['z_v2b','z_v3','z_e06','z_e13']):B=B.with_columns(pl.Series(n,zs[:,j].astype(np.float32)))
 B=B.with_columns(pl.Series('z_spread',zs.max(1)-zs.min(1)),pl.Series('z_std',zs.std(1)),pl.Series('vote05',(zs>0).sum(1)),pl.Series('anchor_z',logit(B['p2'].to_numpy().astype(float))),((pl.col('addr_empty1')>0)|(pl.col('addr_empty2')>0)).cast(pl.Int8).alias('missing_gate'))
 B=B.with_columns((pl.col('z_std')*pl.col('missing_gate')).alias('disagree_missing'),(pl.col('z_std')*(pl.col('num_first_both')-pl.col('num_first_eq'))).alias('disagree_number'),(pl.col('anchor_z')-(pl.col('z_v2b')+pl.col('z_v3')+pl.col('z_e06')+pl.col('z_e13'))/4).alias('anchor_vs_mean'))
 feat=BASE+['p2']+G.AGG_FEATURES+G.G_FEATURES+dec+['z_v2b','z_v3','z_e06','z_e13','z_spread','z_std','vote05','anchor_z','missing_gate','disagree_missing','disagree_number','anchor_vs_mean']
 assert B.select(pl.struct(K).n_unique()).item()==B.height
 B.write_parquet(HERE/'data/anchor_band.parquet');save('features.json',feat)
 save('preparation.json',dict(rows=B.height,features=len(feat),sample=B.filter(pl.col('grp')=='sample').height,val=B.filter(pl.col('grp')!='sample').height,gate='existing v2b band AND .001 < anchor p < .999',fit_early_stop_group='country+normalized S1 name hash, 90/10',sibling_scores='v3anchor, self excluded'))
 log('PREPARED',B.shape,len(feat))
def fit():
 B=pl.read_parquet(HERE/'data/anchor_band.parquet');feat=json.loads((HERE/'features.json').read_text())
 X=np.nan_to_num(B.select(feat).to_numpy().astype(np.float32),nan=-1,posinf=1e6,neginf=-1e6);y=B['label'].to_numpy();z=logit(B['p2'].to_numpy().astype(float))
 tr=((B['grp']=='sample')&~B['sample_es']).to_numpy();es=((B['grp']=='sample')&B['sample_es']).to_numpy()
 outputs=B.select(K+['grp','eval_split','missing_gate','p2']);infos=[]
 for seed in [11,29,47]:
  train=lgb.Dataset(X[tr],label=y[tr],init_score=z[tr],feature_name=feat,free_raw_data=True)
  early=lgb.Dataset(X[es],label=y[es],init_score=z[es],reference=train,free_raw_data=True)
  model=lgb.train(dict(objective='binary',metric='binary_logloss',learning_rate=.035,num_leaves=15,min_data_in_leaf=250,lambda_l2=30,feature_fraction=.85,bagging_fraction=.8,bagging_freq=1,num_threads=4,verbosity=-1,seed=seed),train,num_boost_round=600,valid_sets=[early],callbacks=[lgb.early_stopping(60,verbose=False)])
  model.save_model(str(HERE/f'models/residual_{seed}.txt'))
  pred=model.predict(X,raw_score=True,num_threads=4)
  outputs=outputs.with_columns(pl.Series(f'delta_{seed}',pred.astype(np.float32)))
  info=dict(seed=seed,iterations=model.best_iteration,early_stop_logloss=model.best_score['valid_0']['binary_logloss'],fit_pairs=int(tr.sum()),es_pairs=int(es.sum()),top_features=sorted(zip(feat,model.feature_importance(importance_type='gain').tolist()),key=lambda x:-x[1])[:20]);infos.append(info);log('FIT',info)
 outputs.write_parquet(HERE/'data/residual_scores.parquet');save('fit.json',infos)
 log('FIT DONE')
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='fit':fit()
