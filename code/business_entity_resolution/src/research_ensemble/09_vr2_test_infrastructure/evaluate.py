"""Frozen decoder, shipping parity, TUNE-only selection and post-freeze reporting."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import sys,json,time
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.special import expit,logit
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];M=W/'matching';E=M/'ensemble_v1';K=['s1_idx','cand_idx']
sys.path.insert(0,str(M/'variance_study_20260926'))
import study as S
sys.path.insert(0,str(M/'accuracy_lab_20260925'))
from set_decode import decide_fast

def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def save(n,o):(HERE/n).write_text(json.dumps(o,indent=2))
def inputs():
 meta=pl.read_parquet(M/'variance_study_20260926/val_meta.parquet')
 splits=pl.read_parquet(M/'accuracy_lab_20260925/data/baseline_rows.parquet',columns=['s1_idx','eval_split'])
 meta=meta.join(splits,on='s1_idx');g=pl.read_parquet(M/'variance_study_20260926/val_truth.parquet')
 cache=HERE/'data/val_decoder.parquet'
 if cache.exists():full=pl.read_parquet(cache)
 else:
  p=pl.scan_parquet(E/'data/preds_v3anchor.parquet').filter(pl.col('grp')!='sample').select(K+['p']).collect()
  comp=pl.read_parquet(M/'accuracy_lab_20260925/data/val_decoder.parquet',columns=K+['p_other','src'])
  full=p.join(comp,on=K,how='left');assert full.height==comp.height and full['p_other'].null_count()==0
  full.write_parquet(cache)
 a=S.readsel(E/'results/eval_ENS_v3anchor_val_selected.parquet')
 return full,a,meta,g

def veto():return S.pp.add_pairs('train').filter(pl.col('k').is_in(S.D)).select(K)
def post(sel,full,v):
 s=sel.select(K).join(v,on=K,how='anti').join(full.select(K+['p']),on=K)
 return s.sort(['cand_idx','p','s1_idx'],descending=[False,True,False]).unique('cand_idx',keep='first',maintain_order=True).select(K)
def modified(full,corr,alpha,scope,seed=None):
 cols=[f'delta_{seed}'] if seed else ['delta_11','delta_29','delta_47']
 c=corr.filter(pl.col('missing_gate')>0) if scope=='missing' else corr
 c=c.select(K+[pl.mean_horizontal(cols).alias('delta')])
 d=full.join(c,on=K,how='left');p=d['p'].to_numpy();ix=d['delta'].is_not_null().to_numpy();pn=p.copy()
 pn[ix]=expit(logit(np.clip(p[ix].astype(float),1e-6,1-1e-6))+alpha*d['delta'].to_numpy()[ix]).astype(np.float32)
 assert np.array_equal(p[~ix],pn[~ix])
 return d.drop('delta').with_columns(pl.Series('p',pn))
def tune():
 full,a,meta,g=inputs();v=veto();model=lgb.Booster(model_file=str(E/'results/eval_ENS_v3anchor_R10c_m0.txt'))
 rep=decide_fast(full,model)
 assert rep.join(a,on=K,how='anti').height==a.join(rep,on=K,how='anti').height==0
 a=post(a,full,v);meta=meta.filter(pl.col('eval_split')=='tune');base=S.score(a,meta,g)
 corr=pl.read_parquet(HERE/'data/residual_scores.parquet').filter(pl.col('grp')!='sample')
 baseline=float(base['f'].mean());best=dict(scope='all',alpha=0.,score=baseline,delta=0.);results=[]
 for scope in ['all','missing']:
  for alpha in [.25,.5,1.]:
   p=modified(full,corr,alpha,scope);sel=post(decide_fast(p,model),p,v)
   rows=S.score(sel,meta,g);d=rows['f'].to_numpy()-base['f'].to_numpy()
   by={c:float(d[rows['country'].to_numpy()==c].mean()) for c in ['US','India']}
   r=dict(scope=scope,alpha=alpha,score=float(rows['f'].mean()),delta=float(d.mean()),countries=by,improved=int((d>1e-12).sum()),harmed=int((d < -1e-12).sum()))
   results.append(r);log('TUNE',r)
   sel.write_parquet(HERE/f'data/tune_selected_{scope}_{alpha}.parquet')
   if r['score']>best['score'] and min(by.values())>=0:best=r
 save('frozen_choice.json',dict(baseline_tune=baseline,grid=results,chosen=best,selection='highest TUNE macro F0.5 with nonnegative country point estimates; baseline alpha 0 eligible',report_opened=False))
 log('FROZEN',best)

def report():
 cfg=json.loads((HERE/'frozen_choice.json').read_text());best=cfg['chosen']
 full,a,meta,g=inputs();v=veto();a=post(a,full,v);base=S.score(a,meta,g)
 result=dict(choice=best,baseline=float(base['f'].mean()),experiments={})
 if best['alpha']==0:
  result['decision']='No TUNE candidate clears selection; baseline retained; report labels not used to rescue variants.';save('residual_report.json',result);log(result);return
 corr=pl.read_parquet(HERE/'data/residual_scores.parquet').filter(pl.col('grp')!='sample')
 model=lgb.Booster(model_file=str(E/'results/eval_ENS_v3anchor_R10c_m0.txt'))
 for seed in [None,11,29,47]:
  name='bagged' if seed is None else str(seed)
  p=modified(full,corr,best['alpha'],best['scope'],seed);sel=post(decide_fast(p,model),p,v)
  r,rows=S.compare(sel,base,meta,g)
  for cohort in ['tune','report','locked']:
   mi=meta.filter(pl.col('eval_split')==cohort)
   rb=base.join(mi.select('s1_idx'),on='s1_idx',how='semi')
   rr,_=S.compare(sel,rb,mi,g);r[cohort]=rr
  r['one_owner_violations']=sel.group_by('cand_idx').len().filter(pl.col('len')>1).height
  result['experiments'][name]=r;log('REPORT',name,r['delta'],'held report',r['report']['delta'],r['report']['query_ci95'],'locked',r['locked']['delta'])
  if seed is None:
   sel.write_parquet(HERE/'data/residual_frozen_selected.parquet');rows.write_parquet(HERE/'data/residual_frozen_rows.parquet')
 save('residual_report.json',result)
if __name__=='__main__':{'tune':tune,'report':report}[sys.argv[1]]()
