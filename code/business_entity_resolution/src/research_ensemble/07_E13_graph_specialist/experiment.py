"""Train band specialists and select postpasses on TUNE entities only.

Run one model per invocation. REPORT and locked metrics are deliberately withheld
here; finalize.py evaluates frozen choices after all comparisons are complete.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS']:
    os.environ[k]='4'
import argparse, json, time
from pathlib import Path
import numpy as np
import polars as pl
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
HERE=Path(__file__).resolve().parent
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
def row_f(tp,k,m):
    return np.where(m==0,(k==0).astype(float),5*tp/np.maximum(4*k+m,1))
def tuning(B,prob,name):
    M=pl.read_parquet(HERE/'data/baseline_rows.parquet').sort('s1_idx')
    # TUNE only is accessed for model and postpass selection.
    M=M.filter(pl.col('eval_split')=='tune')
    T=B.filter(pl.col('eval_split')=='tune')
    p=prob[(B['eval_split']=='tune').to_numpy()]
    sid=M['s1_idx'].to_numpy(); idx=np.searchsorted(sid,T['s1_idx'].to_numpy())
    n=M.height; y=T['label'].to_numpy(); old=T['selected'].to_numpy()
    basek=M['k'].to_numpy().astype(np.int64); baset=M['t'].to_numpy().astype(np.int64); m=M['m'].to_numpy()
    restk=basek-np.bincount(idx,weights=old,minlength=n)
    restt=baset-np.bincount(idx,weights=old*y,minlength=n)
    baseline=float(row_f(baset,basek,m).mean())
    basep=T['p2'].to_numpy(); results=[]
    for alpha in ([0.] if name=='control' else [0.25,0.5,0.75,1.]):
        q=expit((1-alpha)*logit(np.clip(basep,1e-6,1-1e-6))+alpha*logit(np.clip(p,1e-6,1-1e-6)))
        for tau in [0.45,0.5,0.55,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85]:
            sel=q>=tau
            k=restk+np.bincount(idx,weights=sel,minlength=n)
            t=restt+np.bincount(idx,weights=sel*y,minlength=n)
            f=float(row_f(t,k,m).mean())
            results.append(dict(alpha=alpha,tau=tau,macro_f05=f,delta=f-baseline))
    best=max(results,key=lambda r:r['macro_f05'])
    out=dict(name=name,n_tune=n,baseline=baseline,best=best,grid=results)
    json.dump(out,open(HERE/f'logs/tune_{name}.json','w'),indent=2)
    log('TUNE',name,best)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['control','lgb','residual','catboost','scoreonly','graph','freq','sibling'],required=True);ap.add_argument('--gpu',default='6');a=ap.parse_args()
    suffix='_graph' if a.model in ['graph','freq','sibling'] else ''
    B=pl.read_parquet(HERE/f'data/band{suffix}.parquet')
    F=json.load(open(HERE/f'data/features{suffix}.json'))
    if a.model=='freq': F=[f for f in F if not f.startswith('g_') or f.startswith('g_s1freq_')]
    if a.model=='sibling': F=[f for f in F if not f.startswith('g_s1freq_')]
    if a.model=='scoreonly': F=[f for f in F if f.startswith(('p1','p2','sib_'))]
    if a.model=='control':
        tuning(B,B['p2'].to_numpy(),a.model);return
    X=B.select(F).to_numpy().astype(np.float32); X=np.nan_to_num(X,nan=-1.,posinf=1e6,neginf=-1e6)
    y=B['label'].to_numpy().astype(np.float32)
    train=((B['grp']=='sample')&~B['es']).to_numpy(); es=((B['grp']=='sample')&B['es']).to_numpy()
    val=(B['grp']!='sample').to_numpy();p0=B['p2'].to_numpy();z0=logit(np.clip(p0,1e-6,1-1e-6))
    log('fit',a.model,'train',int(train.sum()),'es',int(es.sum()),'eval',int(val.sum()),'features',len(F))
    t=time.time()
    if a.model=='catboost':
        # Only this explicitly authorized physical GPU is visible to CatBoost.
        os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
        from catboost import CatBoostClassifier,Pool
        model=CatBoostClassifier(iterations=1600,depth=6,border_count=32,learning_rate=0.04,loss_function='Logloss',eval_metric='Logloss',l2_leaf_reg=8,random_seed=20260925,task_type='GPU',devices='0',gpu_ram_part=0.40,thread_count=4,allow_writing_files=False,verbose=100)
        model.fit(Pool(X[train],y[train],feature_names=F),eval_set=Pool(X[es],y[es],feature_names=F),early_stopping_rounds=100)
        model.save_model(str(HERE/f'models/{a.model}.cbm'))
        p=model.predict_proba(X)[:,1];it=model.best_iteration_
        imp=model.feature_importances_
    else:
        import lightgbm as lgb
        resid=a.model in ['residual','graph','freq','sibling']
        ds=lgb.Dataset(X[train],label=y[train],feature_name=F,init_score=z0[train] if resid else None)
        de=lgb.Dataset(X[es],label=y[es],reference=ds,init_score=z0[es] if resid else None)
        model=lgb.train(dict(objective='binary',metric='binary_logloss',learning_rate=0.035,num_leaves=15 if a.model in ['residual','scoreonly','graph','freq','sibling'] else 31,min_data_in_leaf=150,lambda_l2=10,feature_fraction=0.9,num_threads=4,verbosity=-1,seed=20260925),ds,num_boost_round=1400,valid_sets=[de],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(100)])
        model.save_model(str(HERE/f'models/{a.model}.txt'));it=model.best_iteration
        p=expit(model.predict(X,raw_score=True,num_threads=4)+(z0 if resid else 0))
        imp=model.feature_importance(importance_type='gain')
    B.select('s1_idx','cand_idx').with_columns(pl.Series('prob',p.astype(np.float32))).write_parquet(HERE/f'data/pred_{a.model}.parquet')
    stats=dict(name=a.model,seconds=time.time()-t,iterations=int(it),features=F,train_rows=int(train.sum()),es_rows=int(es.sum()),es_logloss=log_loss(y[es],p[es]),es_base_logloss=log_loss(y[es],p0[es]),es_auc=roc_auc_score(y[es],p[es]),importance=sorted(zip(F,map(float,imp)),key=lambda x:-x[1])[:35])
    json.dump(stats,open(HERE/f'logs/fit_{a.model}.json','w'),indent=2)
    tuning(B,p,a.model)
    log('DONE',a.model,'seconds',stats['seconds'])
if __name__=='__main__':main()
