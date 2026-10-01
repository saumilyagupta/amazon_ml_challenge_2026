"""Vectorized shipped R10c decoder, with frozen-model band-score substitutions.

Exact baseline reproduction is mandatory before comparing new predictions.
Only TUNE metrics are printed/used to select a model or blend.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[k]='4'
import sys,json,time,argparse
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.special import expit,logit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'matching/prod_v2b'))
from pv2b.common import EB
from pv2a.decide import add_comp
from pv1.decide import row_scores
import decide_lib as DL

def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)

def decide_fast(P,model):
    U=P.select(pl.col('s1_idx').alias('s1'),pl.col('cand_idx').alias('qid'),'src',pl.col('p').cast(pl.Float64),(pl.col('p')>=pl.col('p_other')).cast(pl.Int8).replace({0:2}).alias('qr'))
    F=DL.s1_features(U)
    W=U.filter((pl.col('qr')==1)&(pl.col('p')>0.02)).sort(['s1','p'],descending=[False,True])
    W=W.with_columns(pl.col('p').cum_count().over('s1').alias('k'),pl.col('p').cum_sum().over('s1').alias('cs'),(pl.col('src')=='S3').cast(pl.Int32).cum_sum().over('s1').alias('c3'),pl.col('p').shift(-1).over('s1').fill_null(0).alias('pn'),pl.len().over('s1').alias('n'),pl.col('p').sum().over('s1').alias('total'))
    Z=W.group_by('s1',maintain_order=True).first().select('s1',pl.lit(0).cast(pl.UInt32).alias('k'),pl.lit(1.).alias('pk'),pl.col('p').alias('pn'),pl.lit(0.).alias('cs'),'total','n',pl.lit(0).cast(pl.Int32).alias('c3'))
    T=W.filter(pl.col('k')<=10).select('s1','k',pl.col('p').alias('pk'),'pn','cs','total','n','c3')
    T=pl.concat([Z,T],how='vertical_relaxed').join(F,on='s1').with_columns((pl.col('total')-pl.col('cs')).alias('tail'),(pl.col('pk')-pl.col('pn')).alias('gap'),(pl.col('k').cast(pl.Int32)-pl.col('c3')).alias('c2'),(pl.col('cs')/pl.col('k').clip(lower_bound=1)).alias('avg')).sort('s1','k')
    cols=DL.FE+['k','pk','pn','cs','tail','gap','n','c3','c2','avg']
    X=T.select(cols).to_numpy().astype(np.float32)
    T=T.with_columns(pl.Series('u',model.predict(X,num_threads=4)))
    K=T.sort(['s1','u','k'],descending=[False,True,False]).group_by('s1',maintain_order=True).first().select('s1',pl.col('k').alias('kb'))
    return W.join(K,on='s1').filter(pl.col('k')<=pl.col('kb')).select(pl.col('s1').alias('s1_idx'),pl.col('qid').alias('cand_idx'))

def load_val():
    cache=HERE/'data/val_decoder.parquet'
    if cache.exists():return pl.read_parquet(cache)
    P=pl.read_parquet(HERE/'data/val_pairs.parquet').rename({'p2':'p'})
    C=pl.read_parquet(ROOT/'matching/prod_v1/data/preds_train.parquet',columns=['s1_idx','cand_idx','p2']).rename({'p2':'q'})
    log('building full-universe competition',C.height)
    P=add_comp(P,C)
    I=pl.read_parquet(f'{EB}/ids/train_s23.parquet',columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
    P=P.join(I.select('cand_idx',pl.col('entity_id').str.slice(0,2).alias('src')),on='cand_idx')
    P.write_parquet(cache);log('cached decoder input',P.height)
    return P

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='baseline');a=ap.parse_args()
    P=load_val();M=pl.read_parquet(HERE/'data/baseline_rows.parquet').drop('k','t','f')
    model=lgb.Booster(model_file=str(ROOT/'matching/prod_v2b/models/abc_rob_cv2_all_p2_R10c_m0.0.txt'))
    base=pl.read_parquet(HERE/'data/baseline_rows.parquet');base_tune=base.filter(pl.col('eval_split')=='tune')['f'].mean()
    if a.model=='baseline':
        S=decide_fast(P,model)
        expected=P.filter(pl.col('selected')).select('s1_idx','cand_idx')
        extra=S.join(expected,on=['s1_idx','cand_idx'],how='anti').height
        missed=expected.join(S,on=['s1_idx','cand_idx'],how='anti').height
        R=row_scores(S.join(P.select('s1_idx','cand_idx','label'),on=['s1_idx','cand_idx']),M)
        log('REPRO',extra,missed,float(R['f'].mean()))
        json.dump(dict(extra=extra,missed=missed,macro_f05=float(R['f'].mean())),open(HERE/'logs/decoder_reproduction.json','w'),indent=2)
        assert extra==missed==0,'Vectorized decoder must match shipped selections exactly'
        return
    proof=json.load(open(HERE/'logs/decoder_reproduction.json'));assert proof['extra']==proof['missed']==0
    C=pl.read_parquet(HERE/f'data/pred_{a.model}.parquet')
    P=P.join(C,on=['s1_idx','cand_idx'],how='left').with_columns(pl.col('prob').fill_null(pl.col('p')))
    p0=P['p'].to_numpy();pn=P['prob'].to_numpy();results=[]
    for alpha in [0.5,1.]:
        pr=expit((1-alpha)*logit(np.clip(p0,1e-10,1-1e-7))+alpha*logit(np.clip(pn,1e-10,1-1e-7)))
        # Outside the band remain bit-identical, including 0/1 probabilities.
        pr=np.where((p0>0.001)&(p0<0.999),pr,p0)
        Q=P.with_columns(pl.Series('p',pr))
        S=decide_fast(Q,model)
        R=row_scores(S.join(P.select('s1_idx','cand_idx','label'),on=['s1_idx','cand_idx']),M)
        score=float(R.filter(pl.col('eval_split')=='tune')['f'].mean())
        tag=f'{a.model}_{alpha}'
        R.write_parquet(HERE/f'data/set_rows_{tag}.parquet')
        S.write_parquet(HERE/f'data/set_selected_{tag}.parquet')
        results.append(dict(alpha=alpha,macro_f05=score,delta=score-base_tune,tag=tag))
        log('TUNE set',a.model,results[-1])
    json.dump(dict(name=a.model,baseline=base_tune,best=max(results,key=lambda x:x['macro_f05']),grid=results),open(HERE/f'logs/tune_set_{a.model}.json','w'),indent=2)
if __name__=='__main__':main()
