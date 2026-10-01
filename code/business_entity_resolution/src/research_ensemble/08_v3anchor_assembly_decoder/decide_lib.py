"""Decision layer: pair probabilities -> per-S1 match sets (macro F0.5).  Reusable by the production matcher.
Input: polars DataFrame with columns s1, qid, src ('S2'/'S3'), p  -- ALL scored candidate pairs of the universe
(every S1 competing for a record must be present, otherwise the per-record argmax is wrong).
Algorithm (validated in work/research/postproc, see 08_final_*.log):
  1. exclusivity: keep a pair only if it is the argmax-p S1 of its S2/S3 record (ties -> ordinal rank).
  2. S1-level aggregates over the surviving pairs -> LightGBM singleton classifier P0 and Poisson count model C.
  3. if P0 > TAU: predict empty; else k = round(C + SLACK); predict the top-k surviving pairs with p > T.
Defaults (tuned on dev half A): T=0.55, SLACK=-0.25, TAU=0.6.
"""
import numpy as np, polars as pl, lightgbm as lgb
FE=['pmax','psum','n50','n20','n80','ncand_best','p2nd','n50_s2','n50_s3','pmax_all','ncand_all']
def add_qrank(U): return U.with_columns(pl.col('p').rank('ordinal',descending=True).over('qid').alias('qr'))
def s1_features(U):
    """U must already have 'qr' computed over the FULL universe; restrict to the S1s of interest afterwards."""
    Aq=U.filter(pl.col('qr')==1)
    f=Aq.group_by('s1').agg([pl.col('p').max().alias('pmax'),pl.col('p').sum().alias('psum'),(pl.col('p')>0.5).sum().alias('n50'),
        (pl.col('p')>0.2).sum().alias('n20'),(pl.col('p')>0.8).sum().alias('n80'),pl.len().alias('ncand_best'),
        pl.col('p').top_k(2).min().alias('p2nd'),
        ((pl.col('src')=='S2')&(pl.col('p')>0.5)).sum().alias('n50_s2'),((pl.col('src')=='S3')&(pl.col('p')>0.5)).sum().alias('n50_s3')])
    g=U.group_by('s1').agg([pl.col('p').max().alias('pmax_all'),pl.len().alias('ncand_all')])
    return f.join(g,on='s1',how='full',coalesce=True).fill_null(0)
def fit_s1_models(F,n_true,threads=8):
    """F: s1_features on OUT-OF-FOLD pair scores of training S1s; n_true: array of true match counts (0 = singleton)."""
    X=F.select(FE).to_numpy()
    ms=lgb.train(dict(objective='binary',learning_rate=0.05,num_leaves=31,verbose=-1,num_threads=threads),lgb.Dataset(X,(n_true==0).astype(int)),300)
    mc=lgb.train(dict(objective='poisson',learning_rate=0.05,num_leaves=31,verbose=-1,num_threads=threads),lgb.Dataset(X,n_true),300)
    return ms,mc
def decide(U,ms,mc,T=0.55,SLACK=-0.25,TAU=0.6,s1_subset=None):
    """Returns dict s1 -> set(qid). S1s absent from the output must be written as empty rows."""
    if 'qr' not in U.columns: U=add_qrank(U)
    if s1_subset is not None: U=U.filter(pl.col('s1').is_in(list(s1_subset)))
    F=s1_features(U); X=F.select(FE).to_numpy()
    P0=dict(zip(F['s1'].to_list(),ms.predict(X))); C=dict(zip(F['s1'].to_list(),mc.predict(X)))
    out={}
    W=U.filter(pl.col('qr')==1).sort(['s1','p'],descending=[False,True])
    for s1,qids,ps in W.group_by('s1',maintain_order=True).agg('qid','p').iter_rows():
        if P0[s1]>TAU: out[s1]=set(); continue
        k=max(int(np.round(C[s1]+SLACK)),0)
        out[s1]=set(q for q,p in zip(qids[:k],ps[:k]) if p>T)
    return out
def decide_threshold(U,T=0.65):
    """Fallback R1: argmax-per-record + global threshold (no S1-level models)."""
    if 'qr' not in U.columns: U=add_qrank(U)
    out={}
    for s1,qids in U.filter((pl.col('qr')==1)&(pl.col('p')>T)).group_by('s1').agg('qid').iter_rows(): out[s1]=set(qids)
    return out

# ---------------- R10c: learned set decoder (dev numbers: 10_verify_lib.log) ----------------
# NOTE: deliberately no q-side margin feature: it needs the competing S1 rows, which the OOF training table lacks;
# p (stage 2) already contains q-margin information.
KMAX=10
def prefix_table(U,gt=None):
    """U: s1,qid,src,p,qr (qr over the full universe). Returns X (float32), y (realised F0.5 per prefix or None), keys [(s1,k)], lists {s1:[qid sorted by p]}."""
    from score import f05  # work/common/score.py must be on sys.path
    F=s1_features(U); Fd={r[0]:r[1:] for r in F.select(['s1']+FE).iter_rows()}
    U=U.with_columns((pl.col('src')=='S3').cast(pl.Int8).alias('is3'))
    W=U.filter((pl.col('qr')==1)&(pl.col('p')>0.02)).sort(['s1','p'],descending=[False,True]).group_by('s1',maintain_order=True).agg('qid','p','is3')
    X=[];y=[];key=[];lists={}
    for s1,qids,ps,is3 in W.iter_rows():
        lists[s1]=qids; ps=np.array(ps); n=len(ps); cs=np.concatenate([[0],np.cumsum(ps)])
        c3=np.concatenate([[0],np.cumsum(is3)])
        for k in range(0,min(n,KMAX)+1):
            pk=ps[k-1] if k>0 else 1.0; pn=ps[k] if k<n else 0.0
            X.append(list(Fd[s1])+[k,pk,pn,cs[k],cs[n]-cs[k],pk-pn,n,c3[k],k-c3[k],cs[k]/max(k,1)]); key.append((s1,k))
            if gt is not None: y.append(f05(set(qids[:k]),gt[s1]))
    return np.array(X,np.float32),(np.array(y,np.float32) if gt is not None else None),key,lists
def fit_set_decoder(U_oof,gt,threads=8):
    """U_oof: OUT-OF-FOLD scored universe of training S1s (qr computed over the full universe); gt: s1 -> set of true ids."""
    X,y,_,_=prefix_table(U_oof,gt)
    return lgb.train(dict(objective='regression',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,verbose=-1,num_threads=threads),lgb.Dataset(X,y),600)
def decide_set(U,model):
    """Returns s1 -> set(qid); S1s with no surviving candidate are absent (= empty prediction)."""
    if 'qr' not in U.columns: U=add_qrank(U)
    X,_,key,lists=prefix_table(U); pr=model.predict(X); best={}
    for (s1,k),v in zip(key,pr):
        if s1 not in best or v>best[s1][1]: best[s1]=(k,v)
    return {s1:set(lists[s1][:k]) for s1,(k,v) in best.items()}
