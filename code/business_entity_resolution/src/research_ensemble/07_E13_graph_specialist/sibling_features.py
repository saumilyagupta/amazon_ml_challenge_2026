"""Label-free, leave-one-record-out corroboration from high-confidence siblings."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[k]='3'
from pathlib import Path
import time,json
import numpy as np
import polars as pl
from rapidfuzz import process,fuzz
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
def compute_features(B,P,R,R1):
    """P contains model scores, never labels; R/R1 are input-derived record views."""
    C=P.filter(pl.col('p2')>=0.99).sort('p2',descending=True).group_by('s1_idx').head(8).select('s1_idx',pl.col('cand_idx').alias('sib'),pl.col('p2').alias('sp'))
    Q=B.select('s1_idx','cand_idx')
    X=Q.join(C,on='s1_idx').filter(pl.col('cand_idx')!=pl.col('sib'))
    X=X.join(R.select(pl.col('erow').cast(pl.Int32).alias('cand_idx'),pl.col('src').alias('src_a'),*[pl.col(c).alias(c+'_a') for c in ['name_n','name_ns','name_core','name_raw']]),on='cand_idx')
    X=X.join(R.select(pl.col('erow').cast(pl.Int32).alias('sib'),pl.col('src').alias('src_b'),*[pl.col(c).alias(c+'_b') for c in ['name_n','name_ns','name_core','name_raw']]),on='sib')
    log('sibling edges',X.height)
    sims={}
    for c,fn in [('name_n',fuzz.ratio),('name_raw',fuzz.ratio),('name_core',fuzz.ratio),('name_ns',fuzz.ratio)]:
        sims[c]=process.cpdist(X[c+'_a'].to_list(),X[c+'_b'].to_list(),scorer=fn,workers=3,dtype=np.float32)/100
    X=X.with_columns([pl.Series('sim_'+c,v) for c,v in sims.items()])
    expr=[pl.len().cast(pl.Float32).alias('g_n99')]
    for c in sims:
        s=pl.col('sim_'+c);same=pl.col('src_a')==pl.col('src_b')
        expr.extend([s.max().alias('g_'+c+'_max'),pl.when(same).then(s).otherwise(-1).max().alias('g_'+c+'_same_src'),(s==1).sum().cast(pl.Float32).alias('g_'+c+'_exact_n')])
    expr.extend([(pl.col('sim_name_n')*pl.col('sp')).max().alias('g_name_weighted'),(pl.col('sim_name_n')>0.95).sum().cast(pl.Float32).alias('g_name_near_n')])
    G=X.group_by('s1_idx','cand_idx').agg(expr)
    O=Q.join(G,on=['s1_idx','cand_idx'],how='left').fill_null(-1)
    # Counts of full-name variants among all S1, preserving legal-form information.
    for col in ['name_raw','name_n','name_ns']:
        freq=R1.group_by('country',col).agg(pl.len().alias('freq'))
        F=R.select(pl.col('erow').cast(pl.Int32).alias('cand_idx'),'country',col).join(freq,on=['country',col],how='left').select('cand_idx',pl.col('freq').fill_null(0).cast(pl.Float32).log1p().alias('g_s1freq_'+col))
        O=O.join(F,on='cand_idx',how='left')
    return O
if __name__=='__main__':
    B=pl.read_parquet(HERE/'data/band.parquet')
    P=pl.read_parquet(ROOT/'matching/prod_v2b/data/preds_abc_rob_cv2_all.parquet',columns=['s1_idx','cand_idx','p2'])
    cols=['erow','country','src','name_n','name_ns','name_core','name_raw']
    R=pl.read_parquet(ROOT/'matching/prod_v1/data/rec_train_s23.parquet',columns=cols)
    R1=pl.read_parquet(ROOT/'matching/prod_v1/data/rec_train_s1.parquet',columns=cols)
    G=compute_features(B,P,R,R1)
    assert G.height==B.height
    assert G.select(pl.struct('s1_idx','cand_idx').n_unique()).item()==G.height
    B=B.join(G,on=['s1_idx','cand_idx'])
    B.write_parquet(HERE/'data/band_graph.parquet')
    F=json.load(open(HERE/'data/features.json'))+[c for c in G.columns if c not in ['s1_idx','cand_idx']]
    json.dump(F,open(HERE/'data/features_graph.json','w'),indent=2)
    log('DONE',B.shape)
