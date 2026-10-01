"""Build a compact, reproducible hard-band dataset and baseline error audit."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS']:
    os.environ[k]='4'
import sys, json, time
from pathlib import Path
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'matching/prod_v2b'))
from pv2b.common import D, FD, EB
from pv2b.feats import S1_FEATS
from pv2b.evalx import truth_tables
from pv1.decide import row_scores
from score import load_id_lists
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
HERE.joinpath('data').mkdir(exist_ok=True)
HERE.joinpath('models').mkdir(exist_ok=True)
HERE.joinpath('logs').mkdir(exist_ok=True)
P=pl.read_parquet(f'{D}/preds_abc_rob_cv2_all.parquet')
log('predictions',P.shape)
MV,MS=truth_tables()
M=pl.concat([MS.with_columns(pl.lit(False).alias('is_locked')),MV],how='diagonal_relaxed').sort('s1_idx')
M=M.with_columns(pl.when(pl.col('grp')=='sample').then(pl.lit('train')).when(pl.col('is_locked')).then(pl.lit('locked')).when(pl.col('s1_idx')%2==0).then(pl.lit('tune')).otherwise(pl.lit('report')).alias('eval_split'))
M.write_parquet(HERE/'data/entities.parquet')
# Baseline membership comes from the shipped validation file, not a retuned decoder.
base_dir=ROOT/'matching/prod_v2b/output/val_abc_rob_cv2_all_p2'
log('baseline files',[p.name for p in base_dir.iterdir()])
predpath=base_dir/'matching_results.tsv'
if not predpath.exists():
    opts=list(base_dir.glob('*.tsv'))
    predpath=next(p for p in opts if 'candidate' not in p.name)
g=load_id_lists(str(predpath))
B=pl.DataFrame([(s,c) for s,cc in g.items() for c in cc],schema=['s1_id','cand_id'],orient='row')
I1=pl.read_parquet(f'{EB}/ids/train_s1.parquet',columns=['entity_id']).with_row_index('s1_idx').rename({'entity_id':'s1_id'})
I2=pl.read_parquet(f'{EB}/ids/train_s23.parquet',columns=['entity_id']).with_row_index('cand_idx').rename({'entity_id':'cand_id'})
B=B.join(I1,on='s1_id').join(I2,on='cand_id').select(pl.col('s1_idx').cast(pl.Int32),pl.col('cand_idx').cast(pl.Int32),pl.lit(True).alias('selected'))
P=P.join(B,on=['s1_idx','cand_idx'],how='left').with_columns(pl.col('selected').fill_null(False))
PV=P.filter(pl.col('grp')!='sample')
PV.select('s1_idx','cand_idx','label','p2','selected').write_parquet(HERE/'data/val_pairs.parquet')
R=row_scores(PV.filter(pl.col('selected')),M.filter(pl.col('grp')!='sample'))
R.write_parquet(HERE/'data/baseline_rows.parquet')
log('baseline',R.group_by('eval_split').agg(pl.len(),pl.col('f').mean()))
P=P.with_columns(pl.col('p2').sum().over('s1_idx').alias('p2_sum'),(pl.col('p2')>0.5).sum().over('s1_idx').alias('p2_n50'),(pl.col('p2')>0.9).sum().over('s1_idx').alias('p2_n90'),pl.col('p2').max().over('s1_idx').alias('p2_max'),pl.col('p2').rank('ordinal',descending=True).over('s1_idx').alias('p2_rank'))
B=P.filter((pl.col('p2')>0.001)&(pl.col('p2')<0.999)).drop('p1_f0','p2_f0')
log('band',B.shape,B.group_by('grp').agg(pl.len(),pl.col('label').mean()))
basefeats=S1_FEATS['abc_rob']
Fs=[]
for part in sorted(Path(FD).glob('train_[0-9]*.parquet')):
    F=pl.read_parquet(part,columns=['s1_idx','cand_idx']+basefeats)
    C=B.join(F,on=['s1_idx','cand_idx'],how='inner')
    Fs.append(C)
    log('joined',part.name,C.height)
B=pl.concat(Fs).join(M.select('s1_idx','eval_split'),on='s1_idx').sort('s1_idx','cand_idx')
meta=['s1_idx','cand_idx','label','grp','fold','es','country','in_dense','in_dft','selected','eval_split']
features=[c for c in B.columns if c not in meta]
assert len(features)==len(set(features))
assert B.select(pl.struct('s1_idx','cand_idx').n_unique()).item()==B.height
B.write_parquet(HERE/'data/band.parquet')
json.dump(features,open(HERE/'data/features.json','w'),indent=2)
audit=[]
for nm,x in [('all',B.filter(pl.col('grp')!='sample')),('empty_address',B.filter((pl.col('grp')!='sample')&(pl.col('addr_empty2')>0))),('shared_name',B.filter((pl.col('grp')!='sample')&(pl.col('name_freq_s1_1')>1))),('name_unexplained',B.filter((pl.col('grp')!='sample')&(pl.col('x1_name_explained')<0.5)))]:
    audit.append(dict(slice=nm,band_pairs=x.height,tp=x.filter(pl.col('selected')&(pl.col('label')==1)).height,fp=x.filter(pl.col('selected')&(pl.col('label')==0)).height,fn=x.filter(~pl.col('selected')&(pl.col('label')==1)).height))
json.dump(audit,open(HERE/'logs/error_audit.json','w'),indent=2)
log('DONE',B.shape,'features',len(features),audit)
