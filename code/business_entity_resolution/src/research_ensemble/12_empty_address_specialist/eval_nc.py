import os
os.environ['OMP_WAIT_POLICY']='PASSIVE'
from eval_specialist import *

def nc_base(kind):
 p=W/'matching/v2shash/round_c/data'/f'{"ordinary" if kind=="val" else "density"}_blend_e0.25_selected.parquet'
 sel=pl.read_parquet(p).select(K)
 rows=pl.read_parquet(V/('seed_consensus_rows.parquet' if kind=='val' else 'density_seed_consensus_rows.parquet')).drop('k','t','f')
 gt=pl.read_parquet(W/'matching/variance_study_20260926/val_truth.parquet').select(K).with_columns(pl.lit(1).alias('yy'))
 agg=sel.join(gt,on=K,how='left').group_by('s1_idx').agg(pl.len().alias('k'),pl.col('yy').fill_null(0).sum().alias('t'))
 rows=rows.join(agg,on='s1_idx',how='left').with_columns(pl.col('k').fill_null(0),pl.col('t').fill_null(0))
 rows=rows.with_columns(pl.when(pl.col('m')==0).then((pl.col('k')==0).cast(pl.Float64)).when(pl.col('k')==0).then(0.).otherwise(1.25*pl.col('t')/(.25*pl.col('m')+pl.col('k'))).alias('f'))
 d=pl.read_parquet(O/f'{kind}_empty.parquet').drop('selected','owned','other_owned','owner','k','t','m','f')
 d=d.join(sel.with_columns(pl.lit(True).alias('selected')),on=K,how='left').with_columns(pl.col('selected').fill_null(False));d=d.join(sel.select('cand_idx',pl.col('s1_idx').alias('owner')),on='cand_idx',how='left').with_columns(pl.col('owner').is_not_null().alias('owned'));d=d.join(rows.select('s1_idx','k','t','m','f'),on='s1_idx')
 pred=pl.read_parquet(O/f'specialist_{kind}_scores.parquet').select(K+[pl.col('delta_61').alias('mean_delta')]);d=d.join(pred,on=K,how='left')
 return d,rows,sel
if __name__=='__main__':
 import sys
 kind=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'nc_specialist';d,rows,sel=nc_base(kind);cfg=json.loads((O/('nc_robust_frozen.json' if variant=='nc_robust' else 'specialist_frozen.json')).read_text());a=add(corrected(d,cfg['alpha']),cfg);reports={}
 for cohort in ['all','tune','report','locked']:
  rr=rows if cohort=='all' else rows.filter(pl.col('eval_split')==cohort);aa=a.join(rr.select('s1_idx'),on='s1_idx',how='semi');rep,r=describe(aa,rr,True);reports[cohort]=rep
 a.write_parquet(O/f'{kind}_{variant}_additions.parquet');newsel=pl.concat([sel,a.select(K)]);assert newsel.height==newsel['cand_idx'].n_unique();newsel.write_parquet(O/f'{kind}_{variant}_selected.parquet');(O/f'{kind}_{variant}_report.json').write_text(json.dumps(dict(config=cfg,reports=reports),indent=2));print(json.dumps(reports,indent=2))
