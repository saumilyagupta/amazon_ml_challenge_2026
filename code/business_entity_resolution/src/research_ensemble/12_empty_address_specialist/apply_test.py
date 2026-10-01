"""Apply the frozen NC-robust addition policy to US/India only, retaining all baseline owners."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ['OMP_WAIT_POLICY']='PASSIVE'
import polars as pl,numpy as np,json,time,hashlib
from scipy.special import expit,logit
from pathlib import Path
O=Path(__file__).resolve().parent;W=O.parents[2];K=['s1_idx','cand_idx']
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
C=W/'internal_eval/agents/B_archetypes/cache';R=W/'matching/vr2_release_20260926';NC=W/'matching/v2shash/round_c/test_release/data/test_selected.parquet';cfg=json.loads((O/'nc_robust_frozen.json').read_text());assert cfg['alpha']==1. and cfg['threshold']==.8
base=pl.read_parquet(NC).select(K);assert base.height==base['cand_idx'].n_unique()
ids=pl.read_parquet(W/'blocking/embedding/full/ids/test_s1.parquet',columns=['entity_id','country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32));usin=ids.filter(pl.col('country').is_in(['US','India'])).select('s1_idx','country')
empty=pl.scan_parquet(C/'views_test_rec.parquet').filter(pl.col('a_empty')).select('cand_idx')
# Complete eligible candidate universe. Anchor p<=.03 cannot pass because no specialist was fitted/applied there.
d=pl.scan_parquet(W/'matching/ensemble_v1/data/test_v3anchor.parquet').filter(pl.col('p')>.03).select(K+[pl.col('p').alias('p_anchor')]).join(empty,on='cand_idx',how='inner').join(usin.lazy(),on='s1_idx').collect();log('eligible empty anchor candidates',d.height)
pred=pl.read_parquet(O/'specialist_test_scores.parquet').select(K+['p2','delta_61']);d=d.join(pred,on=K,how='left');check=d.filter(pl.col('delta_61').is_not_null()).select((pl.col('p2')-pl.col('p_anchor')).abs().max()).item();assert check==0.
d=d.with_columns(pl.col('delta_61').is_not_null().alias('model_scored'),pl.col('delta_61').fill_null(0.)).drop('p2');newp=expit(logit(np.clip(d['p_anchor'].to_numpy().astype(float),1e-6,1-1e-6))+cfg['alpha']*d['delta_61'].to_numpy());d=d.with_columns(pl.Series('p_specialist',newp))
comp=pl.scan_parquet(R/'data/test_comp.parquet').select(K+['p_other']).join(d.select(K).lazy(),on=K,how='semi').collect();d=d.join(comp,on=K,how='left');assert d['p_other'].null_count()==0
counts=base.group_by('s1_idx').len('k');d=d.join(base.select('cand_idx',pl.col('s1_idx').alias('owner')),on='cand_idx',how='left').join(counts,on='s1_idx',how='left').with_columns(pl.col('k').fill_null(0),pl.col('owner').is_not_null().alias('owned'));d=d.with_columns((~pl.col('owned')&(pl.col('k')>0)&(pl.col('p_specialist')>=pl.col('p_other'))).alias('policy_eligible'))
a=d.filter(pl.col('policy_eligible')&(pl.col('p_specialist')>=cfg['threshold'])).sort(['cand_idx','p_specialist','s1_idx'],descending=[False,True,False]).unique('cand_idx',keep='first');assert a.join(base.select('cand_idx'),on='cand_idx',how='semi').height==0;assert set(a['country'].unique().to_list())<=set(['US','India'])
a.write_parquet(O/'test_nc_robust_additions.parquet');sel=pl.concat([base,a.select(K)]);assert sel.height==sel['cand_idx'].n_unique();sel.write_parquet(O/'test_nc_robust_selected.parquet')
d=d.join(a.select(K).with_columns(pl.lit(True).alias('added')),on=K,how='left').with_columns(pl.col('added').fill_null(False));d.select(K+['country','p_anchor','delta_61','p_specialist','p_other','model_scored','owned','policy_eligible','added']).write_parquet(O/'test_nc_robust_scores.parquet')
report=dict(config=cfg,scope='US/India only; France identical to rawNC; addition-only; no existing owner reassigned',baseline_pairs=base.height,final_pairs=sel.height,added_pairs=a.height,changed_queries=a['s1_idx'].n_unique(),added_by_country=a.group_by('country').agg(pl.len().alias('pairs'),pl.col('s1_idx').n_unique().alias('queries'),pl.col('p_specialist').min().alias('min_p'),pl.col('p_specialist').mean().alias('mean_p')).to_dicts(),score_rows=d.height,model_scored_rows=int(d['model_scored'].sum()),unscored_additions=int((~a['model_scored']).sum()),one_owner_violations=sel.height-sel['cand_idx'].n_unique(),inference_anchor_max_abs_difference=check,model_sha256=hashlib.sha256((O/'specialist_61.txt').read_bytes()).hexdigest(),features_sha256=hashlib.sha256((O/'specialist_features.json').read_bytes()).hexdigest(),probability_policy_note='Scores contain all eligible-domain counterfactual residual probabilities plus an added mask. Selection applies only additions. Use the added mask for a strictly minimal changed-probability diagnostic; broad residual band changes are not the final decision policy.')
(O/'test_nc_robust_audit.json').write_text(json.dumps(report,indent=2));log(report)
