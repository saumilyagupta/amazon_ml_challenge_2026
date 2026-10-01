"""Exploratory whole-query seed-agreement safeguard, not an independent confirmation."""
from evaluate import *
full,a,meta,g=inputs();v=veto();a=post(a,full,v);base=S.score(a,meta,g)
cfg=json.loads((HERE/'frozen_choice.json').read_text())['chosen'];assert cfg['alpha']>0
corr=pl.read_parquet(HERE/'data/residual_scores.parquet').filter(pl.col('grp')!='sample');model=lgb.Booster(model_file=str(E/'results/eval_ENS_v3anchor_R10c_m0.txt'))
q=meta.select('s1_idx');sels=[]
for seed in [11,29,47]:
 p=modified(full,corr,cfg['alpha'],cfg['scope'],seed);sel=post(decide_fast(p,model),p,v);sels.append(sel);sel.write_parquet(HERE/f'data/residual_seed_{seed}_selected.parquet')
 sig=sel.group_by('s1_idx').agg(pl.col('cand_idx').sort().cast(pl.String).str.join(',').alias(f's{seed}'))
 q=q.join(sig,on='s1_idx',how='left').with_columns(pl.col(f's{seed}').fill_null(''));log('seed signatures',seed)
allow=q.filter((pl.col('s11')==pl.col('s29'))&(pl.col('s11')==pl.col('s47'))).select('s1_idx')
sel=pl.concat([sels[0].join(allow,on='s1_idx',how='semi'),a.join(allow,on='s1_idx',how='anti')])
p=modified(full,corr,cfg['alpha'],cfg['scope']);p=pl.concat([p.join(allow,on='s1_idx',how='semi'),full.join(allow,on='s1_idx',how='anti')]);sel=post(sel,p,v)
r,rows=S.compare(sel,base,meta,g)
for cohort in ['tune','report','locked']:
 mi=meta.filter(pl.col('eval_split')==cohort);rb=base.join(mi.select('s1_idx'),on='s1_idx',how='semi');r[cohort],_=S.compare(sel,rb,mi,g)
r['exploratory_after_first_report']=True;r['no_policy_selection_on_report']=True;r['agreed_queries']=allow.height
save('seed_consensus.json',r);sel.write_parquet(HERE/'data/seed_consensus_selected.parquet');rows.write_parquet(HERE/'data/seed_consensus_rows.parquet');log('CONSENSUS',r['delta'],r['report']['delta'],r['report']['query_ci95'],r['perfect_queries_harmed'])
