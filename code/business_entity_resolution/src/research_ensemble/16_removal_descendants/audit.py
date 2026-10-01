"""Independent audit of removal candidates. Never treats test estimates as labels.

Recomputes exact ordinary-validation scores for the UNION of changes, avoiding
double-counting the original US rule when stacking the extended street rule.
Uses Jeffreys Gamma-Poisson rate posteriors for mirror sensitivity, including
nonzero uncertainty when a labelled cell contains zero true pairs.
"""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import json
from pathlib import Path
import numpy as np
import polars as pl

W = Path('/workspace/saumilya/amazon-ml/work')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
S = W / 'winning_strategy_20260926/agents/S1_mirror_sweep/out'
IT = W / 'matching/iterate_20260926'
KEY = ['s1_idx', 'cand_idx']
NTEST = 1732544
NV = {'US': 132373, 'India': 88357}
NT = {'US': 663106, 'India': 809986}
RNG = np.random.default_rng(260926)

def f(tp, k, m):
    tp, k, m = np.broadcast_arrays(tp, k, m)
    return np.where(m == 0, (k == 0).astype(float), np.divide(5*tp, 4*k+m, out=np.zeros(tp.shape, dtype=float), where=(4*k+m)>0))

def nogroupempty(rem, selected):
    k = selected.group_by('s1_idx').len().rename({'len': 'nsel'})
    r = rem.group_by('s1_idx').len().rename({'len': 'nrem'})
    keep = r.join(k, on='s1_idx').filter(pl.col('nrem') < pl.col('nsel')).select('s1_idx')
    return rem.join(keep, on='s1_idx', how='semi')

def original_rule(split, selected):
    fam = pl.read_parquet(W/f'internal_eval/data/pc_{split}_fam.parquet')
    rem = fam.filter((pl.col('country')=='US') & (pl.col('side')=='+') & pl.col('fam').is_in(['LEG12','LEG3'])).select(KEY).unique()
    k = selected.group_by('s1_idx').len()
    return rem.join(selected.select(KEY), on=KEY).join(k.filter(pl.col('len')>1).select('s1_idx'), on='s1_idx')

def add_pattern(p, split):
    pat = pl.read_parquet(W/f'internal_eval/data/pc_{split}_fam.parquet').filter(pl.col('fam').is_in(['LEG12','LEG3','CTAG'])).select(KEY).unique().with_columns(pl.lit(1).alias('pat'))
    return p.join(pat, on=KEY, how='left').with_columns(pl.col('pat').fill_null(0))

def extended(p, broad=False):
    c1 = ((pl.col('country')=='US') & (((pl.col('fam')=='LEGADD') & pl.col('bucket').is_in(['D12','D3'])) | ((pl.col('fam')=='LEGSW') & (pl.col('bucket')=='D3'))))
    c2 = ((pl.col('country')=='US') & (((pl.col('fam')=='LEGSW') & (pl.col('bucket')=='D12')) | ((pl.col('fam')=='SWDW') & (pl.col('bucket')=='D3')) | ((pl.col('fam')=='ADDO') & (pl.col('bucket')=='D12')))) | ((pl.col('country')=='India') & pl.col('fam').is_in(['LEGADD','LEGSW']) & (pl.col('bucket')=='D3') & (pl.col('pat')==1))
    return p.filter((pl.col('side')=='+') & (c1 | c2 if broad else c1)).select(KEY).unique()

def exact_val(rows, rem, selected, gt):
    labelled = rem.join(gt.with_columns(pl.lit(1).alias('truth')), on=KEY, how='left').with_columns(pl.col('truth').fill_null(0))
    counts = labelled.group_by('s1_idx').agg(pl.len().alias('r'), pl.col('truth').sum().alias('rt'))
    rr = rows.join(counts, on='s1_idx', how='left').with_columns(pl.col('r').fill_null(0), pl.col('rt').fill_null(0))
    base = f(rr['tp'].to_numpy(),rr['k'].to_numpy(),rr['m'].to_numpy())
    after = f((rr['tp']-rr['rt']).to_numpy(),(rr['k']-rr['r']).to_numpy(),rr['m'].to_numpy())
    assert np.max(np.abs(base-rr['f'].to_numpy())) < 1e-10
    rr=rr.with_columns(pl.Series('after',after),pl.Series('delta',after-base))
    stats = {'removed_pairs':rem.height, 'removed_true':int(labelled['truth'].sum()), 'removed_false':int(rem.height-labelled['truth'].sum()), 'changed_s1':counts.height, 'new_empty':int(((rr['k']>0)&(rr['k']==rr['r'])).sum())}
    for split, x in [('all',rr),('tune',rr.filter(pl.col('eval_split')=='tune')),('report',rr.filter(pl.col('eval_split')=='report')),('locked',rr.filter(pl.col('is_locked')))]:
        if not x.height: continue
        d=x['delta'].to_numpy(); se=float(d.std(ddof=1)/np.sqrt(len(d)))
        stats[split]={'n':x.height,'baseline':float(x['f'].mean()),'score':float(x['after'].mean()),'delta':float(d.mean()),'paired_normal_ci95':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)]}
    return stats,rr

def main():
    sel_v=pl.read_parquet(IT/'empty_address/val_nc_robust_selected.parquet').select(KEY).unique()
    sel_t=pl.read_parquet(IT/'submissions/nc_specialist_legal_fr/selected.parquet').select(KEY).unique()
    rows=pl.read_parquet(IT/'results/specialist_nc_ordinary_rows.parquet')
    print('eval_split values',rows['eval_split'].value_counts().to_dicts(),flush=True)
    gt=pl.read_parquet(W/'internal_eval/data/val_gt_pairs.parquet').select(KEY).unique()
    pv=add_pattern(pl.read_parquet(S/'pairs_val.parquet').filter(pl.col('acc')==1),'val')
    pt=add_pattern(pl.scan_parquet(S/'pairs_test.parquet').filter((pl.col('acc')==1)&pl.col('country').is_in(['US','India'])).collect(),'test')
    a_v=original_rule('val',sel_v)
    a_t=original_rule('test',sel_t)
    old_a=pl.read_parquet(W/'winning_strategy_20260926/agents/O3_metric_transfer/out/p3_remove_US_LEG.parquet').select(KEY)
    assert a_t.height==4294 and a_t.join(old_a,on=KEY,how='anti').is_empty()
    a_v.write_parquet(OUT/'A_val_removals.parquet')
    a_t.write_parquet(OUT/'A_test_removals.parquet')
    stats={}
    base_selected=sel_t.join(a_t,on=KEY,how='anti')
    for name,broad in [('A_C1',False),('A_C2',True)]:
        extra_t=nogroupempty(extended(pt,broad).join(a_t,on=KEY,how='anti'),base_selected)
        extra_v=nogroupempty(extended(pv,broad).join(a_v,on=KEY,how='anti'),sel_v.join(a_v,on=KEY,how='anti'))
        total_v=pl.concat([a_v,extra_v]).unique()
        total_t=pl.concat([a_t,extra_t]).unique()
        extra_t.write_parquet(OUT/f'{name}_extra_test.parquet')
        total_t.write_parquet(OUT/f'{name}_test_removals.parquet')
        total_v.write_parquet(OUT/f'{name}_val_removals.parquet')
        st,rr=exact_val(rows,total_v,sel_v,gt)
        rr.write_parquet(OUT/f'{name}_val_rows.parquet')
        st.update(test_removed=total_t.height,test_extra=extra_t.height)
        stats[name]=st
    st,rr=exact_val(rows,a_v,sel_v,gt)
    rr.write_parquet(OUT/'A_val_rows.parquet');st['test_removed']=a_t.height;stats['A']=st
    # Rate transfer is an assumption. Output both accepted-minus and all-candidate
    # upper sensitivity; sampling intervals exclude systematic domain shift.
    groups=pt.filter(pl.col('bucket').is_in(['D12','D3'])).select('country','fam','bucket','pat').unique().sort('country','fam','bucket','pat')
    estimates=[]
    for g in groups.iter_rows(named=True):
        filt=pl.lit(True)
        for k,v in g.items(): filt &= pl.col(k)==v
        v=pv.filter(filt);t=pt.filter(filt)
        vp=v.filter(pl.col('side')=='+');vm=v.filter(pl.col('side')=='-')
        tp=t.filter(pl.col('side')=='+');tm=t.filter(pl.col('side')=='-')
        if not tp.height: continue
        y=int(vp['label'].sum()); nminus=vm.height
        # Jeffreys priors on Poisson intensities; never zero-width at y=0.
        direct=RNG.gamma(y+0.5,NT[g['country']]/NV[g['country']],12000)/tp.height
        mirror=RNG.gamma(tm.height+0.5,1,12000)*RNG.gamma(y+0.5,1,12000)/RNG.gamma(nminus+0.5,1,12000)/tp.height
        estimates.append(g|{'val_plus':vp.height,'val_plus_true':y,'val_minus':vm.height,'test_plus':tp.height,'test_minus':tm.height,'q_direct_median':float(np.quantile(np.clip(direct,0,1),.5)),'q_direct_hi95':float(np.quantile(np.clip(direct,0,1),.95)),'q_mirror_median':float(np.quantile(np.clip(mirror,0,1),.5)),'q_mirror_hi95':float(np.quantile(np.clip(mirror,0,1),.95))})
    pl.DataFrame(estimates).write_csv(OUT/'mirror_jeffreys.tsv',separator='\t')
    json.dump(stats,open(OUT/'validation.json','w'),indent=2)
    print(json.dumps(stats,indent=2),flush=True)
    print('All validation numbers measured from labels; mirror estimates are not test scores.',flush=True)

if __name__=='__main__':main()
