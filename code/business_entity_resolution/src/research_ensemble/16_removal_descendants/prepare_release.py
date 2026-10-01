"""Prepare isolated, reproducible candidate files and honest score sensitivity.

R extends A using ONLY LEGADD at generator offsets in the strict-street table.
Sparse LEGSW, ADDO, SWDW and India extensions are excluded. France #9 stays an
independent option; its unknown precision is shown as a scenario, never a label.
"""
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import polars as pl
from audit import W, ROOT, OUT, S, IT, KEY, NTEST, f, exact_val, original_rule, nogroupempty, add_pattern

BASE=IT/'submissions/nc_specialist_legal_fr/matching_results.tsv'
FR9=W/'matching/ensemble_v7/build/spec_swapanch_FR/output/matching_results.tsv'
VALROWS=IT/'results/specialist_nc_ordinary_rows.parquet'

def digest(path,algorithm='sha256'):
    with open(path,'rb') as fh:return hashlib.file_digest(fh,algorithm).hexdigest()

def make_files(remaps):
    release=ROOT/'release'; release.mkdir(exist_ok=True)
    handles={}; stats={}
    for name,rmap in remaps.items():
        folder=release/name;folder.mkdir(exist_ok=True)
        handles[name]=open(folder/'matching_results.tsv','w')
        stats[name]={'removed_pairs':0,'changed_s1':0,'new_empty':0,'rows':0,'pairs':0,'per_country':{c:{'removed':0,'changed_s1':0,'pairs':0,'empty':0,'rows':0} for c in ['US','India','France']}}
    s1=pl.read_parquet(W/'blocking/embedding/full/ids/test_s1.parquet',columns=['entity_id','country'])
    countries=dict(s1.iter_rows())
    with open(BASE) as src:
        header=next(src)
        for h in handles.values():h.write(header)
        for line in src:
            q,values=line.rstrip('\n').split('\t');old=set(values.split(',')) if values else set();c=countries[q]
            for name,rmap in remaps.items():
                st=stats[name];cs=st['per_country'][c];rm=rmap.get(q,set())
                assert rm<=old,(name,q)
                if rm:
                    new=old-rm;handles[name].write(q+'\t'+','.join(sorted(new))+'\n')
                    st['removed_pairs']+=len(rm);st['changed_s1']+=1;st['new_empty']+=int(not new)
                    cs['removed']+=len(rm);cs['changed_s1']+=1
                else:
                    new=old;handles[name].write(line)
                st['rows']+=1;st['pairs']+=len(new);cs['pairs']+=len(new);cs['empty']+=int(not new);cs['rows']+=1
    for name,h in handles.items():
        h.close();folder=release/name
        cp=folder/'candidate_pairs.tsv'
        if not cp.exists():cp.symlink_to((IT/'submissions/nc_specialist_legal_fr/candidate_pairs.tsv').resolve())
        st=stats[name]
        st.update(sha256=digest(folder/'matching_results.tsv'),md5=digest(folder/'matching_results.tsv','md5'),baseline_sha256=digest(BASE),uploaded=False)
        assert st['rows']==NTEST
        for cs in st['per_country'].values():cs['matches_per_s1']=cs['pairs']/cs['rows']
        json.dump(st,open(folder/'manifest.json','w'),indent=2)
    return stats

def expectation(k,qs,missed,kept_precision,correlated=False):
    """Exact expectation over removed truths, kept truths and empirical FN count.
    Correlated uses a shared uniform latent variable for all removals of an S1.
    This is sensitivity analysis, not a calibrated probability of test score.
    """
    r=len(qs);nkeep=k-r
    if correlated:
        cuts=sorted(set([0.,1.]+list(qs)));dist=np.zeros(r+1)
        for a,b in zip(cuts,cuts[1:]):dist[sum(q>(a+b)/2 for q in qs)]+=b-a
    else:
        dist=np.array([1.])
        for q in qs:dist=np.convolve(dist,[1-q,q])
    # Kept false pairs are rare; include their effect rather than assume perfect.
    dy=np.array([math.comb(nkeep,y)*kept_precision**y*(1-kept_precision)**(nkeep-y) for y in range(nkeep+1)])
    result=0.
    for t,pt in enumerate(dist):
        for y,py in enumerate(dy):
            for m0,pm in missed.items():
                m=t+y+m0
                result+=pt*py*pm*float(f(y,nkeep,m)-f(t+y,k,m))
    return result

def simulate(rem,selected,rows):
    kmap=dict(selected.group_by('s1_idx').len().iter_rows())
    ctx={};prec={}
    for c in ['US','India','France']:
        v=rows.filter(pl.col('country')==('US' if c=='France' else c))
        prec[c]=float(v['tp'].sum()/v['k'].sum())
        for key,g in v.group_by('k'):
            z=Counter((g['m']-g['tp']).to_list());ctx[c,key[0]]={x:n/g.height for x,n in z.items()}
    grouped=rem.group_by('s1_idx','country').agg('q','qhi','kind')
    out=[]; cache={}
    for frq in [0.09,0.5,0.9]:
        for stress in ['central','upper','double_upper']:
            for corr in [False,True]:
                sums=defaultdict(float)
                for s,c,qs,qhis,kinds in grouped.iter_rows():
                    pp=[]
                    for q,qhi,kind in zip(qs,qhis,kinds):
                        if kind=='FR9':pp.append(frq)
                        else:pp.append(min(1.,q if stress=='central' else qhi*(2 if stress=='double_upper' else 1)))
                    k=kmap[s];missed=ctx.get((c,k),{0:1.})
                    cachekey=(c,k,tuple(sorted(pp)),corr)
                    if cachekey not in cache:cache[cachekey]=expectation(k,pp,missed,prec[c],corr)
                    sums[c]+=cache[cachekey]/NTEST
                out.append({'fr_removed_ptrue':frq,'assumption':stress,'within_query_perfect_correlation':corr,'estimated_delta_lb':sum(sums.values()),'US_delta':sums.get('US',0.),'India_delta':sums.get('India',0.),'France_delta':sums.get('France',0.)})
    return out

def main():
    sel_v=pl.read_parquet(IT/'empty_address/val_nc_robust_selected.parquet').select(KEY)
    sel_t=pl.read_parquet(IT/'submissions/nc_specialist_legal_fr/selected.parquet').select(KEY)
    rows=pl.read_parquet(VALROWS)
    a_v=pl.read_parquet(OUT/'A_val_removals.parquet');a_t=pl.read_parquet(OUT/'A_test_removals.parquet')
    pv=add_pattern(pl.read_parquet(S/'pairs_val.parquet').filter(pl.col('acc')==1),'val')
    pt=add_pattern(pl.scan_parquet(S/'pairs_test.parquet').filter((pl.col('acc')==1)&(pl.col('country')=='US')).collect(),'test')
    cond=(pl.col('fam')=='LEGADD')&pl.col('bucket').is_in(['D12','D3'])&(pl.col('side')=='+')&(pl.col('country')=='US')
    r_t=nogroupempty(pt.filter(cond).select(KEY).join(a_t,on=KEY,how='anti'),sel_t.join(a_t,on=KEY,how='anti'))
    r_v=nogroupempty(pv.filter(cond).select(KEY).join(a_v,on=KEY,how='anti'),sel_v.join(a_v,on=KEY,how='anti'))
    assert r_t.height==809
    ar_t=pl.concat([a_t,r_t]).unique();ar_v=pl.concat([a_v,r_v]).unique()
    ar_t.write_parquet(OUT/'R_test_removals.parquet');ar_v.write_parquet(OUT/'R_val_removals.parquet')
    gt=pl.read_parquet(W/'internal_eval/data/val_gt_pairs.parquet').select(KEY)
    val,rr=exact_val(rows,ar_v,sel_v,gt);rr.write_parquet(OUT/'R_val_rows.parquet')
    json.dump(val,open(OUT/'R_validation.json','w'),indent=2)
    # Directly derive France #9 mask from the TSVs, independently of its recipe.
    frmap={};frrows=frremoved=0
    with open(BASE) as fi,open(FR9) as fj:
        assert next(fi)==next(fj)
        for old,new in zip(fi,fj,strict=True):
            q,a=old.rstrip('\n').split('\t');q2,b=new.rstrip('\n').split('\t');assert q==q2
            if old==new:continue
            aa=set(a.split(',')) if a else set();bb=set(b.split(',')) if b else set()
            assert bb<=aa and bb
            if aa!=bb:frmap[q]=aa-bb;frrows+=1;frremoved+=len(aa-bb)
    assert frrows==1440 and frremoved==1445
    i1=pl.read_parquet(W/'blocking/embedding/full/ids/test_s1.parquet',columns=['entity_id','country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    i2=pl.read_parquet(W/'blocking/embedding/full/ids/test_s23.parquet',columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
    def to_map(rem):
        mapped=rem.join(i1.rename({'entity_id':'s1_id'}),on='s1_idx').join(i2.rename({'entity_id':'cand_id'}),on='cand_idx')
        return {s:set(cs) for s,cs in mapped.group_by('s1_id').agg('cand_id').iter_rows()}
    amap=to_map(a_t);rmap=to_map(ar_t)
    assert set(frmap).isdisjoint(amap) and set(frmap).isdisjoint(rmap)
    manifests=make_files({'A_US_LEG':amap,'B_US_LEG_FR9':amap|frmap,'R_US_LEGADD':rmap,'R_US_LEGADD_FR9':rmap|frmap})
    assert manifests['A_US_LEG']['md5']=='e63f7fe3a188c67575747fdf2b88e148'
    assert manifests['B_US_LEG_FR9']['md5']=='3648573e946f586a54d9085d482990c4'
    # Save pair masks in IDs and numeric keys for later audit/reapplication.
    for name,m in [('A_US_LEG',amap),('B_US_LEG_FR9',amap|frmap),('R_US_LEGADD',rmap),('R_US_LEGADD_FR9',rmap|frmap)]:
        pl.DataFrame([(q,c) for q,cs in m.items() for c in sorted(cs)],schema=['s1_id','cand_id'],orient='row').write_csv(ROOT/'release'/name/'removed_pairs.tsv',separator='\t')
    # Main A prior: verifier's central ~0.146 and conservative ~0.22. Expose it.
    a=a_t.with_columns(pl.lit('US').alias('country'),pl.lit(.146).alias('q'),pl.lit(.22).alias('qhi'),pl.lit('A').alias('kind'))
    est=pl.read_csv(OUT/'mirror_jeffreys.tsv',separator='\t').with_columns(pl.max_horizontal('q_direct_median','q_mirror_median').alias('q'),pl.max_horizontal('q_direct_hi95','q_mirror_hi95').alias('qhi'))
    r=r_t.join(pt,on=KEY).join(est.select('country','fam','bucket','pat','q','qhi'),on=['country','fam','bucket','pat']).select(KEY+['country','q','qhi']).with_columns(pl.lit('R').alias('kind'))
    assert r.height==809
    fr=pl.DataFrame([(s,c) for s,cs in frmap.items() for c in cs],schema=['s1_id','cand_id'],orient='row').join(i1.rename({'entity_id':'s1_id'}),on='s1_id').join(i2.rename({'entity_id':'cand_id'}),on='cand_id').select(KEY+['country']).with_columns(pl.lit(.5).alias('q'),pl.lit(.9).alias('qhi'),pl.lit('FR9').alias('kind'))
    assert set(fr['country'])=={'France'}
    allrem=pl.concat([a,r,fr]);allrem.write_parquet(OUT/'valued_removals.parquet')
    simulations={}
    for name,rm in [('A_US_LEG',a),('B_US_LEG_FR9',pl.concat([a,fr])),('R_US_LEGADD',pl.concat([a,r])),('R_US_LEGADD_FR9',allrem)]:
        simulations[name]=simulate(rm,sel_t,rows)
    json.dump(simulations,open(OUT/'score_sensitivity.json','w'),indent=2)
    json.dump(manifests,open(OUT/'release_manifest.json','w'),indent=2)
    print('R exact validation',json.dumps(val),flush=True)
    print('RELEASE',json.dumps({k:{z:v[z] for z in ['removed_pairs','changed_s1','new_empty','md5']} for k,v in manifests.items()},indent=2),flush=True)
    for name,res in simulations.items():
        print(name,[x for x in res if x['fr_removed_ptrue']==.5 and not x['within_query_perfect_correlation']],flush=True)

if __name__=='__main__':
    import sys
    if '--simulate-only' in sys.argv:
        allrem=pl.read_parquet(OUT/'valued_removals.parquet')
        selected=pl.read_parquet(IT/'submissions/nc_specialist_legal_fr/selected.parquet').select(KEY)
        rows=pl.read_parquet(VALROWS)
        sims={}
        for name,kinds in [('A_US_LEG',['A']),('B_US_LEG_FR9',['A','FR9']),('R_US_LEGADD',['A','R']),('R_US_LEGADD_FR9',['A','R','FR9'])]:
            sims[name]=simulate(allrem.filter(pl.col('kind').is_in(kinds)),selected,rows)
            print(name,[x for x in sims[name] if x['fr_removed_ptrue']==.5 and not x['within_query_perfect_correlation']],flush=True)
        json.dump(sims,open(OUT/'score_sensitivity.json','w'),indent=2)
    else:main()
