"""Read-only baseline/rule audit; writes only next to this script. CPU, four threads."""
import os
for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import sys, json, time, zlib, unicodedata, re, hashlib
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
W = HERE.parents[1]
M = W / 'matching'
E = M / 'ensemble_v1'
K = ['s1_idx', 'cand_idx']
D = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
sys.dont_write_bytecode = True
sys.path.insert(0, str(W / 'common'))
sys.path.insert(0, str(M / 'v3_postpass/src'))
from score import load_id_lists
import pp
from vx.common import ids, france_proxies

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False))

def norm(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()

def readsel(p):
    return pl.read_parquet(p, columns=K).with_columns(pl.col(K).cast(pl.Int32)).unique()

def bootstrap(d, groups=None, nboot=2000):
    """Exact empirical paired bootstrap; compress identical (delta, size) clusters."""
    if not len(d):
        return None
    if groups is None:
        values, counts = np.unique(d, return_counts=True)
        sizes = np.ones(len(values))
    else:
        _, inv = np.unique(groups, return_inverse=True)
        sums = np.bincount(inv, weights=d)
        ns = np.bincount(inv)
        values_sizes, counts = np.unique(np.column_stack([sums, ns]), axis=0, return_counts=True)
        values, sizes = values_sizes.T
    if np.all(values == 0):
        return [0., 0.]
    rng = np.random.default_rng(20260926)
    weights = rng.multinomial(int(counts.sum()), counts / counts.sum(), size=nboot)
    means = (weights @ values) / (weights @ sizes)
    return np.quantile(means, [.025, .975]).tolist()

def truth():
    gt = load_id_lists(W / 'splits/val_ground_truth.tsv')
    i1, i2 = ids('train', 's1'), ids('train', 's23')
    meta = i1.join(pl.DataFrame({'entity_id': list(gt), 'm': [len(v) for v in gt.values()]}), on='entity_id')
    raw = pl.read_csv(W.parent / 'student_resource/dataset/train/train_source1.tsv', separator='\t', quote_char=None, infer_schema_length=0)
    namecol = next(c for c in raw.columns if c.lower() in ('name', 'business_name', 'entity_name'))
    idcol = next(c for c in raw.columns if c.lower() in ('entity_id', 'id'))
    raw = raw.select(pl.col(idcol).alias('entity_id'), pl.col(namecol).alias('name'))
    meta = meta.join(raw, on='entity_id', how='left')
    keys = [c + ':' + (norm(n) or eid) for c, n, eid in meta.select('country', 'name', 'entity_id').iter_rows()]
    locked = set((M / 'prod_v1/data/locked_val_s1_ids.txt').read_text().split())
    meta = meta.with_columns(pl.Series('name_group', keys), pl.Series('fold', [zlib.crc32(('variance:' + x).encode()) % 5 for x in keys]), pl.col('entity_id').is_in(list(locked)).alias('locked')).sort('s1_idx')
    g = pl.DataFrame({'entity_id': list(gt), 'cid': [sorted(v) for v in gt.values()]}).explode('cid').drop_nulls()
    g = g.join(i1.select('s1_idx', 'entity_id'), on='entity_id').join(i2.select('cand_idx', pl.col('entity_id').alias('cid')), on='cid').select(K).with_columns(pl.lit(1).alias('label'))
    assert meta.height == 220730
    return meta, g

def score(sel, meta, g):
    labeled = sel.select(K).join(g, on=K, how='left').with_columns(pl.col('label').fill_null(0))
    agg = labeled.group_by('s1_idx').agg(pl.len().cast(pl.Int64).alias('k'), pl.col('label').sum().cast(pl.Int64).alias('t'))
    rows = meta.join(agg, on='s1_idx', how='left').with_columns(pl.col('k', 't').fill_null(0))
    rows = rows.with_columns(pl.when(pl.col('m') == 0).then((pl.col('k') == 0).cast(pl.Float64)).otherwise(5 * pl.col('t') / (4 * pl.col('k') + pl.col('m'))).alias('f')).sort('s1_idx')
    return rows

def compare(sel, base, meta, g):
    rows = score(sel, meta, g)
    d = rows['f'].to_numpy() - base['f'].to_numpy()
    out = dict(score=float(rows['f'].mean()), delta=float(d.mean()), query_ci95=bootstrap(d), name_cluster_ci95=bootstrap(d, rows['name_group'].to_numpy()), improved=int((d > 1e-12).sum()), harmed=int((d < -1e-12).sum()), perfect_queries_harmed=int(((base['f'].to_numpy() == 1) & (d < -1e-12)).sum()), tp=int(rows['t'].sum()), fp=int(rows['k'].sum()-rows['t'].sum()), slices={})
    masks = {'US': rows['country'].to_numpy() == 'US', 'India': rows['country'].to_numpy() == 'India', 'locked30k': rows['locked'].to_numpy(), 'truth_empty': rows['m'].to_numpy() == 0, 'truth_one': rows['m'].to_numpy() == 1, 'baseline_empty': base['k'].to_numpy() == 0, 'baseline_perfect': base['f'].to_numpy() == 1}
    masks.update({f'group_fold_{k}': rows['fold'].to_numpy() == k for k in range(5)})
    for key, mask in masks.items():
        dd = d[mask]
        out['slices'][key] = dict(n=int(mask.sum()), delta=float(dd.mean()) if len(dd) else None, improved=int((dd > 1e-12).sum()), harmed=int((dd < -1e-12).sum()))
    return out, rows

def val():
    meta, g = truth()
    meta.write_parquet(HERE / 'val_meta.parquet')
    g.write_parquet(HERE / 'val_truth.parquet')
    paths = {'v2b': M/'prod_v2b/output/val_abc_rob_cv2_all_p2/val_selected.parquet', 'v3': M/'prod_v3/output/val_AD_cv2_all_p2/val_selected.parquet', 'e06': M/'prod_v2c/results/eval_E06_ce_band_xw_p3b_val_selected.parquet', 'e13': M/'prod_v2c/results/eval_E13_cv2_val_selected.parquet', 'anchor': E/'results/eval_ENS_v3anchor_val_selected.parquet'}
    sels = {k: readsel(p) for k, p in paths.items()}
    a = sels['anchor']
    base = score(a, meta, g)
    expected = pl.read_parquet(E/'results/eval_ENS_v3anchor_val_rows.parquet').sort('s1_idx')
    assert base['s1_idx'].equals(expected['s1_idx'])
    err = float(np.abs(base['f'].to_numpy()-expected['f'].to_numpy()).max())
    assert err < 1e-12, err
    base.write_parquet(HERE/'baseline_rows.parquet')
    log('REPRODUCED', base.height, base['f'].mean(), 'max_error', err)
    votes = pl.concat([sels[x].with_columns(pl.lit(x).alias('model')) for x in ['v2b', 'v3', 'e06', 'e13']]).group_by(K).agg(pl.len().alias('votes'))
    av = a.join(votes, on=K, how='left').with_columns(pl.col('votes').fill_null(0)).join(g, on=K, how='left').with_columns(pl.col('label').fill_null(0))
    pairtruth = av.group_by('votes').agg(pl.len().alias('selected'), pl.col('label').sum().alias('true')).sort('votes').to_dicts()
    # Whole-query signatures include empty predictions explicitly.
    sels['v2a'] = pl.read_parquet(M/'prod_v2a/output/val_predictions.parquet', columns=K+['selected']).filter('selected').select(K)
    q = meta.select('s1_idx')
    for name, s in sels.items():
        sig = s.group_by('s1_idx').agg(pl.col('cand_idx').sort().cast(pl.String).str.join(',').alias(name))
        q = q.join(sig, on='s1_idx', how='left').with_columns(pl.col(name).fill_null(''))
    component_names = ['v2b', 'v3', 'e06', 'e13']
    sigs = q.select(component_names+['anchor']).rows()
    q = q.with_columns(pl.Series('unanimous', [len(set(v)) == 1 for v in sigs]), pl.Series('anchor_alone', [all(v[-1] != w for w in v[:-1]) for v in sigs]), pl.Series('four_vs_anchor', [len(set(v[:-1])) == 1 and v[-1] != v[0] for v in sigs]), pl.Series('empty_conflict', ['' in v and any(w != '' for w in v) for v in sigs]))
    q = q.join(base.select('s1_idx','f'), on='s1_idx')
    slices = {}
    for name, mask in [('unanimous', pl.col('unanimous')), ('disagreeing', ~pl.col('unanimous')), ('anchor_alone',pl.col('anchor_alone')), ('four_vs_anchor',pl.col('four_vs_anchor')), ('empty_conflict',pl.col('empty_conflict'))]:
        sub = q.filter(mask)
        slices[name] = dict(n=sub.height, f05=float(sub['f'].mean()), loss=float((1-sub['f']).sum()), loss_share=float((1-sub['f']).sum()/(1-base['f']).sum()))
    q.write_parquet(HERE/'val_disagreement.parquet')
    P = pl.scan_parquet(E/'data/preds_v3anchor.parquet').filter(pl.col('grp') != 'sample').select(K+['p']).collect()
    patterns = pp.pat('train', ['US', 'India']).join(meta.select('s1_idx'), on='s1_idx', how='semi')
    legal = patterns.filter((pl.col('kind') == 'legal') & pl.col('allnum') & pl.col('k').is_in(D))
    us = legal.filter((pl.col('country') == 'US') & (pl.col('k') > 2))
    add = pp.add_pairs('train').filter(pl.col('k').is_in(D)).join(meta.select('s1_idx'), on='s1_idx', how='semi')
    variants = {'ADD_veto': add, 'legal_US_all_offsets': legal.filter(pl.col('country') == 'US'), 'legal_India_all_offsets': legal.filter(pl.col('country') == 'India'), 'legal_US_no12': us, 'legal_US_no12_votes_le1': us.join(votes,on=K,how='left').filter(pl.col('votes').fill_null(0)<=1), 'legal_US_no12_p_lt05': us.join(P,on=K).filter(pl.col('p')<.5), 'legal_US_no12_p_lt08': us.join(P,on=K).filter(pl.col('p')<.8)}
    results = dict(baseline=dict(score=float(base['f'].mean()), n=base.height, max_row_reproduction_error=err), panel_paths={k:str(p) for k,p in paths.items()}, disagreement=slices, anchor_selected_truth_by_component_votes=pairtruth, experiments={})
    for name, veto in variants.items():
        removal = a.join(veto.select(K).unique(), on=K, how='semi')
        out, rows = compare(a.join(removal,on=K,how='anti'), base,meta,g)
        removed_lab = removal.join(g,on=K,how='left').with_columns(pl.col('label').fill_null(0))
        out.update(removed=removal.height, removed_true=int(removed_lab['label'].sum()))
        results['experiments'][name] = out
        rows.select('s1_idx','f').write_parquet(HERE/f'rows_{name}.parquet')
        log(name, json.dumps(out))
    for k in [1,2,3,4]:
        sel = a.join(votes.filter(pl.col('votes')>=k),on=K,how='semi')
        name = f'anchor_require_{k}_votes'
        results['experiments'][name], _ = compare(sel,base,meta,g)
        log(name, json.dumps(results['experiments'][name]))
    for k in [2,3]:
        name=f'replace_with_{k}_of4'
        results['experiments'][name], _ = compare(votes.filter(pl.col('votes')>=k),base,meta,g)
        log(name, json.dumps(results['experiments'][name]))
    # Archetypes on all candidate pairs represented in the precomputed uncertain band.
    b = pl.scan_parquet(M/'accuracy_lab_20260925/data/band.parquet').filter(pl.col('grp')!='sample').select(K+['addr_empty2','num_first_both','num_first_eq','fake2','name_freq_s1_1']).collect()
    archetypes={'candidate_address_missing': pl.col('addr_empty2')>0, 'house_number_mismatch': (pl.col('num_first_both')>0)&(pl.col('num_first_eq')==0), 'fake_name_flag':pl.col('fake2')>0, 'duplicated_S1_name':pl.col('name_freq_s1_1')>np.log(2)+1e-6}
    ar={}
    for name, expr in archetypes.items():
        sids=b.filter(expr).select('s1_idx').unique()
        rr=base.join(sids,on='s1_idx',how='semi')
        ar[name]=dict(n=rr.height,f05=float(rr['f'].mean()),loss_share=float((1-rr['f']).sum()/(1-base['f']).sum()))
    results['overlapping_archetypes_from_uncertain_band']=ar
    save('validation_results.json',results)
    log('VALIDATION DONE')

def test():
    a=readsel(E/'build/v3anchor/data/sel_final.parquet')
    assert a.height==5862675
    patterns=pp.pat('test',['US','India','France'])
    add=pp.add_pairs('test')
    rules=pp.rule_sets(add,patterns)
    comp_paths={'v2b':M/'v2a_experiments/data/test_sel_v2b.parquet','v3':M/'prod_v3/output/test_selected.parquet','e06':M/'prod_v2c/build/interim_v2b_E06/data/sel_final.parquet','e13':M/'prod_v2c/build/interim_v2b_E13/data/sel_final.parquet'}
    union_pat=pl.concat([x.select(K) for pair in rules.values() for x in pair]).unique()
    v=[]
    for name,p in comp_paths.items():
        s=readsel(p).join(union_pat,on=K,how='semi')
        v.append(s)
    votes=pl.concat(v).group_by(K).agg(pl.len().alias('votes'))
    sizes=a.group_by('s1_idx').agg(pl.len().alias('set_size'))
    res={}
    for name,(plus,minus) in rules.items():
        r=a.join(plus,on=K,how='inner').join(votes,on=K,how='left').with_columns(pl.col('votes').fill_null(0))
        mirror=a.join(minus,on=K,how='semi')
        changed=r.group_by('s1_idx').agg(pl.len().alias('removed')).join(sizes,on='s1_idx')
        res[name]=dict(removed=r.height,affected_queries=changed.height,emptied_queries=changed.filter(pl.col('removed')==pl.col('set_size')).height,mirror_selected=mirror.height,by_country=r.group_by('country').len().to_dicts(),by_votes=r.group_by('votes').len().sort('votes').to_dicts(),by_offset=r.group_by('k').len().sort('k').to_dicts())
        r.write_parquet(HERE/f'test_removals_{name}.parquet')
        log('TEST',name,res[name])
    # Audit an exact rebased existing postpass; do not emit a submission.
    veto=pl.concat([p.select(K) for p,m in rules.values()]).unique()
    b=a.join(veto,on=K,how='anti')
    res['combined']=dict(removed=a.height-b.height,baseline_pairs=a.height,new_pairs=b.height,one_owner_violations=b.group_by('cand_idx').len().filter(pl.col('len')>1).height)
    res['proxy_before']=france_proxies(a)
    res['proxy_after']=france_proxies(b)
    res['warning']='France proxies and mirror counts are not ground truth or a proof of gain. Test component postprocessing differs from validation.'
    save('test_audit.json',res)
    log('TEST DONE')

def density():
    meta=pl.read_parquet(HERE/'val_meta.parquet')
    g=pl.read_parquet(HERE/'val_truth.parquet')
    p=M/'density_val/universe/val_matching_results_dens_R10c_shipped.tsv'
    from vx.common import read_sub
    a=read_sub(str(p),split='train').select(K)
    keep=pl.read_parquet(M/'density_val/universe/truth_dens.parquet',columns=['s1_idx'])
    meta=meta.join(keep,on='s1_idx',how='semi')
    base=score(a,meta,g)
    assert abs(float(base['f'].mean())-0.9884288952739365)<2e-6
    pats=pp.pat('train',['US','India'])
    legal=pats.filter((pl.col('kind')=='legal')&pl.col('allnum')&pl.col('k').is_in(D))
    rules={'legal_US_all_offsets':legal.filter(pl.col('country')=='US'),'legal_India_all_offsets':legal.filter(pl.col('country')=='India'),'legal_US_no12':legal.filter((pl.col('country')=='US')&(pl.col('k')>2)),'ADD_veto':pp.add_pairs('train').filter(pl.col('k').is_in(D))}
    res={'baseline':'v2b rebuilt test density, NOT v3anchor','score':float(base['f'].mean()),'n':base.height,'pattern_coverage':'Pattern lists come from original union-v2; new density-only candidate pairs are not covered. Rules leave uncovered pairs unchanged.','experiments':{}}
    for name,v in rules.items():
        rem=a.join(v.select(K).unique(),on=K,how='semi')
        out,_=compare(a.join(rem,on=K,how='anti'),base,meta,g)
        out.update(removed=rem.height,removed_true=rem.join(g,on=K,how='inner').height)
        res['experiments'][name]=out
        log('DENSITY',name,json.dumps(out))
    save('density_transfer.json',res)

if __name__=='__main__':
    {'val':val,'test':test,'density':density}[sys.argv[1]]()
