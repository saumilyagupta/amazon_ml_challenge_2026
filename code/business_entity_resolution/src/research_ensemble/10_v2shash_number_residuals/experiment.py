"""Isolated, TUNE-selected experiments extending the archived three-seed policy."""
import os
for key in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['OMP_WAIT_POLICY']='PASSIVE'
import sys,json,time,hashlib,shutil,argparse
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import polars as pl
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
M=HERE.parent
OLD=M/'variance_research_round2_20260926'
sys.path.insert(0,str(OLD))
import evaluate as R
S=R.S
K=['s1_idx','cand_idx']
SEEDS=[11,29,47]
for folder in ('data','models','logs','snapshot'):
    (HERE/folder).mkdir(exist_ok=True)

def log(*args): print(time.strftime('%H:%M:%S'),*args,flush=True)
def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def snapshot():
    manifest={}
    for p in sorted(OLD.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        rel=p.relative_to(OLD)
        manifest[str(p)]={'size':p.stat().st_size,'sha256':digest(p)}
        if p.suffix in ('.py','.md','.json','.txt'):
            dest=HERE/'snapshot'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
    for p in [M/'ensemble_v1/build/v3anchor/output/matching_results.tsv',M.parent/'results_analysis/submitted_tsv/06_v3anchor_ensemble_2026-09-25.tsv']:
        manifest[str(p)]={'size':p.stat().st_size,'sha256':digest(p)}
    save('preservation.json',manifest)
    log('SNAPSHOT',len(manifest))

def load_environment(density=False):
    full,anchor,meta,truth=R.inputs()
    vf=HERE/'data/veto.parquet'
    if vf.exists():veto=pl.read_parquet(vf)
    else:
        veto=R.veto();veto.write_parquet(vf)
    if density:
        keep=pl.read_parquet(M/'density_val/universe/truth_dens.parquet',columns=['s1_idx'])
        meta=meta.join(keep,on='s1_idx',how='semi')
        full=pl.read_parquet(OLD/'data/density_decoder.parquet')
        anchor=pl.read_parquet(OLD/'data/anchor_density_selected.parquet')
        incumbent=pl.read_parquet(OLD/'data/density_seed_consensus_selected.parquet')
        expected=.9904776169721415
    else:
        anchor=R.post(anchor,full,veto)
        incumbent=pl.read_parquet(OLD/'data/seed_consensus_selected.parquet')
        expected=.9911761274726685
    baseline=S.score(incumbent,meta,truth)
    assert abs(baseline['f'].mean()-expected)<1e-12
    decoder=lgb.Booster(model_file=str(M/'ensemble_v1/results/eval_ENS_v3anchor_R10c_m0.txt'))
    return full,anchor,meta,truth,veto,incumbent,baseline,decoder

def correction(arm,density=False):
    if arm=='prior':
        p=OLD/('data/density_residual.parquet' if density else 'data/residual_scores.parquet')
    else:p=HERE/f'data/{arm}_{"density" if density else "ordinary"}_scores.parquet'
    c=pl.read_parquet(p)
    if 'grp' in c.columns:c=c.filter(pl.col('grp')!='sample')
    return c

def selections(arm,alpha,env,density=False):
    full,anchor,meta,truth,veto,incumbent,baseline,decoder=env
    corr=correction(arm,density)
    prefix=f'{"density" if density else "ordinary"}_{arm}_a{alpha:g}'
    q=meta.select('s1_idx');sels=[]
    for seed in SEEDS:
        path=HERE/f'data/{prefix}_seed{seed}.parquet'
        if path.exists():sel=pl.read_parquet(path)
        elif arm=='prior' and alpha==1:
            oldname=f'{"density_residual" if density else "residual"}_seed_{seed}_selected.parquet'
            sel=pl.read_parquet(OLD/'data'/oldname);sel.write_parquet(path)
        else:
            p=R.modified(full,corr,alpha,'all',seed)
            sel=R.post(R.decide_fast(p,decoder),p,veto)
            sel.write_parquet(path)
            del p
        sels.append(sel)
        sig=sel.group_by('s1_idx').agg(pl.col('cand_idx').sort().cast(pl.String).str.join(',').alias(f's{seed}'))
        q=q.join(sig,on='s1_idx',how='left').with_columns(pl.col(f's{seed}').fill_null(''))
        log('DECODE',prefix,seed,sel.height)
    meanp=R.modified(full,corr,alpha,'all')
    out={}
    for policy in ('average','unanimous','majority'):
        if policy=='average':
            path=HERE/f'data/{prefix}_{policy}_selected.parquet'
            sel=pl.read_parquet(path) if path.exists() else R.post(R.decide_fast(meanp,decoder),meanp,veto)
        else:
            eq12=pl.col('s11')==pl.col('s29');eq13=pl.col('s11')==pl.col('s47');eq23=pl.col('s29')==pl.col('s47')
            if policy=='unanimous':
                allow=q.filter(eq12&eq13).select('s1_idx')
                sel=sels[0].join(allow,on='s1_idx',how='semi')
            else:
                a0=q.filter(eq12|eq13).select('s1_idx')
                a1=q.filter(~(eq12|eq13)&eq23).select('s1_idx')
                allow=pl.concat([a0,a1])
                sel=pl.concat([sels[0].join(a0,on='s1_idx',how='semi'),sels[1].join(a1,on='s1_idx',how='semi')])
            sel=pl.concat([sel,anchor.join(allow,on='s1_idx',how='anti')])
            pp=pl.concat([meanp.join(allow,on='s1_idx',how='semi'),full.join(allow,on='s1_idx',how='anti')])
            sel=R.post(sel,pp,veto)
            del pp
        path=HERE/f'data/{prefix}_{policy}_selected.parquet';sel.write_parquet(path)
        out[policy]=(sel,path)
    if arm=='prior' and alpha==1:
        reproduced=out['unanimous'][0]
        assert reproduced.join(incumbent,on=K,how='anti').height==0
        assert incumbent.join(reproduced,on=K,how='anti').height==0
        log('EXACT INCUMBENT REPRODUCTION',density)
    return out

def tune(arm):
    env=load_environment();meta,truth,baseline=env[2],env[3],env[6]
    tune_meta=meta.filter(pl.col('eval_split')=='tune')
    b=baseline.join(tune_meta.select('s1_idx'),on='s1_idx',how='semi').sort('s1_idx')
    records=[]
    alphas=[1.,.5,1.5,2.] if arm=='prior' else [.5,1.]
    for alpha in alphas:
        for policy,(sel,path) in selections(arm,alpha,env).items():
            rows=S.score(sel,tune_meta,truth);d=rows['f'].to_numpy()-b['f'].to_numpy()
            countries={c:float(d[rows['country'].to_numpy()==c].mean()) for c in ('US','India')}
            r={'arm':arm,'alpha':alpha,'policy':policy,'path':str(path),'score':float(rows['f'].mean()),'delta':float(d.mean()),'countries':countries,'eligible':min(countries.values())>=-0.00002,'improved':int((d>1e-12).sum()),'harmed':int((d < -1e-12).sum()),'previously_perfect_harmed':int(((b['f'].to_numpy()==1)&(d < -1e-12)).sum())}
            records.append(r);save(f'tune_{arm}.json',records);log('TUNE',r)
    log('TUNE COMPLETE',arm)

def freeze():
    arms=['prior','capacity','numbers']
    rows=[]
    for arm in arms:rows+=json.loads((HERE/f'tune_{arm}.json').read_text())
    eligible=[r for r in rows if r['eligible']]
    winner=max(eligible,key=lambda r:r['score'])
    reps={arm:max([r for r in rows if r['arm']==arm and r['eligible']],key=lambda r:r['score']) for arm in arms if any(r['arm']==arm and r['eligible'] for r in rows)}
    save('frozen_selection.json',{'winner':winner,'representatives':reps,'selected_on':'TUNE only','new_REPORT_opened':False,'historical_report_exposure':True,'all_candidates':rows})
    log('FROZEN',winner)

def detailed(sel,env):
    meta,truth,incumbent,base=env[2],env[3],env[5],env[6]
    r,rows=S.compare(sel,base,meta,truth)
    for split in ('tune','report','locked'):
        mi=meta.filter(pl.col('eval_split')==split)
        bi=base.join(mi.select('s1_idx'),on='s1_idx',how='semi').sort('s1_idx')
        r[split],_=S.compare(sel,bi,mi,truth)
    anchorbase=S.score(env[1],meta,truth)
    r['vs_anchor'],_=S.compare(sel,anchorbase,meta,truth)
    r['duplicate_pairs']=sel.height-sel.unique(K).height
    r['ownership_violations']=sel.group_by('cand_idx').len().filter(pl.col('len')>1).height
    r['outside_candidates']=sel.join(env[0].select(K),on=K,how='anti').height
    assert r['duplicate_pairs']==r['ownership_violations']==r['outside_candidates']==0
    return r,rows

def report(density=False):
    cfg=json.loads((HERE/'frozen_selection.json').read_text())
    env=load_environment(density);out={}
    for arm,r in cfg['representatives'].items():
        if density:
            sel=selections(arm,r['alpha'],env,True)[r['policy']][0]
        else:sel=pl.read_parquet(r['path'])
        metrics,rows=detailed(sel,env);out[arm]={'configuration':r,**metrics}
        tag=f'{"density" if density else "ordinary"}_{arm}_final'
        sel.write_parquet(HERE/f'data/{tag}_selected.parquet');rows.write_parquet(HERE/f'data/{tag}_rows.parquet')
        save(f'{"density" if density else "ordinary"}_results.json',out)
        log('REPORT',arm,'density',density,metrics['score'],metrics['delta'],metrics['report']['delta'])
    log('REPORT COMPLETE',density)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['snapshot','tune','freeze','report','density']);ap.add_argument('--arm',default='prior');a=ap.parse_args()
    if a.action=='snapshot':snapshot()
    elif a.action=='tune':tune(a.arm)
    elif a.action=='freeze':freeze()
    else:report(a.action=='density')
