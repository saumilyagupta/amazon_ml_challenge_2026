"""A1: test feature matrix for the round-2 anchor residual (reproduces variance_research_round2_20260926/residual.py prepare()).
usage: prep_test.py test | val <n_s1>
Same code path for both: anchor p from the member matrix (exact BLEND v3anchor formula), p2_aggregates over ALL candidates of each S1,
v2b band keys (0.001 < v2b p2 < 0.999) -> BASE from parts (test: prod_v3x_ash band_test_*_ash = prod_v2b/feats/test rows; val: prod_v2b/feats/train),
p1/sib_* from the v2b prediction table, anchor gate 0.001 < p < 0.999, g_* via e13_block.make_features, dec_* from prod_v3 feats, z features as prepare()."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='8'
os.environ['OMP_WAIT_POLICY']='PASSIVE';os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,resource
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import polars as pl
from scipy.special import logit
M=Path('/workspace/saumilya/amazon-ml/work/matching');E=M/'ensemble_v1';VR=M/'variance_research_round2_20260926'
OUT=M/'vr2_stress_submission';K=['s1_idx','cand_idx']
sys.path.insert(0,str(M/'prod_v2c/exp/E13_codex_refit'));import e13_block as G
sys.path.insert(0,'/workspace/saumilya/amazon-ml/work/hpo_v1/src');import hpo_common
FEAT=json.loads((VR/'features.json').read_text())
# BASE list exactly as residual.py (first 73 entries of features.json)
BASE=FEAT[:FEAT.index('p2')]
V2B_COLS=['p1','p1_max_other_rec','p1_rel_other','p1_rank_rec','sib_n','sib_hn_eq','sib_addr_max','sib_name_max']
DEC=[c for c in FEAT if c.startswith('dec_') and c not in BASE]
T0=time.time();TIM={}
def rss():return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,2)
def log(*a):print(time.strftime('%H:%M:%S'),f'+{time.time()-T0:.0f}s',*a,'peakRSS_GB',rss(),flush=True)
def tick(n):TIM[n]=round(time.time()-T0,1)
def v3anchor(df):
    # ensemble_v1/src/BLEND_run.py v3anchor(h=1) with ens_common.logit/expit (float64, clip 1e-6), final logit clipped [-13.8,13.8], float32
    L=lambda c:(lambda x:np.log(x/(1-x)))(np.clip(df[c].to_numpy().astype(np.float64),1e-6,1-1e-6))
    L2b=L('m_v2b');z=L('m_v3');z+=L('m_e06')-L2b;z+=L('m_e13')-L2b
    return (1.0/(1.0+np.exp(-np.clip(z,-13.8,13.8)))).astype(np.float32)

def build(split,n_s1=None):
    info=dict(split=split)
    if split=='test':
        mem_f=E/'data/members_test.parquet';stored=E/'data/test_v3anchor.parquet';stored_col='p'
        v2b_f=M/'prod_v2b/data/preds_test_abc_rob_cv2_all.parquet'
        base_files=sorted((M/'prod_v3x_ash/data_ash').glob('band_test_[0-9]*_ash.parquet'))
        dec_files=sorted((M/'prod_v3/feats').glob('test_[0-9]*.parquet'));rec='test'
    else:
        mem_f=E/'data/members_val.parquet';stored=E/'data/preds_v3anchor.parquet';stored_col='p'
        v2b_f=M/'prod_v2b/data/preds_abc_rob_cv2_all.parquet'
        base_files=sorted((M/'prod_v2b/feats').glob('train_[0-9]*.parquet'))
        dec_files=sorted((M/'prod_v3/feats').glob('train_[0-9]*.parquet'));rec='train'
    info['sources']=dict(members=str(mem_f),stored_anchor=str(stored),v2b_preds=str(v2b_f),base_parts=f'{len(base_files)} x {base_files[0].parent}/{base_files[0].name}..',dec_parts=f'{len(dec_files)} x {dec_files[0].parent}',records=rec)
    # ---- (1) anchor p on ALL pairs (natural file order kept: p2_rank is an ordinal rank) ----
    Mm=pl.read_parquet(mem_f,columns=K+['country','m_v2b','m_v3','m_e06','m_e13'])
    S=pl.read_parquet(stored,columns=K+[stored_col])
    assert S.height==Mm.height and np.array_equal(S['s1_idx'].to_numpy(),Mm['s1_idx'].to_numpy()) and np.array_equal(S['cand_idx'].to_numpy(),Mm['cand_idx'].to_numpy()),'stored anchor order differs'
    if n_s1:
        ab=pl.read_parquet(VR/'data/anchor_band.parquet',columns=['s1_idx','grp'])
        s1s=np.sort(ab.filter(pl.col('grp')!='sample')['s1_idx'].unique().to_numpy())
        rng=np.random.default_rng(20260926);pick=np.sort(rng.choice(s1s,n_s1,replace=False))
        keep=pl.Series(np.isin(Mm['s1_idx'].to_numpy(),pick));Mm=Mm.filter(keep);S=S.filter(keep);info['val_s1']=int(n_s1)
    p=v3anchor(Mm);ps=S[stored_col].to_numpy()
    d=np.abs(p.astype(np.float64)-ps.astype(np.float64))
    info['anchor_check']=dict(rows=int(len(p)),exact_equal=int((p==ps).sum()),max_abs_diff=float(d.max()),vs=str(stored))
    log('anchor p',info['anchor_check']);del S,ps,d
    Mm=Mm.with_columns(pl.Series('p2',p));del p
    if split=='test':
        Mm.select(K,pl.col('p2').alias('p_anchor')).write_parquet(OUT/'data/test_anchor_p.parquet');log('wrote test_anchor_p')
    tick('anchor_p')
    P=Mm.select(K+['p2'])
    PA=G.p2_aggregates(P)
    # ---- (2) v2b band keys + p1/sib (v2b prediction table) ----
    V=pl.scan_parquet(v2b_f).select(K+V2B_COLS+[pl.col('p2').alias('p2_v2b')]).filter(G.band_expr('p2_v2b'))
    if n_s1:V=V.filter(pl.col('s1_idx').is_in(pick.tolist()))
    V=V.collect(engine='streaming');info['v2b_band_rows']=V.height;log('v2b band',V.height)
    keys=V.select(K).lazy()
    need=[c for c in BASE if c not in V2B_COLS]
    parts=[pl.scan_parquet(f).select(K+need).join(keys,on=K,how='semi').collect(engine='streaming') for f in base_files]
    X=pl.concat(parts);del parts;info['base_part_rows']=X.height;log('base parts rows',X.height)
    B=V.join(X,on=K,how='inner');del X
    assert B.height==V.height,('base rows lost',B.height,V.height)
    tick('base')
    # ---- gate ----
    B=B.join(PA,on=K,how='inner');info['band_with_members']=B.height
    B=B.filter(pl.col('p2').is_between(.001,.999,closed='none'));info['gate_rows']=B.height;log('gated band',B.shape)
    del PA
    # ---- (3) g_* sibling features: anchor p over ALL candidates, whole S1 per chunk ----
    R,R1=G.load_records(rec)
    s1u=np.sort(B['s1_idx'].unique().to_numpy());nch=1 if n_s1 else 6;bounds=np.array_split(s1u,nch);gs=[]
    for i,b in enumerate(bounds):
        Bi=B.filter(pl.col('s1_idx').is_between(int(b[0]),int(b[-1])))
        Pi=P.filter(pl.col('s1_idx').is_between(int(b[0]),int(b[-1])))
        gs.append(G.make_features(Bi,Pi,R,R1));log('g chunk',i,Bi.height)
    Gf=pl.concat(gs);del gs
    B=B.join(Gf,on=K,how='left');assert B.select(G.G_FEATURES).null_count().sum_horizontal().item()==0
    tick('g_features')
    # ---- members + dec ----
    B=B.join(Mm.select(K+['country','m_v2b','m_v3','m_e06','m_e13']),on=K,how='inner')
    del Mm,P
    bk=B.select(K).lazy()
    parts=[pl.scan_parquet(f).select(K+DEC).join(bk,on=K,how='semi').collect(engine='streaming') for f in dec_files]
    n0=B.height;B=B.join(pl.concat(parts),on=K,how='inner');info['dec_rows_lost']=n0-B.height;del parts
    tick('dec')
    # ---- z features exactly as prepare() ----
    zs=np.column_stack([logit(np.clip(B[c].to_numpy().astype(float),1e-6,1-1e-6)) for c in ['m_v2b','m_v3','m_e06','m_e13']])
    for j,n in enumerate(['z_v2b','z_v3','z_e06','z_e13']):B=B.with_columns(pl.Series(n,zs[:,j].astype(np.float32)))
    B=B.with_columns(pl.Series('z_spread',zs.max(1)-zs.min(1)),pl.Series('z_std',zs.std(1)),pl.Series('vote05',(zs>0).sum(1)),pl.Series('anchor_z',logit(B['p2'].to_numpy().astype(float))),((pl.col('addr_empty1')>0)|(pl.col('addr_empty2')>0)).cast(pl.Int8).alias('missing_gate'))
    B=B.with_columns((pl.col('z_std')*pl.col('missing_gate')).alias('disagree_missing'),(pl.col('z_std')*(pl.col('num_first_both')-pl.col('num_first_eq'))).alias('disagree_number'),(pl.col('anchor_z')-(pl.col('z_v2b')+pl.col('z_v3')+pl.col('z_e06')+pl.col('z_e13'))/4).alias('anchor_vs_mean'))
    assert B.select(pl.struct(K).n_unique()).item()==B.height
    B=B.sort(K).select(K+['country',pl.col('p2').alias('p_anchor')]+FEAT)
    tick('z')
    info['rows']=B.height;info['rows_per_country']=dict(B.group_by('country').len().sort('country').iter_rows())
    info['missing_gate_rows']=int(B['missing_gate'].sum())
    nul=B.select(pl.col(FEAT).null_count()).row(0,named=True);nan=B.select([pl.col(c).is_nan().sum().alias(c) for c in FEAT if B.schema[c] in (pl.Float32,pl.Float64)]).row(0,named=True)
    info['nulls']={k:v for k,v in nul.items() if v};info['nans']={k:v for k,v in nan.items() if v}
    return B,info

if __name__=='__main__':
    split=sys.argv[1]
    a,l=hpo_common.wait_for_resources(min_avail_gb=25+60,max_load=185,log=log);log('admitted',a,l)
    if split=='test':
        B,info=build('test')
        B.write_parquet(OUT/'data/test_anchor_band.parquet');tick('write')
        # reference null/nan profile of the training table
        ref=pl.read_parquet(VR/'data/anchor_band.parquet',columns=FEAT)
        info['ref_nulls']={k:v for k,v in ref.select(pl.col(FEAT).null_count()).row(0,named=True).items() if v}
        info['ref_nans']={k:v for k,v in ref.select([pl.col(c).is_nan().sum().alias(c) for c in FEAT if ref.schema[c] in (pl.Float32,pl.Float64)]).row(0,named=True).items() if v}
        info['timings_s']=TIM;info['peak_rss_gb']=rss();info['features']=len(FEAT)
        (OUT/'data/prep_info.json').write_text(json.dumps(info,indent=2));log('DONE',B.shape)
    else:
        n=int(sys.argv[2]);B,info=build('val',n)
        ref=pl.read_parquet(VR/'data/anchor_band.parquet',columns=K+['p2']+FEAT[:0]+[c for c in FEAT if c!='p2']).join(B.select(K).lazy().collect().select('s1_idx').unique(),on='s1_idx',how='semi')
        cmp=dict(ours=B.height,ref=ref.height)
        J=B.join(ref,on=K,how='full',suffix='_ref',coalesce=True)
        cmp['only_ours']=int(J['p_anchor'].is_null().sum()) if False else int(J.filter(pl.col('p1_ref').is_null()).height);cmp['only_ref']=int(J.filter(pl.col('p1').is_null()).height)
        J=J.filter(pl.col('p1').is_not_null()&pl.col('p1_ref').is_not_null())
        diffs={}
        for c in FEAT:
            a=J[c].to_numpy().astype(np.float64);b=J[c+'_ref'].to_numpy().astype(np.float64)
            both_nan=np.isnan(a)&np.isnan(b);dd=np.where(both_nan,0,np.abs(a-b));dd=np.where(np.isnan(dd),np.inf,dd)
            diffs[c]=dict(max_abs=float(dd.max()) if len(dd) else 0.0,n_gt_1e6=int((dd>1e-6).sum()))
        cmp['matched']=J.height;cmp['mismatch_cols']={c:v for c,v in diffs.items() if v['max_abs']>1e-6};cmp['max_abs_all']=max(v['max_abs'] for v in diffs.values())
        info['compare']=cmp;info['per_column']=diffs;info['timings_s']=TIM;info['peak_rss_gb']=rss()
        (OUT/'data/verify_val_info.json').write_text(json.dumps(info,indent=2));log('VERIFY',json.dumps(cmp)[:3000])
