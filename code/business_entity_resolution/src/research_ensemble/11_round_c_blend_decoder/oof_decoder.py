"""New residual cross-fitting and learned prefix utilities, isolated from old runs."""
from experiment import *
import gc,zlib
from scipy.special import logit
import train as T
import decide_lib as DL
D=HERE/'oof_decoder'
for f in ('models','data','logs'):(D/f).mkdir(parents=True,exist_ok=True)
BASE=DL.FE+['k','pk','pn','cs','tail','gap','n','c3','c2','avg']
EV=['missing_gate','n_ratio','a_ratio','num_first_eq','num_first_both','g_name_n_max','g_s1freq_name_n','z_std']
EXTRA=[f'{pos}_{c}' for pos in ('last','next') for c in EV]+['q_band_n','q_missing_n','q_conflict_n','q_spread_max']

def ds(name,value):(D/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def config():return json.loads((HERE/'frozen_selection.json').read_text())['winner']
def band(arm,density=False):
    if arm=='prior':return pl.read_parquet(OLD/('data/density_band.parquet' if density else 'data/anchor_band.parquet')),json.loads((OLD/'features.json').read_text())
    return T.data(arm,density)

def crossfit():
    cfg=config();b,features=band(cfg['arm']);b=b.filter(pl.col('grp')=='sample')
    rec=pl.read_parquet(M/'prod_v1/data/rec_train_s1.parquet',columns=['erow','country','name_n']).select(pl.col('erow').cast(pl.Int32).alias('s1_idx'),'country','name_n')
    groups=b.select('s1_idx').unique().join(rec,on='s1_idx').with_columns(pl.struct('country','name_n').map_elements(lambda r:zlib.crc32(('v2shash_oof:'+r['country']+':'+r['name_n']).encode())%3,return_dtype=pl.Int32).alias('oof_fold'))
    b=b.join(groups.select('s1_idx','oof_fold'),on='s1_idx',maintain_order='left')
    groups.write_parquet(D/'data/folds.parquet')
    x=np.nan_to_num(b.select(features).to_numpy().astype(np.float32),nan=-1,posinf=1e6,neginf=-1e6)
    y=b['label'].to_numpy();z=logit(b['p2'].to_numpy().astype(float));fold=b['oof_fold'].to_numpy();esflag=b['sample_es'].to_numpy()
    pred=np.zeros(b.height,np.float32);coverage=np.zeros(b.height,np.int8);info=[]
    prior=cfg['arm']=='prior';rounds=600 if prior else 1200
    for f,seed in enumerate(SEEDS):
        tr=(fold!=f)&~esflag;es=(fold!=f)&esflag;ho=fold==f
        modelpath=D/f'models/correction_fold{f}.txt'
        if modelpath.exists():model=lgb.Booster(model_file=str(modelpath))
        else:
            params=dict(objective='binary',metric='binary_logloss',learning_rate=.035,num_leaves=15 if prior else 31,min_data_in_leaf=250,lambda_l2=30,feature_fraction=.85,bagging_fraction=.8,bagging_freq=1,num_threads=4,verbosity=-1,seed=seed)
            a=lgb.Dataset(x[tr],label=y[tr],init_score=z[tr],feature_name=features)
            e=lgb.Dataset(x[es],label=y[es],init_score=z[es],reference=a)
            log('OOF FIT',f,int(tr.sum()),int(es.sum()),int(ho.sum()))
            model=lgb.train(params,a,num_boost_round=rounds,valid_sets=[e],callbacks=[lgb.early_stopping(60 if prior else 80,verbose=False),lgb.log_evaluation(200)])
            model.save_model(str(modelpath));del a,e;gc.collect()
        pred[ho]=model.predict(x[ho],raw_score=True,num_threads=4).astype(np.float32);coverage[ho]+=1
        r={'fold':f,'fit':int(tr.sum()),'early_stop':int(es.sum()),'heldout':int(ho.sum()),'iterations':model.current_iteration(),'group_overlap':0}
        assert not np.any((tr|es)&ho)
        info.append(r);ds('crossfit.json',{'configuration':cfg,'folds':info,'method':'3-fold country+normalized-name crossfit of correction over historical base OOF scores'});log('OOF DONE',r)
    assert np.all(coverage==1)
    out=b.select(K+['missing_gate']).with_columns(*[pl.Series(f'delta_{seed}',pred) for seed in SEEDS])
    out.write_parquet(D/'data/sample_correction_oof.parquet');log('CROSSFIT COMPLETE',out.height)

def sample_inputs():
    cache=D/'data/sample_decoder.parquet'
    if cache.exists():return pl.read_parquet(cache),pl.read_parquet(D/'data/sample_meta.parquet')
    p=pl.read_parquet(M/'ensemble_v1/data/preds_v3anchor.parquet').filter(pl.col('grp')=='sample')
    needed=p.select('cand_idx').unique()
    c=pl.scan_parquet(M/'prod_v1/data/preds_train.parquet').select('s1_idx','cand_idx',pl.col('p2').alias('q')).join(needed.lazy(),on='cand_idx',how='semi').collect()
    top=c.sort(['cand_idx','q'],descending=[False,True]).group_by('cand_idx',maintain_order=True).agg(pl.col('s1_idx').first().alias('best_s1'),pl.col('q').first().alias('best_q'),pl.col('q').slice(1,1).first().alias('second_q'))
    del c;gc.collect()
    p=p.join(top,on='cand_idx',how='left').with_columns(pl.when(pl.col('best_s1')==pl.col('s1_idx')).then(pl.col('second_q').fill_null(-1.)).otherwise(pl.col('best_q').fill_null(-1.)).cast(pl.Float32).alias('p_other')).drop('best_s1','best_q','second_q')
    i2=pl.read_parquet(M.parent/'blocking/embedding/full/ids/train_s23.parquet',columns=['entity_id']).with_row_index('cand_idx').select(pl.col('cand_idx').cast(pl.Int32),pl.col('entity_id').str.slice(0,2).alias('src'))
    p=p.join(i2,on='cand_idx',how='left')
    i1=pl.read_parquet(M.parent/'blocking/embedding/full/ids/train_s1.parquet',columns=['entity_id','country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    meta=p.select('s1_idx').unique().join(i1,on='s1_idx');wanted=set(meta['entity_id'].to_list());counts={}
    with (M.parent/'splits/train_split_ground_truth.tsv').open() as fh:
        next(fh)
        for line in fh:
            eid,_,tail=line.rstrip('\n').partition('\t')
            if eid in wanted:counts[eid]=len(set(x for x in tail.split(',') if x))
    assert len(counts)==meta.height
    meta=meta.join(pl.DataFrame({'entity_id':list(counts),'m':list(counts.values())}),on='entity_id')
    p.write_parquet(cache);meta.write_parquet(D/'data/sample_meta.parquet');log('SAMPLE INPUTS',p.height,meta.height)
    return p,meta

def prefixes(p,evidence,meta=None):
    u=p.select(pl.col('s1_idx').alias('s1'),pl.col('cand_idx').alias('qid'),'src',pl.col('p').cast(pl.Float64),(pl.col('p')>=pl.col('p_other')).cast(pl.Int8).replace({0:2}).alias('qr'),*( ['label'] if 'label' in p.columns else []))
    f=DL.s1_features(u)
    ev=evidence.select(K+EV).rename({'s1_idx':'s1','cand_idx':'qid'})
    qa=ev.group_by('s1').agg(pl.len().alias('q_band_n'),pl.col('missing_gate').sum().alias('q_missing_n'),(pl.col('num_first_both')-pl.col('num_first_eq')).sum().alias('q_conflict_n'),pl.col('z_std').max().alias('q_spread_max'))
    w=u.filter((pl.col('qr')==1)&(pl.col('p')>.02)).sort(['s1','p'],descending=[False,True]).join(ev,on=['s1','qid'],how='left',maintain_order='left').with_columns(pl.col(EV).fill_null(-1))
    w=w.with_columns(pl.col('p').cum_count().over('s1').alias('k'),pl.col('p').cum_sum().over('s1').alias('cs'),(pl.col('src')=='S3').cast(pl.Int32).cum_sum().over('s1').alias('c3'),pl.col('p').shift(-1).over('s1').fill_null(0).alias('pn'),pl.len().over('s1').alias('n'),pl.col('p').sum().over('s1').alias('total'),*[pl.col(c).shift(-1).over('s1').fill_null(-1).alias('next_'+c) for c in EV])
    if meta is not None:w=w.with_columns(pl.col('label').cast(pl.Int32).cum_sum().over('s1').alias('ct'))
    zero=w.group_by('s1',maintain_order=True).first().select('s1',pl.lit(0).cast(pl.UInt32).alias('k'),pl.lit(1.).alias('pk'),pl.col('p').alias('pn'),pl.lit(0.).alias('cs'),'total','n',pl.lit(0).cast(pl.Int32).alias('c3'),*[pl.lit(-1.).alias('last_'+c) for c in EV],*[pl.col(c).alias('next_'+c) for c in EV],*([pl.lit(0).cast(pl.Int32).alias('ct')] if meta is not None else []))
    pos=w.filter(pl.col('k')<=10).select('s1','k',pl.col('p').alias('pk'),'pn','cs','total','n','c3',*[pl.col(c).alias('last_'+c) for c in EV],*[pl.col('next_'+c) for c in EV],*(['ct'] if meta is not None else []))
    t=pl.concat([zero,pos],how='vertical_relaxed').join(f,on='s1').join(qa,on='s1',how='left').with_columns(pl.col(['q_band_n','q_missing_n','q_conflict_n','q_spread_max']).fill_null(0),(pl.col('total')-pl.col('cs')).alias('tail'),(pl.col('pk')-pl.col('pn')).alias('gap'),(pl.col('k').cast(pl.Int32)-pl.col('c3')).alias('c2'),(pl.col('cs')/pl.col('k').clip(lower_bound=1)).alias('avg')).sort('s1','k')
    if meta is not None:
        t=t.join(meta.select(pl.col('s1_idx').alias('s1'),'m'),on='s1',maintain_order='left').with_columns(pl.when(pl.col('m')==0).then((pl.col('k')==0).cast(pl.Float64)).otherwise(5*pl.col('ct')/(4*pl.col('k')+pl.col('m'))).alias('target'))
    return t,w

def decoder_fit():
    cfg=config();p,meta=sample_inputs();corr=pl.read_parquet(D/'data/sample_correction_oof.parquet')
    truth_check=p.group_by('s1_idx').agg(pl.col('label').cast(pl.Int64).sum().alias('retrieved_true')).join(meta.select('s1_idx','m'),on='s1_idx')
    assert truth_check.height==meta.height and truth_check.filter(pl.col('retrieved_true')>pl.col('m')).height==0
    p=R.modified(p,corr,cfg['alpha'],'all');b,_=band(cfg['arm']);b=b.filter(pl.col('grp')=='sample')
    t,w=prefixes(p,b,meta);del w,p,b;gc.collect()
    assert t['target'].min()>=0 and t['target'].max()<=1
    # Check the exact empty and truth-count conventions independently on a subset.
    z=t.filter(pl.col('k')==0);assert np.array_equal(z['target'].to_numpy(),(z['m'].to_numpy()==0).astype(float))
    for kind,cols in [('standard',BASE),('evidence',BASE+EXTRA)]:
        x=t.select(cols).to_numpy().astype(np.float32);y=t['target'].to_numpy()
        log('DECODER FIT',kind,x.shape)
        model=lgb.train(dict(objective='regression',learning_rate=.05,num_leaves=63,min_data_in_leaf=50,num_threads=4,verbosity=-1,seed=20260926),lgb.Dataset(x,label=y,feature_name=cols),num_boost_round=600)
        model.save_model(str(D/f'models/decoder_{kind}.txt'));ds(f'decoder_{kind}.json',{'features':cols,'rows':len(y),'source':'fresh correction OOF sample, complete truth counts','rounds':600})
        log('DECODER FIT COMPLETE',kind)

def decide(p,b,model,cols):
    t,w=prefixes(p,b)
    t=t.with_columns(pl.Series('u',model.predict(t.select(cols).to_numpy().astype(np.float32),num_threads=4)))
    best=t.sort(['s1','u','k'],descending=[False,True,False]).group_by('s1',maintain_order=True).first().select('s1',pl.col('k').alias('kb'))
    return w.join(best,on='s1').filter(pl.col('k')<=pl.col('kb')).select(pl.col('s1').alias('s1_idx'),pl.col('qid').alias('cand_idx'))

def candidates(density=False):
    cfg=config();env=load_environment(density);full,anchor,meta,truth,veto,incumbent,baseline,_=env
    b,_=band(cfg['arm'],density);corr=correction(cfg['arm'],density);out={}
    for kind in ('standard','evidence'):
        model=lgb.Booster(model_file=str(D/f'models/decoder_{kind}.txt'));cols=json.loads((D/f'decoder_{kind}.json').read_text())['features']
        sels=[];q=meta.select('s1_idx')
        for seed in SEEDS:
            p=R.modified(full,corr,cfg['alpha'],'all',seed);s=R.post(decide(p,b,model,cols),p,veto);sels.append(s)
            sig=s.group_by('s1_idx').agg(pl.col('cand_idx').sort().cast(pl.String).str.join(',').alias(f's{seed}'))
            q=q.join(sig,on='s1_idx',how='left').with_columns(pl.col(f's{seed}').fill_null(''));log('NEW DECODER SEED',density,kind,seed)
        meanp=R.modified(full,corr,cfg['alpha'],'all')
        avg=R.post(decide(meanp,b,model,cols),meanp,veto)
        allow=q.filter((pl.col('s11')==pl.col('s29'))&(pl.col('s11')==pl.col('s47'))).select('s1_idx')
        # Keep the original safeguard if the new decoder seeds do not agree.
        s=pl.concat([sels[0].join(allow,on='s1_idx',how='semi'),incumbent.join(allow,on='s1_idx',how='anti')])
        # Use original anchor scores for fallback ownership, as an explicit policy.
        p=pl.concat([meanp.join(allow,on='s1_idx',how='semi'),full.join(allow,on='s1_idx',how='anti')]);cons=R.post(s,p,veto)
        for policy,sel in [('average',avg),('unanimous',cons)]:
            name=kind+'_'+policy;path=D/f'data/{"density" if density else "ordinary"}_{name}_selected.parquet';sel.write_parquet(path);out[name]=sel
    return env,out

def tune_decoders():
    env,out=candidates();tm=env[2].filter(pl.col('eval_split')=='tune');base=env[6].join(tm.select('s1_idx'),on='s1_idx',how='semi').sort('s1_idx')
    cfg=config();out['round_a']=pl.read_parquet(cfg['path']);out['original_safeguard']=env[5]
    records=[]
    for name,sel in out.items():
        rows=S.score(sel,tm,env[3]);d=rows['f'].to_numpy()-base['f'].to_numpy();countries={c:float(d[rows['country'].to_numpy()==c].mean()) for c in ('US','India')}
        r={'name':name,'score':float(rows['f'].mean()),'delta':float(d.mean()),'countries':countries,'eligible':min(countries.values())>=-.00002};records.append(r);log('DECODER TUNE',r)
    winner=max([r for r in records if r['eligible']],key=lambda r:r['score'])
    ds('frozen_selection.json',{'winner':winner,'candidates':records,'REPORT_opened':False,'selection':'TUNE only'});log('DECODER FROZEN',winner)

def evaluate_decoders(density=False):
    cfg=json.loads((D/'frozen_selection.json').read_text())
    if density:env,out=candidates(True)
    else:
        env=load_environment();out={r['name']:pl.read_parquet(D/f"data/ordinary_{r['name']}_selected.parquet") for r in cfg['candidates'] if r['name'] not in ('round_a','original_safeguard')}
    a=config();out['round_a']=pl.read_parquet(HERE/f"data/{'density' if density else 'ordinary'}_{a['arm']}_final_selected.parquet");out['original_safeguard']=env[5]
    results={}
    for name,sel in out.items():
        r,rows=detailed(sel,env);results[name]=r
        rows.write_parquet(D/f"data/{'density' if density else 'ordinary'}_{name}_rows.parquet")
        ds(f"{'density' if density else 'ordinary'}_results.json",results);log('DECODER REPORT',density,name,r['score'],r['delta'],r['report']['delta'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['crossfit','fit','tune','report','density']);a=ap.parse_args()
    if a.action=='crossfit':crossfit()
    elif a.action=='fit':decoder_fit()
    elif a.action=='tune':tune_decoders()
    else:evaluate_decoders(a.action=='density')
