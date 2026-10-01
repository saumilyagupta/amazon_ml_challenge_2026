"""Round C; all writes stay here, previous experiment artifacts are read-only."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent
sys.path.insert(0,str(C.parent))
from experiment import *
import oof_decoder as O
import gc
for f in ('models','data','logs'):(C/f).mkdir(exist_ok=True)
FE=O.BASE+O.EXTRA
MODEL_CONFIG={'q63':(63,50,1),'q63e2':(63,50,2),'q31':(31,100,1),'q31e2':(31,100,2)}

def put(name,value):
    (C/name).write_text(json.dumps(value,indent=2,allow_nan=False))

def preserve():
    path=C/'preservation.json'
    if path.exists():return
    previous=json.loads((HERE/'preservation.json').read_text())
    assert all(digest(Path(p))==v['sha256'] for p,v in previous.items())
    for p in HERE.rglob('*'):
        if p.is_file() and C not in p.parents and '__pycache__' not in p.parts:
            previous[str(p)]={'size':p.stat().st_size,'sha256':digest(p)}
    put('preservation.json',previous);log('PRESERVED',len(previous))

def prepare():
    preserve()
    path=C/'data/sample_prefixes.parquet'
    if path.exists():return
    p,meta=O.sample_inputs()
    corr=pl.read_parquet(O.D/'data/sample_correction_oof.parquet')
    p=R.modified(p,corr,1.,'all')
    b,_=O.band('numbers');b=b.filter(pl.col('grp')=='sample')
    t,w=O.prefixes(p,b,meta)
    cols=list(dict.fromkeys(['s1','k','m','target']+FE))
    t=t.select(cols)
    assert t['target'].min()>=0 and t['target'].max()<=1
    assert not set(FE)&{'label','target','m','country','eval_split'}
    t.write_parquet(path)
    put('preparation.json',{'prefix_rows':t.height,'train_queries':t['s1'].n_unique(),'sample_queries':meta.height,'forced_empty_queries_omitted':meta.height-t['s1'].n_unique(),'features':FE,'corrections':'../oof_decoder/data/sample_correction_oof.parquet','correction_sha256':digest(O.D/'data/sample_correction_oof.parquet'),'source':'fresh country/name-group OOF correction, full sample truths'})
    log('PREPARED',t.shape)

def train_model(name):
    path=C/f'models/{name}.txt'
    if path.exists():log('MODEL EXISTS',name);return
    t=pl.read_parquet(C/'data/sample_prefixes.parquet')
    leaves,minleaf,emptyweight=MODEL_CONFIG[name]
    raw=(pl.when(pl.col('m')==0).then(float(emptyweight)).otherwise(1.)/pl.len().over('s1')).alias('weight')
    t=t.with_columns(raw)
    mass=t.group_by('s1').agg(pl.col('weight').sum().alias('mass'),pl.col('m').first())
    want=np.where(mass['m'].to_numpy()==0,emptyweight,1.)
    assert np.max(np.abs(mass['mass'].to_numpy()-want))<1e-6
    weights=t['weight'].to_numpy();weights=weights/weights.mean()
    x=t.select(FE).to_numpy().astype(np.float32);y=t['target'].to_numpy()
    params=dict(objective='regression',learning_rate=.05,num_leaves=leaves,min_data_in_leaf=minleaf,num_threads=4,verbosity=-1,seed=20260926)
    log('TRAIN',name,x.shape,'empty_weight',emptyweight)
    model=lgb.train(params,lgb.Dataset(x,label=y,weight=weights,feature_name=FE),num_boost_round=600)
    model.save_model(str(path))
    put(f'fit_{name}.json',{'parameters':params,'rounds':600,'features':FE,'query_balanced':True,'empty_query_weight':emptyweight,'max_weight_mass_error':float(np.max(np.abs(mass['mass'].to_numpy()-want))),'prefix_rows':t.height,'queries':mass.height,'model_sha256':digest(path)})
    log('TRAIN COMPLETE',name)

def source(name,tag):
    if name=='round_a':return HERE/f'data/{tag}_numbers_final_selected.parquet'
    if name=='round_b':return O.D/f'data/{tag}_evidence_average_selected.parquet'
    return C/f'data/{tag}_{name}_selected.parquet'

def prepared_environment(density):
    tag='density' if density else 'ordinary';env=load_environment(density)
    p=R.modified(env[0],correction('numbers',density),1.,'all')
    cache=C/f'data/{tag}_prefixes.parquet';wpath=C/f'data/{tag}_eligible.parquet'
    if cache.exists():t=pl.read_parquet(cache);w=pl.read_parquet(wpath)
    else:
        b,_=O.band('numbers',density);t,w=O.prefixes(p,b)
        t=t.select(list(dict.fromkeys(['s1','k']+FE)))
        w=w.select('s1','qid','k')
        assert t.select(pl.struct('s1','k').n_unique()).item()==t.height
        t.write_parquet(cache);w.write_parquet(wpath)
    return tag,env,p,t,w

def pick(t,w,u):
    best=t.select('s1','k').with_columns(pl.Series('u',u)).sort(['s1','u','k'],descending=[False,True,False]).group_by('s1',maintain_order=True).first().select('s1',pl.col('k').alias('kb'))
    return w.join(best,on='s1').filter(pl.col('k')<=pl.col('kb')).select(pl.col('s1').alias('s1_idx'),pl.col('qid').alias('cand_idx'))

def guard(s,a,p,veto,meta):
    q=meta.select('s1_idx').join(s.group_by('s1_idx').len().rename({'len':'n'}),on='s1_idx',how='left').join(a.group_by('s1_idx').len().rename({'len':'a'}),on='s1_idx',how='left').with_columns(pl.col('n','a').fill_null(0))
    allow=q.filter((pl.col('n')==0)==(pl.col('a')==0)).select('s1_idx')
    mixed=pl.concat([s.join(allow,on='s1_idx',how='semi'),a.join(allow,on='s1_idx',how='anti')])
    return R.post(mixed,p,veto)

def utilities(t,tag,decoder):
    cache=C/f'data/{tag}_utilities.parquet'
    if cache.exists():
        u=pl.read_parquet(cache)
        assert u.select('s1','k').equals(t.select('s1','k'))
        return {n:u[n].to_numpy() for n in u.columns if n not in ('s1','k')}
    out={}
    for name in ['anchor','standard','evidence']+list(MODEL_CONFIG):
        if name=='anchor':model=decoder;fe=O.BASE
        elif name in ('standard','evidence'):
            model=lgb.Booster(model_file=str(O.D/f'models/decoder_{name}.txt'));fe=O.BASE if name=='standard' else FE
        else:model=lgb.Booster(model_file=str(C/f'models/{name}.txt'));fe=FE
        assert model.num_feature()==len(fe)
        out[name]=model.predict(t.select(fe).to_numpy().astype(np.float32),num_threads=4)
        log('UTILITY',tag,name)
    t.select('s1','k').with_columns(*[pl.Series(n,u) for n,u in out.items()]).write_parquet(cache)
    return out

def generate(density=False):
    tag,env,p,t,w=prepared_environment(density);u=utilities(t,tag,env[7])
    a=pl.read_parquet(source('round_a',tag));b=pl.read_parquet(source('round_b',tag))
    parity={}
    for name,path in [('anchor',HERE/f'data/{tag}_numbers_a1_average_selected.parquet'),('standard',O.D/f'data/{tag}_standard_average_selected.parquet'),('evidence',source('round_b',tag))]:
        s=R.post(pick(t,w,u[name]),p,env[4]);expected=pl.read_parquet(path)
        assert s.join(expected,on=K,how='anti').height==expected.join(s,on=K,how='anti').height==0
        parity[name]=True
    assert guard(a,a,p,env[4],env[2]).sort(K).equals(a.sort(K))
    put(f'parity_{tag}.json',parity)
    configs={f'blend_e{weight:g}':(1-weight)*u['anchor']+weight*u['evidence'] for weight in (.25,.5,.75)}
    configs['blend_standard_evidence']=.5*(u['standard']+u['evidence'])
    configs.update({name:u[name] for name in MODEL_CONFIG})
    configs.update({f'blend_{name}':.5*(u['anchor']+u[name]) for name in ('q63e2','q31e2')})
    outputs={'round_a':a,'round_b':b,'round_b_guard':guard(b,a,p,env[4],env[2])}
    for name,utility in configs.items():
        s=R.post(pick(t,w,utility),p,env[4]);outputs[name]=s;outputs[name+'_guard']=guard(s,a,p,env[4],env[2])
    assert len(outputs)==23
    tm=env[2].filter(pl.col('eval_split')=='tune');base=S.score(a,tm,env[3]);metrics=[]
    for name,s in outputs.items():
        assert s.unique(K).height==s.height and s['cand_idx'].n_unique()==s.height
        assert s.join(p.select(K),on=K,how='anti').height==0
        if name not in ('round_a','round_b'):s.write_parquet(source(name,tag))
        rows=S.score(s,tm,env[3]);assert rows['s1_idx'].equals(base['s1_idx'])
        d=rows['f'].to_numpy()-base['f'].to_numpy();country=rows['country'].to_numpy();empty=rows['m'].to_numpy()==0
        r={'name':name,'score':float(rows['f'].mean()),'delta':float(d.mean()),'countries':{c:float(d[country==c].mean()) for c in ('US','India')},'empty_net_extra_errors':-int(round(float(d[empty].sum()))),'improved':int((d>1e-12).sum()),'harmed':int((d< -1e-12).sum()),'perfect_harmed':int(((base['f'].to_numpy()==1)&(d< -1e-12)).sum())}
        metrics.append(r);log('TUNE',tag,r)
    put(f'tune_{tag}.json',metrics)

def freeze():
    o=json.loads((C/'tune_ordinary.json').read_text());d={x['name']:x for x in json.loads((C/'tune_density.json').read_text())}
    rows=[]
    for x in o:
        y=d[x['name']];r={'name':x['name'],'ordinary':x,'density':y,'robust_delta':min(x['delta'],y['delta']),'average_delta':(x['delta']+y['delta'])/2}
        r['eligible']=r['robust_delta']>=-1e-15 and min(*x['countries'].values(),*y['countries'].values())>=-.00002 and max(x['empty_net_extra_errors'],y['empty_net_extra_errors'])<=1
        rows.append(r)
    winner=sorted([x for x in rows if x['eligible']],key=lambda x:(-x['robust_delta'],-x['average_delta'],x['name']))[0]
    ordinary=sorted([x for x in rows if min(x['ordinary']['countries'].values())>=-.00002],key=lambda x:(-x['ordinary']['score'],x['name']))[0]
    put('frozen_selection.json',{'robust_winner':winner,'ordinary_winner':ordinary,'candidates':rows,'selection':'ordinary and density TUNE only','new_report_opened_at_freeze':False,'historical_holdout_exposure':True})
    log('FROZEN',winner['name'],ordinary['name'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','generate','freeze']);ap.add_argument('--model',choices=MODEL_CONFIG);ap.add_argument('--density',action='store_true');args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='train':train_model(args.model)
    elif args.action=='generate':generate(args.density)
    else:freeze()
