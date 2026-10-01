"""Learn richer residuals without modifying old models or validation labels."""
import os
for key in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
os.environ['OMP_WAIT_POLICY']='PASSIVE'
import sys,json,time,re,unicodedata,argparse,gc
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.special import logit
from rapidfuzz.distance import Levenshtein,DamerauLevenshtein
from rapidfuzz import fuzz
from experiment import HERE,M,OLD,K,SEEDS,log,save

NF=['nf_count1','nf_count2','nf_first_missing1','nf_first_missing2',
    'nf_digit_lev','nf_digit_damerau','nf_digit_norm','nf_len_diff',
    'nf_single_sub','nf_single_indel','nf_transposition','nf_zero_equal',
    'nf_substring','nf_shared_prefix','nf_shared_suffix','nf_firstpos1',
    'nf_firstpos2','nf_startsnum1','nf_startsnum2','nf_unit_before1',
    'nf_unit_before2','nf_compound1','nf_compound2','nf_rest_equal',
    'nf_multiset_equal','nf_order_changed','nf_shared_count','nf_first_in_other',
    'nf_other_first_in_self','nf_alpha_ratio','nf_alpha_tset','nf_alpha_equal',
    'nf_alpha_lenratio','nf_num_only1','nf_num_only2']
RX=re.compile(r'\d+')
UNIT=re.compile(r'\b(?:unit|suite|ste|apt|apartment|floor|fl|bldg|building|room|rm|plot|flat)\b',re.I)

def parse(text):
    s=unicodedata.normalize('NFKC',text or '').lower()
    matches=list(RX.finditer(s));numbers=[x.group() for x in matches]
    first=matches[0] if matches else None
    letters=re.sub(r'[^\w\s]',' ',RX.sub(' ',s));letters=' '.join(letters.split())
    pos=first.start() if first else -1
    compound=bool(first and ((pos>0 and s[pos-1].isalpha()) or (first.end()<len(s) and s[first.end()].isalpha())))
    return numbers,letters,pos,bool(first and not s[:pos].strip()),bool(first and UNIT.search(s[:pos])),compound

def feature_pair(a,b):
    na,la,pa,sa,ua,ca=a;nb,lb,pb,sb,ub,cb=b
    both=bool(na and nb)
    if both:
        x,y=na[0],nb[0];lev=Levenshtein.distance(x,y);dam=DamerauLevenshtein.distance(x,y)
        prefix=0
        for v,w in zip(x,y):
            if v!=w:break
            prefix+=1
        suffix=0
        for v,w in zip(x[::-1],y[::-1]):
            if v!=w:break
            suffix+=1
        s1=set(na);s2=set(nb)
        numeric=[lev,dam,lev/max(len(x),len(y)),len(y)-len(x),int(lev==1 and len(x)==len(y)),int(lev==1 and len(x)!=len(y)),int(dam==1 and lev==2),int(int(x)==int(y)),int(x in y or y in x),prefix,suffix]
        # Empty suffixes contain no corroborating number evidence.
        tail=int(na[1:]==nb[1:]) if len(na)>1 and len(nb)>1 else -1
        relations=[tail,int(sorted(na)==sorted(nb)),int(sorted(na)==sorted(nb) and na!=nb),len(s1&s2),int(x in s2),int(y in s1)]
    else:numeric=[-1.]*11;relations=[-1.]*6
    alphabetic=[fuzz.ratio(la,lb)/100,fuzz.token_set_ratio(la,lb)/100,int(la==lb),min(len(la),len(lb))/max(len(la),len(lb))] if la and lb else [-1.]*4
    return [len(na),len(nb),int(not na),int(not nb),*numeric,pa,pb,int(sa),int(sb),int(ua),int(ub),int(ca),int(cb),*relations,*alphabetic,int(bool(na) and not la),int(bool(nb) and not lb)]

def extra_features(keys,density=False):
    cache=HERE/f'data/{"density" if density else "ordinary"}_number_features.parquet'
    if cache.exists():return pl.read_parquet(cache)
    arrays=[]
    for kind,idcol in [('s1','s1_idx'),('s23','cand_idx')]:
        need=keys.select(pl.col(idcol).cast(pl.UInt32).alias('erow')).unique()
        r=(pl.scan_parquet(M/f'prod_v1/data/rec_train_{kind}.parquet').select('erow','addr_raw')
           .join(need.lazy(),on='erow',how='semi').collect())
        arrays.append({int(i):parse(a) for i,a in r.iter_rows()})
        log('NUMBER RECORDS',kind,len(arrays[-1]))
    mat=np.empty((keys.height,len(NF)),dtype=np.float32)
    for j,(s,c) in enumerate(keys.iter_rows()):
        mat[j]=feature_pair(arrays[0][s],arrays[1][c])
        if j and j%250000==0:log('NUMBER FEATURES',j,keys.height)
    out=keys.hstack(pl.DataFrame(mat,schema=NF))
    assert out.height==keys.height and out.select(NF).null_count().sum_horizontal().item()==0
    out.write_parquet(cache);save('number_features.json',NF)
    log('NUMBER FEATURES DONE',keys.height,len(NF));return out

def data(arm,density=False):
    b=pl.read_parquet(OLD/('data/density_band.parquet' if density else 'data/anchor_band.parquet'))
    features=json.loads((OLD/'features.json').read_text())
    if arm=='numbers':
        extra=extra_features(b.select(K),density)
        b=b.join(extra,on=K,how='left',maintain_order='left');features+=NF
    assert len(features)==len(set(features))
    return b,features

def fit(arm):
    b,features=data(arm)
    x=np.nan_to_num(b.select(features).to_numpy().astype(np.float32),nan=-1,posinf=1e6,neginf=-1e6)
    y=b['label'].to_numpy();z=logit(b['p2'].to_numpy().astype(float))
    tr=((b['grp']=='sample')&~b['sample_es']).to_numpy();es=((b['grp']=='sample')&b['sample_es']).to_numpy()
    out=b.select(K+['grp','eval_split','missing_gate','p2']);info=[]
    save(f'{arm}_features.json',features)
    params=dict(objective='binary',metric='binary_logloss',learning_rate=.035,num_leaves=31,min_data_in_leaf=250,lambda_l2=30,feature_fraction=.85,bagging_fraction=.8,bagging_freq=1,num_threads=4,verbosity=-1)
    for seed in SEEDS:
        path=HERE/f'models/{arm}_{seed}.txt'
        if path.exists():model=lgb.Booster(model_file=str(path))
        else:
            ds=lgb.Dataset(x[tr],label=y[tr],init_score=z[tr],feature_name=features)
            esd=lgb.Dataset(x[es],label=y[es],init_score=z[es],reference=ds)
            log('FIT START',arm,seed,int(tr.sum()),int(es.sum()),len(features))
            model=lgb.train({**params,'seed':seed},ds,num_boost_round=1200,valid_sets=[esd],callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(200)])
            model.save_model(str(path));del ds,esd;gc.collect()
        raw=model.predict(x,raw_score=True,num_threads=4).astype(np.float32)
        out=out.with_columns(pl.Series(f'delta_{seed}',raw))
        row={'seed':seed,'iterations':model.current_iteration(),'best_logloss':model.best_score.get('valid_0',{}).get('binary_logloss'),'fit_pairs':int(tr.sum()),'es_pairs':int(es.sum()),'features':len(features),'params':params,'top_features':sorted(zip(features,model.feature_importance('gain').tolist()),key=lambda a:-a[1])[:30]}
        info.append(row);save(f'fit_{arm}.json',info);log('FIT DONE',arm,seed,row['iterations'],row['best_logloss'])
    out.write_parquet(HERE/f'data/{arm}_ordinary_scores.parquet')
    log('ARM COMPLETE',arm)

def density(arm):
    b,features=data(arm,True)
    assert features==json.loads((HERE/f'{arm}_features.json').read_text())
    x=np.nan_to_num(b.select(features).to_numpy().astype(np.float32),nan=-1,posinf=1e6,neginf=-1e6)
    out=b.select(K+['missing_gate'])
    for seed in SEEDS:
        model=lgb.Booster(model_file=str(HERE/f'models/{arm}_{seed}.txt'))
        out=out.with_columns(pl.Series(f'delta_{seed}',model.predict(x,raw_score=True,num_threads=4).astype(np.float32)))
    out.write_parquet(HERE/f'data/{arm}_density_scores.parquet');log('DENSITY SCORED',arm,b.height)

def check_features():
    # Distinguish a transposition from arithmetic distance and missing evidence
    # from agreement. IDs/labels are deliberately absent from this function.
    a=dict(zip(NF,feature_pair(parse('12 Main Street'),parse('21 Main Street'))))
    assert a['nf_transposition']==1 and a['nf_alpha_equal']==1
    a=dict(zip(NF,feature_pair(parse('0012 Main Street'),parse('12 Main Street'))))
    assert a['nf_zero_equal']==1
    a=dict(zip(NF,feature_pair(parse('Suite 4, 12 Main Street'),parse('12 Main Street'))))
    assert a['nf_unit_before1']==1 and a['nf_other_first_in_self']==1
    a=dict(zip(NF,feature_pair(parse(''),parse(''))))
    assert a['nf_rest_equal']==-1 and a['nf_alpha_equal']==-1 and a['nf_digit_lev']==-1
    a=dict(zip(NF,feature_pair(parse('9N 655 Kendall Road'),parse('9N 660 Kendall Road'))))
    assert a['nf_compound1']==1 and a['nf_rest_equal']==0
    log('NUMBER FEATURE SEMANTICS PASSED')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['fit','density','check']);ap.add_argument('--arm',choices=['capacity','numbers'],default='capacity');a=ap.parse_args()
    check_features()
    if a.action=='fit':fit(a.arm)
    elif a.action=='density':density(a.arm)
