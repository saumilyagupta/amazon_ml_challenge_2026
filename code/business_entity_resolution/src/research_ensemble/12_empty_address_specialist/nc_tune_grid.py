from eval_nc import *
all_results={}
for kind in ['val','density']:
 d,rows,sel=nc_base(kind);d=d.filter(pl.col('eval_split')=='tune');rows=rows.filter(pl.col('eval_split')=='tune');out=[]
 for alpha in [.25,.5,1.]:
  corrected_d=corrected(d,alpha)
  for threshold in [.75,.8,.85,.9]:
   cfg=dict(alpha=alpha,threshold=threshold);a=add(corrected_d,cfg);rep,_=describe(a,rows);rep['config']=cfg;out.append(rep)
 all_results[kind]=out
 (O/'nc_tune_grid.json').write_text(json.dumps(all_results,indent=2))
for v,d in zip(all_results['val'],all_results['density']):
 assert v['config']==d['config']
 print(v['config'], 'ordinary',round(v['delta'],8),'density',round(d['delta'],8),'ordinary_country',v['countries'],'density_country',d['countries'],'counts',v['added_pairs'],d['added_pairs'],flush=True)
