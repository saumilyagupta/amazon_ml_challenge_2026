"""Independent set-based scoring on ordinary and existing density validation.
No fitting, no new threshold selection; rules are frozen before this run.
"""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import sys,json
import polars as pl
from audit import W,ROOT,OUT,S,IT,KEY,original_rule,nogroupempty
sys.path.insert(0,str(W/'common'))
import score

def main():
    gt=score.load_id_lists(W/'splits/val_ground_truth.tsv')
    rec=pl.read_parquet(W/'blocking/embedding/full/ids/train_s23.parquet',columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
    fullpairs=pl.read_parquet(S/'pairs_val.parquet')
    results={}
    for density in [False,True]:
        label='density' if density else 'ordinary'
        rows=pl.read_parquet(IT/f'results/specialist_nc_{label}_rows.parquet')
        ids=rows['entity_id'].to_list()
        sel=pl.read_parquet(IT/f'empty_address/{"density" if density else "val"}_nc_robust_selected.parquet').select(KEY)
        a=original_rule('val',sel)
        plus=fullpairs.filter((pl.col('country')=='US')&(pl.col('fam')=='LEGADD')&pl.col('bucket').is_in(['D12','D3'])&(pl.col('side')=='+')).select(KEY).join(sel,on=KEY).join(a,on=KEY,how='anti')
        plus=nogroupempty(plus,sel.join(a,on=KEY,how='anti'))
        r=pl.concat([a,plus]).unique()
        if not density:
            frozen=pl.read_parquet(OUT/'R_val_removals.parquet')
            assert r.height==frozen.height and r.join(frozen,on=KEY,how='anti').is_empty()
        variants={'baseline':sel,'A':sel.join(a,on=KEY,how='anti'),'R':sel.join(r,on=KEY,how='anti')}
        for name,selected in variants.items():
            mapped=selected.join(rec,on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').alias('matches'))
            qr=rows.select('s1_idx','entity_id','country','eval_split','is_locked').join(mapped,on='s1_idx',how='left')
            pred={q:set(cs or []) for q,cs in qr.select('entity_id','matches').iter_rows()}
            metrics=score.macro_f05(pred,gt,ids=ids,verbose=False)
            assert selected.height==sum(map(len,pred.values()))
            if name=='baseline':assert abs(metrics['macro_f05']-rows['f'].mean())<1e-12
            sub={}
            for split in ['tune','report','locked']:
                subids=rows.filter(pl.col('eval_split')==split)['entity_id'].to_list()
                sub[split]=score.macro_f05(pred,gt,ids=subids,verbose=False)['macro_f05']
            results[label+'_'+name]={'metrics':metrics,'split_scores':sub}
            if name!='baseline':
                selected.write_parquet(OUT/f'{label}_{name}_selected.parquet')
                with open(OUT/f'{label}_{name}_matching_results.tsv','w') as fh:
                    fh.write('source1_entity_id\tmatched_entity_ids\n')
                    for q in ids:fh.write(q+'\t'+','.join(sorted(pred[q]))+'\n')
            print(label,name,metrics['macro_f05'],sub,flush=True)
    json.dump(results,open(OUT/'official_scores.json','w'),indent=2)

if __name__=='__main__':main()
