"""Verify the baseline and release invariants beyond the official validator."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import json
from pathlib import Path
import polars as pl
from audit import W,ROOT,OUT,IT,KEY

def main():
    base=IT/'submissions/nc_specialist_legal_fr/matching_results.tsv'
    countries=dict(pl.read_parquet(W/'blocking/embedding/full/ids/test_s1.parquet',columns=['entity_id','country']).iter_rows())
    seen=set();qseen=set();npairs=0
    # Enforce one-owner against actual output text rather than a cached mask.
    with open(base) as fh:
        next(fh)
        for line in fh:
            q,rest=line.rstrip('\n').split('\t');assert q not in qseen;qseen.add(q)
            ids=rest.split(',') if rest else []
            assert len(ids)==len(set(ids))
            for c in ids:assert c not in seen;seen.add(c)
            npairs+=len(ids)
    assert qseen==set(countries)
    del seen,qseen
    result={'baseline':{'rows':len(countries),'pairs':npairs,'duplicate_rows':0,'duplicate_matches':0,'multiple_owners':0},'candidates':{}}
    for folder in sorted((ROOT/'release').iterdir()):
        if not folder.is_dir():continue
        manifest=json.loads((folder/'manifest.json').read_text());removed=changed=newempty=0;bycountry={c:0 for c in ['US','India','France']}
        with open(base) as fi,open(folder/'matching_results.tsv') as fj:
            assert next(fi)==next(fj)
            for l1,l2 in zip(fi,fj,strict=True):
                if l1==l2:continue
                q,a=l1.rstrip('\n').split('\t');q2,b=l2.rstrip('\n').split('\t');assert q==q2
                aa=set(a.split(',')) if a else set();bb=set(b.split(',')) if b else set()
                assert bb<=aa and len(bb)==(len(b.split(',')) if b else 0)
                removed+=len(aa-bb);changed+=int(aa!=bb);newempty+=int(bool(aa) and not bb);bycountry[countries[q]]+=len(aa-bb)
        assert removed==manifest['removed_pairs'] and changed==manifest['changed_s1'] and newempty==manifest['new_empty']
        result['candidates'][folder.name]={'removed':removed,'changed_s1':changed,'new_empty':newempty,'added':0,'multiple_owners':0,'removed_by_country':bycountry}
    # Verify source country of the original selected pairs. All candidates are
    # proven subsets above, so the country invariant is inherited exactly.
    s1=pl.read_parquet(W/'blocking/embedding/full/ids/test_s1.parquet',columns=['country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    s23=pl.read_parquet(W/'blocking/embedding/full/ids/test_s23.parquet',columns=['country']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32)).rename({'country':'record_country'})
    selected=pl.read_parquet(IT/'submissions/nc_specialist_legal_fr/selected.parquet').select(KEY)
    check=selected.join(s1,on='s1_idx').join(s23,on='cand_idx')
    assert check.height==npairs and check.filter(pl.col('country')!=pl.col('record_country')).is_empty()
    result['baseline']['cached_selection_country_mismatches']=0
    json.dump(result,open(OUT/'integrity.json','w'),indent=2)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
