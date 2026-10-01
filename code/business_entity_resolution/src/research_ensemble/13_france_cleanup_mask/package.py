import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[k]='8'
import json,hashlib,argparse
from pathlib import Path
import polars as pl
ROOT=Path('/workspace/saumilya/amazon-ml/work');OUT=ROOT/'matching/iterate_20260926/france';K=['s1_idx','cand_idx']
src={'vr2':(ROOT/'internal_eval/data/sel/g7_vr2.parquet',ROOT/'results_analysis/submitted_tsv/07_vr2_consensus_2026-09-26.tsv'),'nc':(ROOT/'matching/v2shash/round_c/test_release/data/test_selected.parquet',ROOT/'matching/v2shash/round_c/test_release/output/matching_results.tsv')}
I1=pl.read_parquet(ROOT/'blocking/embedding/full/ids/test_s1.parquet',columns=['entity_id','country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
I2=pl.read_parquet(ROOT/'blocking/embedding/full/ids/test_s23.parquet',columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['ppLegal_FR','ppLegalAnchor_FR'],default='ppLegal_FR');args=ap.parse_args()
manifest={}
for base,(pq,tsv) in src.items():
 name=f'{base}_{args.variant}'; target=OUT/'candidates'/name;target.mkdir(parents=True,exist_ok=True)
 S=pl.read_parquet(pq).select(K);R=pl.read_parquet(OUT/f'data/{name}_removed.parquet');cur=S.join(R,on=K,how='anti');cur.write_parquet(target/'selected.parquet')
 ri=R.join(I1,on='s1_idx').join(I2.rename({'entity_id':'record_id'}),on='cand_idx');assert ri['country'].unique().to_list()==['France'];rem={}
 for q,c in ri.select('entity_id','record_id').iter_rows():rem.setdefault(q,set()).add(c)
 out=target/'matching_results.tsv';rows=changed=removed=emptied=0
 with tsv.open() as fi,out.open('w') as fo:
  fo.write(next(fi))
  for line in fi:
   row=line.rstrip('\n').split('\t');q=row[0];matches=row[1];rows+=1
   if q in rem:
    old=matches.strip('"').split(',');new=[c for c in old if c not in rem[q]];n=len(old)-len(new);assert n==len(rem[q]);removed+=n;changed+=1;emptied+=int(len(new)==0);fo.write(q+'\t'+','.join(new)+'\n')
   else:fo.write(line)
 assert rows==1732544 and removed==R.height and changed==len(rem)
 # mask is removal-only and all candidate/unique-owner validity follows from source.
 assert cur.height==S.height-removed and cur['cand_idx'].n_unique()==cur.height
 sha=hashlib.sha256();md=hashlib.md5()
 with out.open('rb') as f:
  for b in iter(lambda:f.read(1<<22),b''):sha.update(b);md.update(b)
 manifest[name]={'path':str(out),'rows':rows,'pairs':cur.height,'removed_pairs':removed,'changed_queries':changed,'new_empty_queries':emptied,'source':str(tsv),'sha256':sha.hexdigest(),'md5':md.hexdigest(),'mask':('data/mask_ppLegal_FR.parquet' if args.variant=='ppLegal_FR' else f'data/mask_{base}_ppLegalAnchor_FR.parquet'),'rules':['France CTAG positive offsets','France LEG3 positive offsets','France LEG12 positive offsets'],'all_changes_france':True,'ordinary_density_invariant':'France-only; ordinary and density validation have no France queries, so same policy is exactly base'}
 if args.variant=='ppLegalAnchor_FR':
  assert emptied==0
  manifest[name]['rules'].append('Requires a different accepted strict anchor remaining after the veto')
 print(name,manifest[name],flush=True)
(OUT/'out'/('packages.json' if args.variant=='ppLegal_FR' else 'anchor_packages.json')).write_text(json.dumps(manifest,indent=2))
