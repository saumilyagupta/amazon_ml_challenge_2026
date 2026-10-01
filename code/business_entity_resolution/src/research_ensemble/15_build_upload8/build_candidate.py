#!/usr/bin/env python3
"""Combine an existing submission, frozen US/India additions, and France cleanup."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '4'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
import argparse, hashlib, json
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
W = ROOT.parents[1]
K = ['s1_idx','cand_idx']

def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--additions', required=True)
    ap.add_argument('--base-selected', default=str(W/'matching/v2shash/round_c/test_release/data/test_selected.parquet'))
    ap.add_argument('--base-matching', default=str(W/'matching/v2shash/round_c/test_release/output/matching_results.tsv'))
    ap.add_argument('--label', default='nc_specialist_legal_fr')
    a = ap.parse_args()
    out = ROOT/'submissions'/a.label
    out.mkdir(parents=True, exist_ok=True)
    base = pl.read_parquet(a.base_selected).select(K)
    add = pl.read_parquet(a.additions).select(K)
    i1 = pl.read_parquet(W/'blocking/embedding/full/ids/test_s1.parquet',columns=['entity_id','country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    i2 = pl.read_parquet(W/'blocking/embedding/full/ids/test_s23.parquet',columns=['entity_id','country']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
    assert add.unique().height == add.height
    assert add['cand_idx'].n_unique() == add.height
    assert add.join(base.select('cand_idx'),on='cand_idx',how='semi').height == 0, 'addition already claimed'
    ac = add.join(i1.select('s1_idx','country'),on='s1_idx').join(i2.select('cand_idx',pl.col('country').alias('candidate_country')),on='cand_idx')
    assert ac.filter(~pl.col('country').is_in(['US','India']) | (pl.col('country') != pl.col('candidate_country'))).height == 0
    mask = pl.read_parquet(ROOT/'france/data/mask_ppLegal_FR.parquet').select(K)
    remove = base.join(mask,on=K,how='semi')
    assert remove.join(i1,on='s1_idx').filter(pl.col('country')!='France').height == 0
    selected = pl.concat([base,add]).join(mask,on=K,how='anti')
    assert selected.height == base.height + add.height - remove.height
    assert selected['cand_idx'].n_unique() == selected.height
    selected.write_parquet(out/'selected.parquet')
    def as_id_map(pairs):
        mapped = pairs.join(i1.select('s1_idx',pl.col('entity_id').alias('qid')),on='s1_idx').join(i2.select('cand_idx',pl.col('entity_id').alias('rid')),on='cand_idx')
        assert mapped.height == pairs.height
        return {q:set(ids) for q,ids in mapped.group_by('qid').agg('rid').iter_rows()}
    additions, removals = as_id_map(add), as_id_map(remove)
    rows = changed = pairs = new_empty = 0
    with open(a.base_matching) as fi, open(out/'matching_results.tsv','w') as fo:
        header = next(fi)
        assert header.strip() == 'source1_entity_id\tmatched_entity_ids'
        fo.write(header)
        for line in fi:
            q, rest = line.rstrip('\n').split('\t')
            old = set(rest.split(',')) if rest else set()
            assert removals.get(q,set()) <= old
            assert not additions.get(q,set()).intersection(old)
            new = (old-removals.get(q,set())) | additions.get(q,set())
            if new != old:
                fo.write(q+'\t'+','.join(sorted(new))+'\n')
                changed += 1
            else:
                fo.write(line)
            new_empty += bool(old) and not bool(new)
            pairs += len(new)
            rows += 1
    assert rows == 1732544 and pairs == selected.height
    candidate = W/'matching/ensemble_v1/build/v3anchor/output/candidate_pairs.tsv'
    link = out/'candidate_pairs.tsv'
    if not link.exists():
        link.symlink_to(os.path.relpath(candidate, out))
    info = dict(label=a.label, rows=rows, match_pairs=pairs, additions=add.height,
                added_by_country=ac.group_by('country').len().to_dicts(), removed_france_pairs=remove.height,
                changed_queries=changed, new_empty_queries=new_empty,
                base_matching=a.base_matching, base_selected=a.base_selected, additions_file=a.additions,
                matching_sha256=sha(out/'matching_results.tsv'), base_matching_sha256=sha(a.base_matching),
                selected_sha256=sha(out/'selected.parquet'), candidate_sha256=sha(candidate),
                ordinary_density_note='France cleanup is a no-op on labeled validation. Specialist is US/India only.',
                public_score='not uploaded')
    (out/'manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2),flush=True)

if __name__ == '__main__':
    main()
