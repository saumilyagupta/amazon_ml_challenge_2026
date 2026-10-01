#!/usr/bin/env python3
"""Rebuild the submitted `matching_results.tsv` from the FROZEN decision-layer tables shipped in `resources/ens8_frozen/`.

The submitted file (upload #10, `R_US_LEGADD_FR9`) is upload #8 (`nc_specialist_legal_fr`, the v3anchor ensemble chain) minus a frozen
list of label-free decoy removals.  Upload #8 itself is the Round C blend decision ("NC", `decision/test_selected.parquet`)
+ the empty-address specialist additions (`decision/test_nc_robust_additions.parquet`)
- the France legal / country-tag cleanup mask (`decision/mask_ppLegal_FR.parquet`).
This script re-applies exactly those set operations (the logic of the research scripts `iterate_20260926/src/build_candidate.py` and
`winning_strategy_20260926/execution_20260926/src/reproduce_candidate.py`, audit copies under `research_ensemble/`), writes the TSV with
the writer of `v2shash/round_c/test_release/release.py::write` (polars `write_csv`, tab separated, never quoted) and asserts the sha256
of the result against `resources/ens8_frozen/manifest.json`.

It does NOT recompute the pair probabilities, the decoders, the specialist scores or the mask / removal rules: those need the research tree
(see README.md, section 8).  It needs only polars and the standard library; ~2 GB RAM; < 1 min per target at 4 threads.

usage (from anywhere; paths are relative to this file unless --resources / --out-dir are given):
  python3 ens8_reproduce.py                                  # default target R_US_LEGADD_FR9 = the submitted file
  python3 ens8_reproduce.py --target P8                      # upload #8
  python3 ens8_reproduce.py --all                            # all five targets, every sha256 asserted
  python3 ens8_reproduce.py --check-candidate ../../../output/candidate_pairs.tsv   # also verify the shipped candidate file's sha256
Targets: P8, A_US_LEG, B_US_LEG_FR9, R_US_LEGADD, R_US_LEGADD_FR9 (default).  Output: <out-dir>/<target>/matching_results.tsv (+ summary.json).
`candidate_pairs.tsv` is the same union-v2 blocking file for every target (manifest key `candidate_pairs`); it is shipped as
`output/candidate_pairs.tsv` and cannot be derived from these tables (it is the output of block.py, see README.md section 3).
"""
import os
for _k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
sys.dont_write_bytecode = True
import polars as pl

HERE = Path(__file__).resolve().parent
K = ['s1_idx', 'cand_idx']
N_S1 = 1732544
TARGETS = ['P8', 'A_US_LEG', 'B_US_LEG_FR9', 'R_US_LEGADD', 'R_US_LEGADD_FR9']
DEFAULT_TARGET = 'R_US_LEGADD_FR9'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def digest(path, algo='sha256'):
    with open(path, 'rb') as fh:
        return hashlib.file_digest(fh, algo).hexdigest()


def check_file(res, manifest, rel):
    """Assert that a shipped resource is byte-identical to the one recorded in the manifest."""
    entry = manifest['files'][rel]
    path = res / rel
    assert path.exists(), f'missing resource {path}'
    got = digest(path)
    assert got == entry['sha256'], f'{rel}: sha256 {got} != manifest {entry["sha256"]}'
    return path


def load_ids(res):
    i1 = (pl.read_parquet(res / 'ids/test_s1.parquet', columns=['entity_id', 'country'])
          .with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32)))
    i2 = (pl.read_parquet(res / 'ids/test_s23.parquet', columns=['entity_id', 'country'])
          .with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32)))
    assert i1.height == N_S1 and i1['entity_id'].n_unique() == N_S1
    assert i2['entity_id'].n_unique() == i2.height
    return i1, i2


def build_p8(res, i1, i2):
    """#8 = NC selection + specialist additions - France cleanup mask (logic of iterate_20260926/src/build_candidate.py)."""
    base = pl.read_parquet(res / 'decision/test_selected.parquet').select(K)
    add = pl.read_parquet(res / 'decision/test_nc_robust_additions.parquet').select(K)
    mask = pl.read_parquet(res / 'decision/mask_ppLegal_FR.parquet').select(K)
    assert base.height == base['cand_idx'].n_unique(), 'NC selection is not one-owner'
    assert add.unique().height == add.height and add['cand_idx'].n_unique() == add.height
    assert add.join(base.select('cand_idx'), on='cand_idx', how='semi').height == 0, 'addition already claimed'
    ac = (add.join(i1.select('s1_idx', 'country'), on='s1_idx')
             .join(i2.select('cand_idx', pl.col('country').alias('candidate_country')), on='cand_idx'))
    assert ac.height == add.height
    assert ac.filter(~pl.col('country').is_in(['US', 'India']) | (pl.col('country') != pl.col('candidate_country'))).height == 0, \
        'specialist additions must be US / India and same-country'
    remove = base.join(mask, on=K, how='semi')
    assert remove.join(i1, on='s1_idx').filter(pl.col('country') != 'France').height == 0, 'France mask touched a non-France S1'
    selected = pl.concat([base, add]).join(mask, on=K, how='anti')
    assert selected.height == base.height + add.height - remove.height
    assert selected['cand_idx'].n_unique() == selected.height, 'one-owner violated'
    counts = dict(nc_pairs=base.height, additions=add.height, france_mask_pairs=mask.height,
                  france_removed=remove.height, p8_pairs=selected.height,
                  added_by_country=dict(ac.group_by('country').len().sort('country').iter_rows()))
    return selected, counts


def apply_removals(selected, res, manifest, target, i1, i2):
    """A/B/R/R_FR9 = #8 minus the frozen removed_pairs.tsv (logic of execution_20260926/src/reproduce_candidate.py)."""
    rel = f'removals/{target}/removed_pairs.tsv'
    path = check_file(res, manifest, rel)
    rem = pl.read_csv(path, separator='\t', schema={'s1_id': pl.String, 'cand_id': pl.String})
    assert rem.unique().height == rem.height
    mapped = (rem.join(i1.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id', how='left')
                 .join(i2.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id', how='left'))
    assert mapped['s1_idx'].null_count() == 0 and mapped['cand_idx'].null_count() == 0, 'removed pair id not in the test ids'
    rem_idx = mapped.select(K)
    inside = rem_idx.join(selected, on=K, how='semi').height
    assert inside == rem_idx.height, f'{rem_idx.height - inside} removed pairs are not in the #8 selection'
    expected = manifest['targets'][target]['removed_pairs']
    assert rem_idx.height == expected, f'{target}: {rem_idx.height} removals in the list, manifest says {expected}'
    out = selected.join(rem_idx, on=K, how='anti')
    assert out.height == selected.height - rem_idx.height
    by_country = dict(rem_idx.join(i1.select('s1_idx', 'country'), on='s1_idx').group_by('country').len().sort('country').iter_rows())
    return out, dict(removed_pairs=rem_idx.height, removed_by_country=by_country, removed_s1=rem_idx['s1_idx'].n_unique())


def write_tsv(selected, i1, i2, path):
    """Byte-identical to v2shash/round_c/test_release/release.py::write."""
    grouped = (selected.join(i2.select('cand_idx', 'entity_id'), on='cand_idx')
                       .group_by('s1_idx').agg(pl.col('entity_id').sort().str.join(',').alias('matched_entity_ids')))
    table = (i1.join(grouped, on='s1_idx', how='left').sort('s1_idx')
               .select(pl.col('entity_id').alias('source1_entity_id'), pl.col('matched_entity_ids').fill_null('')))
    assert table.height == N_S1
    table.write_csv(path, separator='\t', quote_style='never')
    return table


def reproduce(target, res, manifest, out_dir, i1, i2, p8_cache):
    t0 = time.time()
    info = manifest['targets'][target]
    if p8_cache.get('selected') is None:
        p8_cache['selected'], p8_cache['counts'] = build_p8(res, i1, i2)
        log('#8 selection rebuilt:', json.dumps(p8_cache['counts']))
    selected, counts = p8_cache['selected'], dict(p8_cache['counts'])
    if info.get('removals'):
        selected, rc = apply_removals(selected, res, manifest, info['removals'], i1, i2)
        counts.update(rc)
    out = out_dir / target
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / 'matching_results.tsv'
    table = write_tsv(selected, i1, i2, tsv)
    sha, md5 = digest(tsv), digest(tsv, 'md5')
    ok = sha == info['sha256']
    summary = dict(target=target, label=info.get('label'), output=str(tsv), rows=table.height, pairs=selected.height,
                   empty_rows=int((table['matched_entity_ids'] == '').sum()), bytes=tsv.stat().st_size,
                   sha256=sha, md5=md5, expected_sha256=info['sha256'], expected_md5=info.get('md5'),
                   sha256_match=ok, leaderboard=info.get('leaderboard'), counts=counts, seconds=round(time.time() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    log(f'{target}: pairs {selected.height:,}  sha256 {sha[:16]}...  {"MATCH" if ok else "MISMATCH"}  ({summary["seconds"]} s)')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', choices=TARGETS, default=DEFAULT_TARGET)
    ap.add_argument('--all', action='store_true', help='rebuild all five targets')
    ap.add_argument('--resources', type=Path, default=HERE / 'resources/ens8_frozen', help='directory holding manifest.json')
    ap.add_argument('--out-dir', type=Path, default=HERE / 'output_ens8', help='outputs go to <out-dir>/<target>/')
    ap.add_argument('--check-candidate', type=Path, default=None,
                    help='also verify the sha256 of this candidate_pairs.tsv against the manifest (the shipped union-v2 file)')
    ap.add_argument('--skip-input-hashes', action='store_true', help='do not re-hash the shipped input tables first')
    a = ap.parse_args()
    res = a.resources.resolve()
    manifest = json.loads((res / 'manifest.json').read_text())
    inputs = ['decision/test_selected.parquet', 'decision/test_nc_robust_additions.parquet', 'decision/mask_ppLegal_FR.parquet',
              'ids/test_s1.parquet', 'ids/test_s23.parquet']
    if not a.skip_input_hashes:
        for rel in inputs:
            check_file(res, manifest, rel)
        log('input tables verified against manifest.json:', ', '.join(inputs))
    if a.check_candidate is not None:
        c = manifest['candidate_pairs']
        got = digest(a.check_candidate)
        assert got == c['sha256'], f'candidate_pairs sha256 {got} != manifest {c["sha256"]}'
        log('candidate_pairs.tsv verified:', a.check_candidate, 'sha256', got[:16] + '...')
    i1, i2 = load_ids(res)
    targets = TARGETS if a.all else [a.target]
    cache, results = {}, []
    for t in targets:
        results.append(reproduce(t, res, manifest, a.out_dir.resolve(), i1, i2, cache))
    failed = [r['target'] for r in results if not r['sha256_match']]
    print(json.dumps(results, indent=2))
    if failed:
        log('SHA256 MISMATCH for', failed)
        sys.exit(1)
    log('ALL TARGETS REPRODUCED BYTE-EXACTLY:', ', '.join(targets))


if __name__ == '__main__':
    main()
