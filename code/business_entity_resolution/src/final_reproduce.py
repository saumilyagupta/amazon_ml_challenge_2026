#!/usr/bin/env python3
"""Rebuild the submitted `output/matching_results.tsv` byte-exactly: upload #8 (ens8 frozen tables) +/- the frozen pair lists of the final file.

Lineage.  `ens8_reproduce.py` rebuilds upload #8 (Round C blend "NC" selection + empty-address specialist additions - France legal /
country-tag mask) from `resources/ens8_frozen/`, and upload #10 (= #8 minus 6,548 label-free decoy removals).  Every later leaderboard file
of ours is #8 minus a frozen list of removed pairs (`removed_vs8.tsv`) plus, for the files that carry the CE-NC US / India correction, a
frozen list of added pairs (`added_vs8.tsv`; US / India only unless the target's manifest entry sets `allow_france_add`).  Both lists are (s1_id, cand_id) TSVs, stored in
`resources/final_frozen/<target>/` with their sha256 in `resources/final_frozen/manifest.json`; `layers_vs8.tsv` annotates every listed
pair with the removal / correction layer that produced it (audit only, not read here).

What this script does (polars + standard library only; ~2-3 GB RAM; about 1 min at 4 threads):
  1. re-hashes the ens8 input tables and the final lists against the two manifests;
  2. rebuilds #8 with the functions of `ens8_reproduce.py` (same directory) and #10 from #8 minus the frozen #10 list, writes #10 and
     asserts its sha256 against `ens8_frozen/manifest.json` (proves the shared base is byte-exact in the same run);
  3. target = #8 - removed_vs8 + added_vs8, with the containment / disjointness / one-owner / country assertions below, and reports the
     same set as a delta on top of #10 (removed on top of #10, #10 removals the target keeps, additions);
  4. writes the TSV with `ens8_reproduce.write_tsv` (the writer of every upload since #8: rows in test_source1 order, ids of a row sorted
     bytewise ascending and comma-joined, empty string for no match, tab separated, never quoted) and asserts sha256 + md5.
It does NOT recompute pair probabilities, the CE-NC residual, the decoders or the France removal rules (see README.md sections 8-9).

usage:  python3 final_reproduce.py                    # the default target of resources/final_frozen/manifest.json (the shipped file)
        python3 final_reproduce.py --all              # every target listed in the manifest
        python3 final_reproduce.py --target NAME --out-dir DIR --resources DIR
Output: <out-dir>/<target>/matching_results.tsv + summary.json, and <out-dir>/R_US_LEGADD_FR9/ (the #10 check).
"""
import os
for _k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import sys
sys.dont_write_bytecode = True          # never write __pycache__ next to the shipped code
import json
import time
import argparse
from pathlib import Path
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ens8_reproduce as E               # noqa: E402  (same directory; functions build_p8 / apply_removals / write_tsv / digest)

K = E.K
BASE10 = 'R_US_LEGADD_FR9'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def check(res, manifest, rel):
    entry = manifest['files'][rel]
    path = res / rel
    assert path.exists(), f'missing resource {path}'
    got = E.digest(path)
    assert got == entry['sha256'], f'{rel}: sha256 {got} != manifest {entry["sha256"]}'
    return path


def read_pairs(path, i1, i2):
    df = pl.read_csv(path, separator='\t', schema_overrides={'s1_id': pl.String, 'cand_id': pl.String})
    df = df.select('s1_id', 'cand_id')
    assert df.unique().height == df.height, f'{path.name}: duplicate pairs'
    m = (df.join(i1.select(pl.col('entity_id').alias('s1_id'), 's1_idx'), on='s1_id', how='left')
           .join(i2.select(pl.col('entity_id').alias('cand_id'), 'cand_idx'), on='cand_id', how='left'))
    assert m['s1_idx'].null_count() == 0 and m['cand_idx'].null_count() == 0, f'{path.name}: id not in the test ids'
    return m.select(K)


def by_country(df, i1):
    return dict(df.join(i1.select('s1_idx', 'country'), on='s1_idx').group_by('country').len().sort('country').iter_rows())


def country_profile(sel, i1):
    per = (i1.select('s1_idx', 'country')
             .join(sel.group_by('s1_idx').len(), on='s1_idx', how='left').with_columns(pl.col('len').fill_null(0))
             .group_by('country').agg(pl.len().alias('s1'), pl.col('len').sum().alias('pairs'),
                                      (pl.col('len') == 0).mean().alias('empty_share'))
             .with_columns((pl.col('pairs') / pl.col('s1')).alias('matches_per_s1')).sort('country'))
    return {r['country']: dict(s1=r['s1'], pairs=r['pairs'], matches_per_s1=round(r['matches_per_s1'], 6),
                               empty_share=round(r['empty_share'], 6)) for r in per.iter_rows(named=True)}


def reproduce(name, res, fman, eres, eman, out_dir, i1, i2, cache):
    t0 = time.time()
    info = fman['targets'][name]
    if 'p8' not in cache:
        cache['p8'], c8 = E.build_p8(eres, i1, i2)
        log('#8 selection rebuilt from ens8_frozen:', json.dumps(c8))
        s10, rc10 = E.apply_removals(cache['p8'], eres, eman, BASE10, i1, i2)
        s10_sum = E.reproduce(BASE10, eres, eman, out_dir, i1, i2, {'selected': cache['p8'], 'counts': c8})
        assert s10_sum['sha256_match'], f'#10 ({BASE10}) is not byte-exact: {s10_sum["sha256"]}'
        cache['s10'], cache['s10_summary'] = s10, s10_sum
        rem10 = cache['p8'].join(s10, on=K, how='anti')
        assert rem10.height == rc10['removed_pairs']
        cache['rem10'] = rem10
    p8, s10, rem10 = cache['p8'], cache['s10'], cache['rem10']
    removed = read_pairs(check(res, fman, f'{name}/removed_vs8.tsv'), i1, i2)
    added = read_pairs(check(res, fman, f'{name}/added_vs8.tsv'), i1, i2)
    assert removed.height == info['removed_vs8'] and added.height == info['added_vs8'], 'list sizes differ from the manifest'
    assert removed.join(p8, on=K, how='semi').height == removed.height, 'a removed pair is not in #8'
    assert added.join(p8, on=K, how='semi').height == 0, 'an added pair is already in #8'
    kept = p8.join(removed, on=K, how='anti')
    assert added.join(kept.select('cand_idx'), on='cand_idx', how='semi').height == 0, 'an added record is still owned by another S1'
    ac = (added.join(i1.select('s1_idx', 'country'), on='s1_idx')
               .join(i2.select('cand_idx', pl.col('country').alias('cc')), on='cand_idx'))
    allow_fr = bool(info.get('allow_france_add', False))
    assert ac.filter(pl.col('country') != pl.col('cc')).height == 0, 'additions must be same-country'
    assert allow_fr or ac.filter(pl.col('country') == 'France').height == 0, \
        'additions must be non-France unless the manifest sets allow_france_add'
    sel = pl.concat([kept, added])
    assert sel.height == p8.height - removed.height + added.height
    assert sel['cand_idx'].n_unique() == sel.height, 'one-owner violated'
    # the same set written as a delta on top of #10
    on_top = removed.join(rem10, on=K, how='anti')
    readd = rem10.join(removed, on=K, how='anti')
    alt = pl.concat([s10.join(on_top, on=K, how='anti'), readd, added])
    assert alt.height == sel.height and alt.join(sel, on=K, how='anti').height == 0, '#10-delta route disagrees'
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / 'matching_results.tsv'
    table = E.write_tsv(sel, i1, i2, tsv)
    sha, md5 = E.digest(tsv), E.digest(tsv, 'md5')
    ok = (sha == info['sha256']) and (md5 == info['md5'])
    summary = dict(target=name, output=str(tsv), rows=table.height, pairs=sel.height,
                   empty_rows=int((table['matched_entity_ids'] == '').sum()), bytes=tsv.stat().st_size,
                   sha256=sha, md5=md5, expected_sha256=info['sha256'], expected_md5=info['md5'], byte_exact=ok,
                   base10=dict(target=BASE10, sha256=cache['s10_summary']['sha256'], sha256_match=cache['s10_summary']['sha256_match']),
                   vs8=dict(removed=removed.height, removed_by_country=by_country(removed, i1),
                            added=added.height, added_by_country=by_country(added, i1)),
                   vs10=dict(removed_on_top=on_top.height, removed_on_top_by_country=by_country(on_top, i1),
                             kept_from_10_removals=readd.height, added=added.height),
                   per_country=country_profile(sel, i1), leaderboard=info.get('leaderboard'),
                   seconds=round(time.time() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    log(f'{name}: pairs {sel.height:,}  md5 {md5}  {"BYTE-EXACT" if ok else "MISMATCH"}  ({summary["seconds"]} s)')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', default=None, help='default: manifest default_target (the shipped file)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--resources', type=Path, default=HERE / 'resources/final_frozen')
    ap.add_argument('--ens8-resources', type=Path, default=HERE / 'resources/ens8_frozen')
    ap.add_argument('--out-dir', type=Path, default=HERE / 'output_final')
    ap.add_argument('--skip-input-hashes', action='store_true')
    a = ap.parse_args()
    res, eres = a.resources.resolve(), a.ens8_resources.resolve()
    fman = json.loads((res / 'manifest.json').read_text())
    eman = json.loads((eres / 'manifest.json').read_text())
    if not a.skip_input_hashes:
        for rel in ['decision/test_selected.parquet', 'decision/test_nc_robust_additions.parquet', 'decision/mask_ppLegal_FR.parquet',
                    'ids/test_s1.parquet', 'ids/test_s23.parquet']:
            E.check_file(eres, eman, rel)
        log('ens8 input tables verified against ens8_frozen/manifest.json')
    i1, i2 = E.load_ids(eres)
    targets = list(fman['targets']) if a.all else [a.target or fman['default_target']]
    cache, results = {}, []
    out_dir = a.out_dir.resolve()
    for t in targets:
        results.append(reproduce(t, res, fman, eres, eman, out_dir, i1, i2, cache))
    print(json.dumps(results, indent=2))
    bad = [r['target'] for r in results if not r['byte_exact']]
    if bad:
        log('MISMATCH for', bad)
        sys.exit(1)
    log('BYTE-EXACT:', ', '.join(targets), '(and', BASE10, 'sha256 match)')


if __name__ == '__main__':
    main()
