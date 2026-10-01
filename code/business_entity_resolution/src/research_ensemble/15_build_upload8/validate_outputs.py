#!/usr/bin/env python3
"""Official matching checks plus memory-bounded full candidate containment audit.

Calls the supplied official validator for matching format and ID existence.
Candidate rows are checked in source order without materializing 84M pairs.
Also verifies unique ownership and country consistency from cached source IDs.
"""
import os
for key in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import time
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
W = ROOT.parents[1]

def digest(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matching', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--label', required=True)
    a = ap.parse_args()
    started = time.time()
    official_path = W.parent / 'student_resource/utils/validate_submission.py'
    spec = importlib.util.spec_from_file_location('official_validation', official_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    errors, warnings = official.validate(a.matching, None, str(W.parent / 'student_resource/dataset/test'), check_ids=True)
    assert not errors, errors
    assert not warnings, warnings
    gc.collect()
    ids = W / 'blocking/embedding/full/ids'
    q = pl.read_parquet(ids / 'test_s1.parquet', columns=['entity_id','country'])
    r = pl.read_parquet(ids / 'test_s23.parquet', columns=['entity_id','country'])
    r_country = dict(r.iter_rows())
    del r
    owners = set()
    nrows = nmatches = ncands = 0
    with open(a.matching) as mf, open(a.candidate) as cf:
        assert next(mf).strip() == 'source1_entity_id\tmatched_entity_ids'
        assert next(cf).strip() == 'source1_entity_id\tcandidate_entity_ids'
        for ml, cl, meta in itertools.zip_longest(mf, cf, q.iter_rows()):
            assert ml is not None and cl is not None and meta is not None, 'row count differs'
            mq, mt = ml.rstrip('\n').split('\t')
            cq, ct = cl.rstrip('\n').split('\t')
            assert mq == cq == meta[0], 'query order or IDs differ'
            mm = mt.split(',') if mt else []
            cc = ct.split(',') if ct else []
            cs = set(cc)
            assert len(cs) == len(cc), 'duplicate candidate in row'
            assert set(mm) <= cs, 'selected record missing from matcher candidates'
            assert not owners.intersection(mm), 'record has multiple owners'
            assert all(r_country[c] == meta[1] for c in mm), 'cross-country match'
            owners.update(mm)
            nrows += 1
            nmatches += len(mm)
            ncands += len(cc)
    assert nrows == 1732544
    result = dict(label=a.label, official_matching_with_ids='PASS', candidate_containment='PASS',
                  duplicate_candidate_rows='PASS', ownership='PASS', country='PASS',
                  matching=a.matching, candidate=a.candidate, rows=nrows, match_pairs=nmatches,
                  candidate_pairs=ncands, matching_sha256=digest(a.matching),
                  candidate_sha256=digest(a.candidate), seconds=time.time()-started,
                  note='Official validator checks the matching file with test IDs; separate full streaming audit checks candidate format, query IDs, duplicates, and containment. Candidate ID existence is inherited from the unchanged previously validated candidate file.')
    (ROOT / f'results/validation_{a.label}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
