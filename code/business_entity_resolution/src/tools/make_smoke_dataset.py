#!/opt/conda/bin/python3
"""Build a small self-contained dataset (same file layout as the challenge) from the official TRAIN data, for end-to-end smoke tests.

usage: /opt/conda/bin/python3 tools/make_smoke_dataset.py --data <official dataset dir> --out <dir> [--n_s1 20000] [--n_distractors 150000] [--seed 0]

out/train/{train_source1,2,3.tsv, train_ground_truth.tsv} : n_s1 random train S1 + all their true S2/S3 records + n_distractors other records
out/test/{test_source1,2,3.tsv}                          : a DISJOINT set of n_s1 S1 built the same way; its ground truth is written to
out/test_ground_truth.tsv (never read by the pipeline; use it with tools/score.py to score the smoke prediction).
Distractors = pool records that are not matched to any selected S1 (records of other S1 entities and true singletons alike)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n_s1', type=int, default=20000); ap.add_argument('--n_distractors', type=int, default=150000); ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()
from ber import env
env.setup(8)
import numpy as np, polars as pl
from ber.io import read_tsv, gt_pairs_frame
rng = np.random.default_rng(A.seed)
S1 = read_tsv(f'{A.data}/train/train_source1.tsv'); S2 = read_tsv(f'{A.data}/train/train_source2.tsv'); S3 = read_tsv(f'{A.data}/train/train_source3.tsv')
G = gt_pairs_frame(f'{A.data}/train/train_ground_truth.tsv')
gt_rows = {}
with open(f'{A.data}/train/train_ground_truth.tsv') as f:
    f.readline()
    for line in f:
        s1, _, rest = line.rstrip('\n').partition('\t'); gt_rows[s1] = rest
ids = S1['entity_id'].to_numpy(); perm = rng.permutation(len(ids))
sel = {'train': set(ids[perm[:A.n_s1]].tolist()), 'test': set(ids[perm[A.n_s1:2 * A.n_s1]].tolist())}
for split in ('train', 'test'):
    d = f'{A.out}/{split}'; os.makedirs(d, exist_ok=True)
    s1 = S1.filter(pl.col('entity_id').is_in(list(sel[split])))
    true = set(G.filter(pl.col('s1_id').is_in(list(sel[split])))['cand_id'].to_list())
    out = []
    for k, S in ((2, S2), (3, S3)):
        t = S.filter(pl.col('entity_id').is_in(list(true)))
        rest = S.filter(~pl.col('entity_id').is_in(list(true)))
        nd = int(round(A.n_distractors * S.height / (S2.height + S3.height)))
        dis = rest[rng.choice(rest.height, nd, replace=False)]
        part = pl.concat([t, dis]); part = part[rng.permutation(part.height)]
        part.write_csv(f'{d}/{split}_source{k}.tsv', separator='\t', quote_style='never')
        out.append((k, t.height, dis.height))
    s1.write_csv(f'{d}/{split}_source1.tsv', separator='\t', quote_style='never')
    gtp = f'{d}/train_ground_truth.tsv' if split == 'train' else f'{A.out}/test_ground_truth.tsv'
    with open(gtp, 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s in s1['entity_id'].to_list(): f.write(f'{s}\t{gt_rows.get(s, "")}\n')
    print(split, 'S1', s1.height, 'records (source, true, distractors):', out, 'countries', s1['country'].value_counts().rows(), flush=True)
