#!/opt/conda/bin/python3
"""Re-learn the branch explainer's filler lists (resources/branch_lists/coverage.json) used by the v3 branch features (ber.branch).

usage: /opt/conda/bin/python3 tools/learn_branch_lists.py --data DATASET_DIR --out DIR [--val-ids resources/splits/val_s1_ids.txt]
       [--filler-min 15] [--subst-min 8] [--per-stratum 50000]

Port of the team branch's scripts/make_pair_sample.py + scripts/explain_pairs.py (as run in work/features/branch_port):
 1. labelled pairs from the TRAIN-SPLIT ground truth only (official train_ground_truth.tsv minus the validation S1 ids, so the lists carry
    no validation information); 50,000 pairs per (source S2/S3, country) stratum, pandas sample(random_state=0)   -> DIR/pair_sample.parquet
 2. ber.branch.explain_pairs.main(): learn filler words (leftover tokens seen >= filler-min times) and 1:1 substitutions on one half
    (random_state=1), report explainer coverage on the other half                                                   -> DIR/coverage.json
Only name_filler / addr_filler of coverage.json are used by the features. Runtime on the full data: ~10 min single process, ~6 GB RAM.
(sample(n) is capped at the stratum size so that the tool also runs on small slices; this does not change the full-data behaviour.)"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from ber.paths import RES
from ber.branch import explain_pairs as E

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, help='dataset dir with train/train_source{1,2,3}.tsv and train/train_ground_truth.tsv')
ap.add_argument('--out', required=True); ap.add_argument('--val-ids', default=os.path.join(RES, 'splits', 'val_s1_ids.txt'))
ap.add_argument('--filler-min', type=int, default=15); ap.add_argument('--subst-min', type=int, default=8); ap.add_argument('--per-stratum', type=int, default=50000)
A = ap.parse_args()
D = os.path.join(A.data, 'train'); os.makedirs(A.out, exist_ok=True); SMP = os.path.join(A.out, 'pair_sample.parquet')


def rd(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, quoting=3)


if not os.path.exists(SMP):
    s1 = rd(os.path.join(D, 'train_source1.tsv')).set_index('entity_id')
    other = pd.concat([rd(os.path.join(D, 'train_source2.tsv')), rd(os.path.join(D, 'train_source3.tsv'))]).set_index('entity_id')
    gt = rd(os.path.join(D, 'train_ground_truth.tsv'))
    val = set(l.strip() for l in open(A.val_ids) if l.strip())
    gt = gt[~gt.source1_entity_id.isin(val)]                       # train-split ground truth (validation S1 excluded), file order kept
    ids = gt.matched_entity_ids.str.split(',')
    pairs = pd.DataFrame({'a': gt.source1_entity_id.repeat(ids.str.len()).values, 'b': [x for y in ids for x in y]})
    pairs = pairs[pairs.b != '']
    pairs['src'] = pairs.b.str[:2]
    pairs['country'] = pairs.a.map(s1.country)
    smp = pairs.groupby(['src', 'country'], group_keys=False).apply(lambda g: g.sample(min(A.per_stratum, len(g)), random_state=0))
    for side, tbl, key in [('a', s1, 'a'), ('b', other, 'b')]:
        smp[f'{side}_name'] = smp[key].map(tbl.business_name)
        smp[f'{side}_addr'] = smp[key].map(tbl.business_address)
    smp.reset_index(drop=True).to_parquet(SMP)
    print(smp.groupby(['src', 'country']).size(), flush=True)
sys.argv = [sys.argv[0], '--sample', SMP, '--out', A.out, '--filler-min', str(A.filler_min), '--subst-min', str(A.subst_min)]
E.main()
print('wrote', os.path.join(A.out, 'coverage.json'))
