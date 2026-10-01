#!/opt/conda/bin/python3
"""Step 1: load the challenge TSVs, build the transliteration view, the matcher record views and the lexical-channel views.

usage: /opt/conda/bin/python3 prepare.py --data <dataset dir with train/ and test/> --work <work dir> [--splits train test] [--threads 8]

Writes under WORK/records/: {split}_s1.parquet, {split}_s23.parquet (prepared records; the row order defines s1_idx / cand_idx for the
whole pipeline: S1 file order, and S2 rows followed by S3 rows), rec_{split}_{s1,s23}.parquet + idf/voc (matcher views),
norm_{split}_{s1,pool}.parquet (lexical views), train_gt_pairs.parquet (labels).
Runtime on the full data (2.2M + 10.3M train, 1.7M + 10.0M test records): about 12 min per split at 16 threads, peak RSS about 20 GB."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--work', required=True)
ap.add_argument('--splits', nargs='+', default=['train', 'test']); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--dict', default=None, help='transliteration dictionary TSV (default: resources/translit_dictionary.tsv)')
A = ap.parse_args()
from ber import env
env.setup(A.threads)
import polars as pl
from ber.paths import Work
from ber.io import read_tsv, gt_pairs_frame
from ber.translit import load_dictionary, translit_series, DEFAULT_DICT
from ber import records, lexnorm
log = env.logger(); W = Work(A.work)
D = load_dictionary(A.dict or DEFAULT_DICT); log('dictionary words', len(D))
for split in A.splits:
    t = time.time()
    parts = []
    for k in (1, 2, 3):
        p = f'{A.data}/{split}/{split}_source{k}.tsv'
        df = read_tsv(p)
        assert df.columns[:4] == ['entity_id', 'business_name', 'business_address', 'country'], df.columns
        df = df.select('entity_id', 'business_name', 'business_address', 'country').with_columns(pl.lit(k, pl.Int8).alias('src'))
        df = df.with_columns(translit_series(df['business_name'], D).alias('name_en'), translit_series(df['business_address'], D).alias('addr_en'))
        assert df['entity_id'].n_unique() == df.height, f'duplicate entity ids in {p}'
        parts.append(df); log(split, f'source{k}', df.height, 'rows; indic rows translated:', int((df['name_en'] != df['business_name']).sum() + (df['addr_en'] != df['business_address']).sum()))
    parts[0].write_parquet(W.records(split, 's1'))
    pl.concat(parts[1:]).write_parquet(W.records(split, 's23'))
    log(split, 'records written', f'{time.time()-t:.0f}s')
    if split == 'train':
        gp = f'{A.data}/train/train_ground_truth.tsv'
        if os.path.exists(gp):
            g = gt_pairs_frame(gp)
            s1 = set(parts[0]['entity_id'].to_list()); pool = set(pl.concat(parts[1:])['entity_id'].to_list())
            g = g.filter(pl.col('s1_id').is_in(list(s1)) & pl.col('cand_id').is_in(list(pool)))
            g.write_parquet(W.gt_pairs('train')); log('train gt pairs (present in the files)', g.height)
        else:
            log('WARNING: no train_ground_truth.tsv; training is impossible with this data dir')
    # matcher record views
    records.build(W, split, log=log)
    # lexical views
    for part, src in [('s1', 's1'), ('pool', 's23')]:
        df = pl.read_parquet(W.records(split, src), columns=['entity_id', 'country', 'business_name', 'name_en', 'addr_en'])
        out = lexnorm.norm_frame(df)
        out.write_parquet(W.norm(split, part))
        log(split, part, 'lexical view; addr_empty frac', round(out['addr_empty'].mean(), 4))
log('DONE peak RSS GB', round(env.peak_rss_gb(), 1))
