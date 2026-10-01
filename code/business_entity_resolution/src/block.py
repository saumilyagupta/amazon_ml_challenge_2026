#!/opt/conda/bin/python3
"""Step 2: candidate generation (blocking).

usage: /opt/conda/bin/python3 block.py --work WORK --split train|test [--stage dense lexical union] [--encoders zs ft]
                                       [--device cpu|cuda] [--union v1 v2] [--config v1] [--threads 8]
  dense   : e5 embeddings (encoders zs and/or ft) + exact forward top-50 / reverse top-5 within country
            (GPU: run through your GPU wrapper with --device cuda; CPU works but is slow on the full pool)
  lexical : C1/C2 forward (top-50), C1r/C2r reverse (top-3), C3 forward (top-10), C3r reverse (top-5)
  union   : REC20 union(s): v1 = zero-shot dense member (gate 0.02), v2 = fine-tuned dense member (gate 0.03) [needs both encoders];
            on the train split union v2 flags the fine-tune's own training pairs (ft_seen / ft_seen_s1, from resources/ft_pairs/)
  The fine-tuned encoder is split-aware: train -> resources/models/e5s_ft_trainsplit, test -> resources/models/e5s_ft_all (see ber/dense.py).
The forward lexical / union candidates are built for the 'query' S1 of the split (train: the training sample + validation S1 of
the split config; test: every S1). Dense search always covers every S1 (forward) and every record (reverse)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--split', required=True)
ap.add_argument('--stage', nargs='+', default=['dense', 'lexical', 'union']); ap.add_argument('--encoders', nargs='+', default=['zs'])
ap.add_argument('--device', default='cpu'); ap.add_argument('--union', nargs='+', default=['v1'])
ap.add_argument('--config', default='v1', help='variant config whose split section selects the query S1'); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--ft-model-train', default=None, help='fine-tuned checkpoint for the train split (default resources/models/e5s_ft_trainsplit)')
ap.add_argument('--ft-model-test', default=None, help='fine-tuned checkpoint for the test split (default resources/models/e5s_ft_all)')
A = ap.parse_args()
from ber import env
env.setup(A.threads, gpu='keep' if A.device != 'cpu' else None)
import numpy as np, polars as pl
from ber.paths import Work
from ber import config, splits, dense, lexical, union
log = env.logger(); W = Work(A.work); cfg = config.load(A.config)
dense.set_ft_models(A.ft_model_train, A.ft_model_test)
qmask = splits.query_mask(W, cfg, A.split); log(A.split, 'query S1', int(qmask.sum()), 'of', len(qmask))
UNION_SPECS = {'v1': dict(dense_member='zs', gate=0.02), 'v2': dict(dense_member='ft', gate=0.03)}
if 'dense' in A.stage:
    import torch
    torch.set_num_threads(A.threads)
    dev = A.device if (A.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    if dev != A.device: log('WARNING: no GPU visible, falling back to CPU')
    for enc in A.encoders:
        for part in ('s1', 's23'):
            dense.encode(W, enc, A.split, part, device=dev, batch_size=A.batch_size, log=log)
        dense.search(W, enc, A.split, device=dev, log=log)
if 'lexical' in A.stage:
    lexical.run_c12(W, A.split, qmask, log=log)
    lexical.run_c12r(W, A.split, qmask, log=log)
    lexical.run_c3(W, A.split, qmask, threads=min(16, 2 * A.threads), log=log)
    lexical.run_c3r(W, A.split, qmask, threads=min(16, 2 * A.threads), log=log)
def ft_seen_pairs():
    """train split: the fine-tuned checkpoint's own training pairs (s1_idx, cand_idx) present in this work dir (-> ft_seen flags)."""
    if A.split != 'train' or not os.path.exists(W.forward('ft', A.split)): return None
    done = W.emb('ft', A.split, 's1') + '.done'
    mdl = json.load(open(done))['model'] if os.path.exists(done) else dense.model_path('ft', A.split)
    tp = dense.ft_training_pairs(mdl)
    if tp is None: log('WARNING: no training-pair list for', mdl, '-> ft_seen flags all False'); return None
    i1 = pl.read_parquet(W.records(A.split, 's1'), columns=['entity_id']).with_row_index('s1_idx').rename({'entity_id': 's1_id'})
    i2 = pl.read_parquet(W.records(A.split, 's23'), columns=['entity_id']).with_row_index('cand_idx').rename({'entity_id': 'cand_id'})
    P = tp.join(i1, on='s1_id').join(i2, on='cand_id').select(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
    log('fine-tune training pairs of', os.path.basename(mdl), 'present in this split:', P.height, 'S1', P['s1_idx'].n_unique()); return P
if 'union' in A.stage:
    for u in A.union:
        union.build(W, A.split, u, qmask, log=log, ft_seen_pairs=ft_seen_pairs() if UNION_SPECS[u]['dense_member'] == 'ft' else None, **UNION_SPECS[u])
log('DONE peak RSS GB', round(env.peak_rss_gb(), 1))
