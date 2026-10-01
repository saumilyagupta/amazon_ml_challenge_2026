# Tokenize train/dev pairs ONCE (the MiniLM and XLM-R checkpoints share a byte-identical XLM-R sentencepiece tokenizer.json)
# exactly as train_ce.py does: tok('<name1> | <addr1>'.lower(), '<name2> | <addr2>'.lower(), truncation=True, max_length=128).
# Saves flat int32 ids + int64 offsets -> data/tok_{train,dev}.npz, and measures CPU tokenization throughput (16 rayon threads).
import os
os.environ['RAYON_NUM_THREADS'] = '16'; os.environ['RAYON_RS_NUM_CPUS'] = '16'; os.environ['TOKENIZERS_PARALLELISM'] = 'true'; os.environ['OMP_NUM_THREADS'] = '4'
import time, json, numpy as np, pandas as pd
from transformers import AutoTokenizer
BASE = '/workspace/saumilya/amazon-ml/work/matching/cross_encoder/'
tok = AutoTokenizer.from_pretrained(BASE + 'models/ce_xlmr/ep0')
txt = lambda n, a: [f'{x} | {y}'.lower() for x, y in zip(n, a)]
stats = {}
for split in ['dev', 'train']:
    df = pd.read_parquet(BASE + f'data/pairs_{split}.parquet', columns=['s1', 'cand', 'name1', 'addr1', 'name2', 'addr2'])
    A, B = txt(df.name1, df.addr1), txt(df.name2, df.addr2)
    t = time.time(); X = tok(A, B, truncation=True, max_length=128)['input_ids']; el = time.time() - t
    lens = np.array([len(x) for x in X], np.int64); off = np.concatenate([[0], np.cumsum(lens)])
    np.savez(BASE + f'data/tok_{split}.npz', ids=np.concatenate([np.asarray(x, np.int32) for x in X]), off=off,
             s1=df.s1.values.astype(str), cand=df.cand.values.astype(str))
    # production alternative: tokenize UNIQUE strings once (single-sequence) and splice pairs
    uA, uB = sorted(set(A)), sorted(set(B)); t = time.time(); tok(uA + uB, add_special_tokens=False); el_u = time.time() - t
    stats[split] = dict(n_pairs=len(X), pair_tok_s=round(el, 1), pair_tok_per_s=round(len(X) / el), mean_len=round(float(lens.mean()), 1),
                        p99_len=int(np.percentile(lens, 99)), trunc_frac=round(float((lens >= 128).mean()), 5),
                        n_unique_strings=len(uA) + len(uB), single_seq_tok_per_s=round((len(uA) + len(uB)) / el_u))
    print(split, stats[split], flush=True)
json.dump(stats, open(BASE + 'data/tok_stats.json', 'w'), indent=1)
