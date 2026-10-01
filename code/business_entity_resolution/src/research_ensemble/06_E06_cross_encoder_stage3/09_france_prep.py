# E06 step 9 (CPU, light: 4 threads): label-free France check inputs. From v2b's test pair table (prod_v2b/output/pairs/test): all France pairs
# (s1_idx, cand_idx, p2) -> data/fr_all.parquet; France band pairs 0.02 < p2 < 0.99 with the stage-3 inputs -> data/fr_band.parquet;
# CE tokens for the band pairs (records_test texts, same splice as 02_tok.py) -> data/tok_fr.npz.
import os, sys, glob, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['RAYON_NUM_THREADS'] = '3'; os.environ['TOKENIZERS_PARALLELISM'] = 'true'; os.environ['OMP_NUM_THREADS'] = '3'
os.environ['POLARS_MAX_THREADS'] = '4'; os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2b'); import pv2b.common
import numpy as np, polars as pl
from transformers import AutoTokenizer
from pv2b.newfeats import SIB_FEATS
from pv1.model import S2FEATS
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'
FLAGS = ['addr_empty2', 'name_freq_s1_1', 'n11_junk_name', 'f_url2', 'n10_alias', 'script2', 'x1_name_explained', 'x1_both_explained']
COLS = ['s1_idx', 'cand_idx', 's1_id', 'cand_id', 'country', 'in_dft'] + list(dict.fromkeys(list(S2FEATS) + SIB_FEATS + FLAGS + ['p2']))
T0 = time.time(); log = lambda *a: print(f'[+{time.time()-T0:.0f}s]', *a, flush=True)
alls, band = [], []
for p in sorted(glob.glob('/workspace/saumilya/amazon-ml/work/matching/prod_v2b/output/pairs/test/part-*.parquet')):
    d = pl.read_parquet(p, columns=COLS).filter(pl.col('country') == 'France')
    alls.append(d.select('s1_idx', 'cand_idx', 'p2')); band.append(d.filter((pl.col('p2') > 0.02) & (pl.col('p2') < 0.99)))
A = pl.concat(alls); Bd = pl.concat(band); A.write_parquet(f'{E}/data/fr_all.parquet'); Bd.write_parquet(f'{E}/data/fr_band.parquet')
log('France pairs', A.height, 'S1', A['s1_idx'].n_unique(), 'band', Bd.height)
R = pl.read_parquet('/workspace/saumilya/amazon-ml/work/features/explainer/cache/records_test.parquet', columns=['entity_id', 'name', 'addr'])
tok = AutoTokenizer.from_pretrained('/workspace/saumilya/amazon-ml/work/matching/cross_encoder/models/ce_xlmr/ep0')
BOS, EOS, MAXL = tok.cls_token_id, tok.sep_token_id, 128
def enc(ids):
    u = pl.DataFrame({'entity_id': np.unique(ids)}).join(R, on='entity_id', how='left'); assert u.filter(pl.col('name').is_null() & pl.col('addr').is_null()).height == 0
    t = [f'{x} | {y}'.lower() for x, y in zip(u['name'].fill_null('').to_list(), u['addr'].fill_null('').to_list())]
    return dict(zip(u['entity_id'].to_list(), tok(t, add_special_tokens=False)['input_ids'])), dict(zip(u['entity_id'].to_list(), t))
T1, X1 = enc(Bd['s1_id'].to_numpy()); T2, X2 = enc(Bd['cand_id'].to_numpy()); log('tokenised', len(T1), len(T2))
def splice(a, b):
    a = list(a); b = list(b)
    while len(a) + len(b) + 4 > MAXL:
        if len(a) > len(b): a.pop()
        else: b.pop()
    return [BOS] + a + [EOS, EOS] + b + [EOS]
s1, ca = Bd['s1_id'].to_list(), Bd['cand_id'].to_list()
seqs = [splice(T1[a], T2[b]) for a, b in zip(s1, ca)]
rs = np.random.default_rng(0).choice(len(seqs), min(2000, len(seqs)), replace=False)
ref = tok([X1[s1[i]] for i in rs], [X2[ca[i]] for i in rs], truncation=True, max_length=MAXL)['input_ids']
bad = sum(list(ref[k]) != seqs[i] for k, i in enumerate(rs)); log('splice check mismatches', bad); assert bad <= 4
lens = np.array([len(x) for x in seqs], np.int64); off = np.concatenate([[0], np.cumsum(lens)])
np.savez(f'{E}/data/tok_fr.npz', ids=np.fromiter((t for x in seqs for t in x), np.int32, count=int(off[-1])), off=off,
         s1_idx=Bd['s1_idx'].to_numpy().astype(np.int32), cand_idx=Bd['cand_idx'].to_numpy().astype(np.int32))
log('wrote tok_fr', len(seqs), 'mean len', round(float(lens.mean()), 1))
