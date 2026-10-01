# E06 step 12 (CPU, light: 3 tokenizer threads, 4 polars threads): CE inputs for the TEST band 0.02 < p2 < 0.99 of v2b (B1's band table
# build/interim_v2b_E13/data/band_base.parquet). France band pairs were already tokenised + scored (09/10: data/tok_fr.npz, scores/ce_xw_fr_f*),
# so only US/India pairs are tokenised here (the France key set is checked for exact equality first).
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['RAYON_NUM_THREADS'] = '3'; os.environ['TOKENIZERS_PARALLELISM'] = 'true'; os.environ['OMP_NUM_THREADS'] = '3'
os.environ['POLARS_MAX_THREADS'] = '4'; os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import numpy as np, polars as pl
from transformers import AutoTokenizer
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'; EB = '/workspace/saumilya/amazon-ml/work/blocking/embedding/full'
T0 = time.time(); log = lambda *a: print(f'[+{time.time()-T0:.0f}s]', *a, flush=True)
K = ['s1_idx', 'cand_idx']
B = pl.read_parquet('/workspace/saumilya/amazon-ml/work/matching/prod_v2c/build/interim_v2b_E13/data/band_base.parquet', columns=[*K, 'country', 'p2']).filter((pl.col('p2') > 0.02) & (pl.col('p2') < 0.99))
log('test band', B.height, B.group_by('country').len().sort('country').to_dicts())
fr = B.filter(pl.col('country') == 'France').select(K); fr0 = pl.read_parquet(f'{E}/data/fr_band.parquet', columns=K)
same = fr.height == fr0.height and fr.join(fr0, on=K, how='anti').height == 0; log('France key sets identical:', same, fr.height, fr0.height)
todo = B.filter(pl.col('country') != 'France') if same else B
i1 = pl.read_parquet(f'{EB}/ids/test_s1.parquet', columns=['entity_id']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
i2 = pl.read_parquet(f'{EB}/ids/test_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
todo = todo.join(i1.rename({'entity_id': 's1_id'}), on='s1_idx', how='left').join(i2.rename({'entity_id': 'cand_id'}), on='cand_idx', how='left')
assert todo['s1_id'].null_count() == 0 and todo['cand_id'].null_count() == 0
R = pl.read_parquet('/workspace/saumilya/amazon-ml/work/features/explainer/cache/records_test.parquet', columns=['entity_id', 'name', 'addr'])
tok = AutoTokenizer.from_pretrained('/workspace/saumilya/amazon-ml/work/matching/cross_encoder/models/ce_xlmr/ep0')
BOS, EOS, MAXL = tok.cls_token_id, tok.sep_token_id, 128
def enc(ids):
    u = pl.DataFrame({'entity_id': np.unique(ids)}).join(R, on='entity_id', how='left'); assert u.filter(pl.col('name').is_null() & pl.col('addr').is_null()).height == 0
    t = [f'{x} | {y}'.lower() for x, y in zip(u['name'].fill_null('').to_list(), u['addr'].fill_null('').to_list())]
    return dict(zip(u['entity_id'].to_list(), tok(t, add_special_tokens=False)['input_ids'])), dict(zip(u['entity_id'].to_list(), t))
T1, X1 = enc(todo['s1_id'].to_numpy()); log('S1 strings', len(T1)); T2, X2 = enc(todo['cand_id'].to_numpy()); log('cand strings', len(T2))
def splice(a, b):
    a = list(a); b = list(b)
    while len(a) + len(b) + 4 > MAXL:
        if len(a) > len(b): a.pop()
        else: b.pop()
    return [BOS] + a + [EOS, EOS] + b + [EOS]
s1, ca = todo['s1_id'].to_list(), todo['cand_id'].to_list()
seqs = [splice(T1[a], T2[b]) for a, b in zip(s1, ca)]
rs = np.random.default_rng(0).choice(len(seqs), min(2000, len(seqs)), replace=False)
ref = tok([X1[s1[i]] for i in rs], [X2[ca[i]] for i in rs], truncation=True, max_length=MAXL)['input_ids']
bad = sum(list(ref[k]) != seqs[i] for k, i in enumerate(rs)); log('splice check mismatches', bad); assert bad <= 4
lens = np.array([len(x) for x in seqs], np.int64); off = np.concatenate([[0], np.cumsum(lens)])
tmp = f'{E}/data/tok_test_rest.tmp.npz'
np.savez(tmp, ids=np.fromiter((t for x in seqs for t in x), np.int32, count=int(off[-1])), off=off, s1_idx=todo['s1_idx'].to_numpy().astype(np.int32), cand_idx=todo['cand_idx'].to_numpy().astype(np.int32))
os.replace(tmp, f'{E}/data/tok_test_rest.npz'); log('wrote tok_test_rest', len(seqs), 'mean len', round(float(lens.mean()), 1), 'france reused', same)
