# E06 step 2 (CPU, 3 tokenizer threads): CE inputs for the band pairs. Text = '<name> | <addr>'.lower() of the ORIGINAL records
# (features/explainer/cache/records_train.parquet), tokenised ONCE per unique string with the ce_xlmr/ep0 tokenizer (= xlm-roberta-base
# sentencepiece), spliced per pair as <s> A </s></s> B </s> with longest-first truncation to 128 (= HF tok(A, B, truncation=True,
# max_length=128); verified on 2,000 random pairs per set). Writes (atomically) data/tok_sample.npz first (CE training/dev + OOF scoring),
# then data/tok_val.npz and data/tok_dens.npz (scoring only). Optional --view en: use name_en/addr_en (dictionary transliteration) -> *_en.npz.
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['RAYON_NUM_THREADS'] = '3'; os.environ['RAYON_RS_NUM_CPUS'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'; os.environ['OMP_NUM_THREADS'] = '3'; os.environ['POLARS_MAX_THREADS'] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; os.environ['WANDB_MODE'] = 'disabled'
import numpy as np, polars as pl
from transformers import AutoTokenizer
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'
CE = '/workspace/saumilya/amazon-ml/work/matching/cross_encoder'
EB = '/workspace/saumilya/amazon-ml/work/blocking/embedding/full'
REC = '/workspace/saumilya/amazon-ml/work/features/explainer/cache/records_train.parquet'
VIEW = 'en' if '--view' in sys.argv and sys.argv[sys.argv.index('--view') + 1] == 'en' else 'raw'
SUF = '' if VIEW == 'raw' else '_en'
T0 = time.time(); log = lambda *a: print(f'[+{time.time()-T0:.0f}s]', *a, flush=True)
nc, ac = ('name', 'addr') if VIEW == 'raw' else ('name_en', 'addr_en')
R = pl.read_parquet(REC, columns=['entity_id', nc, ac]).rename({nc: 'n', ac: 'a'})
i1 = pl.read_parquet(f'{EB}/ids/train_s1.parquet', columns=['entity_id']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
i2 = pl.read_parquet(f'{EB}/ids/train_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
tok = AutoTokenizer.from_pretrained(f'{CE}/models/ce_xlmr/ep0')
BOS, EOS, MAXL = tok.cls_token_id, tok.sep_token_id, 128
T1, T2 = {}, {}   # s1_idx -> token list ; cand_idx -> token list
TXT1, TXT2 = {}, {}
def texts(idx, ids_tab, key):
    u = pl.DataFrame({key: np.unique(idx).astype(np.int32)}).join(ids_tab, on=key, how='left').join(R, on='entity_id', how='left')
    assert u['entity_id'].null_count() == 0 and u.filter(pl.col('n').is_null() & pl.col('a').is_null()).height == 0
    t = [f'{x} | {y}'.lower() for x, y in zip(u['n'].fill_null('').to_list(), u['a'].fill_null('').to_list())]
    return u[key].to_list(), t
def ensure(idx, ids_tab, key, T, TXT):
    new = np.setdiff1d(np.unique(idx), np.fromiter(T.keys(), np.int64, len(T)) if T else np.zeros(0, np.int64))
    if len(new) == 0: return
    keys, t = texts(new, ids_tab, key)
    enc = tok(t, add_special_tokens=False)['input_ids']
    for k, s, e in zip(keys, t, enc): T[k] = e; TXT[k] = s
def splice(a, b):
    a = list(a); b = list(b)
    while len(a) + len(b) + 4 > MAXL:
        if len(a) > len(b): a.pop()
        else: b.pop()
    return [BOS] + a + [EOS, EOS] + b + [EOS]
def build(df, name, extra):
    s1 = df['s1_idx'].to_numpy(); ca = df['cand_idx'].to_numpy()
    ensure(s1, i1, 's1_idx', T1, TXT1); log(name, 'S1 strings ready', len(T1))
    ensure(ca, i2, 'cand_idx', T2, TXT2); log(name, 'cand strings ready', len(T2))
    seqs = [splice(T1[a], T2[b]) for a, b in zip(s1.tolist(), ca.tolist())]
    lens = np.array([len(x) for x in seqs], np.int64); off = np.concatenate([[0], np.cumsum(lens)])
    ids = np.fromiter((t for x in seqs for t in x), np.int32, count=int(off[-1]))
    rs = np.random.default_rng(0).choice(len(seqs), min(2000, len(seqs)), replace=False)
    ref = tok([TXT1[int(s1[i])] for i in rs], [TXT2[int(ca[i])] for i in rs], truncation=True, max_length=MAXL)['input_ids']
    bad = sum(list(ref[k]) != seqs[i] for k, i in enumerate(rs)); log(name, 'splice check mismatches', bad, 'of', len(rs))
    assert bad <= 0.002 * len(rs), bad
    arrs = dict(ids=ids, off=off, s1_idx=s1.astype(np.int32), cand_idx=ca.astype(np.int32))
    for c in extra: arrs[c] = df[c].to_numpy()
    tmp = f'{E}/data/tok_{name}{SUF}.tmp.npz'; np.savez(tmp, **arrs); os.replace(tmp, f'{E}/data/tok_{name}{SUF}.npz')
    log('wrote', name, len(seqs), 'pairs, mean len', round(float(lens.mean()), 1), 'trunc share', round(float((lens >= MAXL).mean()), 5))
B = pl.read_parquet(f'{E}/data/band_train.parquet', columns=['s1_idx', 'cand_idx', 's1_id', 'cand_id', 'grp', 'fold', 'es', 'label', 'bs', 'p2'])
# consistency of ids tables with the stored ids
chk = B.sample(20000, seed=0).join(i1.rename({'entity_id': 'e1'}), on='s1_idx').join(i2.rename({'entity_id': 'e2'}), on='cand_idx')
assert (chk['e1'] == chk['s1_id']).all() and (chk['e2'] == chk['cand_id']).all(); log('ids consistent')
EX = ['label', 'fold', 'es', 'bs', 'p2']
build(B.filter(pl.col('grp') == 'sample').with_columns(pl.col('label').cast(pl.Int8), pl.col('fold').cast(pl.Int8)), 'sample', EX)
build(B.filter(pl.col('grp') != 'sample').with_columns(pl.col('label').cast(pl.Int8), pl.col('fold').cast(pl.Int8)), 'val', EX)
Dn = pl.read_parquet(f'{E}/data/band_dens.parquet', columns=['s1_idx', 'cand_idx', 'label', 'bs', 'p2']).with_columns(pl.col('label').cast(pl.Int8))
build(Dn, 'dens', ['label', 'bs', 'p2'])
log('done')
