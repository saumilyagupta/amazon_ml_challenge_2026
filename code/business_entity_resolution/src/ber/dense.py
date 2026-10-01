"""Dense retrieval channel: multilingual-e5-small embeddings + exact within-country inner-product search.

Recipe (identical for the zero-shot and the fine-tuned encoder): text = 'query: {business_name} | {business_address}'.strip() on the
RAW strings (the same 'query: ' prefix on both sides), max_seq_length 64, mean pooling, L2-normalised, stored as float16.
Search: forward top-50 (S1 -> S2/S3 of the same country) and reverse top-5 (each S2/S3 record -> S1 of the same country) with an exact
matmul + top-k (fp16 on GPU with fp32 re-scoring of the winners; fp32 on CPU). GPU: about 2e10 scores/s on a V100 (full test set
about 10 GPU-min). CPU: exact as well but ~50x slower (see README for the estimate); intended for smoke tests / small pools.

Encoders: 'zs' = intfloat/multilingual-e5-small (MIT, 118M params, HF snapshot 614241f622f53c4eeff9890bdc4f31cfecc418b3);
          'ft' = the same model contrastively fine-tuned on 150k ground-truth pairs (see finetune.py). The checkpoint is SPLIT-AWARE,
          exactly as in the validated research run of union v2 / matcher v2b:
            split 'train' -> resources/models/e5s_ft_trainsplit (fine-tuned on 150k pairs of TRAIN-SPLIT S1 only, none of the 220,730
                             validation S1; its training pairs are listed in resources/ft_pairs/e5s_ft_trainsplit_pairs.parquet so that
                             union.build can flag them (ft_seen / ft_seen_s1) and train.py can drop those S1 from the training sample)
            split 'test'  -> resources/models/e5s_ft_all (same recipe on 150k pairs of ALL train S1; pairs in e5s_ft_all_pairs.parquet)
          Override with set_ft_models(train=..., test=...) (block.py --ft-model-train / --ft-model-test)."""
import os, time, json
import numpy as np, polars as pl
from .paths import RES

ZS_MODEL = 'intfloat/multilingual-e5-small'
ZS_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
FT_MODEL = os.path.join(RES, 'models', 'e5s_ft_all')
FT_MODEL_TRAIN = os.path.join(RES, 'models', 'e5s_ft_trainsplit')
FT_PAIRS_DIR = os.path.join(RES, 'ft_pairs')
MAXLEN = 64; DIM = 384; PREFIX = 'query: '
_FT = {'train': FT_MODEL_TRAIN, 'test': FT_MODEL}


def set_ft_models(train=None, test=None):
    if train: _FT['train'] = train
    if test: _FT['test'] = test


def model_path(enc, split=None):
    if enc != 'ft':
        return ZS_MODEL
    p = _FT.get(split, FT_MODEL)
    if not os.path.isdir(p):
        print(f'WARNING: fine-tuned checkpoint {p} not found; using {FT_MODEL} (on the train split its scores are then in-sample '
              f'for the S1 of its training pairs)', flush=True)
        p = FT_MODEL
    return p


def ft_training_pairs(model_dir):
    """(s1_id, cand_id) pairs the fine-tuned checkpoint was trained on (resources/ft_pairs/<checkpoint name>_pairs.parquet), or None."""
    f = os.path.join(FT_PAIRS_DIR, os.path.basename(os.path.normpath(model_dir)) + '_pairs.parquet')
    return pl.read_parquet(f) if os.path.exists(f) else None


def text_of(names, addrs):
    return [f'{PREFIX}{a} | {b}'.strip() for a, b in zip(names, addrs)]


def load_model(enc, device, split=None):
    from sentence_transformers import SentenceTransformer
    kw = {'revision': ZS_REVISION} if enc == 'zs' else {}
    m = SentenceTransformer(model_path(enc, split), device=device, **kw)
    m.max_seq_length = MAXLEN
    if device != 'cpu': m.half()
    m.eval()
    return m


def encode(W, enc, split, part, device='cpu', batch_size=None, log=print):
    """records/{split}_{part}.parquet -> dense/{enc}/emb_{split}_{part}.f16.npy (row i == record row i)."""
    out = W.emb(enc, split, part)
    if os.path.exists(out) and os.path.exists(out + '.done'):
        log('skip (exists)', out); return
    import torch
    df = pl.read_parquet(W.records(split, part), columns=['business_name', 'business_address'])
    txt = text_of(df['business_name'].to_list(), df['business_address'].to_list()); N = len(txt)
    m = load_model(enc, device, split); log('encoder', enc, split, model_path(enc, split))
    bs = batch_size or (1024 if device != 'cpu' else 256)
    t0 = time.time(); E = np.lib.format.open_memmap(out + '.tmp', mode='w+', dtype=np.float16, shape=(N, DIM))
    CH = 200_000; tl = t0
    with torch.inference_mode():
        for s in range(0, N, CH):
            e = m.encode(txt[s:s + CH], batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            E[s:s + CH] = e.astype(np.float16)
            if time.time() - tl > 60 or s + CH >= N:
                log(f'encode {enc} {split}_{part}: {min(s + CH, N)}/{N} {min(s + CH, N) / (time.time() - t0):.0f} rec/s'); tl = time.time()
    E.flush(); del E
    os.rename(out + '.tmp', out)
    open(out + '.done', 'w').write(json.dumps({'rows': N, 'sec': time.time() - t0, 'device': device, 'model': model_path(enc, split)}) + '\n')
    log(f'encode {enc} {split}_{part} DONE {N} rows {time.time() - t0:.0f}s')


def _topk_ip(Q, P, k, qbs, pbs, dev):
    import torch
    nq, npool = Q.shape[0], P.shape[0]; k = min(k, npool)
    V = torch.empty((nq, k), dtype=torch.float32, device=dev); I = torch.empty((nq, k), dtype=torch.int64, device=dev)
    for qs in range(0, nq, qbs):
        q = Q[qs:qs + qbs]; bv = bi = None
        for ps in range(0, npool, pbs):
            S = q @ P[ps:ps + pbs].T
            v, i = torch.topk(S, min(k, S.shape[1]), dim=1, sorted=False); i += ps; del S
            if bv is None: bv, bi = v, i
            else:
                v = torch.cat([bv, v], 1); i = torch.cat([bi, i], 1)
                bv, j = torch.topk(v, k, dim=1, sorted=False); bi = torch.gather(i, 1, j)
        ex = torch.einsum('bd,bkd->bk', q.float(), P[bi].float())     # exact fp32 re-score of the selected
        ex, j = torch.sort(ex, dim=1, descending=True)
        V[qs:qs + qbs] = ex; I[qs:qs + qbs] = torch.gather(bi, 1, j)
    return V, I


def search(W, enc, split, device='cpu', fwd_k=50, rev_k=5, log=print, query_rows=None):
    """forward_{split}.parquet: s1_idx, cand_idx, rank (1 = best), score ; reverse_{split}.parquet: cand_idx, s1_idx, rank, score.
    Forward queries = all S1 (or query_rows), pool = all S2/S3 of the same country; reverse = every record against ALL S1 of its country."""
    import torch
    dev = torch.device(device)
    c1 = pl.read_parquet(W.records(split, 's1'), columns=['country'])['country'].to_numpy()
    c23 = pl.read_parquet(W.records(split, 's23'), columns=['country'])['country'].to_numpy()
    M1 = np.load(W.emb(enc, split, 's1'), mmap_mode='r'); M23 = np.load(W.emb(enc, split, 's23'), mmap_mode='r')
    dt = torch.float16 if device != 'cpu' else torch.float32
    qbs, pbs = (4096, 1 << 19) if device != 'cpu' else (1024, 1 << 17)
    def to_dev(M, rows): return torch.from_numpy(np.ascontiguousarray(M[rows])).to(dev).to(dt)
    for kind, out in [('fwd', W.forward(enc, split)), ('rev', W.reverse(enc, split))]:
        if os.path.exists(out): log('skip (exists)', out); continue
        parts = []; t0 = time.time(); nsc = 0
        if kind == 'fwd':
            qc, pc, MQ, MP, k = c1, c23, M1, M23, fwd_k
            qsel = np.arange(len(c1)) if query_rows is None else np.asarray(query_rows)
        else:
            qc, pc, MQ, MP, k = c23, c1, M23, M1, rev_k
            qsel = np.arange(len(c23))
        for c in np.unique(qc[qsel]):
            prow = np.where(pc == c)[0]
            if len(prow) == 0: continue
            X = to_dev(MP, prow)
            sel = qsel[qc[qsel] == c]
            for s in range(0, len(sel), 200_000):
                ss = sel[s:s + 200_000]
                V, I = _topk_ip(to_dev(MQ, ss), X, k, qbs, pbs, dev)
                kk = V.shape[1]
                parts.append(pl.DataFrame({'q': np.repeat(ss, kk).astype(np.int32), 'p': prow[I.cpu().numpy()].ravel().astype(np.int32),
                                           'rank': np.tile(np.arange(1, kk + 1, dtype=np.uint8), len(ss)), 'score': V.cpu().numpy().ravel().astype(np.float32)}))
                nsc += len(ss) * len(prow)
                log(f'{kind} {enc} {split} {c}: {s + len(ss)}/{len(sel)} queries, {nsc / (time.time() - t0):.2e} scores/s')
            del X
            if device != 'cpu': torch.cuda.empty_cache()
        df = pl.concat(parts)
        df = df.rename({'q': 's1_idx', 'p': 'cand_idx'}) if kind == 'fwd' else df.rename({'q': 'cand_idx', 'p': 's1_idx'})
        df.write_parquet(out + '.tmp'); os.rename(out + '.tmp', out)
        log(f'{kind} {enc} {split} DONE rows {df.height} {time.time() - t0:.0f}s')
