# Fine-tune a multilingual binary cross-encoder on data/pairs_train.parquet, evaluate on data/pairs_dev.parquet.
# Input text: '<name1> | <address1>' [SEP] '<name2> | <address2>'.  fp16 autocast, dynamic padding, length bucketing.
import os
os.environ.setdefault('RAYON_RS_NUM_CPUS', '3'); os.environ.setdefault('RAYON_NUM_THREADS', '3'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, time, math, json, argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score
torch.set_num_threads(2)
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--max_len', type=int, default=128)
ap.add_argument('--out', default='models/ce_minilm'); ap.add_argument('--max_train', type=int, default=0)
ap.add_argument('--eval_every', type=int, default=0, help='extra dev eval every N steps (0=only at epoch end)')
a = ap.parse_args()
BASE = '/workspace/saumilya/amazon-ml/work/matching/cross_encoder/'
OUT = BASE + a.out; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
dev = torch.device('cuda')
print('device', torch.cuda.get_device_name(0), 'free/total GB', [round(x/2**30, 1) for x in torch.cuda.mem_get_info()], vars(a), flush=True)
def txt(n, ad): return [f'{x} | {y}'.lower() for x, y in zip(n, ad)]   # lowercase: removes ALL-CAPS noise (S2 US) and shortens token sequences
tr = pd.read_parquet(BASE+'data/pairs_train.parquet', columns=['name1', 'addr1', 'name2', 'addr2', 'label'])
dv = pd.read_parquet(BASE+'data/pairs_dev.parquet', columns=['s1', 'cand', 'name1', 'addr1', 'name2', 'addr2', 'label'])
if a.max_train: tr = tr.iloc[:a.max_train]
tok = AutoTokenizer.from_pretrained(a.model)
def enc(df):
    return tok(txt(df.name1, df.addr1), txt(df.name2, df.addr2), truncation=True, max_length=a.max_len)['input_ids']
t0 = time.time(); Xtr = enc(tr); t1 = time.time(); Xdv = enc(dv); ytr = tr.label.values.astype(np.float32); ydv = dv.label.values
print(f'tokenized train {len(Xtr)} in {t1-t0:.0f}s ({len(Xtr)/(t1-t0):.0f} pairs/s, 3 rayon threads), dev {len(Xdv)}; mean len',
      round(float(np.mean([len(x) for x in Xtr])), 1), 'trunc frac', round(float(np.mean([len(x) >= a.max_len for x in Xtr])), 4), flush=True)
pad = tok.pad_token_id
def collate(seqs):
    L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s)
    return ids, (ids != pad).long()
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
print('params', sum(p.numel() for p in model.parameters()), flush=True)
no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
groups = [{'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': 0.01},
          {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0}]
opt = torch.optim.AdamW(groups, lr=a.lr)
steps = a.epochs * math.ceil(len(Xtr) / a.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler('cuda'); lossf = torch.nn.BCEWithLogitsLoss()
@torch.no_grad()
def predict(X, bs=512):
    model.eval(); order = np.argsort([len(x) for x in X], kind='stable'); out = np.zeros(len(X), dtype=np.float32)
    torch.cuda.synchronize(); t = time.time()
    for i in range(0, len(X), bs):
        idx = order[i:i+bs]; ids, m = collate([X[j] for j in idx])
        with torch.autocast('cuda', dtype=torch.float16):
            lo = model(input_ids=ids.to(dev, non_blocking=True), attention_mask=m.to(dev, non_blocking=True)).logits.float().squeeze(-1)
        out[idx] = lo.cpu().numpy()
    torch.cuda.synchronize(); el = time.time() - t; model.train()
    return out, len(X) / el
def evaluate(tag):
    p, thr = predict(Xdv)
    r = dict(tag=tag, auc=round(roc_auc_score(ydv, p), 5), ap=round(average_precision_score(ydv, p), 5), infer_pairs_per_s_bs512=round(thr))
    print('EVAL', r, flush=True); return p, r
hist = []; step = 0
# smoke-test the eval + save path before training (untrained model)
_p, _t = predict(Xdv[:4096]); print('pre-train smoke eval auc', round(roc_auc_score(ydv[:4096], _p), 4) if len(set(ydv[:4096])) > 1 else None, f'{_t:.0f} pairs/s', flush=True)
tok.save_pretrained(OUT + '/tokenizer_check')
for ep in range(a.epochs):
    model.train(); perm = np.random.permutation(len(Xtr))
    chunks = [perm[i:i+a.bs*100] for i in range(0, len(perm), a.bs*100)]   # length-bucket within chunks of 100 batches
    batches = []
    for c in chunks:
        c = c[np.argsort([len(Xtr[j]) for j in c], kind='stable')]
        batches += [c[i:i+a.bs] for i in range(0, len(c), a.bs)]
    np.random.shuffle(batches); t = time.time(); run = 0.0; seen = 0
    for bi, b in enumerate(batches):
        ids, m = collate([Xtr[j] for j in b]); y = torch.tensor(ytr[b], device=dev)
        with torch.autocast('cuda', dtype=torch.float16):
            lo = model(input_ids=ids.to(dev, non_blocking=True), attention_mask=m.to(dev, non_blocking=True)).logits.float().squeeze(-1)
        loss = lossf(lo, y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); step += 1; seen += len(b)
        run = 0.98 * run + 0.02 * loss.item() if bi else loss.item()
        if bi % 500 == 0:
            print(f'ep {ep} step {bi}/{len(batches)} loss {run:.4f} {seen/(time.time()-t):.0f} pairs/s lr {sch.get_last_lr()[0]:.2e} '
                  f'mem {torch.cuda.max_memory_allocated()/2**30:.2f}GB elapsed {time.time()-t:.0f}s', flush=True)
        if a.eval_every and step % a.eval_every == 0: hist.append(evaluate(f'step{step}')[1])
    tr_time = time.time() - t
    p, r = evaluate(f'epoch{ep}'); r['train_s'] = round(tr_time); r['train_pairs_per_s'] = round(len(Xtr)/tr_time); hist.append(r)
    model.save_pretrained(OUT + f'/ep{ep}'); tok.save_pretrained(OUT + f'/ep{ep}')
    pd.DataFrame({'s1': dv.s1.values, 'cand': dv.cand.values, 'ce_logit': p}).to_parquet(OUT + f'/dev_scores_ep{ep}.parquet')
    json.dump(hist, open(OUT + '/history.json', 'w'), indent=1)
hist.append(dict(peak_train_mem_gb=round(torch.cuda.max_memory_allocated()/2**30, 2)))
json.dump(hist, open(OUT + '/history.json', 'w'), indent=1)
print('TRAIN_DONE', flush=True)
