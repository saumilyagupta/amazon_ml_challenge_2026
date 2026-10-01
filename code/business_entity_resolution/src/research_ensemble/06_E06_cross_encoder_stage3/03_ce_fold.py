# E06 step 3 (GPU, run through work/common/gpu_run.sh): fine-tune the XLM-R cross-encoder on v2b's OWN hard band of ONE sample fold, then
# score everything this fold's model is needed for, in the same process (GPU held once):
#   train = sample band pairs (0.005 < OOF p2 < 0.995) of S1 fold f with es = False; dev (epoch selection) = the fold's es rows (same band);
#   after training: best epoch by dev AUC -> score the OTHER fold's sample band (OOF logits for stage 3), the val band and the density band.
# Warm start from cross_encoder/models/ce_xlmr/ep0 (trained on 100k train-split S1 + TF-IDF top-10 pairs; fine for OOF sample + val stacking).
# Recipe (adapted from cross_encoder/train_ce.py): BCE, AdamW (wd 0.01, no decay on bias/LayerNorm), linear warmup 5% + decay, fp16 autocast +
# GradScaler, grad clip 1.0, bs 64, length bucketing within chunks of 100 batches; scoring with fp16 weights, length-sorted, bs 512.
# usage: python3 03_ce_fold.py --fold 0 [--lr 2e-5] [--epochs 2] [--init <model dir>] [--tag xw]
import os, sys, time, math, json, argparse
os.environ['OMP_NUM_THREADS'] = '3'; os.environ['MKL_NUM_THREADS'] = '3'; os.environ['RAYON_NUM_THREADS'] = '3'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; os.environ['WANDB_MODE'] = 'disabled'; os.environ['WANDB_DISABLED'] = 'true'
os.environ['POLARS_MAX_THREADS'] = '3'
import numpy as np, torch, polars as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score
torch.set_num_threads(3)
ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, required=True); ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--bs', type=int, default=64); ap.add_argument('--init', default='/workspace/saumilya/amazon-ml/work/matching/cross_encoder/models/ce_xlmr/ep0')
ap.add_argument('--tag', default='xw'); ap.add_argument('--evals_per_epoch', type=int, default=4); ap.add_argument('--view', default='')
ap.add_argument('--max_train', type=int, default=0); ap.add_argument('--max_score', type=int, default=0)
ap.add_argument('--subset', default='all', choices=['all', 'nonempty'], help='nonempty: train/dev only on pairs with a non-empty candidate address (addr_empty2 <= 0)')
a = ap.parse_args(); f = a.fold
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'
OUT = f'{E}/models/ce_{a.tag}_f{f}'; os.makedirs(OUT, exist_ok=True); SC = f'{E}/scores'; os.makedirs(SC, exist_ok=True)
T0 = time.time(); log = lambda *x: print(f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s]', *x, flush=True)
torch.manual_seed(f); np.random.seed(f)
dev = torch.device('cuda'); PAD = 1
log('device', torch.cuda.get_device_name(0), 'free/total GB', [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()], vars(a))
V = a.view
Z = dict(np.load(f'{E}/data/tok_sample{V}.npz'))
ids_all, off_all = Z['ids'], Z['off']; lens_all = np.diff(off_all)
def rows(mask): return np.flatnonzero(mask)
sub = np.ones(len(Z['fold']), bool)
if a.subset == 'nonempty':
    Bs = pl.read_parquet(f'{E}/data/band_train.parquet', columns=['s1_idx', 'cand_idx', 'grp', 'addr_empty2']).filter(pl.col('grp') == 'sample')
    assert (Bs['s1_idx'].to_numpy() == Z['s1_idx']).all() and (Bs['cand_idx'].to_numpy() == Z['cand_idx']).all()
    sub = Bs['addr_empty2'].to_numpy() <= 0
tr_rows = rows((Z['fold'] == f) & (~Z['es']) & sub); dv_rows = rows((Z['fold'] == f) & Z['es'] & sub); ot_rows = rows(Z['fold'] == 1 - f)
if a.max_train: tr_rows = tr_rows[:a.max_train]
if a.max_score: dv_rows = dv_rows[:a.max_score]; ot_rows = ot_rows[:a.max_score]
ytr = Z['label'][tr_rows].astype(np.float32); ydv = Z['label'][dv_rows]; bsdv = Z['bs'][dv_rows]; p2dv = Z['p2'][dv_rows]
log('train pairs', len(tr_rows), 'pos rate', round(float(ytr.mean()), 4), '| dev pairs', len(dv_rows), 'pos', round(float(ydv.mean()), 4), '| other-fold pairs', len(ot_rows))
def collate(ids, off, lens, idx):
    L = int(lens[idx].max()); arr = np.full((len(idx), L), PAD, np.int64)
    for k, j in enumerate(idx): arr[k, :lens[j]] = ids[off[j]:off[j + 1]]
    x = torch.from_numpy(arr).pin_memory().to(dev, non_blocking=True)
    return x, (x != PAD).long()
model = AutoModelForSequenceClassification.from_pretrained(a.init, num_labels=1).to(dev)
no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
groups = [{'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': 0.01},
          {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0}]
try: opt = torch.optim.AdamW(groups, lr=a.lr, fused=True)
except Exception as e: log('fused AdamW unavailable', e); opt = torch.optim.AdamW(groups, lr=a.lr)
nb = math.ceil(len(tr_rows) / a.bs); steps = a.epochs * nb
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler('cuda'); lossf = torch.nn.BCEWithLogitsLoss()
@torch.no_grad()
def predict(ids, off, rws, bs=512, half=False):
    model.eval(); lens = np.diff(off); ln = lens[rws]; order = np.argsort(ln, kind='stable'); out = np.zeros(len(rws), np.float32)
    torch.cuda.synchronize(); t = time.time()
    for i in range(0, len(rws), bs):
        o = order[i:i + bs]; x, m = collate(ids, off, lens, rws[o])
        if half: lo = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)
        else:
            with torch.autocast('cuda', dtype=torch.float16): lo = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)
        out[o] = lo.cpu().numpy()
    torch.cuda.synchronize(); el = time.time() - t; model.train()
    return out, len(rws) / max(el, 1e-6)
def m2(y, p): return dict(auc=round(float(roc_auc_score(y, p)), 5), ap=round(float(average_precision_score(y, p)), 5))
def evaluate(tag, step):
    p, thr = predict(ids_all, off_all, dv_rows)
    r = dict(tag=tag, step=step, dev=m2(ydv, p), dev_scoring_band=m2(ydv[bsdv], p[bsdv]), infer_pairs_per_s=round(thr))
    log('EVAL', json.dumps(r)); return p, r
hist = dict(args=vars(a), n_train=len(tr_rows), n_dev=len(dv_rows), pos_train=float(ytr.mean()), pos_dev=float(ydv.mean()),
            p2_dev=m2(ydv, p2dv), p2_dev_scoring_band=m2(ydv[bsdv], p2dv[bsdv]), evals=[], epochs=[])
log('p2 on dev', hist['p2_dev'], 'scoring band', hist['p2_dev_scoring_band'])
_, r0 = evaluate('init', 0); hist['evals'].append(r0)
step = 0; ev_every = max(1, nb // a.evals_per_epoch); dev_p = {}
for ep in range(a.epochs):
    model.train(); perm = np.random.permutation(len(tr_rows))
    chunks = [perm[i:i + a.bs * 100] for i in range(0, len(perm), a.bs * 100)]
    batches = []
    for c in chunks:
        c = c[np.argsort(lens_all[tr_rows[c]], kind='stable')]
        batches += [c[i:i + a.bs] for i in range(0, len(c), a.bs)]
    np.random.shuffle(batches); t = time.time(); run = 0.0; seen = 0
    for bi, b in enumerate(batches):
        x, m = collate(ids_all, off_all, lens_all, tr_rows[b]); y = torch.from_numpy(ytr[b]).to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            lo = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)
        loss = lossf(lo, y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); step += 1; seen += len(b)
        run = 0.98 * run + 0.02 * loss.item() if bi else loss.item()
        if bi % 500 == 0:
            log(f'ep {ep} step {bi}/{len(batches)} loss {run:.4f} {seen / (time.time() - t):.0f} pairs/s lr {sch.get_last_lr()[0]:.2e} '
                f'mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GB')
        if (bi + 1) % ev_every == 0 and (bi + 1) < len(batches):
            _, r = evaluate(f'ep{ep}_b{bi + 1}', step); r['loss_ema'] = round(run, 4); hist['evals'].append(r)
    tr_time = time.time() - t
    p, r = evaluate(f'epoch{ep}', step); r.update(train_s=round(tr_time), train_pairs_per_s=round(len(tr_rows) / tr_time), loss_ema=round(run, 4))
    hist['evals'].append(r); hist['epochs'].append(r); dev_p[ep] = p
    model.save_pretrained(f'{OUT}/ep{ep}')
    json.dump(hist, open(f'{OUT}/history.json', 'w'), indent=1)
hist['peak_train_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
best = max(range(a.epochs), key=lambda e: hist['epochs'][e]['dev']['auc']); hist['best_epoch'] = best; log('best epoch', best)
if best != a.epochs - 1:
    del model, opt; torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(f'{OUT}/ep{best}', num_labels=1).to(dev)
model.half().eval()
def save(name, s1, ca, lo): pl.DataFrame({'s1_idx': s1, 'cand_idx': ca, 'ce_logit': lo}).write_parquet(f'{SC}/ce_{a.tag}_f{f}_{name}.parquet')
save('dev', Z['s1_idx'][dv_rows], Z['cand_idx'][dv_rows], dev_p[best])
lo, thr = predict(ids_all, off_all, ot_rows, half=True); save('oof', Z['s1_idx'][ot_rows], Z['cand_idx'][ot_rows], lo)
hist['oof'] = dict(n=len(ot_rows), **m2(Z['label'][ot_rows], lo), scoring_band=m2(Z['label'][ot_rows][Z['bs'][ot_rows]], lo[Z['bs'][ot_rows]]), pairs_per_s=round(thr))
log('OOF other fold', hist['oof'])
for nm in ['val', 'dens']:
    fn = f'{E}/data/tok_{nm}{V}.npz'
    while not os.path.exists(fn): log('waiting for', fn); time.sleep(30)
    Y = np.load(fn); rw = np.arange(len(Y['off']) - 1)
    if a.max_score: rw = rw[:a.max_score]
    lo, thr = predict(Y['ids'], Y['off'], rw, half=True); save(nm, Y['s1_idx'][rw], Y['cand_idx'][rw], lo)
    yl, bb = Y['label'][rw], Y['bs'][rw]
    hist[nm] = dict(n=len(rw), **m2(yl, lo), scoring_band=m2(yl[bb], lo[bb]), pairs_per_s=round(thr)); log(nm, hist[nm])
hist['total_s'] = round(time.time() - T0)
json.dump(hist, open(f'{OUT}/history.json', 'w'), indent=1)
log('DONE')
