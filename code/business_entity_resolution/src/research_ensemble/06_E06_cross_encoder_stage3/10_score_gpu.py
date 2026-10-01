# E06 step 10 (GPU via gpu_run.sh): score token files with the two fine-tuned fold CEs (fp16 weights, length-sorted, bs 512).
# usage: python3 10_score_gpu.py <tok.npz> <out prefix>   -> <out prefix>_f{0,1}.parquet (s1_idx, cand_idx, ce_logit) + timing json
import os, sys, time, json
os.environ['OMP_NUM_THREADS'] = '3'; os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; os.environ['WANDB_MODE'] = 'disabled'
import numpy as np, torch, polars as pl
from transformers import AutoModelForSequenceClassification
torch.set_num_threads(3)
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'; PAD = 1; dev = torch.device('cuda')
Z = np.load(sys.argv[1]); ids, off = Z['ids'], Z['off']; lens = np.diff(off); n = len(lens); T = {}
for f in (0, 1):
    t0 = time.time(); model = AutoModelForSequenceClassification.from_pretrained(f'{E}/models/ce_xw_f{f}/ep1', num_labels=1).to(dev).half().eval(); tl = time.time() - t0
    order = np.argsort(lens, kind='stable'); out = np.zeros(n, np.float32); torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        for i in range(0, n, 512):
            idx = order[i:i + 512]; L = int(lens[idx].max()); arr = np.full((len(idx), L), PAD, np.int64)
            for k, j in enumerate(idx): arr[k, :lens[j]] = ids[off[j]:off[j + 1]]
            x = torch.from_numpy(arr).pin_memory().to(dev, non_blocking=True)
            out[idx] = model(input_ids=x, attention_mask=(x != PAD).long()).logits.float().squeeze(-1).cpu().numpy()
    torch.cuda.synchronize(); el = time.time() - t
    pl.DataFrame({'s1_idx': Z['s1_idx'], 'cand_idx': Z['cand_idx'], 'ce_logit': out}).write_parquet(f'{sys.argv[2]}_f{f}.parquet')
    T[f'f{f}'] = dict(n=n, load_s=round(tl, 1), score_s=round(el, 1), pairs_per_s=round(n / el)); print(T[f'f{f}'], flush=True)
    del model; torch.cuda.empty_cache()
json.dump(T, open(f'{sys.argv[2]}_timing.json', 'w'), indent=1)
