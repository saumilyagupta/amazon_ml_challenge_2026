#!/opt/conda/bin/python3
"""Optional: re-create the fine-tuned dense encoders from the base model (GPU, about 11 minutes each on a V100).

usage: CUDA_VISIBLE_DEVICES=<gpu> /opt/conda/bin/python3 finetune.py --data <dataset dir> [--out resources/models/e5s_ft_all] [--pairs 150000]
       ... --exclude-s1 resources/splits/val_s1_ids.txt --out resources/models/e5s_ft_trainsplit      (the train-split checkpoint)
The sampled training pairs are identical (same pairs, same order) to resources/ft_pairs/{e5s_ft_all,e5s_ft_trainsplit}_pairs.parquet
(checked in the package build), and are written next to the model as training_pairs.parquet.

Recipe (identical to the research run that produced the shipped checkpoint): sample 150,000 ground-truth (S1, S2/S3) pairs from
train_ground_truth.tsv with pandas .sample(random_state=1); two training examples per pair, ('query: name | address' of the record,
the same for the S1) and a name-only view ('query: name', 'query: name'); random.seed(0) shuffle; MultipleNegativesSymmetricRankingLoss
(in-batch negatives), batch 128, drop_last, 1 epoch (2,343 steps), warmup 200, AMP fp16, max_seq_length 96, sentence-transformers fit()
defaults otherwise (lr 2e-5 linear); base model intfloat/multilingual-e5-small (MIT). No telemetry: report_to='none', WANDB disabled.
Two checkpoints are shipped: e5s_ft_all (pairs of all train S1; used for the TEST split) and e5s_ft_trainsplit (pairs of train-split
S1 only, i.e. excluding the 220,730 validation S1; used for the TRAIN split, so that validation and the training sample see
out-of-sample scores except for the S1 of its own training pairs, which are flagged ft_seen_s1 and dropped)."""
import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', default=None); ap.add_argument('--pairs', type=int, default=150000)
ap.add_argument('--exclude-s1', default=None, help='file of S1 ids whose ground-truth pairs are excluded (train-split checkpoint: resources/splits/val_s1_ids.txt)')
A = ap.parse_args()
from ber import env
env.setup(8, gpu='keep')
import pandas as pd, torch
from ber.paths import RES
from ber.dense import ZS_MODEL, ZS_REVISION
import sentence_transformers.fit_mixin as FM
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
OUT = A.out or os.path.join(RES, 'models', 'e5s_ft_all')
_TA = FM.SentenceTransformerTrainingArguments
class _NoReportArgs(_TA):
    def __init__(self, *a, **k): k['report_to'] = 'none'; super().__init__(*a, **k)
FM.SentenceTransformerTrainingArguments = _NoReportArgs
t0 = time.time()
rd = lambda p: pd.read_csv(p, sep='\t', quoting=3, dtype=str, keep_default_na=False)
s1 = rd(f'{A.data}/train/train_source1.tsv').set_index('entity_id')
r = pd.concat([rd(f'{A.data}/train/train_source2.tsv'), rd(f'{A.data}/train/train_source3.tsv')]).set_index('entity_id')
gt = rd(f'{A.data}/train/train_ground_truth.tsv')
gt = gt[gt.matched_entity_ids != ''].assign(rid=lambda d: d.matched_entity_ids.str.split(',')).explode('rid').rename(columns={'source1_entity_id': 's1'})[['s1', 'rid']].reset_index(drop=True)
if A.exclude_s1:
    ex_ids = set(l.strip() for l in open(A.exclude_s1) if l.strip()); gt = gt[~gt.s1.isin(ex_ids)]
p = gt.sample(A.pairs, random_state=1)
A_ = s1.loc[p.s1]; B_ = r.loc[p.rid]
ex = []
for an, aa, bn, ba in zip(A_.business_name, A_.business_address, B_.business_name, B_.business_address):
    ex.append(InputExample(texts=['query: ' + bn + ' | ' + ba, 'query: ' + an + ' | ' + aa]))
    ex.append(InputExample(texts=['query: ' + bn, 'query: ' + an]))
random.seed(0); random.shuffle(ex)
print(f'{len(p)} pairs ({p.s1.nunique()} S1), {len(ex)} examples, data prep {time.time() - t0:.0f}s', flush=True)
torch.manual_seed(0)
m = SentenceTransformer(ZS_MODEL, revision=ZS_REVISION, device='cuda' if torch.cuda.is_available() else 'cpu'); m.max_seq_length = 96
dl = DataLoader(ex, shuffle=True, batch_size=128, drop_last=True)
loss = losses.MultipleNegativesSymmetricRankingLoss(m)
os.makedirs(OUT + '_work', exist_ok=True); cwd = os.getcwd(); os.chdir(OUT + '_work')
t1 = time.time(); m.fit(train_objectives=[(dl, loss)], epochs=1, warmup_steps=200, use_amp=True, show_progress_bar=False); os.chdir(cwd)
m.save(OUT)
import polars as pl
pl.DataFrame({'s1_id': p.s1.values, 'cand_id': p.rid.values}).write_parquet(os.path.join(OUT, 'training_pairs.parquet'))
print(f'trained in {time.time() - t1:.0f}s, saved {OUT}', flush=True)
