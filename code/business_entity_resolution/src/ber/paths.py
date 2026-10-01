"""Work-directory layout. Everything the pipeline writes lives under ONE work root (--work), so runs are self-contained.

WORK/
  records/{split}_s1.parquet, {split}_s23.parquet      prepared records (row order = record index used everywhere below)
  records/rec_{split}_{s1,s23}.parquet, idf_*, voc_*    matcher record views (ber.records)
  records/norm_{split}_{s1,pool}.parquet                lexical-channel views (ber.lexnorm)
  records/{split}_gt_pairs.parquet                      labelled pairs (train only)
  records/hn_{split}.npz                                multi-part house numbers (v2b, ber.v2bfeats)
  dense/{enc}/emb_{split}_{s1,s23}.f16.npy              e5 embeddings (enc = zs | ft; ft is split-aware, see ber.dense)
  dense/{enc}/forward_{split}.parquet, reverse_{split}.parquet
  lexical/{c1,c2,c3,c1r,c2r,c3r}_{split}.parquet
  union/{union}_{split}.parquet                         blocking union with provenance
  cands/{variant}_{split}.parquet                       candidate table (labels / groups / folds on train)
  feats/{variant}/{split}_NNN.parquet                   feature parts
  models/{variant}/                                     LightGBM models + decision.json
  preds/{variant}_{split}.parquet                       stage-1/2 probabilities (+ stage-2 features)
  output/{variant}/matching_results.tsv, candidate_pairs.tsv
"""
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(SRC, 'resources')


class Work:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        for d in ('records', 'dense', 'lexical', 'union', 'cands', 'feats', 'models', 'preds', 'output', 'logs'):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    # records
    def records(self, split, part): return self.p('records', f'{split}_{part}.parquet')
    def rec(self, split, part): return self.p('records', f'rec_{split}_{part}.parquet')
    def idf(self, split, nm): return self.p('records', f'idf_{split}_{nm}.npy')
    def voc(self, split, nm): return self.p('records', f'voc_{split}_{nm}.parquet')
    def norm(self, split, part): return self.p('records', f'norm_{split}_{part}.parquet')
    def gt_pairs(self, split): return self.p('records', f'{split}_gt_pairs.parquet')
    # dense
    def dense_dir(self, enc):
        d = self.p('dense', enc); os.makedirs(d, exist_ok=True); return d
    def emb(self, enc, split, part): return os.path.join(self.dense_dir(enc), f'emb_{split}_{part}.f16.npy')
    def forward(self, enc, split): return os.path.join(self.dense_dir(enc), f'forward_{split}.parquet')
    def reverse(self, enc, split): return os.path.join(self.dense_dir(enc), f'reverse_{split}.parquet')
    # lexical / union
    def lexical(self, ch, split): return self.p('lexical', f'{ch}_{split}.parquet')
    def union(self, name, split): return self.p('union', f'{name}_{split}.parquet')
    # matcher
    def cands(self, variant, split): return self.p('cands', f'{variant}_{split}.parquet')
    def dense_table(self, enc, split): return self.p('cands', f'dense_table_{enc}_{split}.parquet')
    def feats_dir(self, variant):
        d = self.p('feats', variant); os.makedirs(d, exist_ok=True); return d
    def models_dir(self, variant):
        d = self.p('models', variant); os.makedirs(d, exist_ok=True); return d
    def preds(self, variant, split): return self.p('preds', f'{variant}_{split}.parquet')
    def output_dir(self, variant):
        d = self.p('output', variant); os.makedirs(d, exist_ok=True); return d
