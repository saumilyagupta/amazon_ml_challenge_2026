"""Matcher-variant configs (src/configs/*.yaml). A config fully determines candidates, features, models and the decision layer,
so switching the final model = pointing --variant at another YAML. Defaults below are the production v1 recipe."""
import copy, os
import yaml
from .paths import SRC

DEFAULTS = {
    'name': None,
    'candidates': {            # what the matcher scores (= candidate_pairs.tsv)
        'kind': 'dense',       # dense : e5 forward@fwd_k + margin-gated reverse (rank 1, or rank<=5 within rev_gate of the record's top-1)
                               # union : REC20 blocking union file union/{union}_{split}.parquet (see ber.union)
        'encoder': 'zs',       # dense encoder: zs = intfloat/multilingual-e5-small zero-shot | ft = resources/models/e5s_ft_all (fine-tuned)
        'fwd_k': 20, 'rev_gate': 0.02,
        'union': None,         # union name (kind=union), e.g. 'v1' (zero-shot dense member) or 'v2' (fine-tuned dense member)
        'cap': 0,              # optional per-S1 cap by fused_rank (0 = none)
    },
    'features': {
        'dense_table_encoder': 'zs',   # record-side competition features are always computed over the dense table (fwd@20 + gate) of ALL S1 of this encoder
        'dense_table_fwd_k': 20, 'dense_table_gate': 0.02,
        'channel_feats': False,        # + 37 union channel features (u_*) [+ 7 ft channel features when the union has them]
        'explainer': False,            # + 127 noise-inversion explainer features (ber.explainer, word lists in resources/explainer_wordlists)
        'v2b_extra': False,            # + 14 v2b additions (multi-part house numbers, decoy, admin conflict, provenance, hub flag; ber.v2bfeats)
        'sibling': False,              # stage 2: + 6 sibling-corroboration features (ber.v2bfeats.sibling_features)
        'drop': [],                    # stage-1 features to leave out (v2b: ber.featsets.DENSITY_SHIFT; v3: + dec_num_shift, dec_flag)
        'v3_decoy': False,             # + 15 offset-conditioned decoy features (ber.v3decoy; v3)
        'branch_q': False,             # + 14 France-safe branch features (ber.branch.Q14; v3)
        'pack_gate': [],               # countries whose pairs take the France-pack views of the 43 affected features at scoring time (v3: [France])
        'part_rows': 3_000_000,
    },
    'split': {                 # how train S1 are partitioned into training sample / validation / locked holdout
        'mode': 'files',       # files : resources/splits/{train_sample,val,locked_val}_s1_ids.txt (intersected with the S1 present)
                               # hash  : deterministic hash split (val_frac, locked_frac, sample_size) for arbitrary subsets / smoke tests
        'val_frac': 0.10, 'locked_frac': 0.136, 'sample_size': 0,
    },
    'model': {
        'lgb': {'objective': 'binary', 'learning_rate': 0.1, 'num_leaves': 127, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
                'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbose': -1, 'seed': 7, 'max_bin': 63},
        'rounds': 400, 'early_stopping': 30, 'folds': 2,
        'competitors': None,   # variant whose stage-1 / stage-2 probabilities on the dense table of ALL S1 define record-side competition
                               # (None = this variant's own candidate table, valid only for kind=dense)
        'drop_ft_seen_s1': True,   # train split: drop the S1 whose pairs were used to fine-tune the train-split encoder (in-sample ft scores)
        'lexneg': 1.0,             # keep this share of lexical-only (not dense-ft member) negatives (deterministic pair hash), weight 1/share
    },
    'decision': {
        'taus': [round(0.2 + 0.025 * i, 3) for i in range(31)],
        'deltas': [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
        'row_model': True, 'efs': True, 'efs_L': 12, 'efs_draws': 256, 'efs_seed': 0,
        'force': None,         # e.g. {'policy': 'e', 'mode': 'plain', 'delta': 0.0, 'tau': 0.7} to skip tuning
        'set_decoder': False,  # + R10c learned prefix set-decoder (ber.setdecoder) with argmax-vs-best-other exclusivity, margins below
        'set_margins': [0.0, 0.2],
        'competition': 'v1',   # exclusivity competitor: 'v1' = competitor variant's p2 (v2b) | 'own' = this variant's p2 where it scores (v3)
        'one_owner': False,    # strict one-owner post-pass (each S2/S3 record keeps only its highest-p2 selecting S1; v3)
        'final_policy': None,  # e.g. 'R10c:m0.0' to fix the final policy instead of taking the best on validation
    },
    'hybrid': None,            # {'dense': 'v1', 'lexical_only': 'v2a'}: probabilities of variant `dense` on pairs of its dense table, of `lexical_only` elsewhere
    'postpass': None,          # v3: {'enabled': True, 'rules': [...], 'one_owner': True} = label-free neighbour-decoy vetoes after the decision (ber.postpass)
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(name_or_path):
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(SRC, 'configs', f'{name_or_path}.yaml')
    with open(path) as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    cfg['name'] = cfg['name'] or os.path.splitext(os.path.basename(path))[0]
    cfg['_path'] = path
    return cfg
