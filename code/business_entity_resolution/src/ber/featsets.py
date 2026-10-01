"""Feature lists per variant, derived from the config (so that the order always equals the order the models were trained with).

stage 1 = 89 pair features (ber.pairfeats) [+ 37 union channel features + 7 fine-tuned-member channel features (ber.chan), if channel_feats]
          [+ 127 explainer features (ber.explainer), if features.explainer] [+ 14 v2b additions (ber.v2bfeats.C_FEATS), if features.v2b_extra]
          minus features.drop (v2b: the 7 density-shifting reverse-membership / count features; v3: + dec_num_shift, dec_flag)
          [+ 15 offset-conditioned decoy features (ber.v3decoy.DEC_FEATS), if features.v3_decoy]
          [+ 14 France-safe branch features (ber.branch.Q14, 'bp_' prefix), if features.branch_q]
stage 2 = stage 1 + 13 list / record-competition features (ber.model.S2FEATS) [+ 6 sibling-corroboration features, if features.sibling]
v2b: 267 / 286 features; v3: 294 / 313 (= research variant 'AD').
France-gated pack (features.pack_gate, v3): the feature NAMES do not change; at scoring time the PACK_AFFECTED columns of gated rows are
replaced by their 'pk_' pack views (pack_gate_expr)."""
import polars as pl
from .pairfeats import ALL_FEATS
from .chan import CHAN_FEATS, CHAN_FEATS_FT
from .model import S2FEATS
from .v2bfeats import C_FEATS, SIB_FEATS
from .v3decoy import DEC_FEATS as V3_DEC_FEATS

CATEGORICAL = ['script2', 'a3_num_rel', 'prov', 'bp_a_num_rel']
DENSITY_SHIFT = ['u_in_rev', 'u_in_c1r', 'u_in_c2r', 'u_in_c3r', 'u_n_lexrev_s1', 'u_n_member_channels', 'u_in_rev_ft']
PK = 'pk_'


def explainer_feats():
    from .explainer import ALL_FEATURES
    return list(ALL_FEATURES)


def branch_feats():
    from .branch import Q14
    return list(Q14)


def pack_affected():
    from .packfeats import AFFECTED
    return list(AFFECTED)


def stage1(cfg, cols0):
    fc = cfg['features']
    F = list(ALL_FEATS) + ([c for c in CHAN_FEATS + CHAN_FEATS_FT if c in cols0] if fc['channel_feats'] else [])
    if fc.get('explainer'): F += explainer_feats()
    if fc.get('v2b_extra'): F += list(C_FEATS)
    drop = set(fc.get('drop') or [])
    F = [f for f in F if f not in drop]
    if fc.get('v3_decoy'): F += list(V3_DEC_FEATS)
    if fc.get('branch_q'): F += branch_feats()
    return F


def stage2_extra(cfg):
    return list(S2FEATS) + (list(SIB_FEATS) if cfg['features'].get('sibling') else [])


def pack_gate_cols(cfg, F1):
    """the stage-1 features that the France-gated pack replaces (empty when features.pack_gate is off)."""
    return [f for f in pack_affected() if f in F1] if cfg['features'].get('pack_gate') else []


def pack_gate_expr(cfg, F1, cols):
    """polars expressions that substitute the pack view pk_<f> for <f> on rows whose country is gated (research 08_test.py --gate-pack);
    no-op when the part carries no pk_ columns (e.g. a split without gated countries)."""
    g = [f for f in pack_gate_cols(cfg, F1) if PK + f in cols]
    if not g: return [], []
    fr = pl.col('country').is_in(list(cfg['features']['pack_gate']))
    return [pl.when(fr).then(pl.col(PK + f)).otherwise(pl.col(f)).alias(f) for f in g], [PK + f for f in g]
