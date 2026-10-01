#!/opt/conda/bin/python3
"""Step 5: score the candidates of a split with a trained variant, apply its decision policy, write the two submission TSVs.

usage: /opt/conda/bin/python3 predict.py --work WORK --variant v1|v2a|H|v2b|v3|<cfg.yaml> --split test [--out DIR] [--threads 8]
       [--from-preds P.parquet] [--comp-preds C.parquet] [--decision D.json] [--s1-ids I1.parquet] [--s23-ids I2.parquet]
       [--models-dir DIR] [--feats-dir DIR] [--parts i j ..] [--rec-s23 R.parquet] [--preds-out P.parquet] [--score-only]

* normal mode: feats/{variant}/{split}_*.parquet -> stage 1 (average of the fold models) -> stage-2 features (record-side competition
  from the competitor variant's preds/{comp}_{split}.parquet, i.e. run predict.py for v1 before v2a / H) -> stage 2 -> preds/{variant}_{split}.parquet
* hybrid variant: combines preds/{dense}_{split}.parquet and preds/{lexical_only}_{split}.parquet by provenance (in_dense)
* --from-preds: skip scoring and only run the decision layer on a given probability table (s1_idx, cand_idx, [country, in_dense,] p1, p2);
  --comp-preds gives the competitor table (s1_idx, cand_idx, p1, p2) and --s1-ids / --s23-ids the id tables (entity_id, country) in
  s1_idx / cand_idx row order when the work dir has no prepared records (used for the reproduction check of the stored production run).
* --models-dir: take the fold models / decision.json / decoder from another directory, e.g. the SHIPPED v2b models in
  resources/models/v2b/ (predict.py --variant v2b --models-dir resources/models/v2b reproduces the submitted file from the v2b features)
* v2b: stage 2 adds the 6 sibling-corroboration features (--rec-s23: record views with nums / num_first / addr_n / name_n in cand_idx order,
  default WORK/records/rec_{split}_s23.parquet); decision policy 'R10c' = argmax-vs-best-other exclusivity against the competitor's p2
  (margin from decision.json) + the learned prefix set-decoder (ber.setdecoder).
* v3 (configs/v3.yaml, shipped models resources/models/v3): same scoring chain on 294 / 313 features; France-gated pack = for rows whose
  country is in features.pack_gate the 43 pack-affected features are read from their pk_ columns (ber.featsets.pack_gate_expr) at both
  stages; decision.json 'competition: own' = exclusivity against this variant's OWN p2 over all scored pairs (no competitor table needed
  for the decision; v1's p1 is still needed for the stage-2 features), then 'one_owner' = every record keeps only its highest-p2 S1.
* --feats-dir / --parts / --preds-out / --score-only: score feature parts from another directory (also files named part-NNN.parquet),
  optionally only some of them, and stop after writing the probabilities (reproduction checks on subsets).
* post-pass (config block 'postpass', enabled in configs/v3.yaml = THE SUBMITTED FILE): after the decision, the label-free neighbour-decoy
  vetoes of ber.postpass remove pairs (removal only; per-record explainer views + pattern pairs derived from the split's records and scored
  pairs, cached under WORK/records/pp_*). --no-postpass writes plain v3; --pre-postpass-out DIR also writes the decision's
  matching_results.tsv before the post-pass. Report: OUT/postpass_report.json (+ 'postpass' in stats.json).
Output (default WORK/output/{variant}/): matching_results.tsv (one row per S1 of the split) and candidate_pairs.tsv (= exactly the scored pairs)."""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--variant', required=True); ap.add_argument('--split', default='test')
ap.add_argument('--out', default=None); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--from-preds', default=None); ap.add_argument('--comp-preds', default=None); ap.add_argument('--decision', default=None)
ap.add_argument('--s1-ids', default=None); ap.add_argument('--s23-ids', default=None)
ap.add_argument('--models-dir', default=None); ap.add_argument('--feats-dir', default=None); ap.add_argument('--parts', type=int, nargs='+', default=None)
ap.add_argument('--rec-s23', default=None); ap.add_argument('--preds-out', default=None); ap.add_argument('--score-only', action='store_true')
ap.add_argument('--no-postpass', action='store_true'); ap.add_argument('--pre-postpass-out', default=None)
A = ap.parse_args()
from ber import env
NT = env.setup(A.threads)
import numpy as np, polars as pl, lightgbm as lgb
from ber.paths import Work
from ber import config
from ber.featsets import stage1, stage2_extra, pack_gate_expr
from ber.model import S2FEATS, stage2_features
from ber import decide as D, v2bfeats as VB, setdecoder as SD
log = env.logger(); W = Work(A.work); cfg = config.load(A.variant); V = cfg['name']; split = A.split; MD = A.models_dir or W.models_dir(V)
mc = cfg['model']; OUT = A.out or W.output_dir(V); os.makedirs(OUT, exist_ok=True)
comp_variant = mc['competitors']


def comp_table(col):
    if A.comp_preds: return pl.read_parquet(A.comp_preds, columns=['s1_idx', 'cand_idx', col]).rename({col: 'q'})
    if comp_variant is None: return None
    return pl.read_parquet(W.preds(comp_variant, split), columns=['s1_idx', 'cand_idx', col]).rename({col: 'q'})


FDIR = A.feats_dir or W.feats_dir(V)
parts = sorted(glob.glob(f'{FDIR}/{split}_[0-9]*.parquet')) or sorted(glob.glob(f'{FDIR}/part-[0-9]*.parquet'))
if A.parts: parts = [parts[i] for i in A.parts]
if A.from_preds:
    P = pl.read_parquet(A.from_preds); log('probabilities loaded from', A.from_preds, P.height)
elif cfg['hybrid']:
    hd, hl = cfg['hybrid']['dense'], cfg['hybrid']['lexical_only']
    P = pl.read_parquet(W.preds(hl, split))
    Q = pl.read_parquet(W.preds(hd, split), columns=['s1_idx', 'cand_idx', 'p1', 'p2']).rename({'p1': '_q1', 'p2': '_q2'})
    P = P.join(Q, on=['s1_idx', 'cand_idx'], how='left'); assert P.filter(pl.col('in_dense'))['_q2'].null_count() == 0
    P = P.with_columns(pl.when(pl.col('in_dense')).then(pl.col('_q1')).otherwise(pl.col('p1')).alias('p1'),
                       pl.when(pl.col('in_dense')).then(pl.col('_q2')).otherwise(pl.col('p2')).alias('p2')).drop('_q1', '_q2')
    P.write_parquet(W.preds(V, split)); log('hybrid probabilities', P.height)
else:
    assert parts, f'no feature parts for {V}/{split}: run features.py first'
    cols0 = pl.scan_parquet(parts[0]).collect_schema().names()
    F1 = stage1(cfg, cols0); F2X = stage2_extra(cfg)
    META = [c for c in ['s1_idx', 'cand_idx', 'country', 'in_dense', 'in_dft'] if c in cols0]
    GX, GC = pack_gate_expr(cfg, F1, cols0)   # v3: France-gated pack (pk_<f> replaces <f> on gated-country rows); ([], []) otherwise
    if GX: log('France-gated pack:', len(GX), 'features replaced on rows with country in', cfg['features']['pack_gate'])
    def read(p, cols):
        X = pl.read_parquet(p, columns=list(dict.fromkeys(cols + (['country'] + GC if GX else []))))
        return X.with_columns(GX).drop(GC) if GX else X
    M1 = [lgb.Booster(model_file=f'{MD}/s1_fold{f}.txt') for f in range(mc['folds'])]
    M2 = [lgb.Booster(model_file=f'{MD}/s2_fold{f}.txt') for f in range(mc['folds'])]
    assert M1[0].feature_name() == F1, 'feature list mismatch between the model and the feature parts'
    P = []
    for p in parts:
        X = read(p, META + F1); Xa = X.select(F1).to_numpy().astype(np.float32)
        P.append(X.select(META).with_columns(pl.Series('p1', (sum(m.predict(Xa) for m in M1) / len(M1)).astype(np.float32)))); log('p1', p)
    P = pl.concat(P)
    P = stage2_features(P, comp_table('p1')); log('stage-2 features done')
    if cfg['features'].get('sibling'):
        rec = VB.sibling_records_from(pl.read_parquet(A.rec_s23, columns=['nums', 'num_first', 'addr_n', 'name_n'])) if A.rec_s23 else VB.sibling_records(W, split)
        SB = VB.sibling_features(P.select('s1_idx', 'cand_idx', 'p1'), rec, threads=NT); del rec
        P = P.join(SB, on=['s1_idx', 'cand_idx'], how='left'); del SB; log('sibling features done')
    assert M2[0].feature_name() == F1 + F2X, 'stage-2 feature list mismatch'
    S2 = P.select(['s1_idx', 'cand_idx'] + F2X); out = []
    for p in parts:
        X = read(p, ['s1_idx', 'cand_idx'] + F1).join(S2, on=['s1_idx', 'cand_idx'], how='left')
        Xa = X.select(F1 + F2X).to_numpy().astype(np.float32)
        out.append(X.select('s1_idx', 'cand_idx').with_columns(pl.Series('p2', (sum(m.predict(Xa) for m in M2) / len(M2)).astype(np.float32)))); log('p2', p)
    P = P.join(pl.concat(out), on=['s1_idx', 'cand_idx'], how='left'); P.write_parquet(A.preds_out or W.preds(V, split)); log('preds written', P.height)
    if A.score_only: log('score-only: stop after the probabilities'); sys.exit(0)
# ---- decision ----
DJ = json.load(open(A.decision or f'{MD}/decision.json')); DEC, ep = DJ['final'], DJ['efs_params']; log('policy', DEC, ep)
OWN = DEC.get('competition', 'v1') == 'own'
comp2 = SD.own_competition(P) if OWN else comp_table('p2')   # v3: exclusivity against the variant's own p2 over ALL scored pairs of the split
Pc = P.select('s1_idx', 'cand_idx', 'p1', pl.col('p2').alias('p'), *[c for c in ('country', 'in_dense') if c in P.columns])
Pc = D.add_comp(Pc, comp2) if comp2 is not None else D.add_competition(Pc)
row_model = feats = None
if DEC['policy'] == 'd':
    RF = json.load(open(f'{MD}/row_empty_feats.json')); row_model = (lgb.Booster(model_file=f'{MD}/row_empty.txt'), RF)
    src = parts or sorted(glob.glob(f'{W.feats_dir(cfg["hybrid"]["lexical_only"])}/{split}_[0-9]*.parquet'))
    feats = pl.concat([pl.read_parquet(p, columns=['s1_idx', 'cand_idx'] + D.ROW_FE) for p in src])
i1 = pl.read_parquet(A.s1_ids or W.records(split, 's1'), columns=['entity_id', 'country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
i2 = pl.read_parquet(A.s23_ids or W.records(split, 's23'), columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
if DEC['policy'] == 'R10c':
    # argmax-vs-best-other exclusivity (qr = 1 iff p2 >= best other S1's competitor p2 + margin) + learned prefix set-decoder
    U = SD.universe(Pc.join(i2.select('cand_idx', pl.col('entity_id').str.slice(0, 2).alias('src')), on='cand_idx', how='left'), DEC['margin'])
    pred = SD.to_frame(SD.decide_set(U, lgb.Booster(model_file=f'{MD}/{DEC["decoder"]}'))); del U
else:
    pred = D.apply_decision(Pc, DEC, ep, row_model, feats)
n_before = pred.height; mo = {}
if DEC.get('one_owner'):   # v3: strict one-owner, each S2/S3 record keeps only the selecting S1 with the highest p2
    pred = pred.select('s1_idx', 'cand_idx').join(Pc.select('s1_idx', 'cand_idx', 'p'), on=['s1_idx', 'cand_idx'], how='left')
    mo['before'] = SD.multi_owner(pred); pred = SD.one_owner(pred); mo['after'] = SD.multi_owner(pred)
    log('one-owner: records with > 1 selecting S1 before / after', mo['before'], mo['after'])
log('decision done, matches', pred.height, '(before one-owner', n_before, ')')
# ---- write the two TSVs ----
assert P['s1_idx'].n_unique() == i1.height, f'{i1.height - P["s1_idx"].n_unique()} S1 without candidates'
def write(df, path, col):
    g = df.join(i2, on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').sort().str.join(',').alias('ids'))
    g = i1.join(g, on='s1_idx', how='left').with_columns(pl.col('ids').fill_null('')).sort('s1_idx')
    g.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('ids').alias(col)).write_csv(path, separator='\t', quote_style='never')
    return g
# ---- post-pass (v3): label-free neighbour-decoy vetoes, removal only (ber.postpass) ----
PPC = cfg.get('postpass') or {}; ppres = None; n_decision = pred.height
if PPC.get('enabled') and not A.no_postpass:
    from ber import postpass as PPm
    if A.pre_postpass_out:
        os.makedirs(A.pre_postpass_out, exist_ok=True); write(pred, f'{A.pre_postpass_out}/matching_results.tsv', 'matched_entity_ids')
        log('decision before the post-pass written to', A.pre_postpass_out)
    sel = pred.select('s1_idx', 'cand_idx').join(Pc.select('s1_idx', 'cand_idx', 'p'), on=['s1_idx', 'cand_idx'], how='left')
    pp = PPm.run(W, split, P.select('s1_idx', 'cand_idx'), sel, PPC, threads=NT, log=log)
    pred, ppres = pp['selected'], pp['report']; del pp, sel
    json.dump(ppres, open(f'{OUT}/postpass_report.json', 'w'), indent=1, default=str)
    log('post-pass done: matches', n_decision, '->', pred.height)
gm = write(pred, f'{OUT}/matching_results.tsv', 'matched_entity_ids')
gc = write(P, f'{OUT}/candidate_pairs.tsv', 'candidate_entity_ids'); log('wrote', OUT)
st = gm.with_columns((pl.col('ids') == '').alias('empty'), pl.when(pl.col('ids') == '').then(0).otherwise(pl.col('ids').str.count_matches(',') + 1).alias('k')) \
       .group_by('country').agg(pl.len().alias('n_s1'), pl.col('empty').mean().alias('empty_share'), pl.col('k').mean().alias('matches_per_s1')).sort('country')
tot = dict(n_s1=gm.height, candidate_pairs=P.height, cands_per_s1=P.height / gm.height, matches=pred.height, empty_share=float((gm['ids'] == '').mean()), matches_per_s1=pred.height / gm.height)
band = P.group_by('s1_idx').agg(((pl.col('p2') > 0.2) & (pl.col('p2') < 0.8)).any().alias('band'))
tot['share_s1_with_uncertain_pair'] = float(band['band'].mean())
res = dict(variant=V, split=split, per_country=st.to_dicts(), total=tot, policy=DEC, efs_params=ep, matches_before_one_owner=n_before, multi_owner=mo,
           matches_before_postpass=n_decision, postpass=ppres)
log(json.dumps(res, indent=1, default=str)); json.dump(res, open(f'{OUT}/stats.json', 'w'), indent=1, default=str)
log('DONE peak RSS GB', round(env.peak_rss_gb(), 1))
