#!/opt/conda/bin/python3
"""Step 3: candidate table + matcher features for one variant and split.

usage: /opt/conda/bin/python3 features.py --work WORK --variant v1|v2a|v2b|v3|<cfg.yaml> --split train|test [--threads 8] [--limit N]

* dense table (cached per encoder): fwd@20 + margin-gated reverse for ALL S1 of the split, with the dense / competition table features
  (cos, ranks, S1-side and record-side gaps and claimant counts) -> cands/dense_table_{enc}_{split}.parquet
* candidate table of the variant: kind=dense -> the dense table itself; kind=union -> union/{union}_{split}.parquet rows of the query S1
  (+ in_dense flag, channel features u_*; record-side table features re-computed over the dense table of ALL S1 + the pair)
* labels (train: official ground truth), groups sample / val / locked / other, 2 hash folds, ES flag, country
* 89 pair features per candidate (ber.pairfeats), streamed to feats/{variant}/{split}_NNN.parquet in 3M-row parts (restartable)
* v2b (features.explainer / v2b_extra): + the 127 explainer features (ber.explainer, --threads fork workers per part) + the 14 (c) additions
  (ber.v2bfeats: multi-part house numbers, decoy, France-safe admin conflict, provenance prov / in_dft, hub flag); union v2 train S1 whose
  pairs trained the train-split encoder (ft_seen_s1) are dropped (candidates.drop_ft_seen_s1)
* v3 (features.v3_decoy / branch_q / pack_gate), on top of v2b: + 15 offset-conditioned decoy features (ber.v3decoy; uses the explainer
  flags of the pair) + 14 France-safe branch features (ber.branch, --threads fork workers per part) + for pairs whose S1 country is in
  pack_gate (France) the France-pack views of the 43 affected v1 features as pk_<name> columns (null on other rows; the record views
  WORK/records/pack_{split}_*.parquet are built once per split that contains a gated country, ber.packrecords). predict.py substitutes
  pk_<name> for <name> on the gated rows; training rows (US / India) never carry pack values.
Runtime, full data: v1 train (48.0M pairs) about 15 min at 16 threads / 21 GB; v2a test union (88.0M pairs) about 46 min at 8 threads / 41 GB;
v2b: + explainer at ~20k pairs/s with 16 workers (train 28.3M pairs ~25 min, test 83.8M ~75 min), peak RSS about 45 GB.
v3: + branch features at 14-20k pairs/s with 16 workers (train ~25 min, test ~70-100 min), decoy block (~5 / ~25 min), France-pack record
views of the test split (~20 min) + pack features of the ~12.8M France test pairs (~10 min); peak RSS about 60 GB (estimates from the
research runs, which materialised these blocks separately)."""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--variant', required=True); ap.add_argument('--split', required=True)
ap.add_argument('--threads', type=int, default=8); ap.add_argument('--limit', type=int, default=0)
A = ap.parse_args()
from ber import env
NT = env.setup(A.threads)
import numpy as np, polars as pl
from ber.paths import Work
from ber import config, splits, labels
from ber.cands import load_forward, load_reverse, dense_candidates, table_features
from ber.chan import channel_features, recside_table, RAW_COLS, FT_RAW_COLS, FT_PASSTHROUGH, CHAN_FEATS, CHAN_FEATS_FT
from ber.pairfeats import FeatureBuilder
log = env.logger(); W = Work(A.work); cfg = config.load(A.variant); V = cfg['name']; split = A.split
fc, cc = cfg['features'], cfg['candidates']; enc = fc['dense_table_encoder']
country = pl.read_parquet(W.records(split, 's1'), columns=['country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
# ---- dense table of ALL S1 (shared across variants) ----
dtp = W.dense_table(enc, split)
if not os.path.exists(dtp):
    F = load_forward(W, enc, split); R = load_reverse(W, enc, split)
    U = dense_candidates(F, R, fc['dense_table_fwd_k'], fc['dense_table_gate'])
    T = table_features(U, F, R, W, enc, split, gate=fc['dense_table_gate']).join(country, on='s1_idx', how='left')
    assert T['s1_idx'].n_unique() == country.height, 'some S1 without dense candidates'
    T.write_parquet(dtp); log('dense table', T.height, 'pairs for', country.height, 'S1'); del F, R, U, T
DT = pl.read_parquet(dtp); log('dense table rows', DT.height)
# ---- candidate table ----
if cc['kind'] == 'dense':
    assert cc['encoder'] == enc and cc['fwd_k'] == fc['dense_table_fwd_k'] and cc['rev_gate'] == fc['dense_table_gate'], 'dense variant must use the dense-table parameters'
    T = DT.with_columns(pl.lit(True).alias('in_dense')); CH = []
else:
    sch = pl.scan_parquet(W.union(cc['union'], split)).collect_schema().names()
    ft = all(c in sch for c in FT_RAW_COLS)
    cols = ['s1_idx', 'cand_idx'] + RAW_COLS + (FT_RAW_COLS + [c for c in FT_PASSTHROUGH if c in sch] if ft else [])
    U = pl.read_parquet(W.union(cc['union'], split), columns=cols)
    if cc.get('cap'): U = U.filter(pl.col('fused_rank') <= cc['cap'])
    C = channel_features(U); CH = CHAN_FEATS + (CHAN_FEATS_FT if ft else []) + ([c for c in FT_PASSTHROUGH if c in C.columns])
    U = U.select('s1_idx', 'cand_idx', *([(pl.col('in_fwd_ft') | pl.col('in_rev_ft')).alias('in_dft')] if ft else [])).join(DT.select('s1_idx', 'cand_idx', pl.lit(True).alias('in_dense')), on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('in_dense').fill_null(False))
    log('union rows', U.height, 'S1', U['s1_idx'].n_unique(), 'in dense table', int(U['in_dense'].sum()), 'ft cols', ft)
    F = load_forward(W, enc, split); R = load_reverse(W, enc, split)
    Ua = DT.select('s1_idx', 'cand_idx', 'ch_fwd', 'ch_rev').join(U.select('s1_idx', 'cand_idx'), on=['s1_idx', 'cand_idx'], how='full', coalesce=True) \
           .with_columns(pl.col('ch_fwd').fill_null(False), pl.col('ch_rev').fill_null(False))
    T = table_features(Ua, F, R, W, enc, split, gate=fc['dense_table_gate']).join(U, on=['s1_idx', 'cand_idx'], how='semi'); del Ua, F, R
    T = recside_table(T, DT.select('s1_idx', 'cand_idx', 'cos', 'rank_rec_tab', 'n_s1_rec_tab', 'rel_rec_tab'))
    T = T.join(U, on=['s1_idx', 'cand_idx'], how='left').join(C, on=['s1_idx', 'cand_idx'], how='left').join(country, on='s1_idx', how='left')
    log('table features for union pairs', T.height)
# ---- meta: labels, groups, folds ----
META = ['country', 'in_dense'] + (['in_dft'] if 'in_dft' in T.columns else [])
if split == 'train':
    G = labels.gt_idx(W, split)
    T = T.join(G, on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('label').fill_null(0).cast(pl.Int8))
    g = splits.assign(W, cfg, split)
    T = T.join(g.select('s1_idx', 'grp', 'fold', 'es'), on='s1_idx', how='left')
    META = ['label', 'grp', 'fold', 'es'] + META
    if cc.get('drop_ft_seen_s1') and 'ft_seen_s1' in T.columns:
        # training-sample S1 whose pairs fine-tuned the train-split encoder carry in-sample ft scores -> removed from the train file
        # (validation S1 are never in its training pairs with the shipped split files; on hash splits they are kept and only flagged)
        n0 = T.height; T = T.filter(~(pl.col('ft_seen_s1') & (pl.col('grp') == 'sample')))
        log('dropped training-sample S1 with fine-tune training pairs (ft_seen_s1):', n0 - T.height, 'pairs')
    for gname in ['sample', 'val', 'locked', 'other']:
        x = T.filter(pl.col('grp') == gname)
        if x.height: log(f'  {gname}: S1 {x["s1_idx"].n_unique()} pairs {x.height} pos {int(x["label"].sum())} pos-rate {x["label"].mean():.4f} cands/S1 {x.height / x["s1_idx"].n_unique():.2f}')
    pr = {'sample': 0, 'val': 1, 'locked': 2, 'other': 3}
    T = T.with_columns(pl.col('grp').replace_strict(pr, return_dtype=pl.Int8).alias('_p')).sort('_p', 's1_idx', 'cand_idx').drop('_p')
else:
    T = T.sort('s1_idx', 'cand_idx')
META += [c for c in FT_PASSTHROUGH if c in T.columns]
T.select('s1_idx', 'cand_idx', *META).write_parquet(W.cands(V, split)); log('wrote candidate table', W.cands(V, split), T.height)
# ---- pair features in parts ----
if A.limit: T = T.head(A.limit)
fb = FeatureBuilder(W, split, threads=max(NT, 8), log=log); CHp = [c for c in CH if c in T.columns and c not in META]
EXPL, V2X = bool(fc.get('explainer')), bool(fc.get('v2b_extra'))
if EXPL:
    from ber import explainer as EX
    REC = EX.load_records(W, split); EX.all_tables(); log('explainer records', REC.height)
    ID1 = pl.read_parquet(W.records(split, 's1'), columns=['entity_id'])['entity_id'].to_numpy()
    ID2 = pl.read_parquet(W.records(split, 's23'), columns=['entity_id'])['entity_id'].to_numpy()
if V2X:
    from ber import v2bfeats as VB
    HN = VB.build_records(W, split); log('multi-part house numbers', HN[0].shape, HN[2].shape)
V3D, BQ, PG = bool(fc.get('v3_decoy')), bool(fc.get('branch_q')), list(fc.get('pack_gate') or [])
if V3D:
    assert EXPL, 'v3_decoy needs the explainer features (x1_name_explained, op_filler_add, op_legal_add, op_legal_restyle)'
    from ber import v3decoy as VD
    DREC = VD.build_records(W, split); log('decoy record views', DREC['P1'].shape, DREC['P2'].shape)
if BQ:
    from ber import branch as BR
    if not EXPL:
        from ber import explainer as EX
        REC = EX.load_records(W, split)
        ID1 = pl.read_parquet(W.records(split, 's1'), columns=['entity_id'])['entity_id'].to_numpy()
        ID2 = pl.read_parquet(W.records(split, 's23'), columns=['entity_id'])['entity_id'].to_numpy()
    BCTX = BR.load_ctx(W, log, need=country['country'].unique().to_list()); log('branch ctx: filler', len(BCTX['name_filler']), len(BCTX['addr_filler']), 'idf tables', len(BCTX['idf']))
PFB = None
if PG and country.filter(pl.col('country').is_in(PG)).height:
    from ber import packrecords as PR, packfeats as PF
    PFB = PF.PackFeatures(split, prefix=PR.ensure(W, split, log=log), idf='global', log=log); log('France-pack features for countries', PG)
CHUNK = fc['part_rows']; nparts = (T.height + CHUNK - 1) // CHUNK; FD = W.feats_dir(V); tm = {}; t1 = time.time()
for k in range(nparts):
    out = f'{FD}/{split}_{k:03d}.parquet'
    if os.path.exists(out) and not A.limit: continue
    t = time.time(); c = T.slice(k * CHUNK, CHUNK)
    X = fb.pair_features(c, tm)
    X = pl.concat([X, c.select(CHp + [m for m in META if m not in X.columns])], how='horizontal')
    if EXPL:
        t2 = time.time(); ia = X['s1_idx'].to_numpy(); ib = X['cand_idx'].to_numpy()
        E = EX.run_parallel(pl.DataFrame({'s1_id': ID1[ia], 'cand_id': ID2[ib]}), REC, n_workers=min(16, NT), chunk_size=20000)
        assert E.height == X.height and (E['cand_id'].to_numpy() == ID2[ib]).all()
        X = pl.concat([X, E.drop('s1_id', 'cand_id')], how='horizontal'); del E; tm['explainer'] = tm.get('explainer', 0) + time.time() - t2
    if V2X:
        t3 = time.time(); X = VB.c_features(X, HN, fb); tm['v2b_extra'] = tm.get('v2b_extra', 0) + time.time() - t3
    if V3D:
        t3 = time.time(); X = VD.add_features(X, DREC); tm['v3_decoy'] = tm.get('v3_decoy', 0) + time.time() - t3
    if BQ:
        t3 = time.time(); ia = X['s1_idx'].to_numpy(); ib = X['cand_idx'].to_numpy()
        Q = BR.run_parallel(pl.DataFrame({'s1_id': ID1[ia], 'cand_id': ID2[ib]}), REC, BCTX, n_workers=min(16, NT), chunk=20000)
        assert Q.height == X.height
        X = pl.concat([X, Q.with_columns(pl.all().cast(pl.Float32))], how='horizontal'); del Q; tm['branch_q'] = tm.get('branch_q', 0) + time.time() - t3
    if PFB is not None:
        t3 = time.time(); g = np.nonzero(X['country'].is_in(PG).to_numpy())[0]
        if len(g):
            Fp = PFB.pair_features(X['s1_idx'].to_numpy()[g], X['cand_idx'].to_numpy()[g], new=False)
            G = pl.DataFrame({'_r': g.astype(np.int64), **{'pk_' + f: Fp[f] for f in PF.AFFECTED}})
            X = X.with_row_index('_r').with_columns(pl.col('_r').cast(pl.Int64)).join(G, on='_r', how='left', maintain_order='left').drop('_r')
        else:
            X = X.with_columns([pl.lit(None, pl.Float32).alias('pk_' + f) for f in PF.AFFECTED])
        tm['pack'] = tm.get('pack', 0) + time.time() - t3
    if A.limit: log('limit run:', X.height, 'rows', X.width, 'cols', f'{X.height / (time.time() - t):.0f} pairs/s'); print(X.head(3)); break
    X.write_parquet(out + '.tmp'); os.rename(out + '.tmp', out)
    log(f'part {k + 1}/{nparts} rows {X.height} {X.height / (time.time() - t):.0f} pairs/s cols {X.width}', {a: round(b) for a, b in tm.items()})
json.dump({'variant': V, 'split': split, 'rows': T.height, 'parts': nparts, 'sec': time.time() - t1, 'peak_rss_gb': env.peak_rss_gb(), 'channel_feats': CHp},
          open(f'{FD}/{split}_meta.json', 'w'), indent=1)
log('DONE peak RSS GB', round(env.peak_rss_gb(), 1))
