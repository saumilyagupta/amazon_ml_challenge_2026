#!/opt/conda/bin/python3
"""Step 4: fit the matcher of one variant and tune its decision layer.

usage: /opt/conda/bin/python3 train.py --work WORK --variant v1|v2a|H|v2b|v3|<cfg.yaml> [--threads 8] [--stage fit decide]

fit    : 2-fold cross-fit (folds = hash of the S1 id) LightGBM stage 1 on the training-sample S1 (early stopping on the 15% ES entities
         of the fold), stage-1 probability for EVERY row of the train table (out-of-fold on the sample, fold average elsewhere), 13
         stage-2 list / record-competition features, LightGBM stage 2, stage-2 probability -> preds/{variant}_train.parquet, models/{variant}/
         (hybrid variants only combine the probabilities of their component variants by provenance).
decide : on the validation S1 (never trained on): (a) threshold sweep, (b) one-to-one, (c) source-side margin, (d) row-level empty model
         (fit on the training sample's out-of-fold aggregates), (e) expected-F0.5 prefix choice; keeps the best, checks it on the locked
         holdout, writes models/{variant}/decision.json + report.json + val_matching_results.tsv.
         v2b (decision.set_decoder): + R10c learned prefix set-decoder (ber.setdecoder) trained on the training sample's OUT-OF-FOLD p2 with
         argmax-vs-best-other exclusivity (margins decision.set_margins); stage 2 also gets the 6 sibling features; lexical-only negatives
         are subsampled (model.lexneg, weight 1/lexneg) at both stages.
         v3 (decision.competition: own, decision.one_owner): the exclusivity competitor of every policy is this variant's OWN p2 wherever it
         scores (validation + training-sample S1), v1's p2 over the dense table elsewhere (research prod_v3/07_decide.py 'own'); the R10c
         decoder (R10c_m{margin}_own.txt) is refit with that competition; one-owner is applied over the validation + sample decisions and
         the validation part is scored; decision.final_policy fixes the final policy (v3: 'R10c:m0.0').
Runtime, full data (400k-S1 sample, 8 threads per fold): v1 stage 1 about 11 min per fold, stage 2 about 2.5 min; predicting the
48M-pair train table about 15 min; decision layer about 7 min. Peak RSS about 21 GB."""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--variant', required=True)
ap.add_argument('--threads', type=int, default=8); ap.add_argument('--stage', nargs='+', default=['fit', 'decide'])
A = ap.parse_args()
from ber import env
NT = env.setup(A.threads)
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from ber.paths import Work
from ber import config, splits, labels
from ber.featsets import CATEGORICAL, stage1, stage2_extra
from ber.model import S2FEATS, train_lgb, stage2_features
from ber import decide as D, v2bfeats as VB, setdecoder as SD
log = env.logger(); W = Work(A.work); cfg = config.load(A.variant); V = cfg['name']; MD = W.models_dir(V)
mc, dc = cfg['model'], cfg['decision']; split = 'train'
parts = sorted(glob.glob(f'{W.feats_dir(V)}/{split}_[0-9]*.parquet'))
cols0 = pl.scan_parquet(parts[0]).collect_schema().names() if parts else []
F1 = stage1(cfg, cols0); F2X = stage2_extra(cfg)
META = [c for c in ['s1_idx', 'cand_idx', 'label', 'grp', 'fold', 'es', 'country', 'in_dense', 'in_dft', 'ft_seen_s1'] if c in cols0]
LEXNEG = float(mc.get('lexneg', 1.0))
ld = lambda n: lgb.Booster(model_file=f'{MD}/{n}.txt')
comp_variant = mc['competitors']


def comp_table(col):
    if comp_variant is None: return None
    return pl.read_parquet(W.preds(comp_variant, split), columns=['s1_idx', 'cand_idx', col]).rename({col: 'q'})


def oof(X, pa, pb):
    samp = (X['grp'] == 'sample').to_numpy(); fo = X['fold'].to_numpy()
    return np.where(samp, np.where(fo == 0, pb, pa), 0.5 * (pa + pb)).astype(np.float32)


def fit(stage, fold, feats, extra=None):
    filt = (pl.col('grp') == 'sample') & (pl.col('fold') == fold)
    if 'ft_seen_s1' in cols0 and mc.get('drop_ft_seen_s1'): filt = filt & ~pl.col('ft_seen_s1')
    base = [f for f in feats if f not in F2X]
    lx = ['in_dft'] if (LEXNEG < 1.0 and 'in_dft' in cols0) else []
    S = pl.concat([pl.scan_parquet(p).filter(filt).select(['s1_idx', 'cand_idx', 'label', 'es'] + lx + base).collect() for p in parts])
    if extra is not None:
        S = S.join(extra, on=['s1_idx', 'cand_idx'], how='left'); assert S['p1'].null_count() == 0
    if lx:   # lexical-only negatives: keep a deterministic hash share LEXNEG, weight 1/LEXNEG (v2b: 25%, weight 4)
        h = (pl.col('s1_idx').cast(pl.UInt64) * 1000003 + pl.col('cand_idx').cast(pl.UInt64)).hash(seed=7) % 1000000
        S = S.filter(pl.col('in_dft') | (pl.col('label') == 1) | (h < int(LEXNEG * 1e6)))
        S = S.with_columns(pl.when(~pl.col('in_dft') & (pl.col('label') == 0)).then(1.0 / LEXNEG).otherwise(1.0).cast(pl.Float32).alias('_w'))
    else:
        S = S.with_columns(pl.lit(1.0, pl.Float32).alias('_w'))
    tr = S.filter(~pl.col('es')); va = S.filter(pl.col('es')); del S
    log(f'{V} {stage} fold {fold}: {len(feats)} feats; train S1 {tr["s1_idx"].n_unique()} rows {tr.height} pos {int(tr["label"].sum())}; ES S1 {va["s1_idx"].n_unique()} rows {va.height}')
    xy = lambda d: (d.select(feats).to_numpy().astype(np.float32), d['label'].to_numpy(), d['_w'].to_numpy())
    Xtr, ytr, wtr = xy(tr); Xva, yva, wva = xy(va); del tr, va
    t = time.time()
    m = train_lgb(Xtr, ytr, Xva, yva, feats, dict(mc['lgb'], num_threads=NT), rounds=mc['rounds'], es=mc['early_stopping'], cat=CATEGORICAL, log=log,
                  wtr=wtr if lx else None, wva=wva if lx else None)
    m.save_model(f'{MD}/{stage}_fold{fold}.txt')
    json.dump(dict(best_iter=m.best_iteration, n_feats=len(feats), feats=feats, train_rows=len(ytr), es_rows=len(yva), sec=time.time() - t,
                   gain=sorted([(n, float(v)) for n, v in zip(m.feature_name(), m.feature_importance('gain'))], key=lambda x: -x[1])),
              open(f'{MD}/{stage}_fold{fold}.json', 'w'), indent=1)


def predict_table(stage, feats, extra=None):
    M = [ld(f'{stage}_fold{f}') for f in range(mc['folds'])]; out = []
    for p in parts:
        X = pl.read_parquet(p, columns=META + [f for f in feats if f not in F2X])
        if extra is not None: X = X.join(extra, on=['s1_idx', 'cand_idx'], how='left')
        Xa = X.select(feats).to_numpy().astype(np.float32)
        pa, pb = M[0].predict(Xa), M[1].predict(Xa)
        out.append(X.select(META).with_columns(pl.Series('p' + stage[1], oof(X, pa, pb)))); log(stage, 'predicted', p)
    return pl.concat(out)


if 'fit' in A.stage:
    if cfg['hybrid']:
        hd, hl = cfg['hybrid']['dense'], cfg['hybrid']['lexical_only']
        P = pl.read_parquet(W.preds(hl, split))
        Q = pl.read_parquet(W.preds(hd, split), columns=['s1_idx', 'cand_idx', 'p1', 'p2']).rename({'p1': '_q1', 'p2': '_q2'})
        P = P.join(Q, on=['s1_idx', 'cand_idx'], how='left')
        assert P.filter(pl.col('in_dense'))['_q2'].null_count() == 0, 'dense pairs without dense-variant probabilities'
        P = P.with_columns(pl.when(pl.col('in_dense')).then(pl.col('_q1')).otherwise(pl.col('p1')).alias('p1'),
                           pl.when(pl.col('in_dense')).then(pl.col('_q2')).otherwise(pl.col('p2')).alias('p2')).drop('_q1', '_q2')
        P.write_parquet(W.preds(V, split)); log('hybrid preds written', P.height)
    else:
        assert parts, 'run features.py first'
        for f in range(mc['folds']): fit('s1', f, F1)
        P = predict_table('s1', F1)
        P = stage2_features(P, comp_table('p1')); log('stage-2 features done')
        if cfg['features'].get('sibling'):
            t = time.time(); SB = VB.sibling_features(P.select('s1_idx', 'cand_idx', 'p1'), VB.sibling_records(W, split), threads=NT)
            P = P.join(SB, on=['s1_idx', 'cand_idx'], how='left'); del SB; log('sibling features', f'{time.time() - t:.0f}s')
        S2 = P.select(['s1_idx', 'cand_idx'] + F2X)
        for f in range(mc['folds']): fit('s2', f, F1 + F2X, extra=S2)
        P2 = predict_table('s2', F1 + F2X, extra=S2)
        P = P.join(P2.select('s1_idx', 'cand_idx', 'p2'), on=['s1_idx', 'cand_idx'], how='left')
        P.write_parquet(W.preds(V, split)); log('preds written', P.height)
        Vv = P.filter(pl.col('grp').is_in(['val', 'locked'])); y = Vv['label'].to_numpy(); res = {}
        for c in ['p1', 'p2']:
            res[c] = dict(auc=float(roc_auc_score(y, Vv[c].to_numpy())), ap=float(average_precision_score(y, Vv[c].to_numpy()))); log('val pair metrics', c, res[c])
        json.dump(res, open(f'{MD}/pair_metrics.json', 'w'), indent=1)

if 'decide' in A.stage:
    P = pl.read_parquet(W.preds(V, split), columns=['s1_idx', 'cand_idx', 'label', 'grp', 'country', 'in_dense', 'p1', 'p2'])
    G = labels.gt_idx(W, split); g = splits.assign(W, cfg, split)
    MV = labels.truth_counts(W, P, G, g, ['val', 'locked']).with_columns((pl.col('grp') == 'locked').alias('is_locked'))
    MS = labels.truth_counts(W, P, G, g, ['sample']).join(P.select('s1_idx').unique(), on='s1_idx', how='semi')   # S1 dropped at feature time excluded
    assert MV.height > 0, 'no validation S1 in this work dir'
    log('val S1', MV.height, 'singletons', int((MV['m'] == 0).sum()), 'locked', int(MV['is_locked'].sum()), '| sample S1', MS.height)
    RC = ['s1_idx', 'm', 'r', 'country']; RES = {'variant': V, 'sweeps': {}}; vids = MV.select('s1_idx'); best = {}
    comp2 = comp_table('p2') if comp_variant else None
    OWN = dc.get('competition', 'v1') == 'own'
    if OWN:   # v3: own-probability competition (the variant's p2 for every pair it scores; the competitor's p2 elsewhere)
        comp2 = SD.own_competition(P, comp2); log('own competition table', comp2.height, 'rows')
    SFX = '_own' if OWN else ''
    def with_comp(Pq):
        return D.add_comp(Pq, comp2) if comp2 is not None else D.add_competition(Pq)
    for pcol in ['p1', 'p2']:
        Pc = with_comp(P.select('s1_idx', 'cand_idx', 'label', pl.col(pcol).alias('p'))).join(vids, on='s1_idx', how='semi')
        sw = []
        for tau in dc['taus']:
            sw.append(('plain', 0.0, tau, D.macro(D.policy(Pc, tau, 'plain'), MV))); sw.append(('o2o', 0.0, tau, D.macro(D.policy(Pc, tau, 'o2o'), MV)))
            if pcol == 'p2':
                for d in dc['deltas']: sw.append(('margin', d, tau, D.macro(D.policy(Pc, tau, 'margin', d), MV)))
        RES['sweeps'][pcol] = sw
        for mode in ['plain', 'o2o', 'margin']:
            x = [s for s in sw if s[0] == mode]
            if x: best[(pcol, mode)] = max(x, key=lambda s: s[3]); log(f'{pcol} best {mode}: delta {best[(pcol, mode)][1]} tau {best[(pcol, mode)][2]} macro {best[(pcol, mode)][3]:.5f}')
    RES['best'] = {f'{k[0]}|{k[1]}': v for k, v in best.items()}
    if dc.get('force'):
        f = dc['force']; mode, delta, tau, sc = f['mode'], f['delta'], f['tau'], None
    else:
        mode, delta, tau, sc = max([best[('p2', m)] for m in ['plain', 'o2o', 'margin']], key=lambda s: s[3])
    log('best (a-c) on p2:', mode, delta, tau, sc)
    PA = with_comp(P.select('s1_idx', 'cand_idx', 'label', 'grp', pl.col('p2').alias('p'), 'p1')).with_columns(D.elig_expr(mode, delta).alias('elig'))
    Pv = PA.join(vids, on='s1_idx', how='semi')
    base_pred = Pv.filter(pl.col('elig') & (pl.col('p') >= tau)); sc = D.macro(base_pred, MV)
    cands = {'a-c': (sc, lambda: base_pred)}
    # (d) row-level empty / non-empty model on per-S1 aggregates of the training sample (OOF probabilities)
    if dc['row_model'] and MS.height:
        feat = pl.concat([pl.scan_parquet(p).select(['s1_idx', 'cand_idx'] + D.ROW_FE).collect() for p in parts]) if parts else \
               pl.concat([pl.scan_parquet(p).select(['s1_idx', 'cand_idx'] + D.ROW_FE).collect() for p in sorted(glob.glob(f'{W.feats_dir(cfg["hybrid"]["lexical_only"])}/{split}_[0-9]*.parquet'))])
        RS = D.row_agg(PA.join(MS.select('s1_idx'), on='s1_idx', how='semi').join(feat, on=['s1_idx', 'cand_idx'], how='left')).join(MS.select('s1_idx', 'm'), on='s1_idx')
        RV = D.row_agg(Pv.join(feat, on=['s1_idx', 'cand_idx'], how='left')).join(MV.select('s1_idx', 'm'), on='s1_idx'); del feat
        RF = [c for c in RS.columns if c not in ('s1_idx', 'm')]
        rm = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9, num_threads=NT, verbose=-1, seed=3),
                       lgb.Dataset(RS.select(RF).to_numpy().astype(np.float32), (RS['m'] == 0).to_numpy().astype(int)), 300)
        rm.save_model(f'{MD}/row_empty.txt'); json.dump(RF, open(f'{MD}/row_empty_feats.json', 'w'))
        RV = RV.with_columns(pl.Series('p_empty', rm.predict(RV.select(RF).to_numpy().astype(np.float32))))
        RES['row_model_auc_val'] = float(roc_auc_score((RV['m'] == 0).to_numpy(), RV['p_empty'].to_numpy())) if 0 < int((RV['m'] == 0).sum()) < RV.height else None
        Pv = Pv.join(RV.select('s1_idx', 'p_empty'), on='s1_idx', how='left').with_columns(pl.col('p_empty').fill_null(1.0))
        base_pred = Pv.filter(pl.col('elig') & (pl.col('p') >= tau))
        dsw = [(te, tn, tl, D.macro(D.apply_row(Pv, base_pred, te, tn, tl), MV)) for te in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01] for tn, tl in [(0, 0), (0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.2, 0.2), (0.3, 0.2)]]
        bd = max(dsw, key=lambda s: s[3]); RES['row_best'] = bd; log('best (d):', bd, 'vs (a-c)', sc)
        cands['d'] = (bd[3], lambda: D.apply_row(Pv, base_pred, *bd[:3]))
    # (e) expected-F0.5 prefix choice; missing-truth model estimated on the training sample
    lam = float(MS.filter(pl.col('r') > 0).select((pl.col('m') - pl.col('r')).mean())[0, 0]) if MS.height else 0.0
    q0 = float(MS.filter(pl.col('r') == 0).select((pl.col('m') > 0).mean())[0, 0]) if MS.filter(pl.col('r') == 0).height else 0.0
    ep = dict(lam=lam, q0=q0, L=dc['efs_L'], draws=dc['efs_draws'], seed=dc['efs_seed']); RES['efs_params'] = ep; log('(e) lambda', lam, 'q0', q0)
    if dc['efs']:
        pr = D.efs_choose(Pv, lam, q0, L=ep['L'], NS=ep['draws'], seed=ep['seed']); cands['e'] = (D.macro(pr, MV), lambda: pr); log('(e) macro', cands['e'][0])
    # R10c learned prefix set-decoder on the training sample's OUT-OF-FOLD p2, argmax-vs-best-other exclusivity (qr = 1 iff p >= p_other + margin)
    if dc.get('set_decoder') and MS.height:
        src = pl.read_parquet(W.records(split, 's23'), columns=['entity_id']).with_row_index('cand_idx').select(pl.col('cand_idx').cast(pl.Int32), pl.col('entity_id').str.slice(0, 2).alias('src'))
        Ps = PA.join(MS.select('s1_idx'), on='s1_idx', how='semi').join(src, on='cand_idx', how='left'); Pvs = Pv.join(src, on='cand_idx', how='left')
        GS = SD.truth_sets(MS, Ps); RES['set_decoder'] = {}
        for mg in dc.get('set_margins', [0.0]):
            t = time.time(); sd = SD.fit_set_decoder(SD.universe(Ps, mg), GS, threads=NT); sd.save_model(f'{MD}/R10c_m{mg}{SFX}.txt')
            pr_ = SD.to_frame(SD.decide_set(SD.universe(Pvs, mg), sd)).join(Pv.select('s1_idx', 'cand_idx', 'label', 'p'), on=['s1_idx', 'cand_idx'], how='left')
            if dc.get('one_owner'):   # v3: strict one-owner over the validation + training-sample decisions (sample decided on its OOF p2)
                ps_ = SD.to_frame(SD.decide_set(SD.universe(Ps, mg), sd)).join(Ps.select('s1_idx', 'cand_idx', 'label', 'p'), on=['s1_idx', 'cand_idx'], how='left')
                allsel = pl.concat([pr_, ps_]); kb = SD.one_owner(allsel)
                RES['set_decoder'][f'm{mg}_one_owner'] = dict(multi_before=SD.multi_owner(allsel), removed=allsel.height - kb.height,
                                                              removed_true=int(allsel['label'].sum() - kb['label'].sum()), val_before=D.macro(pr_, MV))
                pr_ = kb.join(Pv.select('s1_idx').unique(), on='s1_idx', how='semi'); log('one-owner (val + sample):', RES['set_decoder'][f'm{mg}_one_owner'])
            sc_ = D.macro(pr_, MV); cands[f'R10c:m{mg}'] = (sc_, (lambda pr_=pr_: pr_)); RES['set_decoder'][f'm{mg}'] = sc_
            log(f'R10c margin {mg}{SFX}: macro {sc_:.5f}', f'{time.time() - t:.0f}s')
        del Ps, Pvs
    fin = dc['force']['policy'] if dc.get('force') else (dc['final_policy'] if dc.get('final_policy') else max(cands, key=lambda k: cands[k][0]))
    log('policies on validation:', {k: round(v[0], 6) for k, v in cands.items()}, '-> final', fin); pred = cands[fin][1]()
    rep, R = D.report(pred, MV.select(RC))
    RES['final'] = dict(policy=fin, mode=mode, delta=delta, tau=tau, row=list(bd[:3]) if fin == 'd' else None)
    if fin.startswith('R10c'):
        mg = float(fin.split(':m')[1]); RES['final'].update(policy='R10c', margin=mg, decoder=f'R10c_m{mg}{SFX}.txt', competition=dc.get('competition', 'v1'),
                                                              one_owner=bool(dc.get('one_owner')))
    RES['report'] = rep; log('final policy', RES['final']); log(json.dumps(rep, indent=1))
    MVl = MV.filter(pl.col('is_locked')); MVn = MV.filter(~pl.col('is_locked'))
    RES['locked'] = dict(n_locked=MVl.height, final_policy_on_locked=D.macro(pred, MVl) if MVl.height else None, final_policy_on_tuning_part=D.macro(pred, MVn) if MVn.height else None)
    log('locked check', RES['locked'])
    json.dump({'final': RES['final'], 'efs_params': ep, 'row': RES.get('row_best')}, open(f'{MD}/decision.json', 'w'), indent=1)
    json.dump(RES, open(f'{MD}/report.json', 'w'), indent=1, default=str)
    # val predictions in the official format (scored by tools/score.py against the ground truth restricted to the val ids)
    i1 = pl.read_parquet(W.records(split, 's1'), columns=['entity_id']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    i2 = pl.read_parquet(W.records(split, 's23'), columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
    gm = pred.join(i2, on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').sort().str.join(',').alias('ids'))
    gm = MV.select('s1_idx').join(i1, on='s1_idx').join(gm, on='s1_idx', how='left').with_columns(pl.col('ids').fill_null('')).sort('s1_idx')
    gm.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('ids').alias('matched_entity_ids')).write_csv(f'{MD}/val_matching_results.tsv', separator='\t', quote_style='never')
    open(f'{MD}/val_s1_ids.txt', 'w').write('\n'.join(gm['entity_id'].to_list()) + '\n')
    R.write_parquet(f'{MD}/val_row_scores.parquet')
log('DONE peak RSS GB', round(env.peak_rss_gb(), 1))
