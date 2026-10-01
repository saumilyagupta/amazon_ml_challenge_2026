# E06 step 4 (CPU; LightGBM fits run through prod_v2c/common/fitlock.sh at 4 threads; band-level tables only, <= 0.9M rows).
# Stage 3 on v2b's band 0.02 < p2 < 0.99 ('b'; plus the CE training band 0.005 < p2 < 0.995 = 'w' for the main variant) with fine-tuned CE logits.
#  CE logits: sample rows = OOF (fold-0 rows scored by CE_f1, fold-1 rows by CE_f0); val / density rows = mean of the two fold CEs.
#    ce_logit = CE trained on ALL band pairs (tag xw); ce_ne = CE trained on NON-EMPTY-address band pairs only (tag xne).
#  Residual LightGBM: init_score = logit(p2), features [logit(p2), <ce>, 13 S2FEATS (incl. p1), 6 SIB_FEATS, 8 stage-1 slice flags incl. addr_empty2, in_dft],
#    trained on SAMPLE band rows only, 2-fold by v2b's S1 fold (model_f fit on fold f non-es rows, early stopping on fold f es rows, predicts fold 1-f = OOF;
#    val / density = mean raw score of the two fold models); p = expit(logit(p2) + raw) on the rows it covers, p2 elsewhere.
#  Variants (column names):
#    p3b   all band pairs, ce_logit                         (main)
#    p3b_ne  = p3b on non-empty-address pairs, p2 on empty-address pairs (coordinator request: CE applied to non-empty pairs only)
#    q3b   trained + applied on NON-EMPTY band pairs only, ce_ne (CE specialist retrained on non-empty pairs), p2 on empty-address pairs
#    r3b   trained + applied on NON-EMPTY band pairs only, ce_logit (isolates the CE specialisation from the stage-3 specialisation)
#    p3nb  all band pairs, NO CE feature (control: what a residual stage 3 gains without the CE)
#    p3w   as p3b on the wide band 0.005 < p2 < 0.995
#    pbb   Platt blend w * p2 + (1 - w) * sigmoid(a * ce_logit + b) (a, b, w fitted on the sample band OOF, as prod_v2b/09_ce_stage3.py)
# Output: data/stage3_<tag>.parquet (+ _dens), models/s3_<tag>_*.txt, logs/04_stage3_<tag>.json
# usage: python3 04_stage3.py [--tag xw] [--netag xne] [--leaves 15]
import os, sys, json, time, argparse
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2b')
from pv2b.common import envcap; envcap(4)
import numpy as np, polars as pl, lightgbm as lgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from pv2b.newfeats import SIB_FEATS
from pv1.model import S2FEATS
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='xw'); ap.add_argument('--netag', default='xne'); ap.add_argument('--leaves', type=int, default=15)
ap.add_argument('--lr', type=float, default=0.03); A = ap.parse_args()
E = '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E06_ce_band'; SC = f'{E}/scores'
V2B = '/workspace/saumilya/amazon-ml/work/matching/prod_v2b'
T0 = time.time(); log = lambda *a: print(f'[+{time.time()-T0:.0f}s]', *a, flush=True)
FLAGS = ['addr_empty2', 'name_freq_s1_1', 'n11_junk_name', 'f_url2', 'n10_alias', 'script2', 'x1_name_explained', 'x1_both_explained']
BASEF = list(S2FEATS) + SIB_FEATS + FLAGS + ['in_dft']
K = ['s1_idx', 'cand_idx']
def ce_tables(tag, col):
    rd = lambda nm: pl.read_parquet(f'{SC}/ce_{tag}_{nm}.parquet')
    oof = pl.concat([rd('f0_oof'), rd('f1_oof')]).rename({'ce_logit': col})
    avg = lambda nm: rd(f'f0_{nm}').join(rd(f'f1_{nm}'), on=K, suffix='_1').select(*K, ((pl.col('ce_logit') + pl.col('ce_logit_1')) / 2).alias(col))
    return oof, avg('val'), avg('dens')
o1, v1, d1 = ce_tables(A.tag, 'ce_logit'); o2, v2, d2 = ce_tables(A.netag, 'ce_ne')
old = pl.read_parquet(f'{V2B}/data/ce_train_abc_rob_cv2_all.parquet').rename({'ce_logit': 'ce_old'})
B = pl.read_parquet(f'{E}/data/band_train.parquet')
Bs = B.filter(pl.col('grp') == 'sample').join(o1, on=K, how='left').join(o2, on=K, how='left')
Bv = B.filter(pl.col('grp') != 'sample').join(v1, on=K, how='left').join(v2, on=K, how='left')
B = pl.concat([Bs, Bv], how='vertical_relaxed').join(old, on=K, how='left')
Dn = pl.read_parquet(f'{E}/data/band_dens.parquet').join(d1, on=K, how='left').join(d2, on=K, how='left')
for c in ['ce_logit', 'ce_ne']: assert B[c].null_count() == 0 and Dn[c].null_count() == 0, c
def prep(df): return df.with_columns(pl.Series('lp2', logit(np.clip(df['p2'].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6))), pl.col('in_dft').cast(pl.Float32),
                                     (pl.col('addr_empty2') <= 0).alias('ne'))
B = prep(B); Dn = prep(Dn)
RES = dict(tag=A.tag, netag=A.netag, leaves=A.leaves, lr=A.lr, base_features=BASEF, n_band_wide=dict(sample=Bs.height, val=Bv.height, dens=Dn.height))
PAR = dict(objective='binary', learning_rate=A.lr, num_leaves=A.leaves, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, num_threads=4, verbose=-1, seed=7, max_bin=255)
def X(df, feats): return df.select(feats).to_numpy().astype(np.float32)
def m2(y, p): return dict(auc=round(float(roc_auc_score(y, p)), 5), ap=round(float(average_precision_score(y, p)), 5))
cols_tr, cols_dn = {}, {}   # column -> DataFrame (K + column) for the rows it covers
def residual(c, rows_filter, feats):
    Bb = B.filter(rows_filter); Db = Dn.filter(rows_filter); S = Bb.filter(pl.col('grp') == 'sample'); Vv = Bb.filter(pl.col('grp') != 'sample')
    ys = S['label'].to_numpy(); fs = S['fold'].to_numpy(); es = S['es'].to_numpy()
    Xs, Xv, Xd = X(S, feats), X(Vv, feats), X(Db, feats); rs = np.zeros(S.height); rv = np.zeros(Vv.height); rdn = np.zeros(Db.height); its = []
    for f in (0, 1):
        tr = (fs == f) & ~es; ev = (fs == f) & es
        dtr = lgb.Dataset(Xs[tr], ys[tr], init_score=S['lp2'].to_numpy()[tr], feature_name=feats, free_raw_data=False)
        dev = lgb.Dataset(Xs[ev], ys[ev], init_score=S['lp2'].to_numpy()[ev], reference=dtr)
        m = lgb.train(PAR, dtr, 4000, valid_sets=[dev], callbacks=[lgb.early_stopping(100, verbose=False)])
        m.save_model(f'{E}/models/s3_{A.tag}_{c}_fold{f}.txt'); its.append(m.best_iteration)
        o = fs == 1 - f; rs[o] = m.predict(Xs[o], raw_score=True, num_iteration=m.best_iteration)
        rv += 0.5 * m.predict(Xv, raw_score=True, num_iteration=m.best_iteration); rdn += 0.5 * m.predict(Xd, raw_score=True, num_iteration=m.best_iteration)
        RES[f'gain_{c}_f{f}'] = dict(sorted(zip(feats, m.feature_importance('gain').round(1).tolist()), key=lambda t: -t[1])[:12])
    RES[f'best_iter_{c}'] = its; RES[f'rows_{c}'] = dict(sample=S.height, val=Vv.height, dens=Db.height)
    cols_tr[c] = pl.concat([S.select(*K).with_columns(pl.Series(c, expit(S['lp2'].to_numpy() + rs))), Vv.select(*K).with_columns(pl.Series(c, expit(Vv['lp2'].to_numpy() + rv)))])
    cols_dn[c] = Db.select(*K).with_columns(pl.Series(c, expit(Db['lp2'].to_numpy() + rdn)))
    log(c, 'best iters', its, RES[f'rows_{c}'])
inb = (pl.col('p2') > 0.02) & (pl.col('p2') < 0.99); inw = (pl.col('p2') > 0.005) & (pl.col('p2') < 0.995)
residual('p3b', inb, ['lp2', 'ce_logit'] + BASEF)
residual('q3b', inb & pl.col('ne'), ['lp2', 'ce_ne'] + BASEF)
residual('r3b', inb & pl.col('ne'), ['lp2', 'ce_logit'] + BASEF)
residual('p3nb', inb, ['lp2'] + BASEF)
residual('p3w', inw, ['lp2', 'ce_logit'] + BASEF)
# p3b_ne: p3b restricted to non-empty-address pairs
ne_tr = B.filter(inb & pl.col('ne')).select(*K); ne_dn = Dn.filter(inb & pl.col('ne')).select(*K)
cols_tr['p3b_ne'] = cols_tr['p3b'].join(ne_tr, on=K, how='semi').rename({'p3b': 'p3b_ne'}); cols_dn['p3b_ne'] = cols_dn['p3b'].join(ne_dn, on=K, how='semi').rename({'p3b': 'p3b_ne'})
# Platt blend control on band b
S = B.filter(inb & (pl.col('grp') == 'sample')); ys = S['label'].to_numpy()
lr = LogisticRegression(C=1e3).fit(S['ce_logit'].to_numpy()[:, None], ys); a, b0 = float(lr.coef_[0, 0]), float(lr.intercept_[0])
bl0 = lambda df, w: w * df['p2'].to_numpy() + (1 - w) * expit(a * df['ce_logit'].to_numpy() + b0)
ws = {w: log_loss(ys, np.clip(bl0(S, w), 1e-6, 1 - 1e-6)) for w in np.round(np.arange(0, 1.01, 0.05), 2)}
w = float(min(ws, key=ws.get)); RES['platt_b'] = dict(a=a, b=b0, w=w)
Bb = B.filter(inb); Db = Dn.filter(inb)
cols_tr['pbb'] = Bb.select(*K).with_columns(pl.Series('pbb', bl0(Bb, w))); cols_dn['pbb'] = Db.select(*K).with_columns(pl.Series('pbb', bl0(Db, w)))
# assemble: every column = its model on the rows it covers, p2 elsewhere (within the wide band table; outside the wide band everything = p2)
T = B.select(*K, 's1_id', 'cand_id', 'grp', 'fold', 'es', 'country', 'label', 'p1', 'p2', 'lp2', 'ce_logit', 'ce_ne', 'ce_old', 'in_dft', 'ne', *FLAGS)
Td = Dn.select(*K, 'label', 'country', 'p2', 'lp2', 'ce_logit', 'ce_ne', 'in_dft', 'ne', *FLAGS)
for c in cols_tr:
    T = T.join(cols_tr[c], on=K, how='left').with_columns(pl.coalesce(c, 'p2').alias(c)); Td = Td.join(cols_dn[c], on=K, how='left').with_columns(pl.coalesce(c, 'p2').alias(c))
T.write_parquet(f'{E}/data/stage3_{A.tag}.parquet'); Td.write_parquet(f'{E}/data/stage3_dens_{A.tag}.parquet')
# in-band metrics (band b), val + sample OOF + density; E11 exclusive slice labels for val
E11 = pl.read_parquet('/workspace/saumilya/amazon-ml/work/matching/prod_v2c/exp/E11_error_analysis/data/val_slice.parquet', columns=[*K, 'slice'])
T = T.join(E11, on=K, how='left')
SL = {'all': pl.lit(True), 'empty_addr': pl.col('addr_empty2') > 0, 'nonempty_addr': pl.col('addr_empty2') <= 0, 'shared_name': pl.col('name_freq_s1_1') > 1,
      'remainder(not empty, not shared)': (pl.col('addr_empty2') <= 0) & (pl.col('name_freq_s1_1') <= 1),
      'junk_url_alias': (pl.col('n11_junk_name') > 0) | (pl.col('f_url2') > 0) | (pl.col('n10_alias') > 0), 'nonlatin': pl.col('script2') >= 2,
      'name_unexplained': pl.col('x1_name_explained') <= 0, 'lexical_only': pl.col('in_dft') < 0.5, 'US': pl.col('country') == 'US', 'India': pl.col('country') == 'India'}
for s in ['shared_name', 'junk_name', 'other', 'translit', 'url_handle', 'decoy_twin']: SL[f'E11:{s}'] = pl.col('slice') == s
SCN = ['p2', 'ce_old', 'ce_logit', 'ce_ne', 'p3nb', 'p3b', 'q3b', 'r3b', 'p3w', 'pbb']
RES['inband'] = {}
for nm, df in [('val', T.filter((pl.col('grp') != 'sample') & inb)), ('sample_oof', T.filter((pl.col('grp') == 'sample') & inb)), ('density', Td.filter(inb))]:
    RES['inband'][nm] = {}
    for sn, cond in SL.items():
        if sn.startswith('E11') and nm != 'val': continue
        d = df.filter(cond); y = d['label'].to_numpy()
        if d.height < 50 or y.min() == y.max(): continue
        r = dict(n=d.height, pos=int(y.sum()))
        for c in SCN:
            if c in d.columns and d[c].null_count() == 0: r[c] = m2(y, d[c].to_numpy())
        RES['inband'][nm][sn] = r
    log(nm, json.dumps(RES['inband'][nm]['all'])); log(nm, 'nonempty', json.dumps(RES['inband'][nm].get('nonempty_addr')))
RES['seconds'] = round(time.time() - T0)
json.dump(RES, open(f'{E}/logs/04_stage3_{A.tag}.json', 'w'), indent=1, default=str); log('done')
