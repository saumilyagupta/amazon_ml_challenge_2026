#!/opt/conda/bin/python3
"""BLEND agent pipeline for one blend: val preds -> run_eval (thread) || test preds -> run_stress -> summarize -> progress.
usage: BLEND_run.py <name>   name in v3anchor, v3anchor_h, wopt, wopt_band, pmin_agree, frgate
Blends over MEMBERS = m_v2b, m_v3, m_e06, m_e13 (probabilities; logits clipped at 1e-6 via C.logit):
  v3anchor    logit p = L(v3) + [L(e06)-L(v2b)] + [L(e13)-L(v2b)], final logit clipped to [-13.8, 13.8]. No fitting.
  v3anchor_h  same with both corrections x 0.5. No fitting.
  wopt        logit p = b + sum_k w_k L(m_k), w = softmax(a) on the simplex; (a, b) minimise binary log-loss on fold-f SAMPLE rows,
              applied to SAMPLE rows of fold 1-f; val/locked/test use the mean of the two folds' (w, b).
  wopt_band   same fit restricted to fitting-fold sample rows with 0.02 < mean-of-4 p < 0.99; applied to all rows.
  pmin_agree  p = mean-of-4 if max-min <= 0.15 else second-lowest of the 4. No fitting.
  frgate      test only: US/India rows = best-by-stress blend among the five above (read from results/stress_*.json), France rows = v3anchor.
              Its val table = that blend's (val has no France): eval json/decoder copied, stress recomputed with its own test file.
Selection only ever uses grp == 'sample' labels; val/locked labels are only read by the evaluator."""
import os, sys, json, time, threading, shutil
H = '/workspace/saumilya/amazon-ml/work/matching/ensemble_v1'
sys.path.insert(0, H + '/common'); import ens_common as C
import numpy as np, polars as pl

NAME = sys.argv[1]; AG = 'BLEND'
WJ = f'{C.RD}/BLEND_params_{NAME}.json'  # one file per blend (no cross-process races)
M = C.MEMBERS


def logits(df):
    return np.column_stack([C.logit(df[m].to_numpy()) for m in M])  # float64 (n, 4)


def v3anchor(df, h=1.0):
    L2b = C.logit(df['m_v2b'].to_numpy()); z = C.logit(df['m_v3'].to_numpy())
    z += h * (C.logit(df['m_e06'].to_numpy()) - L2b); z += h * (C.logit(df['m_e13'].to_numpy()) - L2b)
    return C.expit(np.clip(z, -13.8, 13.8)).astype(np.float32)


def pmin_agree(df):
    P = np.column_stack([df[m].to_numpy() for m in M]).astype(np.float64)
    rng = P.max(1) - P.min(1); mean4 = P.mean(1); sec = np.sort(P, axis=1)[:, 1]
    C.log('pmin_agree: share of rows with max-min > 0.15 =', float((rng > 0.15).mean()))
    return np.where(rng <= 0.15, mean4, sec).astype(np.float32)


def fit_wopt(X, y):
    """min mean log-loss of sigmoid(b + X @ softmax(a)); returns (w, b, loss)."""
    from scipy.optimize import minimize
    y = y.astype(np.float64)

    def f(th):
        a, b = th[:4], th[4]; e = np.exp(a - a.max()); w = e / e.sum()
        z = X @ w + b
        loss = np.mean(np.logaddexp(0, z) - y * z)
        r = C.expit(z) - y
        gw = X.T @ r / len(y); gb = r.mean()
        ga = w * (gw - w @ gw)
        return loss, np.concatenate([ga, [gb]])
    th0 = np.zeros(5)
    res = minimize(f, th0, jac=True, method='L-BFGS-B', options=dict(maxiter=500, ftol=1e-15, gtol=1e-12))
    a = res.x[:4]; e = np.exp(a - a.max()); w = e / e.sum()
    return w, float(res.x[4]), float(res.fun), int(res.nit)


def blend_val(df):
    if NAME == 'v3anchor': return v3anchor(df, 1.0), None
    if NAME == 'v3anchor_h': return v3anchor(df, 0.5), None
    if NAME == 'pmin_agree': return pmin_agree(df), None
    if NAME in ('wopt', 'wopt_band'):
        X = logits(df); y = df['label'].to_numpy(); grp = np.where((df['grp'] == 'sample').to_numpy(), 'sample', 'other'); fold = df['fold'].to_numpy()
        P = np.column_stack([df[m].to_numpy() for m in M]).astype(np.float64); mean4 = P.mean(1); del P
        par = {}
        for f in (0, 1):
            fit = (grp == 'sample') & (fold == f)
            if NAME == 'wopt_band': fit &= (mean4 > 0.02) & (mean4 < 0.99)
            t = time.time(); w, b, loss, nit = fit_wopt(X[fit], y[fit])
            base = {m: float(np.mean(np.logaddexp(0, X[fit, k]) - y[fit] * X[fit, k])) for k, m in enumerate(M)}
            par[f] = dict(w=dict(zip(M, map(float, w))), b=b, logloss=loss, member_logloss=base, n_fit=int(fit.sum()), pos_fit=int(y[fit].sum()), nit=nit, sec=round(time.time() - t, 1))
            C.log(NAME, 'fold', f, par[f])
        wm = np.mean([[par[f]['w'][m] for m in M] for f in (0, 1)], axis=0); bm = float(np.mean([par[f]['b'] for f in (0, 1)]))
        par['mean'] = dict(w=dict(zip(M, map(float, wm))), b=bm)
        z = X @ wm + bm  # val/locked rows: mean of the two folds' parameters
        for f in (0, 1):  # sample rows of fold 1-f get the fold-f parameters (honest OOF)
            sel = (grp == 'sample') & (fold == 1 - f)
            z[sel] = X[sel] @ np.array([par[f]['w'][m] for m in M]) + par[f]['b']
        return C.expit(z).astype(np.float32), par
    raise SystemExit(NAME)


def blend_test(df, par):
    if NAME == 'v3anchor': return v3anchor(df, 1.0)
    if NAME == 'v3anchor_h': return v3anchor(df, 0.5)
    if NAME == 'pmin_agree': return pmin_agree(df)
    if NAME in ('wopt', 'wopt_band'):
        wm = np.array([par['mean']['w'][m] for m in M]); return C.expit(logits(df) @ wm + par['mean']['b']).astype(np.float32)
    raise SystemExit(NAME)


def test_proxy(T):
    """cheap test-side behaviour: per S1 count of pairs with p >= 0.5 (matches-per-S1 proxy) and share of S1 with none (empty proxy), per country."""
    g = T.group_by('s1_idx', 'country').agg((pl.col('p') >= 0.5).sum().alias('k'))
    return {r['country']: dict(k05_per_s1=r['k05'], empty05=r['e05']) for r in g.group_by('country').agg(pl.col('k').mean().alias('k05'), (pl.col('k') == 0).mean().alias('e05')).to_dicts()}


def save_json(path, key, val):
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[key] = val; json.dump(d, open(path, 'w'), indent=1)


def fmt_line(name):
    e = json.load(open(f'{C.RD}/eval_ENS_{name}.json'))['val']; s = json.load(open(f'{C.RD}/stress_{name}.json')); pv = e['paired_vs_v2b']
    return f"{name}: val {e['macro_f05']:.6f} locked30k {e.get('locked30k', float('nan')):.5f} d_vs_v2b {pv['diff']:+.5f} [{pv['ci95'][0]:+.5f},{pv['ci95'][1]:+.5f}] stress {s['overall_test_mix']:.5f}"


if NAME == 'frgate':
    cands = ['v3anchor', 'v3anchor_h', 'wopt', 'wopt_band', 'pmin_agree']
    done = {n: json.load(open(f'{C.RD}/stress_{n}.json'))['overall_test_mix'] for n in cands if os.path.exists(f'{C.RD}/stress_{n}.json') and os.path.exists(f'{C.RD}/eval_ENS_{n}.json')}
    best = max(done, key=done.get); C.log('frgate: finished blends and stress', done, '-> US/India blend =', best)
    T = pl.read_parquet(f'{C.DD}/test_{best}.parquet').rename({'p': 'p_ui'})
    A = pl.read_parquet(f'{C.DD}/test_v3anchor.parquet', columns=['p']).rename({'p': 'p_fr'})
    T = pl.concat([T, A], how='horizontal').with_columns(pl.when(pl.col('country') == 'France').then(pl.col('p_fr')).otherwise(pl.col('p_ui')).alias('p'))
    C.write_test_preds(T, 'frgate'); prox = test_proxy(T.select('s1_idx', 'country', 'p')); del T, A
    for suf in ('.json', '_val_rows.parquet', '_val_selected.parquet', '_R10c_m0.txt'):
        shutil.copyfile(f'{C.RD}/eval_ENS_{best}{suf}', f'{C.RD}/eval_ENS_frgate{suf}')
    ej = json.load(open(f'{C.RD}/eval_ENS_frgate.json')); ej['tag'] = 'ENS_frgate'
    ej['note'] = f'frgate = {best} on US/India (the whole val table); eval copied from eval_ENS_{best} (identical val decision); France test rows = v3anchor'
    json.dump(ej, open(f'{C.RD}/eval_ENS_frgate.json', 'w'), indent=1, default=str)
    s = C.run_stress('frgate', val_preds=f'{C.DD}/preds_{best}.parquet', rows=f'{C.RD}/eval_ENS_{best}_val_rows.parquet')
    save_json(WJ, 'frgate', dict(us_india_blend=best, candidates_stress=done)); save_json(WJ, 'test_proxy', prox)
    C.summarize('frgate'); C.progress(AG, fmt_line('frgate') + f' (US/India = {best}, France = v3anchor)')
    C.log('DONE', fmt_line('frgate')); sys.exit(0)

C.wait_mem(60, f'{NAME} val')
V = pl.read_parquet(C.MEMBERS_VAL)
p, par = blend_val(V)
assert len(p) == V.height and np.isfinite(p).all()
C.log(NAME, 'val p mean', float(p.mean()), 'sample OOF mean', float(p[(V['grp'] == 'sample').to_numpy()].mean()))
if par is not None: save_json(WJ, 'params', {str(k): v for k, v in par.items()})
C.write_val_preds(V.select(C.VAL_KEEP).with_columns(pl.Series('p', p)), NAME); del V, p
C.log('wrote val preds; peak RSS GB', round(C.peak_rss_gb(), 1))

ERR = []
def _ev():
    try: C.run_eval(NAME)
    except Exception as ex: ERR.append(ex)
th = threading.Thread(target=_ev); th.start()
time.sleep(5)
T = pl.read_parquet(C.MEMBERS_TEST)
pt = blend_test(T, par); assert np.isfinite(pt).all()
T = T.select(C.TEST_KEEP).with_columns(pl.Series('p', pt)); del pt
C.write_test_preds(T, NAME); prox = test_proxy(T.select('s1_idx', 'country', 'p')); del T
save_json(WJ, 'test_proxy', prox)
C.log('wrote test preds', prox, 'peak RSS GB', round(C.peak_rss_gb(), 1))
th.join()
if ERR: raise ERR[0]
C.run_stress(NAME); C.summarize(NAME); C.progress(AG, fmt_line(NAME))
C.log('DONE', fmt_line(NAME), 'peak RSS GB', round(C.peak_rss_gb(), 1))
