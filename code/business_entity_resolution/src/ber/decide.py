"""Decision layer: vectorised exact macro F0.5 (official singleton rule) + policies.
(a) global threshold tau  (b) one-to-one: keep only the arg-max claiming S1 of a record  (c) source-side margin: p >= p_other + delta
(d) row-level empty / non-empty LightGBM model on per-S1 aggregates  (e) expected-F0.5 prefix choice (Monte-Carlo under independence).
Predictions are polars frames of (s1_idx, cand_idx, p, label) over the FULL candidate table; scoring is restricted to a set of S1 with
truth counts m (incl. truths outside the candidate set)."""
import numpy as np, polars as pl
from .chan import recside_stage2


def row_scores(pred: pl.DataFrame, M: pl.DataFrame) -> pl.DataFrame:
    a = pred.group_by('s1_idx').agg(pl.len().alias('k'), pl.col('label').cast(pl.Int32).sum().alias('t'))
    r = M.join(a, on='s1_idx', how='left').with_columns(pl.col('k').fill_null(0), pl.col('t').fill_null(0))
    return r.with_columns(pl.when(pl.col('m') == 0).then((pl.col('k') == 0).cast(pl.Float64))
                          .when(pl.col('t') > 0).then(5 * pl.col('t') / (4 * pl.col('k') + pl.col('m'))).otherwise(0.0).alias('f'))


def macro(pred, M):
    return float(row_scores(pred, M)['f'].mean())


def add_competition(P: pl.DataFrame, pcol='p'):
    """v1: best OTHER S1's probability for the same record, over P itself (P = full dense table)."""
    P = P.with_columns(pl.col(pcol).rank('ordinal', descending=True).over('cand_idx').alias('_rk'),
                       pl.col(pcol).max().over('cand_idx').alias('_m1'), pl.col(pcol).top_k(2).min().over('cand_idx').alias('_m2'),
                       pl.len().over('cand_idx').alias('_nr'))
    return P.with_columns(pl.when(pl.col('_nr') == 1).then(-1.0).when(pl.col('_rk') == 1).then(pl.col('_m2')).otherwise(pl.col('_m1')).alias('p_other'),
                          (pl.col('_rk') == 1).alias('is_argmax')).drop('_m1', '_m2', '_nr', '_rk')


def add_comp(P: pl.DataFrame, comp: pl.DataFrame, pcol='p') -> pl.DataFrame:
    """union variants: p_other / is_argmax against competitor table comp (s1_idx, cand_idx, q = competitor's stage-2 prob over the dense
    table of ALL S1); -1 if the record has no other claimant."""
    R = recside_stage2(P.select('s1_idx', 'cand_idx', pcol), comp, pcol, '_c').select('s1_idx', 'cand_idx', '_c_max_other_rec', '_c_rank_rec')
    n_other = comp.group_by('cand_idx').agg(pl.len().alias('_nc'))
    P = P.join(R, on=['s1_idx', 'cand_idx'], how='left').join(n_other, on='cand_idx', how='left')
    P = P.join(comp.select('s1_idx', 'cand_idx', pl.lit(1).alias('_self')), on=['s1_idx', 'cand_idx'], how='left')
    nother = pl.col('_nc').fill_null(0) - pl.col('_self').fill_null(0)
    return P.with_columns(pl.when(nother <= 0).then(-1.0).otherwise(pl.col('_c_max_other_rec')).alias('p_other'),
                          (pl.col('_c_rank_rec') == 1).alias('is_argmax')).drop('_c_max_other_rec', '_c_rank_rec', '_nc', '_self')


def elig_expr(mode, delta):
    return pl.col('is_argmax') if mode == 'o2o' else ((pl.col('p') >= pl.col('p_other') + delta) if mode == 'margin' else pl.lit(True))


def policy(P, tau, mode='plain', delta=0.0, pcol='p'):
    sel = pl.col(pcol) >= tau
    if mode == 'o2o': sel = sel & pl.col('is_argmax')
    elif mode == 'margin': sel = sel & (pl.col(pcol) >= pl.col('p_other') + delta)
    return P.filter(sel)


def report(pred, M):
    """full breakdown for a final policy. M: s1_idx, m, r (truths retained in candidates), country."""
    R = row_scores(pred, M)
    out = {'macro_f05': R['f'].mean(), 'n': R.height}
    for c in R['country'].unique().sort().to_list():
        out[f'macro_{c}'] = R.filter(pl.col('country') == c)['f'].mean()
    R = R.with_columns(pl.when(pl.col('m') >= 4).then(pl.lit('4+')).otherwise(pl.col('m').cast(pl.Utf8)).alias('mb'))
    for b in ['0', '1', '2', '3', '4+']:
        x = R.filter(pl.col('mb') == b); out[f'by_m_{b}'] = x['f'].mean(); out[f'n_m_{b}'] = x.height
    out['singleton_acc'] = R.filter(pl.col('m') == 0)['f'].mean()
    ns = R.filter(pl.col('m') > 0); out['nonsingleton_wrongly_empty'] = float((ns['k'] == 0).mean()) if ns.height else None
    out['macro_precision'] = R.select(pl.when(pl.col('k') > 0).then(pl.col('t') / pl.col('k')).otherwise(1.0))[:, 0].mean()
    out['macro_recall'] = R.select(pl.when(pl.col('m') > 0).then(pl.col('t') / pl.col('m')).otherwise(pl.when(pl.col('k') == 0).then(1.0).otherwise(0.0)))[:, 0].mean()
    out['micro_precision'] = R['t'].sum() / max(R['k'].sum(), 1); out['micro_recall'] = R['t'].sum() / max(R['m'].sum(), 1)
    out['frac_pred_nonempty'] = float((R['k'] > 0).mean()); out['matches_per_s1'] = R['k'].mean()
    R = R.with_columns(
        pl.when(pl.col('m') == 0).then(1.0).when(pl.col('r') > 0).then(5 * pl.col('r') / (4 * pl.col('r') + pl.col('m'))).otherwise(0.0).alias('C'),
        pl.when(pl.col('m') == 0).then(1.0).when(pl.col('t') > 0).then(5 * pl.col('t') / (4 * pl.col('t') + pl.col('m'))).otherwise(0.0).alias('Bn'))
    out['oracle_ceiling'] = R['C'].mean(); out['loss_not_in_candidates'] = 1 - R['C'].mean()
    out['loss_below_threshold'] = R['C'].mean() - R['Bn'].mean(); out['loss_false_positives'] = R['Bn'].mean() - R['f'].mean()
    return {k: (float(v) if v is not None else None) for k, v in out.items()}, R


ROW_FE = ['cos', 's1_top1', 's1_gap12', 'n_tset', 'a_tset', 'num_first_eq', 'rev_rank', 'name_freq_s1_1', 'addr_freq_s1_1', 'fake2', 'addr_empty2', 'n_cands_s1', 'script2']


def row_agg(q: pl.DataFrame) -> pl.DataFrame:
    """per-S1 aggregates for the row-level empty/non-empty model. q: s1_idx, cand_idx, p, p1, p_other, elig + ROW_FE columns."""
    q = q.sort('s1_idx', 'p', descending=[False, True])
    e = q.filter(pl.col('elig'))
    top = e.group_by('s1_idx', maintain_order=True).agg(
        pl.col('p').first().alias('r_p1'), pl.col('p').slice(1, 1).first().fill_null(0).alias('r_p2'), pl.col('p').slice(2, 1).first().fill_null(0).alias('r_p3'),
        (pl.col('p') > 0.3).sum().alias('r_n03'), (pl.col('p') > 0.5).sum().alias('r_n05'), (pl.col('p') > 0.8).sum().alias('r_n08'), pl.col('p').sum().alias('r_sum'),
        pl.len().alias('r_nelig'), *[pl.col(c).first().alias('t_' + c) for c in ROW_FE], pl.col('p1').first().alias('t_p1'), pl.col('p_other').first().alias('t_pother'),
        (pl.col('n_tset').first() - pl.col('a_tset').first()).alias('t_name_minus_addr'))
    allp = q.group_by('s1_idx').agg(pl.col('p').max().alias('r_pmax_all'), pl.len().alias('r_ncand'))
    return allp.join(top, on='s1_idx', how='left').with_columns((pl.col('r_p1') - pl.col('r_p2')).alias('r_gap12')).fill_null(0)


def apply_row(Pv, base_pred, te, tn, tl):
    pr = base_pred.filter(pl.col('p_empty') <= te)
    if tn > 0:
        have = pr.select('s1_idx').unique()
        add = Pv.filter(pl.col('elig') & (pl.col('p_empty') < tn) & (pl.col('p') >= tl)).join(have, on='s1_idx', how='anti') \
                .sort('p', descending=True).group_by('s1_idx').head(1)
        pr = pl.concat([pr.select(base_pred.columns), add.select(base_pred.columns)])
    return pr


def efs_choose(Pv, lam, q0, L=12, NS=256, seed=0, chunk=20000):
    """(e) expected-F0.5 prefix choice under independence (Monte Carlo). Pv: s1_idx, cand_idx, p, elig (+ label).
    Missing truths outside the candidate list: Poisson(lam) if the S1 has >= 1 candidate truth, else Bernoulli(q0).
    Returns the predicted pairs (prefix of the p-sorted eligible list; possibly empty)."""
    rng = np.random.default_rng(seed)
    E = Pv.filter(pl.col('elig')).sort('s1_idx', 'p', descending=[False, True]).group_by('s1_idx', maintain_order=True).head(L)
    E = E.with_columns(pl.int_range(pl.len()).over('s1_idx').alias('j'))
    sid = E['s1_idx'].unique(maintain_order=True).to_numpy(); pos = np.searchsorted(sid, E['s1_idx'].to_numpy())
    Pm = np.zeros((len(sid), L), np.float32); Pm[pos, E['j'].to_numpy()] = E['p'].to_numpy()
    kbest = np.zeros(len(sid), np.int32)
    for s in range(0, len(sid), chunk):
        p = Pm[s:s + chunk]
        Y = rng.random((NS,) + p.shape, dtype=np.float32) < p[None]
        Mc = Y.sum(2)
        miss = np.where(Mc > 0, rng.poisson(lam, Mc.shape), (rng.random(Mc.shape) < q0).astype(np.int64))
        Mt = Mc + miss; T = np.cumsum(Y, 2)
        k = np.arange(1, L + 1)[None, None, :]
        F = np.where(T > 0, 5 * T / (4 * k + Mt[..., None]), 0.0).mean(0)
        F0 = (Mt == 0).mean(0)
        kbest[s:s + chunk] = np.concatenate([F0[:, None], F], 1).argmax(1)
    kdf = pl.DataFrame({'s1_idx': sid, 'kb': kbest})
    return E.join(kdf, on='s1_idx').filter(pl.col('j') < pl.col('kb')).drop('j', 'kb')


def apply_decision(Pc, DEC, ep, row_model=None, feats=None):
    """Pc: s1_idx, cand_idx, p, p1, p_other, is_argmax (+ ROW_FE via feats for policy d). DEC: decision dict; ep: efs params."""
    Pc = Pc.with_columns(elig_expr(DEC['mode'], DEC['delta']).alias('elig'))
    pred = Pc.filter(pl.col('elig') & (pl.col('p') >= DEC['tau']))
    if DEC['policy'] == 'd':
        rm, RF = row_model
        RT = row_agg(Pc.join(feats, on=['s1_idx', 'cand_idx'], how='left'))
        RT = RT.with_columns(pl.Series('p_empty', rm.predict(RT.select(RF).to_numpy().astype(np.float32))))
        Pc = Pc.join(RT.select('s1_idx', 'p_empty'), on='s1_idx', how='left').with_columns(pl.col('p_empty').fill_null(1.0))
        pred = apply_row(Pc, Pc.filter(pl.col('elig') & (pl.col('p') >= DEC['tau'])), *DEC['row'])
    elif DEC['policy'] == 'e':
        pred = efs_choose(Pc, ep['lam'], ep['q0'], L=ep.get('L', 12), NS=ep.get('draws', 256), seed=ep.get('seed', 0))
    return pred
