"""S1_mirror_sweep step 3: (a) model-conditioned sub-families (family x bucket x [v3anchor < 0.5 | >= 0.5]) with the same
selection-side mirror; (b) per-S1 F0.5 simulated first-order dLB of removing each (sub-)family's +k accepted pairs on top of
O3's cand_US_LEG (O3-removed pairs excluded, never empties an S1), with q = mirror P(true) and q_hi (95th pct);
(c) exact val dF (labels) of the same removal applied to #8's val selection.
Writes out/s03_rank.tsv (all units: dLB = weight x dF_country; val dF in val-macro units over 220,730 val S1)."""
import os, sys
os.environ.setdefault('POLARS_MAX_THREADS', '4')
import polars as pl, numpy as np
W = '/workspace/saumilya/amazon-ml/work'
OUT = f'{W}/winning_strategy_20260926/agents/S1_mirror_sweep/out'
IT = f'{W}/matching/iterate_20260926'
NV = {'US': 132373, 'India': 88357}
NT = {'US': 663106, 'India': 809986}
WLB = {'US': 0.3827, 'India': 0.4675}
NVAL = 220730
rng = np.random.default_rng(11)
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_float_precision(3)
K2 = ['s1_idx', 'cand_idx']


def f05(tp, k, m):
    if m == 0:
        return 1.0 if k == 0 else 0.0
    if k == 0 or tp == 0:
        return 0.0
    p = tp / k; r = tp / m
    return 1.25 * p * r / (0.25 * p + r)


def dist_true(qs):
    d = np.zeros(len(qs) + 1); d[0] = 1.0
    for q in qs:
        d[1:] = d[1:] * (1 - q) + d[:-1] * q; d[0] *= (1 - q)
    return d


def sim(rem, ksel):
    """rem: DataFrame s1_idx, q (removal set); ksel: dict s1 -> k selected in base. Returns (sum dF, n_pairs, n_s1, skipped_s1)."""
    tot = 0.0; npair = 0; ns1 = 0; skip = 0
    for s, qs in rem.group_by('s1_idx').agg('q').iter_rows():
        k = ksel[s]; r = len(qs)
        if r >= k:
            skip += 1; continue
        d = dist_true(qs); e = 0.0
        for t, pt in enumerate(d):
            e += pt * (f05(k - r, k - r, k - r + t) - f05(k - r + t, k, k - r + t))
        tot += e; npair += r; ns1 += 1
    return tot, npair, ns1, skip


pv = pl.read_parquet(f'{OUT}/pairs_val.parquet').filter(pl.col('acc') == 1)
pt = pl.read_parquet(f'{OUT}/pairs_test.parquet').filter((pl.col('acc') == 1) & pl.col('country').is_in(['US', 'India']))
mv = pl.scan_parquet(f'{W}/matching/ensemble_v5/data/members_val.parquet').select(K2 + ['m_v3anchor']).join(pv.select(K2).lazy(), on=K2).collect()
mt = pl.scan_parquet(f'{W}/matching/ensemble_v5/data/members_test.parquet').select(K2 + ['m_v3anchor']).join(pt.select(K2).lazy(), on=K2).collect()
pv = pv.join(mv, on=K2, how='left'); pt = pt.join(mt, on=K2, how='left')
print('v3anchor coverage val', pv['m_v3anchor'].is_not_null().mean(), 'test', pt['m_v3anchor'].is_not_null().mean())
v3 = pl.when(pl.col('m_v3anchor') < 0.5).then(pl.lit('lo')).otherwise(pl.lit('hi')).alias('v3a')
pv = pv.with_columns(v3); pt = pt.with_columns(v3)
# pattern-table LEG12/LEG3/CTAG membership (O3's LEG definition, both sides, val + test)
for nm in ('val', 'test'):
    pf = pl.read_parquet(f'{W}/internal_eval/data/pc_{nm}_fam.parquet', columns=K2 + ['fam']).filter(pl.col('fam').is_in(['LEG12', 'LEG3', 'CTAG'])).select(K2).unique().with_columns(pl.lit('1').alias('pat'))
    if nm == 'val':
        pv = pv.join(pf, on=K2, how='left').with_columns(pl.col('pat').fill_null('0'))
    else:
        pt = pt.join(pf, on=K2, how='left').with_columns(pl.col('pat').fill_null('0'))
print('pat share val', (pv['pat'] == '1').mean(), 'test', (pt['pat'] == '1').mean())

# base selections (test: #8 minus O3 US_LEG; val: #8 val selection)
tsel = pl.read_parquet(f'{IT}/submissions/nc_specialist_legal_fr/selected.parquet')
o3 = pl.read_parquet(f'{W}/winning_strategy_20260926/agents/O3_metric_transfer/out/p3_remove_US_LEG.parquet', columns=K2)
tsel = tsel.join(o3, on=K2, how='anti')
kt = dict(tsel.group_by('s1_idx').len().iter_rows())
vsel = pl.read_parquet(f'{IT}/empty_address/val_nc_robust_selected.parquet')
gt = pl.read_parquet(f'{W}/internal_eval/data/val_gt_pairs.parquet').select(K2)
vs = vsel.join(gt.with_columns(pl.lit(1).alias('t')), on=K2, how='left').with_columns(pl.col('t').fill_null(0))
kv = dict(vs.group_by('s1_idx').len().iter_rows())
tpv = dict(vs.group_by('s1_idx').agg(pl.col('t').sum()).iter_rows())
mvv = dict(gt.group_by('s1_idx').len().iter_rows())


def val_exact(rem):
    """exact val dF sum for removing rem (s1_idx, cand_idx, label) from #8's val selection (never empties)."""
    tot = 0.0
    for s, labs in rem.group_by('s1_idx').agg('label').iter_rows():
        k = kv[s]; r = len(labs)
        if r >= k:
            continue
        tp = tpv.get(s, 0); m = mvv.get(s, 0); tr = int(sum(labs))
        tot += f05(tp - tr, k - r, m) - f05(tp, k, m)
    return tot


def mirror(av, at, keys):
    """selection-side mirror per keys; av/at are accepted pairs (val with label)."""
    gv = av.group_by(keys + ['side']).agg(pl.len().alias('va'), pl.col('label').sum().alias('vat'))
    gt_ = at.group_by(keys + ['side']).agg(pl.len().alias('ta'), (pl.col('o3rm') == 0).sum().alias('tax'))
    P = gt_.filter(pl.col('side') == '+').drop('side').join(gt_.filter(pl.col('side') == '-').drop('side').rename({'ta': 'taM', 'tax': 'taxM'}), on=keys, how='left') \
        .join(gv.filter(pl.col('side') == '+').drop('side'), on=keys, how='left') \
        .join(gv.filter(pl.col('side') == '-').drop('side').rename({'va': 'vaM', 'vat': 'vatM'}), on=keys, how='left').fill_null(0)
    out = []
    for r in P.iter_rows(named=True):
        c = r['country']; nt = NT[c] / 1000; nv = NV[c] / 1000
        if r['vaM'] >= 5:
            true_t = r['taM'] / nt * r['vat'] / r['vaM']
            b = rng.poisson(r['taM'], 4000) / nt * rng.poisson(r['vat'], 4000) / np.maximum(rng.poisson(r['vaM'], 4000), 1)
            meth = 'mirror'
        else:
            true_t = r['vat'] / nv; b = rng.poisson(r['vat'], 4000) / nv; meth = 'O3fb'
        accP = r['ta'] / nt
        ab = rng.poisson(r['ta'], 4000) / nt
        qb = np.clip(b / np.maximum(ab, 1e-9), 0, 1)
        q = min(1.0, true_t / accP) if accP > 0 else 1.0
        d = {kk: r[kk] for kk in keys}
        d.update(n_vaP=r['va'], v_prec=(r['vat'] / r['va']) if r['va'] else None, n_vaM=r['vaM'], n_taP=r['ta'], n_taP_afterO3=r['tax'], n_taM=r['taM'],
                 v_accP_1k=r['va'] / nv, t_accP_1k=accP, t_accM_1k=r['taM'] / nt, v_accM_1k=r['vaM'] / nv,
                 method=meth, t_fp_1k=accP - true_t, t_fp_lo=float(np.quantile(ab - b, 0.05)), t_fp_hi=float(np.quantile(ab - b, 0.95)),
                 q=q, q_lo=float(np.quantile(qb, 0.05)), q_hi=float(np.quantile(qb, 0.95)))
        out.append(d)
    return pl.DataFrame(out, infer_schema_length=None)


res = []
for keys in (['country', 'fam', 'bucket'], ['country', 'fam', 'bucket', 'v3a'], ['country', 'fam', 'bucket', 'pat']):
    M = mirror(pv, pt, keys)
    if len(keys) == 3:
        M = M.with_columns(pl.lit('all').alias('sub'))
    else:
        M = M.with_columns((pl.lit(keys[3] + ':') + pl.col(keys[3])).alias('sub')).drop(keys[3])
    for r in M.iter_rows(named=True):
        if r['n_taP_afterO3'] < 20:
            continue
        cond = (pl.col('country') == r['country']) & (pl.col('fam') == r['fam']) & (pl.col('bucket') == r['bucket']) & (pl.col('side') == '+')
        if r['sub'] != 'all':
            kk, vv = r['sub'].split(':')
            cond = cond & (pl.col(kk) == vv)
        rt = pt.filter(cond & (pl.col('o3rm') == 0)).select('s1_idx')
        c = r['country']
        dF, npair, ns1, skip = sim(rt.with_columns(pl.lit(r['q']).alias('q')), kt)
        dFh, _, _, _ = sim(rt.with_columns(pl.lit(r['q_hi']).alias('q')), kt)
        vdf = val_exact(pv.filter(cond).select('s1_idx', 'label'))
        r.update(pairs_removed=npair, s1_changed=ns1, s1_skipped_sole=skip, dLB=WLB[c] * dF / NT[c], dLB_qhi=WLB[c] * dFh / NT[c], val_dF=vdf / NVAL)
        res.append(r)
R = pl.DataFrame(res, infer_schema_length=None).sort('dLB', descending=True)
R.write_csv(f'{OUT}/s03_rank.tsv', separator='\t')
print(R.select('country', 'fam', 'bucket', 'sub', 'n_vaP', 'v_prec', 'n_vaM', 'v_accP_1k', 't_accP_1k', 'v_accM_1k', 't_accM_1k', 'n_taP_afterO3', 'method', 't_fp_1k', 't_fp_lo', 't_fp_hi',
               'q', 'q_lo', 'q_hi', 'pairs_removed', 's1_skipped_sole', 'dLB', 'dLB_qhi', 'val_dF'))
