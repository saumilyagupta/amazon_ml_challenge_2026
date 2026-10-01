"""S1_mirror_sweep step 2: per family x country x offset-bucket val-vs-test table, +-k mirror estimate of test false acceptances,
Poisson parametric-bootstrap interval, mirror P(true | accepted, test), per-S1 F0.5 simulated first-order dLB of removing the family's
+k accepted pairs ON TOP OF O3's cand_US_LEG (pairs already removed by O3 excluded; never empties an S1).
Mirror (primary, selection side): test true+ /1k = test accepted(-k) /1k x [val true accepted(+k) / val accepted(-k)].
Mirror (alt, candidate side):     test true+ /1k = test cand(-k) /1k   x [val true accepted(+k) / val cand(-k)].
Fallback when val accepted(-k) < 5: O3's estimator test true+ /1k = val true accepted(+k) /1k.
"""
import os, sys
os.environ.setdefault('POLARS_MAX_THREADS', '4')
import polars as pl, numpy as np
W = '/workspace/saumilya/amazon-ml/work'
OUT = f'{W}/winning_strategy_20260926/agents/S1_mirror_sweep/out'
IT = f'{W}/matching/iterate_20260926'
NV = {'US': 132373, 'India': 88357}
NT = {'US': 663106, 'India': 809986, 'France': 259452}
WLB = {'US': 0.3827, 'India': 0.4675, 'France': 0.1498}
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_float_precision(3)
rng = np.random.default_rng(7)

pv = pl.read_parquet(f'{OUT}/pairs_val.parquet')
pt = pl.read_parquet(f'{OUT}/pairs_test.parquet')
K = ['country', 'fam', 'bucket']
gv = pv.group_by(K + ['side']).agg(pl.len().alias('vc'), pl.col('label').sum().alias('vct'), pl.col('acc').sum().alias('va'),
                                   ((pl.col('acc') == 1) & (pl.col('label') == 1)).sum().alias('vat'))
gt = pt.group_by(K + ['side']).agg(pl.len().alias('tc'), pl.col('acc').sum().alias('ta'),
                                   ((pl.col('acc') == 1) & (pl.col('o3rm') == 0)).sum().alias('ta_x'), pl.col('o3rm').sum().alias('o3'))


def wide(g, cols):
    p = g.filter(pl.col('side') == '+').drop('side').rename({c: c + 'P' for c in cols})
    m = g.filter(pl.col('side') == '-').drop('side').rename({c: c + 'M' for c in cols})
    return p.join(m, on=K, how='full', coalesce=True)


J = wide(gt, ['tc', 'ta', 'ta_x', 'o3']).join(wide(gv, ['vc', 'vct', 'va', 'vat']), on=K, how='left').fill_null(0)
rows = []
for r in J.iter_rows(named=True):
    c = r['country']; nt = NT[c] / 1000
    d = dict(country=c, fam=r['fam'], bucket=r['bucket'])
    d.update(t_candP=r['tcP'] / nt, t_candM=r['tcM'] / nt, t_accP=r['taP'] / nt, t_accM=r['taM'] / nt, t_accP_afterO3=r['ta_xP'] / nt, n_t_accP_afterO3=r['ta_xP'])
    if c in NV:
        nv = NV[c] / 1000
        d.update(v_candP=r['vcP'] / nv, v_candM=r['vcM'] / nv, v_trueP=r['vctP'] / nv, v_accP=r['vaP'] / nv, v_accM=r['vaM'] / nv,
                 v_precP=(r['vatP'] / r['vaP']) if r['vaP'] else None, v_trueaccP=r['vatP'] / nv, n_v_accM=r['vaM'], n_v_accP=r['vaP'])
        # mirror on the after-O3 accepted set (for LEG families O3 already removed most; after-O3 share applied proportionally)
        keep = (r['ta_xP'] / r['taP']) if r['taP'] else 1.0
        if r['vaM'] >= 5:
            ratio = r['vatP'] / r['vaM']
            true_t = (r['taM'] / nt) * ratio
            method = 'mirror_acc'
        else:
            ratio = None; true_t = r['vatP'] / nv; method = 'O3_fallback'
        true_alt = (r['tcM'] / nt) * (r['vatP'] / r['vcM']) if r['vcM'] >= 5 else None
        accP = r['taP'] / nt
        fp = accP - true_t
        q = min(1.0, true_t / accP) if accP > 0 else None
        # Poisson parametric bootstrap
        B = 4000
        aP = rng.poisson(max(r['taP'], 0), B) / nt
        if method == 'mirror_acc':
            tt = rng.poisson(r['taM'], B) / nt * rng.poisson(r['vatP'], B) / np.maximum(rng.poisson(r['vaM'], B), 1)
        else:
            tt = rng.poisson(r['vatP'], B) / nv
        fpb = aP - tt
        qb = np.clip(np.where(aP > 0, tt / np.maximum(aP, 1e-9), 1.0), 0, 1)
        ctrl_ratio = (r['tcM'] / nt) / (r['vcM'] / nv) if r['vcM'] else None
        d.update(method=method, t_true_mirror=true_t, t_true_alt=true_alt, t_fp_mirror=fp, t_fp_lo=float(np.quantile(fpb, 0.05)), t_fp_hi=float(np.quantile(fpb, 0.95)),
                 q_mirror=q, q_lo=float(np.quantile(qb, 0.05)), q_hi=float(np.quantile(qb, 0.95)),
                 q_alt=(min(1.0, true_alt / accP) if (true_alt is not None and accP > 0) else None), minus_ctrl_ratio_t_over_v=ctrl_ratio,
                 keep_afterO3=keep)
    rows.append(d)
T = pl.DataFrame(rows, infer_schema_length=None).sort(['country', 'fam', 'bucket'])
T.write_csv(f'{OUT}/s02_family_table.tsv', separator='\t')
show = ['country', 'fam', 'bucket', 'v_candP', 'v_candM', 't_candP', 't_candM', 'v_trueP', 'v_accP', 'v_precP', 'v_accM', 't_accP', 't_accM', 't_accP_afterO3',
        'n_v_accM', 'method', 't_fp_mirror', 't_fp_lo', 't_fp_hi', 'q_mirror', 'q_lo', 'q_hi', 'q_alt', 'minus_ctrl_ratio_t_over_v']
print(T.filter(pl.col('country') != 'France').select(show))
print(T.filter(pl.col('country') == 'France').select(['country', 'fam', 'bucket', 't_candP', 't_candM', 't_accP', 't_accM']))
