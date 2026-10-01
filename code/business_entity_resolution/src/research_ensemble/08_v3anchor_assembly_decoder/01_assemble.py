#!/opt/conda/bin/python3
"""Build the member matrices (see common/ens_common.py docstring). Verifies identical (s1_idx, cand_idx) order across the full-length files,
left-joins the band-only test specialists and fills outside-band rows with v2b's p2 (their definition)."""
import sys; sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/ensemble_v1/common')
import ens_common as C
import numpy as np, polars as pl, json, os
log = C.log; K2 = ['s1_idx', 'cand_idx']; RES = {}
# ---------------- train side (sample OOF + val + locked)
base = pl.read_parquet(f'{C.V2B}/data/preds_abc_rob_cv2_all.parquet', columns=C.VAL_KEEP + ['p2']).rename({'p2': 'm_v2b'})
log('base', base.height)
def col(path, c, new):
    d = pl.read_parquet(path, columns=K2 + [c])
    assert d.height == base.height
    assert (d['s1_idx'] == base['s1_idx']).all() and (d['cand_idx'] == base['cand_idx']).all(), f'row order differs: {path}'
    return d[c].cast(pl.Float32).alias(new)
M = base.with_columns(col(f'{C.V3}/data/preds_AD_cv2_all.parquet', 'p2', 'm_v3'),
                      col(f'{C.V2C}/exp/E06_ce_band/data/preds_E06_xw.parquet', 'p3b', 'm_e06'),
                      col(f'{C.V2C}/exp/E13_codex_refit/data/preds_E13_cv2.parquet', 'p3', 'm_e13'))
M = M.with_columns(pl.col('m_v2b').cast(pl.Float32))
for m in C.MEMBERS: assert M[m].null_count() == 0, m
M.write_parquet(C.MEMBERS_VAL); log('wrote', C.MEMBERS_VAL, M.height, M.columns)
RES['val_rows'] = M.height; RES['val_grp'] = dict(M.group_by('grp').len().sort('grp').iter_rows())
X = M.select(C.MEMBERS).to_numpy().astype(np.float64)
RES['val_corr_p'] = np.corrcoef(X.T).round(5).tolist(); Lg = C.logit(X); RES['val_corr_logit'] = np.corrcoef(Lg.T).round(5).tolist()
RES['val_mean_p'] = dict(zip(C.MEMBERS, X.mean(0).round(6).tolist()))
RES['val_pairs_any_disagree_0.5'] = int(((X > 0.5).any(1) & ~(X > 0.5).all(1)).sum())
RES['val_band_rows_0.02_0.99'] = {m: int(((X[:, i] > 0.02) & (X[:, i] < 0.99)).sum()) for i, m in enumerate(C.MEMBERS)}
del X, Lg, M
# ---------------- test side
T = pl.read_parquet(f'{C.V2B}/data/preds_test_abc_rob_cv2_all.parquet', columns=C.TEST_KEEP + ['p2']).rename({'p2': 'm_v2b'})
v3 = pl.read_parquet(f'{C.V3}/data/preds_test_AD_cv2_all_gate.parquet', columns=K2 + ['p2'])
assert v3.height == T.height and (v3['s1_idx'] == T['s1_idx']).all() and (v3['cand_idx'] == T['cand_idx']).all()
T = T.with_columns(v3['p2'].cast(pl.Float32).alias('m_v3'), pl.col('m_v2b').cast(pl.Float32)); del v3
e06 = pl.read_parquet(f'{C.V2C}/build/p3_test_E06.parquet').select(*K2, pl.col('p3').cast(pl.Float32).alias('m_e06'))
e13 = pl.read_parquet(f'{C.V2C}/build/interim_v2b_E13/data/p3_all_variants_test.parquet').select(*K2, pl.col('p_cv2').cast(pl.Float32).alias('m_e13'))
RES['test_e06_band_rows'] = e06.height; RES['test_e13_band_rows'] = e13.height
T = T.join(e06, on=K2, how='left').join(e13, on=K2, how='left')
RES['test_e06_joined'] = int(T['m_e06'].is_not_null().sum()); RES['test_e13_joined'] = int(T['m_e13'].is_not_null().sum())
assert RES['test_e06_joined'] == e06.height and RES['test_e13_joined'] == e13.height, RES
T = T.with_columns(pl.coalesce('m_e06', 'm_v2b').alias('m_e06'), pl.coalesce('m_e13', 'm_v2b').alias('m_e13'))
assert T.height == 83761275
T.write_parquet(C.MEMBERS_TEST); log('wrote', C.MEMBERS_TEST, T.height, T.columns)
X = T.select(C.MEMBERS).to_numpy().astype(np.float64)
RES['test_corr_p'] = np.corrcoef(X.T).round(5).tolist(); RES['test_mean_p'] = dict(zip(C.MEMBERS, X.mean(0).round(6).tolist()))
RES['test_pairs_any_disagree_0.5'] = int(((X > 0.5).any(1) & ~(X > 0.5).all(1)).sum())
ct = T['country'].to_numpy()
RES['test_band_share_s1_by_member'] = {}
for i, m in enumerate(C.MEMBERS):
    b = T.select('s1_idx', 'country', ((pl.col(m) > 0.2) & (pl.col(m) < 0.8)).alias('b')).group_by('s1_idx', 'country').agg(pl.col('b').any()).group_by('country').agg(pl.col('b').mean())
    RES['test_band_share_s1_by_member'][m] = dict(b.iter_rows())
RES['peak_rss_gb'] = C.peak_rss_gb()
json.dump(RES, open(f'{C.LD}/01_assemble.json', 'w'), indent=1); log(json.dumps(RES, indent=1))
open(f'{C.DD}/_SUCCESS', 'w').write('ok\n'); log('DONE')
