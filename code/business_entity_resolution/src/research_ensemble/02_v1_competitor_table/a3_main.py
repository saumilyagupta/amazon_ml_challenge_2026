"""A3: apply the round-2 anchor residual on test, decode with the FROZEN v3anchor R10c decoder + the shipped post-pass, write submissions.
Chain replicated from prod_v2c/build/interim_v2b_E13/src/03_decide.py (imported read-only; B1_HOME -> vr2_stress_submission/build):
  p (anchor = ensemble_v1/data/test_v3anchor.parquet, bit-identical to A1's recomputation) -> U(s1,qid,src,p,qr = 1 if p >= p_other else 2)
  -> PP.decide_fast (verified == decide_lib.decide_set in the shipped build log) -> ADD-decoy veto (PP.decoy_pairs('test') kind ADD) -> strict one-owner by p
Step 0: reproduce the shipped v3anchor matching_results.tsv (md5 cdc0c756...) from the uncorrected anchor p; abort if different.
Variants: residual_all, residual_frfallback, consensus (variance_research_round2_20260926/seed_consensus.py logic)."""
import os, sys, time, json, hashlib, gc
os.environ['B1_HOME'] = '/workspace/saumilya/amazon-ml/work/matching/vr2_stress_submission/build'; os.environ['B1_THREADS'] = '8'
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2c/build/interim_v2b_E13/src')
import b1common as C
import numpy as np, polars as pl, lightgbm as lgb
from scipy.special import expit, logit
import pp_common as PP
H = '/workspace/saumilya/amazon-ml/work/matching/vr2_stress_submission'; VR = '/workspace/saumilya/amazon-ml/work/matching/variance_research_round2_20260926'
E = '/workspace/saumilya/amazon-ml/work/matching/ensemble_v1'
K2 = ['s1_idx', 'cand_idx']; log = C.log; t0 = time.time(); T = {}
md5 = lambda f: hashlib.md5(open(f, 'rb').read()).hexdigest()
RES = dict(timings_s=T)
SHIPPED = f'{E}/build/v3anchor/output/matching_results.tsv'; SHIPPED_CAND = f'{E}/build/v3anchor/output/candidate_pairs.tsv'
i1 = pl.read_parquet(f'{C.EB}/ids/test_s1.parquet', columns=['entity_id', 'country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
i2 = pl.read_parquet(f'{C.EB}/ids/test_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
# ---- load competition table (a3_comp.py) + anchor p
P = pl.read_parquet(f'{H}/data/test_comp.parquet', columns=['s1_idx', 'cand_idx', 'p_other', 'src'])
A = pl.read_parquet(f'{E}/data/test_v3anchor.parquet', columns=['s1_idx', 'cand_idx', 'p'])
P = P.join(A.select(*K2, pl.col('p').cast(pl.Float32).alias('p_anchor')), on=K2, how='left'); del A
assert P.height == C.N_PAIRS and P['p_anchor'].null_count() == 0
P = P.join(i1.select('s1_idx', pl.col('country').cast(pl.Categorical)), on='s1_idx', how='left')
T['load'] = round(time.time() - t0); log('loaded', P.height, f'RSS {C.rss_gb():.1f} GB')
# ---- residual deltas on the gated band
t = time.time(); feat = json.load(open(f'{VR}/features.json')); assert len(feat) == 124
B = pl.read_parquet(f'{H}/data/test_anchor_band.parquet'); assert B.columns[4:] == feat, 'feature order'
X = np.nan_to_num(B.select(feat).to_numpy().astype(np.float32), nan=-1, posinf=1e6, neginf=-1e6)
D = B.select(*K2, 'country', 'p_anchor', 'missing_gate')
for s in (11, 29, 47):
    m = lgb.Booster(model_file=f'{VR}/models/residual_{s}.txt'); assert m.feature_name() == feat, f'model {s} feature names'
    D = D.with_columns(pl.Series(f'delta_{s}', m.predict(X, raw_score=True, num_threads=8).astype(np.float32)))
del X, B; gc.collect()
D = D.with_columns(pl.mean_horizontal('delta_11', 'delta_29', 'delta_47').alias('delta'))
D.write_parquet(f'{H}/data/test_residual_deltas.parquet')
RES['deltas'] = {c: D.select(pl.col(c).mean().alias('mean'), pl.col(c).std().alias('std'), pl.col(c).abs().max().alias('absmax'), (pl.col(c) > 0).mean().alias('pos_share')).to_dicts()[0] for c in ('delta_11', 'delta_29', 'delta_47', 'delta')}
RES['deltas_by_country'] = D.group_by('country').agg(pl.len().alias('n'), pl.col('delta').mean().alias('mean'), pl.col('delta').abs().mean().alias('absmean')).sort('country').to_dicts()
T['residual_predict'] = round(time.time() - t); log('residual deltas', RES['deltas'], f'{time.time()-t:.0f}s')
P = P.join(D.select(*K2, 'p_anchor', 'delta_11', 'delta_29', 'delta_47', 'delta').rename({'p_anchor': 'pa_band'}), on=K2, how='left'); del D
ix = P['delta'].is_not_null().to_numpy(); assert ix.sum() == 3325555, ix.sum()
pa = P['p_anchor'].to_numpy(); assert np.array_equal(P['pa_band'].to_numpy()[ix], pa[ix]), 'band p_anchor != test_v3anchor p'
za = logit(np.clip(pa[ix].astype(float), 1e-6, 1 - 1e-6))
def corr(dcol):
    pn = pa.copy(); pn[ix] = expit(za + P[dcol].to_numpy()[ix]).astype(np.float32); assert np.array_equal(pa[~ix], pn[~ix]); return pn
fr = (P['country'] == 'France').to_numpy()
P = P.with_columns(pl.Series('p_corr', corr('delta')), *[pl.Series(f'p_s{s}', corr(f'delta_{s}')) for s in (11, 29, 47)]).drop('pa_band', 'delta_11', 'delta_29', 'delta_47', 'delta')
P = P.with_columns(pl.Series('p_fr', np.where(fr, pa, P['p_corr'].to_numpy()).astype(np.float32)))
RES['pairs_changed_p'] = {c: int(((P[c] != P['p_anchor']) ).sum()) for c in ('p_corr', 'p_fr', 'p_s11', 'p_s29', 'p_s47')}
log('p columns', RES['pairs_changed_p'], f'RSS {C.rss_gb():.1f} GB')
# ---- decoding machinery
dec_m = lgb.Booster(model_file=f'{E}/results/eval_ENS_v3anchor_R10c_m0.txt'); RES['decoder_md5'] = md5(f'{E}/results/eval_ENS_v3anchor_R10c_m0.txt')
V = PP.decoy_pairs('test').filter(pl.col('kind') == 'ADD').select(K2); RES['veto_pairs'] = V.height
def U_of(pcol):
    return P.select(pl.col('s1_idx').alias('s1'), pl.col('cand_idx').alias('qid'), 'src', pl.col(pcol).alias('p'),
                    pl.when(pl.col(pcol) >= pl.col('p_other') + 0.0).then(1).otherwise(2).alias('qr'))
def decode(pcol):
    t = time.time(); s = PP.decide_fast(U_of(pcol), dec_m, C.TH); log('decode', pcol, s.height, f'{time.time()-t:.0f}s'); return s
def post(sel, pcol):
    sv = sel.select(K2).join(V, on=K2, how='anti'); S_ = sv.join(P.select(*K2, pl.col(pcol).alias('p')), on=K2, how='left'); assert S_['p'].null_count() == 0
    return PP.one_owner(S_).select(K2)
def write(df, path):
    g = df.join(i2, on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').sort().str.join(',').alias('ids'))
    g = i1.join(g, on='s1_idx', how='left').with_columns(pl.col('ids').fill_null('')).sort('s1_idx')
    g.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('ids').alias('matched_entity_ids')).write_csv(path, separator='\t', quote_style='never')
def sig(sel, name):
    s = sel.group_by('s1_idx').agg(pl.col('cand_idx').sort().cast(pl.String).str.join(',').alias(name))
    return i1.select('s1_idx', 'country').join(s, on='s1_idx', how='left').with_columns(pl.col(name).fill_null(''))
# ---- step 0: reproduce the shipped v3anchor file
t = time.time(); a_raw = decode('p_anchor'); a = post(a_raw, 'p_anchor')
os.makedirs(f'{H}/output_v3anchor_repro', exist_ok=True); fa = f'{H}/output_v3anchor_repro/matching_results.tsv'; write(a, fa)
RES['repro'] = dict(pairs_r10c=a_raw.height, pairs_final=a.height, md5=md5(fa), shipped_md5=md5(SHIPPED))
ship = PP.read_sub(SHIPPED).select(K2)
RES['repro'].update(only_ours=a.join(ship, on=K2, how='anti').height, only_shipped=ship.join(a, on=K2, how='anti').height, identical=RES['repro']['md5'] == 'cdc0c756b0c8d24c7589a6ed6f6ad81e')
T['repro'] = round(time.time() - t); log('REPRO', RES['repro'])
json.dump(RES, open(f'{H}/logs/a3_main.json', 'w'), indent=1, default=str)
if not RES['repro']['identical']:
    log('ABORT: reproduction differs from shipped v3anchor'); sys.exit(2)
a_raw_fr = a_raw.join(i1.select('s1_idx', 'country'), on='s1_idx').filter(pl.col('country') == 'France').select(K2); del a_raw; gc.collect()
SA = sig(a, 'sa')
out = {}
def finish(name, sel, pcol_series_name):
    t = time.time(); d = f'{H}/output_{name}'; os.makedirs(d, exist_ok=True); f = f'{d}/matching_results.tsv'; write(sel, f)
    ss = sig(sel, 'sv').join(SA.select('s1_idx', 'sa'), on='s1_idx')
    ch = ss.filter(pl.col('sv') != pl.col('sa'))
    sel.write_parquet(f'{H}/data/sel_{name}.parquet')
    out[name] = dict(pairs=sel.height, md5=md5(f), changed_rows=ch.height, changed_rows_by_country=dict(ch.group_by('country').len().sort('country').iter_rows()),
                     pairs_added_vs_anchor=sel.join(a, on=K2, how='anti').height, pairs_removed_vs_anchor=a.join(sel, on=K2, how='anti').height,
                     stats=PP.test_stats(sel), p_column=pcol_series_name)
    log('VARIANT', name, {k: v for k, v in out[name].items() if k != 'stats'}, f'{time.time()-t:.0f}s')
# V1 residual_all
t = time.time(); s1 = post(decode('p_corr'), 'p_corr'); finish('residual_all', s1, 'p_corr'); T['V1_residual_all'] = round(time.time() - t)
# V2 residual_frfallback (France pairs keep p_anchor -> France S1 decode identical to v3anchor; one-owner over all with p_fr)
t = time.time(); s2r = decode('p_fr'); s2 = post(s2r, 'p_fr')
chk = s2r.join(i1.select('s1_idx', 'country'), on='s1_idx').filter(pl.col('country') == 'France').select(K2)
RES['frfallback_france_raw_decode_symdiff_vs_anchor'] = chk.join(a_raw_fr, on=K2, how='anti').height + a_raw_fr.join(chk, on=K2, how='anti').height; log('France raw decode symdiff', RES['frfallback_france_raw_decode_symdiff_vs_anchor'])
finish('residual_frfallback', s2, 'p_fr'); T['V2_residual_frfallback'] = round(time.time() - t)
# V3 consensus (seed_consensus.py): per-seed post-passed decodes; accept a query's change only if all three seeds give the identical set
t = time.time(); sels = []; Q = i1.select('s1_idx')
for s in (11, 29, 47):
    ss = post(decode(f'p_s{s}'), f'p_s{s}'); sels.append(ss); ss.write_parquet(f'{H}/data/sel_seed_{s}.parquet')
    out[f'seed_{s}'] = dict(pairs=ss.height, changed_rows=int(sig(ss, 'sv').join(SA.select('s1_idx', 'sa'), on='s1_idx').filter(pl.col('sv') != pl.col('sa')).height))
    g = ss.group_by('s1_idx').agg(pl.col('cand_idx').sort().cast(pl.String).str.join(',').alias(f's{s}'))
    Q = Q.join(g, on='s1_idx', how='left').with_columns(pl.col(f's{s}').fill_null(''))
allow = Q.filter((pl.col('s11') == pl.col('s29')) & (pl.col('s11') == pl.col('s47'))).select('s1_idx')
sel = pl.concat([sels[0].join(allow, on='s1_idx', how='semi'), a.join(allow, on='s1_idx', how='anti')])
am = P['s1_idx'].is_in(allow['s1_idx']).to_numpy()
P = P.with_columns(pl.Series('p_cons', np.where(am, P['p_corr'].to_numpy(), pa).astype(np.float32)))
s3 = post(sel, 'p_cons'); RES['consensus_agreed_queries'] = allow.height
# queries where the three seeds agree on a set different from the anchor's
RES['consensus_agreed_changed_queries'] = Q.join(allow, on='s1_idx', how='semi').join(SA.select('s1_idx', 'sa'), on='s1_idx').filter(pl.col('s11') != pl.col('sa')).height
finish('consensus', s3, 'p_cons'); T['V3_consensus'] = round(time.time() - t)
RES['variants'] = out
# ---- per-pair test p for the band stress (s1_idx, country, p)
t = time.time()
for name, col in (('v3anchor', 'p_anchor'), ('residual_all', 'p_corr'), ('residual_frfallback', 'p_fr'), ('consensus', 'p_cons')):
    P.select('s1_idx', pl.col('country').cast(pl.String), pl.col(col).alias('p')).write_parquet(f'{H}/data/test_p_{name}.parquet', compression='lz4')
    RES.setdefault('test_band_share_s1', {})[name] = P.group_by('s1_idx', 'country').agg(((pl.col(col) > 0.2) & (pl.col(col) < 0.8)).any().alias('b')).group_by('country').agg(pl.col('b').mean()).sort('country').to_dicts()
T['test_p_write'] = round(time.time() - t)
RES['seconds'] = round(time.time() - t0); RES['peak_rss_gb'] = round(C.peak_rss_gb(), 1)
json.dump(RES, open(f'{H}/logs/a3_main.json', 'w'), indent=1, default=str); log('DONE', RES['seconds'], 'peak RSS', RES['peak_rss_gb'])
