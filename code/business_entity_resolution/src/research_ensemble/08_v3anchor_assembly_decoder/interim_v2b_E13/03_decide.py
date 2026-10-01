#!/opt/conda/bin/python3
"""B1 steps 3-5: test decision (prod_v2b/08_test.py R10c branch) on p = p3 (E13 band specialist, band rows) / p2 (elsewhere), then the two verified
post-pass rules (v2a_experiments V3d ADD-decoy veto, all countries + V1 strict one-owner by p), TSVs exactly as 08_test.py, official validator.
  exclusivity: competition = v1's p2 over the zero-shot dense table of ALL test S1 (pv2a.stage2.v1_competitors('test', 'p2')) via pv2a.decide.add_comp;
               p_other does not depend on p (max v1 q of the OTHER claimants of the record), so it is computed once.
  U = (s1, qid, src = record id prefix, p, qr = 1 if p >= p_other + margin else 2) -> decide_lib.decide_set(U, decoder)
  checks: (i) the same pipeline with p = p2 + the FROZEN v2b decoder (vectorised decide_fast) reproduces v2b's shipped file exactly;
          (ii) decide_fast(U, decoder) == decide_set(U, decoder); (iii) diagnostics: E13 p3 + frozen decoder, Codex single + frozen decoder.
usage: 03_decide.py [--decoder results/eval_E13_cv2_R10c_m0.txt] [--margin 0] [--p3 data/p3_test.parquet] [--outdir output] [--no-validate]"""
import b1common as C
import os, sys, json, time, argparse, gc, hashlib, subprocess
ap = argparse.ArgumentParser()
ap.add_argument('--decoder', default=f'{C.RESD}/eval_E13_cv2_R10c_m0.txt'); ap.add_argument('--margin', default='0')
ap.add_argument('--p3', default=f'{C.DD}/p3_test.parquet'); ap.add_argument('--outdir', default=C.OD); ap.add_argument('--no-validate', action='store_true')
ap.add_argument('--no-diag', action='store_true'); ap.add_argument('--skip-checks', action='store_true', help='skip (i) baseline reproduction and (ii) decide_fast==decide_set (already proven on the E13 run)'); ap.add_argument('--no-band-check', action='store_true', help='p3 rows need not be exactly the E13 band (e.g. E06 CE block)'); ap.add_argument('--p3col', default='p3'); ap.add_argument('--tag', default='', help="suffix for data/sel_*, logs/03_decide*.json (alternative runs)"); A = ap.parse_args()
import numpy as np, polars as pl, lightgbm as lgb
import pv2b.common  # noqa
from pv2a.stage2 import v1_competitors
from pv2a.decide import add_comp
import decide_lib as DL
import pp_common as PP
log = C.log; K2 = ['s1_idx', 'cand_idx']; t0 = time.time(); os.makedirs(A.outdir, exist_ok=True)
md5 = lambda f: hashlib.md5(open(f, 'rb').read()).hexdigest()
MARGIN = None if A.margin.lower() == 'none' else float(A.margin)
RES = dict(mode=C.MODE, decoder=A.decoder, decoder_md5=md5(A.decoder), margin=A.margin, p3=A.p3, outdir=A.outdir)
log('decoder', A.decoder, 'margin', A.margin, 'p3', A.p3)
i1 = pl.read_parquet(f'{C.EB}/ids/test_s1.parquet', columns=['entity_id', 'country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
i2 = pl.read_parquet(f'{C.EB}/ids/test_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
C.gate('load')
P = pl.read_parquet(C.PREDS, columns=['s1_idx', 'cand_idx', 'country', 'in_dft', 'p1', 'p2']); assert P.height == C.N_PAIRS
B3 = pl.read_parquet(A.p3).select(*K2, pl.col(A.p3col).cast(pl.Float32).alias('p3')); nb = B3.height; RES['p3col'] = A.p3col; RES['tag'] = A.tag
P = P.join(B3, on=K2, how='left'); del B3
band = (pl.col('p2') > 0.001) & (pl.col('p2') < 0.999)
RES['p3_rows'] = nb; RES['p3_rows_outside_band'] = P.filter(pl.col('p3').is_not_null() & ~band).height; RES['band_rows_without_p3'] = P.filter(pl.col('p3').is_null() & band).height
RES['p3_rows_differing_from_p2'] = P.filter(pl.col('p3').is_not_null() & (pl.col('p3') != pl.col('p2'))).height
assert P['p3'].is_not_null().sum() == nb, (P['p3'].is_not_null().sum(), nb)
if not A.no_band_check: assert RES['p3_rows_outside_band'] == 0 and RES['band_rows_without_p3'] == 0, RES
log('p3 coverage', {k: RES[k] for k in ('p3_rows', 'p3_rows_outside_band', 'band_rows_without_p3', 'p3_rows_differing_from_p2')})
P = P.with_columns(pl.coalesce('p3', 'p2').cast(pl.Float32).alias('p')).drop('p1', 'p3')
log('pairs', P.height, 'band rows with p3', nb, f'RSS {C.rss_gb():.1f} GB')
comp = v1_competitors('test', 'p2'); log('v1 competitors', comp.height)
t = time.time(); Pc = add_comp(P.select('s1_idx', 'cand_idx', pl.col('p2').alias('p')), comp).select(*K2, 'p_other'); del comp; gc.collect()
P = P.join(Pc, on=K2, how='left').join(i2.select('cand_idx', pl.col('entity_id').str.slice(0, 2).alias('src')), on='cand_idx', how='left'); del Pc; gc.collect()
assert P['p_other'].null_count() == 0 and P['src'].null_count() == 0
log('competition added', f'{time.time()-t:.0f}s', f'RSS {C.rss_gb():.1f} GB')
def U_of(pcol, margin):
    qr = pl.lit(1, dtype=pl.Int32) if margin is None else pl.when(pl.col(pcol) >= pl.col('p_other') + margin).then(1).otherwise(2)
    return P.select(pl.col('s1_idx').alias('s1'), pl.col('cand_idx').alias('qid'), 'src', pl.col(pcol).alias('p'), qr.alias('qr'))
def symdiff(a, b):
    return dict(only_a=a.join(b, on=K2, how='anti').height, only_b=b.join(a, on=K2, how='anti').height)
frozen = lgb.Booster(model_file=C.FROZEN_DECODER); dec_m = lgb.Booster(model_file=A.decoder)
V2B_SEL = PP.read_sub(f'{C.V2B}/output/matching_results.tsv').select(K2)
# (i) baseline reproduction: p2 + frozen decoder, margin 0
if not A.skip_checks:
    t = time.time(); S0 = PP.decide_fast(U_of('p2', 0.0), frozen, C.TH)
    RES['baseline_reproduction_vs_v2b_file'] = symdiff(S0, V2B_SEL); RES['baseline_reproduction_vs_v2b_file']['pairs'] = S0.height
    log('(i) baseline reproduction (p2 + frozen decoder) vs shipped v2b file', RES['baseline_reproduction_vs_v2b_file'], f'{time.time()-t:.0f}s')
    assert RES['baseline_reproduction_vs_v2b_file']['only_a'] == 0 and RES['baseline_reproduction_vs_v2b_file']['only_b'] == 0
    del S0; gc.collect()
# main decision: decide_lib.decide_set (as 08_test.py)
C.gate('decide'); t = time.time(); U = U_of('p', MARGIN)
dec = DL.decide_set(U, dec_m)
pred = pl.DataFrame([(s, q) for s, qs in dec.items() for q in qs], schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32}, orient='row'); del dec; gc.collect()
RES['decide_set_seconds'] = time.time() - t; log('decision done (decide_set), matches', pred.height, f'{time.time()-t:.0f}s', f'RSS {C.rss_gb():.1f} GB')
if not A.skip_checks:
    t = time.time(); Sf = PP.decide_fast(U, dec_m, C.TH); RES['decide_fast_vs_decide_set'] = symdiff(Sf, pred); log('(ii) decide_fast vs decide_set', RES['decide_fast_vs_decide_set'], f'{time.time()-t:.0f}s')
    del Sf
del U; gc.collect()
pred.write_parquet(f'{C.DD}/sel_r10c{A.tag}.parquet')
ctry = i1.select('s1_idx', 'country')
DPt = PP.decoy_pairs('test'); V = DPt.filter(pl.col('kind') == 'ADD').select(K2)
def postpass(sel, pcol):
    sv = sel.join(V, on=K2, how='anti'); S_ = sv.join(P.select(*K2, pl.col(pcol).alias('p')), on=K2, how='left'); assert S_['p'].null_count() == 0
    return sv, PP.one_owner(S_).select(K2)
if not A.no_diag:   # (iii) decoder / model sensitivity on test (selection level; diagnostics only)
    D = {}
    S = PP.decide_fast(U_of('p', 0.0), frozen, C.TH); D['main_p3_frozen_v2b_decoder_m0'] = dict(pairs=S.height, vs_main=symdiff(S, pred), vs_v2b=symdiff(S, V2B_SEL))
    S.write_parquet(f'{C.DD}/sel_diag_main_p3_frozen{A.tag}.parquet')
    fv = f'{C.DD}/p3_cv2_test.parquet'
    if os.path.exists(fv):   # the brief's original configuration: E13 2-fold mean-of-folds score + the E13 refit decoder (kept as a VARIANT)
        Cx = pl.read_parquet(fv).select(*K2, pl.col('p3').cast(pl.Float32).alias('p_var'))
        P = P.join(Cx, on=K2, how='left').with_columns(pl.coalesce('p_var', 'p2').cast(pl.Float32).alias('p_var')); del Cx
        S = PP.decide_fast(U_of('p_var', MARGIN), dec_m, C.TH); _, Sf_var = postpass(S, 'p_var')
        S.write_parquet(f'{C.DD}/sel_variant_cv2_r10c{A.tag}.parquet'); Sf_var.write_parquet(f'{C.DD}/sel_variant_cv2_final{A.tag}.parquet')
        D['variant_cv2_p3_refit_decoder'] = dict(pairs=S.height, vs_main=symdiff(S, pred), vs_v2b=symdiff(S, V2B_SEL), after_postpass_pairs=Sf_var.height,
                                                 stats=PP.test_stats(Sf_var.join(ctry, on='s1_idx')), france=PP.france_proxies(Sf_var))
        P = P.drop('p_var')
    RES['diagnostics'] = D; log('(iii) diagnostics', D); del S; gc.collect()
# post-pass (a): offset-conditioned ADD-decoy veto (all countries), (b): strict one-owner by p
RES['decoy_words'] = PP.decoy_words(); RES['decoy_pattern_pairs_ADD_in_table'] = V.height
rem_v = pred.join(V, on=K2, how='semi'); sel_v = pred.join(V, on=K2, how='anti')
RES['veto_removed'] = dict(total=rem_v.height, s1=rem_v['s1_idx'].n_unique(), by_country=dict(rem_v.join(ctry, on='s1_idx').group_by('country').len().sort('country').iter_rows()))
log('(a) decoy veto removed', RES['veto_removed'])
S = sel_v.join(P.select(*K2, 'p'), on=K2, how='left'); assert S['p'].null_count() == 0
sel_f = PP.one_owner(S).select(K2); rem_o = sel_v.join(sel_f, on=K2, how='anti')
RES['one_owner_removed'] = dict(total=rem_o.height, records=rem_o['cand_idx'].n_unique(), by_country=dict(rem_o.join(ctry, on='s1_idx').group_by('country').len().sort('country').iter_rows()))
log('(b) one-owner removed', RES['one_owner_removed'])
sel_v.write_parquet(f'{C.DD}/sel_r10c_veto{A.tag}.parquet'); sel_f.write_parquet(f'{C.DD}/sel_final{A.tag}.parquet')
# TSVs exactly as prod_v2b/08_test.py
assert P['s1_idx'].n_unique() == i1.height, 'some test S1 without candidates'
def write(df, path, col):
    g = df.join(i2, on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').sort().str.join(',').alias('ids'))
    g = i1.join(g, on='s1_idx', how='left').with_columns(pl.col('ids').fill_null('')).sort('s1_idx')
    g.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('ids').alias(col)).write_csv(path, separator='\t', quote_style='never')
    return g
t = time.time()
gm = write(sel_f, f'{A.outdir}/matching_results.tsv', 'matched_entity_ids')
write(P.select(K2), f'{A.outdir}/candidate_pairs.tsv', 'candidate_entity_ids'); log('wrote TSVs', A.outdir, f'{time.time()-t:.0f}s')
RES['md5'] = dict(matching=md5(f'{A.outdir}/matching_results.tsv'), candidate=md5(f'{A.outdir}/candidate_pairs.tsv'), v2b_candidate=md5(f'{C.V2B}/output/candidate_pairs.tsv'))
RES['candidate_pairs_identical_to_v2b'] = RES['md5']['candidate'] == RES['md5']['v2b_candidate']; log('md5', RES['md5'], 'candidate identical to v2b:', RES['candidate_pairs_identical_to_v2b'])
# 08_test.py statistics (pc = p = p3/p2)
st = gm.with_columns((pl.col('ids') == '').alias('empty'), pl.when(pl.col('ids') == '').then(0).otherwise(pl.col('ids').str.count_matches(',') + 1).alias('k')) \
       .group_by('country').agg(pl.len().alias('n_s1'), pl.col('empty').mean().alias('empty_share'), pl.col('k').mean().alias('matches_per_s1')).sort('country')
bandx = P.group_by('s1_idx', 'country').agg(((pl.col('p') > 0.2) & (pl.col('p') < 0.8)).any().alias('band'), ((pl.col('p') > 0.02) & (pl.col('p') < 0.99)).sum().alias('nband_ce'))
bs = bandx.group_by('country').agg(pl.col('band').mean().alias('band_share_s1'), pl.col('nband_ce').mean().alias('ce_band_pairs_per_s1')).sort('country')
pairband = P.group_by('country').agg(((pl.col('p') > 0.2) & (pl.col('p') < 0.8)).mean().alias('uncertain_pair_share'), ((pl.col('p2') > 0.2) & (pl.col('p2') < 0.8)).mean().alias('uncertain_pair_share_v2b_p2'), pl.len().alias('pairs')).sort('country')
selx = sel_f.join(P.select(*K2, 'in_dft', 'country', 'p'), on=K2)
sel = selx.group_by('country').agg(pl.len().alias('matches'), (~pl.col('in_dft')).mean().alias('lexical_only_share_of_matches'), pl.col('p').min().alias('min_selected_p')).sort('country')
selband = selx.group_by('s1_idx', 'country').agg(((pl.col('p') > 0.2) & (pl.col('p') < 0.8)).any().alias('sb')).group_by('country').agg(pl.col('sb').sum().alias('n')).sort('country')
RES['stats'] = dict(per_country=st.to_dicts(), band=bs.to_dicts(), pair_band=pairband.to_dicts(), selected=sel.to_dicts(),
                    s1_with_selected_pair_in_band={r['country']: r['n'] / i1.filter(pl.col('country') == r['country']).height for r in selband.to_dicts()},
                    total=dict(n_s1=gm.height, candidate_pairs=P.height, matches=sel_f.height, empty_share=float((gm['ids'] == '').mean()), matches_per_s1=sel_f.height / gm.height,
                               band_share_s1=float(bandx['band'].mean())))
log('stats', RES['stats'])
json.dump(RES, open(f'{C.LD}/03_decide{A.tag}.json', 'w'), indent=1, default=str)
if not A.no_validate:
    t = time.time()
    r = subprocess.run(['/opt/conda/bin/python3', 'utils/validate_submission.py', '--matching', f'{A.outdir}/matching_results.tsv', '--candidate', f'{A.outdir}/candidate_pairs.tsv',
                        '--test-dir', 'dataset/test', '--check-ids'], cwd=C.SR, capture_output=True, text=True)
    open(f'{C.LD}/validate_final{A.tag}.log', 'w').write(r.stdout + r.stderr)
    RES['validator'] = dict(exit=r.returncode, verdict='PASS' if r.returncode == 0 and 'PASS' in r.stdout else 'FAIL', seconds=round(time.time() - t), tail=r.stdout.strip().splitlines()[-6:])
    log('validator', RES['validator'])
RES['seconds'] = time.time() - t0; RES['peak_rss_gb'] = C.peak_rss_gb()
json.dump(RES, open(f'{C.LD}/03_decide{A.tag}.json', 'w'), indent=1, default=str); log('DONE', f'{time.time()-t0:.0f}s', 'peak RSS', round(C.peak_rss_gb(), 1))
