#!/opt/conda/bin/python3
"""B1 step 6: release-gate statistics of output/matching_results.tsv vs v2b's shipped file and the v2b post-pass file (v2a_experiments/output/v2b_V3dALL_V1):
per-country matches/S1, empty share, match-count profile, one-owner violations (test_stats); France proxies (strict anchors incl. filler-free / filler-added
split and op slices, mined decoys, copy filler) + pseudo-label proxies (P1s, N1/N2, SWAP_GEN/SWAP_ABBR) for all three files and the E13 selection
before the post-pass; rows / pairs changed per country; uncertain-band shares; 10 verbatim added / removed examples vs v2b (p2 -> p3).
usage: 04_gate.py [--outdir output]"""
import b1common as C
import os, sys, json, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--outdir', default=C.OD); ap.add_argument('--tag', default=''); ap.add_argument('--p3', default=f'{C.DD}/p3_test.parquet'); ap.add_argument('--p3col', default='p3'); ap.add_argument('--extra-ref', action='append', default=[], help='label=path of another submission TSV to diff against'); A = ap.parse_args()
import numpy as np, polars as pl
import pp_common as PP
log = C.log; K2 = ['s1_idx', 'cand_idx']; t0 = time.time(); RES = {}
i1 = PP.ids('test', 's1'); ctry = i1.select('s1_idx', 'country')
FILES = {'v2b': f'{C.V2B}/output/matching_results.tsv', 'v2b_postpass': f'{C.XA}/output/v2b_V3dALL_V1/matching_results.tsv', 'B1_final': f'{A.outdir}/matching_results.tsv'}
for e in A.extra_ref: FILES[e.split('=', 1)[0]] = e.split('=', 1)[1]
SEL = {k: PP.read_sub(v).select(K2) for k, v in FILES.items()}
SEL['B1_before_postpass'] = pl.read_parquet(f'{C.DD}/sel_r10c{A.tag}.parquet').select(K2)
chk = pl.read_parquet(f'{C.DD}/sel_final{A.tag}.parquet').select(K2)
RES['final_tsv_vs_sel_final_parquet'] = dict(only_tsv=SEL['B1_final'].join(chk, on=K2, how='anti').height, only_parquet=chk.join(SEL['B1_final'], on=K2, how='anti').height)
log('final TSV vs sel_final.parquet', RES['final_tsv_vs_sel_final_parquet'])
TRAIN_PROFILE = [5.6, 5.4, 17, 24, 22, 14.6, 7.5, 2.9, 1.0]
def profile(S):
    k = i1.join(S.group_by('s1_idx').agg(pl.len().alias('k')), on='s1_idx', how='left').with_columns(pl.col('k').fill_null(0)); out = {}
    for c in ('US', 'India', 'France', 'all'):
        kc = k if c == 'all' else k.filter(pl.col('country') == c)
        out[c] = [round(100 * float((kc['k'] == j).mean()), 2) for j in range(8)] + [round(100 * float((kc['k'] >= 8).mean()), 2)]
    return out
for nm, S in SEL.items():
    t = time.time()
    RES[nm] = dict(pairs=S.height, stats=PP.test_stats(S.join(ctry, on='s1_idx')), france=PP.france_proxies(S), pseudo=PP.extra_proxies(S), profile_pct_0_to_8plus=profile(S))
    log(nm, f'{time.time()-t:.0f}s', json.dumps({k: v for k, v in RES[nm].items() if k != 'profile_pct_0_to_8plus'}, default=str)[:1800])
RES['train_profile'] = TRAIN_PROFILE
# rows (S1 lines) and pairs changed per country
def rows(path):
    d = pl.read_csv(path, separator='\t', quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True)
    return d.rename({d.columns[0]: 'entity_id', d.columns[1]: 'ids'}).with_columns(pl.col('ids').fill_null(''))
Rf = rows(FILES['B1_final'])
for ref in [k for k in FILES if k != 'B1_final']:
    J = rows(FILES[ref]).join(Rf, on='entity_id', how='full', coalesce=True, suffix='_new').join(i1.select('entity_id', 'country'), on='entity_id', how='left')
    assert J.height == C.N_S1 and J['ids'].null_count() == 0 and J['ids_new'].null_count() == 0
    J = J.with_columns((pl.col('ids') != pl.col('ids_new')).alias('chg'))
    a = SEL['B1_final'].join(SEL[ref], on=K2, how='anti').join(ctry, on='s1_idx'); r = SEL[ref].join(SEL['B1_final'], on=K2, how='anti').join(ctry, on='s1_idx')
    RES[f'changed_vs_{ref}'] = dict(rows_changed=int(J['chg'].sum()), rows_changed_share=float(J['chg'].mean()),
                                   rows_changed_by_country={c['country']: dict(n=c['n'], share=round(c['share'], 5)) for c in J.group_by('country').agg(pl.col('chg').sum().alias('n'), pl.col('chg').mean().alias('share')).to_dicts()},
                                   pairs_added=a.height, pairs_removed=r.height,
                                   added_by_country=dict(a.group_by('country').len().sort('country').iter_rows()), removed_by_country=dict(r.group_by('country').len().sort('country').iter_rows()),
                                   empty_rows_ref=int((J['ids'] == '').sum()), empty_rows_new=int((J['ids_new'] == '').sum()))
    log(f'changed vs {ref}', RES[f'changed_vs_{ref}'])
# band-score effect alone (selection before the post-pass) vs v2b, per country
a = SEL['B1_before_postpass'].join(SEL['v2b'], on=K2, how='anti').join(ctry, on='s1_idx'); r = SEL['v2b'].join(SEL['B1_before_postpass'], on=K2, how='anti').join(ctry, on='s1_idx')
RES['before_postpass_vs_v2b'] = dict(added=dict(a.group_by('country').len().sort('country').iter_rows()), removed=dict(r.group_by('country').len().sort('country').iter_rows()),
                                         changed_s1=pl.concat([a.select('s1_idx'), r.select('s1_idx')])['s1_idx'].n_unique(),
                                         changed_s1_by_country=dict(pl.concat([a.select('s1_idx', 'country'), r.select('s1_idx', 'country')]).unique().group_by('country').len().sort('country').iter_rows()))
log('B1 selection before post-pass vs v2b', RES['before_postpass_vs_v2b'])
# examples: 10 added, 10 removed vs v2b (verbatim S1 / record text), with p2 -> p3 and the removal cause
P3 = pl.read_parquet(A.p3).select(*K2, pl.col(A.p3col).alias('p3'))
Pp = pl.scan_parquet(C.PREDS).select(*K2, 'p2')
rec = pl.read_parquet(f'{C.W}/features/explainer/cache/records_test.parquet', columns=['entity_id', 'name', 'addr'])
i2 = PP.ids('test', 's23').select('cand_idx', pl.col('entity_id').alias('cand_id'))
veto_rm = SEL['B1_before_postpass'].join(pl.read_parquet(f'{C.DD}/sel_r10c_veto{A.tag}.parquet'), on=K2, how='anti').with_columns(pl.lit('decoy veto').alias('cause'))
oo_rm = pl.read_parquet(f'{C.DD}/sel_r10c_veto{A.tag}.parquet').join(SEL['B1_final'], on=K2, how='anti').with_columns(pl.lit('one-owner').alias('cause'))
def ex(D, n, seed):
    D = D.join(ctry, on='s1_idx')
    parts = [D.filter(pl.col('country') == c).sample(min(k, D.filter(pl.col('country') == c).height), seed=seed) for c, k in (('US', n // 3 + n % 3), ('India', n // 3), ('France', n // 3))]
    X = pl.concat(parts).join(Pp.join(D.select(K2).lazy(), on=K2, how='semi').collect(), on=K2, how='left').join(P3, on=K2, how='left').join(i1.select('s1_idx', pl.col('entity_id').alias('s1_id')), on='s1_idx').join(i2, on='cand_idx')
    X = X.join(rec.select(pl.col('entity_id').alias('s1_id'), pl.col('name').alias('s1_name'), pl.col('addr').alias('s1_addr')), on='s1_id', how='left') \
         .join(rec.select(pl.col('entity_id').alias('cand_id'), pl.col('name').alias('rec_name'), pl.col('addr').alias('rec_addr')), on='cand_id', how='left')
    return X.select('country', 's1_id', 's1_name', 's1_addr', 'cand_id', 'rec_name', 'rec_addr', 'p2', 'p3', *(['cause'] if 'cause' in X.columns else [])).to_dicts()
added = SEL['B1_final'].join(SEL['v2b'], on=K2, how='anti')
removed_e13 = SEL['v2b'].join(SEL['B1_before_postpass'], on=K2, how='anti').with_columns(pl.lit('band score (p3)').alias('cause'))
RES['examples_added'] = ex(added, 10, 1); RES['examples_removed_E13'] = ex(removed_e13, 10, 2)
RES['examples_removed_postpass'] = ex(pl.concat([veto_rm.join(SEL['v2b'], on=K2, how='semi'), oo_rm.join(SEL['v2b'], on=K2, how='semi')]), 4, 3)
RES['removed_vs_v2b_by_cause'] = dict(E13_score=removed_e13.height, decoy_veto_of_v2b_pairs=veto_rm.join(SEL['v2b'], on=K2, how='semi').height,
                                      decoy_veto_of_new_pairs=veto_rm.join(SEL['v2b'], on=K2, how='anti').height,
                                      one_owner_of_v2b_pairs=oo_rm.join(SEL['v2b'], on=K2, how='semi').height, one_owner_of_new_pairs=oo_rm.join(SEL['v2b'], on=K2, how='anti').height)
log('removed vs v2b by cause', RES['removed_vs_v2b_by_cause'])
for k in ('examples_added', 'examples_removed_E13', 'examples_removed_postpass'):
    for e in RES[k]: log(k, e)
RES['seconds'] = time.time() - t0; RES['peak_rss_gb'] = C.peak_rss_gb()
json.dump(RES, open(f'{C.LD}/04_gate{A.tag}.json', 'w'), indent=1, default=str); log('DONE', f'{time.time()-t0:.0f}s', 'peak RSS', round(C.peak_rss_gb(), 1))
