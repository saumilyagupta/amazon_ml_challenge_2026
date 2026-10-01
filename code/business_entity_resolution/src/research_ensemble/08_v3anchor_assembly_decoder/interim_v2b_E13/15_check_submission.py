# Independent checks of <outdir>/matching_results.tsv + candidate_pairs.tsv against the scored test table data/preds_test_<name>.parquet:
#   candidate pairs == exactly the scored union-v2 pairs; matches subset of candidates; S1 rows == test_source1 ids (every country);
#   format (header, tabs, no quotes, sorted comma-joined ids, empty field for singletons); per-country empty share, matches/S1,
#   share of S1 with a pair in 0.2 < p2 < 0.8 (all pairs / selected pairs); S2/S3 records matched to > 1 S1 (diagnostic).
# usage: python3 15_check_submission.py [outdir]   -> logs/15_check_submission.json
# B1 COPY of prod_v2b/15_check_submission.py: output dir + log dir rebound to build/interim_v2b_E13, and the score used for the band / min-selected
# statistics is p = p3 (E13 specialist, band rows of data/p3_test.parquet) else v2b's p2 (column still called 'p2' below).
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1common as B1
import polars as pl
from pv2b.common import B, D, EB, SR, logger, peak_rss_gb
LD = B1.LD
log = logger(); od = sys.argv[1] if len(sys.argv) > 1 else B1.OD; name = 'abc_rob_cv2_all'
P3F = sys.argv[2] if len(sys.argv) > 2 else f'{B1.DD}/p3_test.parquet'; P3C = sys.argv[3] if len(sys.argv) > 3 else 'p3'
res = {'outdir': od, 'p3_file': P3F, 'p3_col': P3C}
def read(path, col):
    raw = open(path, encoding='utf-8').read(); lines = raw.split('\n')
    assert lines[-1] == '' and '"' not in raw and '\r' not in raw, 'trailing newline / quotes / CR'
    hdr = lines[0].split('\t'); rows = [l.split('\t') for l in lines[1:-1]]
    assert all(len(r) == 2 for r in rows), 'every row must have exactly 2 tab-separated fields'
    ids = [r[1] for r in rows]
    assert all((x == '') or (x.split(',') == sorted(set(x.split(',')))) for x in ids), 'ids must be unique, sorted, comma-joined'
    df = pl.DataFrame({'s1': [r[0] for r in rows], col: ids})
    return hdr, df
hm, M = read(f'{od}/matching_results.tsv', 'm'); hc, C = read(f'{od}/candidate_pairs.tsv', 'c'); log('read TSVs', M.height, C.height)
res['headers'] = [hm, hc]
t1 = pl.read_csv(f'{SR}/dataset/test/test_source1.tsv', separator='\t', columns=['entity_id', 'country'], quote_char=None, infer_schema=False)
res['s1_rows'] = dict(matching=M.height, candidate=C.height, test_source1=t1.height, matching_unique=M['s1'].n_unique(),
                      same_set_matching=set(M['s1']) == set(t1['entity_id']), same_set_candidate=set(C['s1']) == set(t1['entity_id']))
ex = lambda df, col: df.filter(pl.col(col) != '').with_columns(pl.col(col).str.split(',')).explode(col).rename({col: 'e'})
Mp = ex(M, 'm'); Cp = ex(C, 'c')
res['pairs'] = dict(matches=Mp.height, candidates=Cp.height, matches_not_in_candidates=Mp.join(Cp, on=['s1', 'e'], how='anti').height,
                    candidate_dup=int(Cp.select(pl.struct('s1', 'e').is_duplicated().sum()).item()))
# scored pairs -> ids
i1 = pl.read_parquet(f'{EB}/ids/test_s1.parquet', columns=['entity_id']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
i2 = pl.read_parquet(f'{EB}/ids/test_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
P = pl.read_parquet(f'{D}/preds_test_{name}.parquet', columns=['s1_idx', 'cand_idx', 'country', 'p2']).join(pl.read_parquet(P3F).select('s1_idx', 'cand_idx', pl.col(P3C).alias('p3')), on=['s1_idx', 'cand_idx'], how='left') \
      .with_columns(pl.coalesce('p3', 'p2').cast(pl.Float32).alias('p2')).drop('p3').join(i1.rename({'entity_id': 's1'}), on='s1_idx').join(i2.rename({'entity_id': 'e'}), on='cand_idx')
res['pairs'].update(scored=P.height, scored_not_in_candidates=P.join(Cp, on=['s1', 'e'], how='anti').height, candidates_not_scored=Cp.join(P, on=['s1', 'e'], how='anti').height)
# per-country stats
g = M.join(t1.rename({'entity_id': 's1'}), on='s1', how='left').with_columns((pl.col('m') == '').alias('empty'),
        pl.when(pl.col('m') == '').then(0).otherwise(pl.col('m').str.count_matches(',') + 1).alias('k'))
band = P.group_by('s1').agg(((pl.col('p2') > 0.2) & (pl.col('p2') < 0.8)).any().alias('band'))
sel = Mp.join(P.select('s1', 'e', 'p2'), on=['s1', 'e'], how='left')
selband = sel.group_by('s1').agg(((pl.col('p2') > 0.2) & (pl.col('p2') < 0.8)).any().alias('selband'), pl.col('p2').min().alias('pmin_sel'))
g = g.join(band, on='s1', how='left').join(selband, on='s1', how='left').with_columns(pl.col('selband').fill_null(False))
agg = [pl.len().alias('n_s1'), pl.col('empty').mean().alias('empty_share'), pl.col('k').mean().alias('matches_per_s1'),
       pl.col('band').mean().alias('s1_share_with_pair_in_0.2_0.8'), pl.col('selband').mean().alias('s1_share_with_selected_pair_in_0.2_0.8'),
       (pl.col('k') == 1).mean().alias('share_k1'), (pl.col('k') >= 7).mean().alias('share_k7plus'), pl.col('k').max().alias('k_max')]
res['per_country'] = g.group_by('country').agg(agg).sort('country').to_dicts(); res['total'] = g.select(agg).to_dicts()[0]
res['k_hist'] = {key[0]: dict(d.group_by('k').len().sort('k').iter_rows()) for key, d in g.group_by('country')}
res['selected_p2_min'] = float(sel['p2'].min()); res['selected_p2_null'] = int(sel['p2'].null_count())
multi = Mp.group_by('e').len().filter(pl.col('len') > 1)
res['records_matched_to_multiple_s1'] = dict(records=multi.height, share_of_matched_records=multi.height / max(Mp['e'].n_unique(), 1))
res['pass'] = bool(res['s1_rows']['same_set_matching'] and res['s1_rows']['same_set_candidate'] and res['s1_rows']['matching'] == t1.height
                   and res['pairs']['matches_not_in_candidates'] == 0 and res['pairs']['scored_not_in_candidates'] == 0 and res['pairs']['candidates_not_scored'] == 0
                   and res['pairs']['candidates'] == P.height and res['pairs']['candidate_dup'] == 0)
for k, v in res.items(): log(k, v)
json.dump(res, open(f'{LD}/15_check_submission.json', 'w'), indent=1, default=str); log('DONE peak RSS GB', peak_rss_gb())
