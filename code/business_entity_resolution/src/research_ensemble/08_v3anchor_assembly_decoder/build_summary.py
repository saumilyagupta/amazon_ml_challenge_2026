#!/opt/conda/bin/python3
"""BUILD: condense the B1-chain JSONs of one ensemble test-file build into build/<name>/SUMMARY.json (read-only on everything else).
usage: build_summary.py <name> <wall_seconds> <rc_decide> <rc_check> <rc_gate> <start_utc> <end_utc>"""
import sys, os, json, hashlib
sys.dont_write_bytecode = True
H = '/workspace/saumilya/amazon-ml/work/matching/ensemble_v1'
name, wall, rc_d, rc_c, rc_g, t_start, t_end = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
BH = f'{H}/build/{name}'; LD = f'{BH}/logs'; OD = f'{BH}/output'
V3F = '/workspace/saumilya/amazon-ml/work/matching/prod_v3/output/matching_results.tsv'
V2B_CAND_MD5_PREFIX = '56ce4634b217'
def load(f):
    try: return json.load(open(f))
    except Exception as e: return {'_missing': f'{f}: {e}'}
def md5(f):
    h = hashlib.md5()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
D = load(f'{LD}/03_decide.json'); K = load(f'{LD}/15_check_submission.json'); G = load(f'{LD}/04_gate.json')
S = dict(name=name, matching_results=f'{OD}/matching_results.tsv', candidate_pairs=f'{OD}/candidate_pairs.tsv',
         p_file=f'{H}/data/test_{name}.parquet', decoder=D.get('decoder'), decoder_md5=D.get('decoder_md5'), margin=D.get('margin'),
         start_utc=t_start, end_utc=t_end, wall_seconds=round(wall), wall_minutes=round(wall / 60, 1),
         return_codes={'03_decide': rc_d, '15_check_submission': rc_c, '04_gate': rc_g},
         decide_seconds=round(D.get('seconds', -1)), decide_peak_rss_gb=round(D.get('peak_rss_gb', -1), 1))
S['validator'] = {k: D.get('validator', {}).get(k) for k in ('verdict', 'exit', 'seconds', 'tail')}
md = D.get('md5', {})
S['md5'] = dict(matching=md.get('matching'), candidate=md.get('candidate'), v2b_candidate=md.get('v2b_candidate'),
                candidate_identical_to_v2b=D.get('candidate_pairs_identical_to_v2b'),
                v2b_candidate_md5_prefix_expected=V2B_CAND_MD5_PREFIX, v2b_candidate_prefix_ok=str(md.get('v2b_candidate', '')).startswith(V2B_CAND_MD5_PREFIX))
S['sanity'] = dict(baseline_reproduction_vs_v2b_file=D.get('baseline_reproduction_vs_v2b_file'), decide_fast_vs_decide_set=D.get('decide_fast_vs_decide_set'),
                   p_rows=D.get('p3_rows'), p_rows_differing_from_p2=D.get('p3_rows_differing_from_p2'),
                   check15_pass=K.get('pass'), check15_pairs=K.get('pairs'), check15_s1_rows=K.get('s1_rows'),
                   final_tsv_vs_sel_final_parquet=G.get('final_tsv_vs_sel_final_parquet'))
st = D.get('stats', {}); tot = st.get('total', {})
S['total'] = dict(n_s1=tot.get('n_s1'), matches=tot.get('matches'), matches_per_s1=round(tot.get('matches_per_s1', float('nan')), 5),
                  empty_share=round(tot.get('empty_share', float('nan')), 5), candidate_pairs=tot.get('candidate_pairs'))
gs = G.get('B1_final', {}).get('stats', {})
pc = {}
for r in st.get('per_country', []):
    c = r['country']
    pc[c] = dict(n_s1=r['n_s1'], matches=gs.get(c, {}).get('matches'), matches_per_s1=round(r['matches_per_s1'], 5), empty_share=round(r['empty_share'], 5),
                 one_owner_violating_records=gs.get(c, {}).get('multi_assigned_records'))
S['per_country'] = pc
S['one_owner_violations'] = dict(gate_all_countries=gs.get('all', {}).get('multi_assigned_records'),
                                 check15_records_matched_to_multiple_s1=K.get('records_matched_to_multiple_s1', {}).get('records'))
S['postpass_removed'] = dict(decoy_veto=D.get('veto_removed'), one_owner=D.get('one_owner_removed'))
FK = ('strict_anchor_n', 'strict_anchor_accept', 'strict_anchor_filler_free_accept', 'strict_anchor_filler_added_accept', 'decoy_accept', 'copy_filler_accept',
      'france_one_owner_violating_records')
S['france_proxies'] = {ref: {k: G.get(ref, {}).get('france', {}).get(k) for k in FK}
                       for ref in ('B1_final', 'B1_before_postpass', 'v3', 'v3x_ash', 'v2b', 'v2b_postpass') if ref in G}
S['france_proxies_note'] = 'decoy_accept / copy_filler_accept = (n pairs, acceptance share, accepted count); lower decoy, higher copy_filler/strict = better'
for ref in ('v3', 'v2b', 'v3x_ash'):
    c = G.get(f'changed_vs_{ref}')
    if c: S[f'changed_vs_{ref}'] = {k: c.get(k) for k in ('rows_changed', 'rows_changed_share', 'rows_changed_by_country', 'pairs_added', 'pairs_removed',
                                                          'added_by_country', 'removed_by_country', 'empty_rows_ref', 'empty_rows_new')}
S['v3_reference_file'] = dict(path=V3F, md5=md5(V3F) if os.path.exists(V3F) else None)
S['uncertain_band'] = dict(band=st.get('band'), pair_band=st.get('pair_band'))
S['ok'] = bool(S['validator'].get('verdict') == 'PASS' and S['md5']['candidate_identical_to_v2b'] and K.get('pass') is True
               and (S['one_owner_violations']['gate_all_countries'] in (0, None)) and rc_d == 0)
json.dump(S, open(f'{BH}/SUMMARY.json', 'w'), indent=1, default=str)
fp = S['france_proxies'].get('B1_final', {}); cv = S.get('changed_vs_v3', {})
line = (f"build {name} {'OK' if S['ok'] else 'NOT-OK'}: validator {S['validator'].get('verdict')}, cand==v2b {S['md5']['candidate_identical_to_v2b']}, "
        f"matches {S['total']['matches']} ({S['total']['matches_per_s1']}/S1, empty {S['total']['empty_share']}), "
        + ', '.join(f"{c} {v['matches_per_s1']}/{v['empty_share']}" for c, v in pc.items())
        + f"; one-owner viol {S['one_owner_violations']['gate_all_countries']}; FR strict {fp.get('strict_anchor_accept')} decoy {fp.get('decoy_accept')} copy {fp.get('copy_filler_accept')}; "
        f"rows chg vs v3 {cv.get('rows_changed')} {cv.get('rows_changed_by_country')}; wall {S['wall_minutes']} min; {S['matching_results']} md5 {S['md5']['matching']}")
print(line)
