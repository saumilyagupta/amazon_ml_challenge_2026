#!/opt/conda/bin/python3
"""Round 7: decision-level hybrids of the iterate_20260926 file nc_specialist_legal_fr (= blend_e0.25 + empty-address specialist + France CTAG/LEG3/LEG12 cleanup). Base selection = its selected.parquet. Variants are removal-only (France) except where noted. Adapted from round 6: (v2shash round_c; best single on val / band stress / IE-3 US-India leg) with
(a) the internal_eval France decoy-family vetoes (mirror-controlled; validated on vr2 by that session) and (b) vr2 as a second opinion restricted to the
decoy classes. All variants are REMOVAL-ONLY subsets of blend_e0.25's decision (one-owner preserved; candidate_pairs unchanged = 56ce4634b217).
Writes build/<name>/output/matching_results.tsv + data/sel_<name>.parquet + val-side selections for US/India-touching variants + results/hybrids.json."""
import os, sys, json, time
IE = '/workspace/saumilya/amazon-ml/work/internal_eval'; sys.path.insert(0, f'{IE}/src'); from ie_common import *
envcap(8)
import polars as pl
V6 = '/workspace/saumilya/amazon-ml/work/matching/ensemble_v7'; W = '/workspace/saumilya/amazon-ml/work/matching'
T0 = time.time(); log = lambda *a: print(f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s]', *a, flush=True)
NC = pl.read_parquet(f'{W}/iterate_20260926/submissions/nc_specialist_legal_fr/selected.parquet').select(K2).with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32)).unique()
VR = pl.read_parquet(f'{W}/vr2_stress_submission/data/sel_consensus.parquet').select(K2).with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32)).unique()
log('blend sel', NC.height, 'vr2 sel', VR.height)
FAM = pl.read_parquet(f'{DATA}/pc_test_fam.parquet').with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
SW = pl.read_parquet(f'{DATA}/pc_test_pl.parquet', columns=K2 + ['country', 'swap']).filter(pl.col('swap') == 'SWAP_GEN').with_columns(pl.col('country').cast(pl.String), pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
FS = pl.read_parquet(f'{DATA}/pc_test_frstrict.parquet').select(K2).with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
ALL = ['US', 'India', 'France']
def fam(f, side, countries): return FAM.filter((pl.col('fam') == f) & (pl.col('side') == side) & pl.col('country').is_in(countries)).select(K2)
sel = NC.with_columns(pl.lit(True).alias('sel')); vsel = VR.with_columns(pl.lit(True).alias('vsel'))
RULES = {'ADDW_all': (fam('ADDW', '+', ALL), fam('ADDW', '-', ALL)), 'CTAG_FR': (fam('CTAG', '+', ['France']), fam('CTAG', '-', ['France'])),
         'LEG3_FR': (fam('LEG3', '+', ['France']), fam('LEG3', '-', ['France'])), 'LEG3_US': (fam('LEG3', '+', ['US']), fam('LEG3', '-', ['US'])),
         'LEG12_FR': (fam('LEG12', '+', ['France']), fam('LEG12', '-', ['France'])), 'EQ12_FR': (fam('EQ12', '+', ['France']), fam('EQ12', '-', ['France'])),
         'EQ3_FR': (fam('EQ3', '+', ['France']), fam('EQ3', '-', ['France']))}
FSsel = FS.join(NC, on=K2, how='semi').select('s1_idx').unique()
RULES['SWAPGEN_FR_anch'] = (SW.filter(pl.col('country') == 'France').select(K2).join(FSsel, on='s1_idx', how='semi'), pl.DataFrame(schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32}))
m = 3.46; D_MISS = 1 - 5 * (m - 1) / (5 * (m - 1) + 1); D_FP = 1 - 5 * m / (5 * m + 4)
MIR = {}
for k, (plus, minus) in RULES.items():
    a = plus.join(sel, on=K2, how='left')['sel'].fill_null(False); b = minus.join(sel, on=K2, how='left')['sel'].fill_null(False)
    r = dict(plus_pairs=plus.height, plus_selected=int(a.sum()), minus_pairs=minus.height, minus_selected=int(b.sum()))
    c = 'US' if k.endswith('_US') else 'France'; n_s1 = N_TEST[c] if k != 'ADDW_all' else N_TEST['France']
    t = 0.092 * r['plus_selected'] if k.startswith('SWAPGEN') else (0.0 if k == 'ADDW_all' else min(r['plus_selected'], r['minus_selected']))
    r['E_true_removed'] = t; r['E_false_removed'] = r['plus_selected'] - t; r['dF_country'] = ((r['plus_selected'] - t) * D_FP - t * D_MISS) / n_s1; r['dLB'] = MIX[c] * r['dF_country']
    MIR[k] = r; log(f"mirror {k}: +k {r['plus_selected']}/{r['plus_pairs']}  -k {r['minus_selected']}/{r['minus_pairs']}  E[true] {t:.0f}  dLB {r['dLB']:+.6f}")
# vr2 second opinion on decoy classes: pairs in the +k decoy families that blend selected but vr2 did not
DEC_FR = pl.concat([fam(f, '+', ['France']) for f in ('CTAG', 'LEG3', 'LEG12', 'EQ12', 'EQ3', 'ADDW', 'ADDO')]).unique()
DEC_ALL = pl.concat([fam(f, '+', ALL) for f in ('CTAG', 'LEG3', 'LEG12', 'EQ12', 'EQ3', 'ADDW', 'ADDO')]).unique()
def not_in_vr2(dec): return dec.join(NC, on=K2, how='semi').join(VR, on=K2, how='anti')
RULES['SWAPGEN_FR_full'] = (SW.filter(pl.col('country') == 'France').select(K2), pl.DataFrame(schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32}))
RULES['LEG3_US_anch'] = (RULES['LEG3_US'][0], RULES['LEG3_US'][1])
VAR = {'spec_swapanch_FR': ['SWAPGEN_FR_anch'],                      # + anchored France SWAP_GEN guard (round-6 ppD lever)
       'spec_swapfull_FR': ['SWAPGEN_FR_full'],                      # + full France SWAP_GEN guard (no anchor condition; riskier)
       'spec_vr2hyb_FR': ['__hyb_FR'],                               # vr2 as second opinion on the remaining France +k decoy-family pairs
       'spec_swapanch_vr2hyb_FR': ['SWAPGEN_FR_anch', '__hyb_FR'],
       'spec_residual_check_FR': ['CTAG_FR', 'LEG3_FR', 'LEG12_FR']}  # control: the base's own rules re-applied (must remove 0)
i1 = ids('test', 's1'); i2 = ids('test', 's23').select('cand_idx', 'entity_id')
def write_tsv(S, path):
    g = S.join(i2, on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').sort().str.join(',').alias('ids'))
    g = i1.join(g, on='s1_idx', how='left').with_columns(pl.col('ids').fill_null('')).sort('s1_idx')
    g.select(pl.col('entity_id').alias('source1_entity_id'), pl.col('ids').alias('matched_entity_ids')).write_csv(path, separator='\t', quote_style='never')
OUT = {'mirror': MIR, 'variants': {}}
for name, rules in VAR.items():
    cur = NC; removed = {}
    for r in rules:
        rem = not_in_vr2(DEC_FR) if r == '__hyb_FR' else (not_in_vr2(DEC_ALL) if r == '__hyb_all' else RULES[r][0])
        nxt = cur.join(rem, on=K2, how='anti'); removed[r] = cur.height - nxt.height; cur = nxt
    by_c = dict(NC.join(cur, on=K2, how='anti').join(i1.select('s1_idx', 'country'), on='s1_idx').group_by('country').len().sort('country').iter_rows())
    os.makedirs(f'{V6}/build/{name}/output', exist_ok=True); cur.write_parquet(f'{V6}/data/sel_{name}.parquet'); write_tsv(cur, f'{V6}/build/{name}/output/matching_results.tsv')
    mo = cur.group_by('cand_idx').len().filter(pl.col('len') > 1).height
    OUT['variants'][name] = dict(rules=rules, pairs=cur.height, removed=removed, removed_by_country=by_c, s1_changed=int(NC.join(cur, on=K2, how='anti')['s1_idx'].n_unique()), multi_owner_records=mo,
                                 expected_dLB_mirror=sum(MIR[r]['dLB'] for r in rules if r in MIR), tsv=f'{V6}/build/{name}/output/matching_results.tsv')
    log(name, OUT['variants'][name])
json.dump(OUT, open(f'{V6}/results/hybrids.json', 'w'), indent=1, default=str); log('DONE'); raise SystemExit(0)
# val-side selections (train index space) for the variants that touch US/India (val has no France): blend val selection minus the same classes
VNC = pl.read_parquet(f'{W}/v2shash/round_c/data/ordinary_blend_e0.25_selected.parquet').select(K2).with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
VVR = pl.read_parquet(f'{W}/variance_research_round2_20260926/data/seed_consensus_selected.parquet').select(K2).with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
VFAM = pl.read_parquet(f'{DATA}/pc_val_fam.parquet').with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
def vfam(f, side, countries): return VFAM.filter((pl.col('fam') == f) & (pl.col('side') == side) & pl.col('country').is_in(countries)).select(K2)
vdec_all = pl.concat([vfam(f, '+', ['US', 'India']) for f in ('CTAG', 'LEG3', 'LEG12', 'EQ12', 'EQ3', 'ADDW', 'ADDO')]).unique()
VRULE = {'ADDW_all': vfam('ADDW', '+', ['US', 'India']), 'LEG3_US': vfam('LEG3', '+', ['US'])}
for name, rules in VAR.items():
    cur = VNC
    for r in rules:
        if r in VRULE: cur = cur.join(VRULE[r], on=K2, how='anti')
        if r == '__hyb_all': cur = cur.join(vdec_all.join(VNC, on=K2, how='semi').join(VVR, on=K2, how='anti'), on=K2, how='anti')
    cur.write_parquet(f'{V6}/data/val_sel_{name}.parquet'); OUT['variants'][name]['val_pairs'] = cur.height; OUT['variants'][name]['val_pairs_removed'] = VNC.height - cur.height
    # France-only variants: val identical to blend_e0.25's (0.991305) by construction
json.dump(OUT, open(f'{V6}/results/hybrids.json', 'w'), indent=1, default=str); log('DONE')
