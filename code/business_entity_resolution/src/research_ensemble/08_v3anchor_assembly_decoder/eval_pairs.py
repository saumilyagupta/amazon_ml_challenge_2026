#!/opt/conda/bin/python3
"""v2c shared evaluator (BRIEF §2). One protocol for every experiment: R10c m0 set-decoder refit on the SAMPLE's out-of-fold scores
(competition = v1's p2 over the zero-shot dense table of ALL train S1, exactly as prod_v2b/07_decide.py), decision on the 220,730 val S1,
slices, loss decomposition, paired bootstrap vs v2b's final val decision; optionally the same decoder applied at test density on the
178,891 density_val queries (competition = v1dense_dens p2) with a paired bootstrap vs v2b at density (0.98843).
usage:
  eval_pairs.py --preds P.parquet --tag E0x_name [--pcol p2] [--density Pd.parquet] [--decoder model.txt] [--threads 6] [--no-sweep]
  eval_pairs.py --selected val_selected.parquet --tag E0x_name          # decoder-only experiments: evaluate a given val selection
P: s1_idx, cand_idx, label, grp, country, in_dft, p1, <pcol>  for ALL sample rows (OOF scores) + ALL val rows (fold average).
Pd: s1_idx, cand_idx, <pcol> [, label] for the union_dens pairs of the density queries.
Outputs (results/): eval_<tag>.json, eval_<tag>_val_rows.parquet (s1_idx, f), eval_<tag>_val_selected.parquet, eval_<tag>_R10c_m0.txt (the refit decoder),
and with --density eval_<tag>_dens_rows.parquet / eval_<tag>_dens_selected.parquet."""
import os, sys, json, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument('--preds'); ap.add_argument('--selected'); ap.add_argument('--tag', required=True); ap.add_argument('--pcol', default='p2')
ap.add_argument('--density'); ap.add_argument('--selected-dens', help='evaluate a given density selection (s1_idx, cand_idx) on the 178,891 queries'); ap.add_argument('--decoder'); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--no-sweep', action='store_true')
ap.add_argument('--margin', default='0', help="exclusivity margin for the R10c eligibility: float (v2b = 0) or 'none' (no exclusivity)"); ap.add_argument('--outdir', default='/workspace/saumilya/amazon-ml/work/matching/prod_v2c/results')
A = ap.parse_args()
sys.path.insert(0, '/workspace/saumilya/amazon-ml/work/matching/prod_v2b')
from pv2b.common import envcap; envcap(A.threads)
import numpy as np, polars as pl, lightgbm as lgb
from pv2b.common import D, V1, EB, SPL, logger, peak_rss_gb
from pv2b.evalx import truth_tables, comp_train, lam_q0, paired, TAUS
from pv2a.decide import add_comp
from pv1.decide import macro, policy, report, efs_choose
import decide_lib as DL
log = logger(); T0 = time.time()
V2B = '/workspace/saumilya/amazon-ml/work/matching/prod_v2b'; DV = '/workspace/saumilya/amazon-ml/work/matching/density_val'
os.makedirs(A.outdir, exist_ok=True); out = lambda s: f'{A.outdir}/eval_{A.tag}{s}'
MV, MS = truth_tables(); lam, q0 = lam_q0()
i2 = pl.read_parquet(f'{EB}/ids/train_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
SRC = i2.select('cand_idx', pl.col('entity_id').str.slice(0, 2).alias('src'))
RES = {'tag': A.tag, 'pcol': A.pcol, 'preds': A.preds, 'selected': A.selected, 'density': A.density, 'decoder_in': A.decoder}


MARGIN = None if str(A.margin).lower() == 'none' else float(A.margin)
RES['margin'] = A.margin


def U_of(Pl, margin='default'):
    m = MARGIN if margin == 'default' else margin
    qr = pl.lit(1, dtype=pl.Int32) if m is None else pl.when(pl.col('p') >= pl.col('p_other') + m).then(1).otherwise(2)
    return Pl.select(pl.col('s1_idx').alias('s1'), pl.col('cand_idx').alias('qid'), 'src', 'p', qr.alias('qr'))


def gt_idx(M, Pl):
    tr = Pl.filter(pl.col('label') == 1).group_by('s1_idx').agg(pl.col('cand_idx'))
    d = {s: set(c) for s, c in tr.iter_rows()}
    return {s: d.get(s, set()) | {-(k + 1) for k in range(m - len(d.get(s, set())))} for s, m in M.select('s1_idx', 'm').iter_rows()}


def to_pred(dec, Pl):
    rows = [(s, q) for s, qs in dec.items() for q in qs]
    pr = pl.DataFrame(rows, schema={'s1_idx': pl.Int32, 'cand_idx': pl.Int32}, orient='row')
    return pr.join(Pl.select('s1_idx', 'cand_idx', 'label'), on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('label').fill_null(0))


def evaluate(pred, M, base_rows, extra_slices=True):
    """pred: s1_idx, cand_idx, label. M: s1_idx, m, r, country (+ is_locked, ncand). Returns (dict, R)."""
    pred = pred.join(M.select('s1_idx'), on='s1_idx', how='semi')
    rep, R = report(pred.select('s1_idx', 'cand_idx', 'label'), M.select('s1_idx', 'm', 'r', 'country'))
    rep['empty_row_share'] = 1 - rep['frac_pred_nonempty']; n = M.height
    rep['fp_per_1000_s1'] = 1000 * float((R['k'] - R['t']).sum()) / n
    rep['missed_truths_per_1000_s1'] = 1000 * float((R['m'] - R['t']).sum()) / n
    if extra_slices:
        Rx = R.join(M.select('s1_idx', *[c for c in ('is_locked', 'ncand') if c in M.columns]), on='s1_idx', how='left')
        if 'is_locked' in Rx.columns:
            rep['locked30k'] = float(Rx.filter(pl.col('is_locked'))['f'].mean()); rep['tune190k'] = float(Rx.filter(~pl.col('is_locked'))['f'].mean())
        if 'ncand' in Rx.columns:
            rep['dense_half'] = float(Rx.filter(pl.col('ncand') >= 48)['f'].mean()); rep['sparse_half'] = float(Rx.filter(pl.col('ncand') < 48)['f'].mean())
    for c in ('US', 'India'):
        Rc = R.filter(pl.col('country') == c)
        if Rc.height: rep[f'matches_per_s1_{c}'] = float(Rc['k'].mean()); rep[f'empty_row_share_{c}'] = float((Rc['k'] == 0).mean())
    if base_rows is not None:
        rep['paired_vs_v2b'] = paired(R.select('s1_idx', 'f'), base_rows)
    return rep, R


# ------------------------------------------------------------------ val
V2B_ROWS = pl.read_parquet(f'{V2B}/output/val_abc_rob_cv2_all_p2/val_row_scores.parquet', columns=['s1_idx', 'f'])
if A.selected:
    S = pl.read_parquet(A.selected, columns=['s1_idx', 'cand_idx']).with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
    L = pl.read_parquet(f'{D}/union_train_lab.parquet', columns=['s1_idx', 'cand_idx', 'label', 'grp']).filter(pl.col('grp') != 'sample')
    pred = S.join(L.select('s1_idx', 'cand_idx', pl.col('label').cast(pl.Int8)), on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('label').fill_null(0))
    RES['note'] = 'selection evaluated as given (no decoder refit)'; sd = None
else:
    cols = ['s1_idx', 'cand_idx', 'label', 'grp', 'country', 'in_dft', A.pcol] + (['p1'] if A.pcol != 'p1' else [])
    sch = pl.read_parquet_schema(A.preds); cols = [c for c in cols if c in sch]
    P = pl.read_parquet(A.preds, columns=cols).rename({A.pcol: 'p'})
    assert P['p'].null_count() == 0, 'null probabilities'
    comp = comp_train('p2')
    PA = add_comp(P, comp).join(SRC, on='cand_idx', how='left'); del P
    Pv = PA.filter(pl.col('grp') != 'sample'); Ps = PA.filter(pl.col('grp') == 'sample'); del PA
    log(f'val rows {Pv.height} S1 {Pv["s1_idx"].n_unique()}; sample rows {Ps.height} S1 {Ps["s1_idx"].n_unique()}')
    assert Pv['s1_idx'].n_unique() == 220730, Pv['s1_idx'].n_unique()
    y = Pv['label'].to_numpy(); pp = Pv['p'].to_numpy()
    from sklearn.metrics import roc_auc_score, average_precision_score
    RES['val_pair_auc'] = float(roc_auc_score(y, pp)); RES['val_pair_ap'] = float(average_precision_score(y, pp))
    if not A.no_sweep:
        sw = [(t, macro(policy(Pv, t, 'plain'), MV)) for t in TAUS]; b = max(sw, key=lambda s: s[1])
        RES['a_plain'] = dict(tau=b[0], macro=b[1]); log('(a) plain best', RES['a_plain'])
        pe = efs_choose(Pv.with_columns((pl.col('p') >= pl.col('p_other') + 0.2).alias('elig')), lam, q0)
        RES['e_margin0.2'] = macro(pe, MV); log('(e) margin 0.2', RES['e_margin0.2'])
    Uv = U_of(Pv)
    if A.decoder:
        sd = lgb.Booster(model_file=A.decoder); RES['decoder'] = A.decoder
    else:
        t = time.time(); Us = U_of(Ps); GS = gt_idx(MS, Ps)
        sd = DL.fit_set_decoder(Us, GS, threads=A.threads); sd.save_model(out('_R10c_m0.txt')); RES['decoder'] = out('_R10c_m0.txt')
        log('R10c m0 refit on sample OOF', f'{time.time()-t:.0f}s')
    pred = to_pred(DL.decide_set(Uv, sd), Pv)
rep, R = evaluate(pred, MV, V2B_ROWS)
RES['val'] = rep
R.select('s1_idx', 'f').write_parquet(out('_val_rows.parquet')); pred.select('s1_idx', 'cand_idx').write_parquet(out('_val_selected.parquet'))
log('VAL macro', round(rep['macro_f05'], 6), 'US', round(rep['macro_US'], 5), 'India', round(rep['macro_India'], 5), 'locked', round(rep.get('locked30k', float('nan')), 5),
    'singletons', round(rep['singleton_acc'], 5), 'm=1', round(rep['by_m_1'], 5), 'paired vs v2b', rep['paired_vs_v2b'])
# ------------------------------------------------------------------ density
if A.density or A.selected_dens:
    Md = pl.read_parquet(f'{DV}/universe/truth_dens.parquet', columns=['s1_idx', 'm', 'r']).with_columns(pl.col('m').cast(pl.Int64), pl.col('r').cast(pl.Int64))
    Md = Md.join(MV.select('s1_idx', 'country', 'is_locked'), on='s1_idx', how='left')
    assert Md.height == 178891, Md.height
    if A.selected_dens:
        Sd = pl.read_parquet(A.selected_dens, columns=['s1_idx', 'cand_idx']).with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
        Ld = pl.read_parquet(f'{DV}/universe/preds_v2b_dens.parquet', columns=['s1_idx', 'cand_idx', 'label'])
        predd = Sd.join(Ld, on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('label').fill_null(0).cast(pl.Int8))
        RES['density_note'] = 'density selection evaluated as given'
    else:
        if sd is None: raise SystemExit('--density needs a decoder (use --preds or --decoder)')
        sch = pl.read_parquet_schema(A.density); pc = A.pcol if A.pcol in sch else 'p2'
        Pd = pl.read_parquet(A.density, columns=['s1_idx', 'cand_idx', pc] + (['label'] if 'label' in sch else [])).rename({pc: 'p'})
        if 'label' not in Pd.columns:
            Pd = Pd.join(pl.read_parquet(f'{DV}/universe/preds_v2b_dens.parquet', columns=['s1_idx', 'cand_idx', 'label']), on=['s1_idx', 'cand_idx'], how='left').with_columns(pl.col('label').fill_null(0))
        compd = pl.read_parquet(f'{DV}/universe/v1dense_dens.parquet', columns=['s1_idx', 'cand_idx', 'p2']).rename({'p2': 'q'})
        PAd = add_comp(Pd, compd).join(SRC, on='cand_idx', how='left'); del Pd, compd
        PAd = PAd.join(Md.select('s1_idx'), on='s1_idx', how='semi'); log('density pairs', PAd.height, 'S1', PAd['s1_idx'].n_unique())
        predd = to_pred(DL.decide_set(U_of(PAd), sd), PAd)
    base_d = pl.read_parquet(f'{DV}/out/row_scores_v2b.parquet').filter((pl.col('policy') == 'dens:R10c_m0.0') & (pl.col('set') == 'Q')).select('s1_idx', 'f')
    repd, Rd = evaluate(predd, Md, base_d)
    RES['density'] = repd; RES['density']['v2b_reference'] = float(base_d['f'].mean())
    Rd.select('s1_idx', 'f').write_parquet(out('_dens_rows.parquet')); predd.select('s1_idx', 'cand_idx').write_parquet(out('_dens_selected.parquet'))
    # same queries at original density from the val decision -> the model's own density loss
    Rq = R.join(Md.select('s1_idx'), on='s1_idx', how='semi').select('s1_idx', 'f')
    RES['density']['own_density_loss_paired'] = paired(Rd.select('s1_idx', 'f'), Rq)
    log('DENSITY macro', round(repd['macro_f05'], 6), 'paired vs v2b@density', repd['paired_vs_v2b'], 'own density loss', RES['density']['own_density_loss_paired'])
RES['seconds'] = time.time() - T0; RES['peak_rss_gb'] = peak_rss_gb()
json.dump(RES, open(out('.json'), 'w'), indent=1, default=str)
log('DONE ->', out('.json'), 'peak RSS GB', round(peak_rss_gb(), 1))
