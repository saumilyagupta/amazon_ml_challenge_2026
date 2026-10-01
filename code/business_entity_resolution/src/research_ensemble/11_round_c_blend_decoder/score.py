#!/usr/bin/env python3
"""Macro F0.5 scorer for the ML Challenge 2026 entity-resolution task.

Usage:
    python3 score.py --pred preds.tsv --gt train_ground_truth.tsv [--ids val_s1_ids.txt]

pred TSV: columns source1_entity_id, matched_entity_ids (comma-separated, may be empty).
gt   TSV: same format (the official train_ground_truth.tsv).
--ids: optional file with one S1 id per line; scoring is restricted to those ids.
       Any id in --ids missing from pred is scored as an EMPTY prediction (and counted).
Also importable: from score import macro_f05, oracle_ceiling, load_id_lists
--oracle: treat --pred as a candidate_pairs-style file and report the exact macro-F0.5 oracle ceiling 5r/(4r+m).
"""
import argparse, sys

def load_id_lists(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            s1, _, rest = line.partition('\t')
            out[s1] = set(x for x in rest.split(',') if x) if rest.strip() else set()
    return out

def f05(pred, true):
    """pred/true are sets. Singleton rule: true empty -> 1.0 if pred empty else 0.0."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred); r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)

def macro_f05(pred_map, gt_map, ids=None, verbose=True):
    ids = list(ids) if ids is not None else list(gt_map)
    tot = 0.0; n = 0
    tp_all = fp_all = fn_all = 0
    sing_tot = sing_ok = 0; nonsing_tot = 0; nonsing_f = 0.0
    n_pred_nonempty = 0
    for s1 in ids:
        true = gt_map[s1]; pred = pred_map.get(s1, set())
        sc = f05(pred, true); tot += sc; n += 1
        tp = len(pred & true); tp_all += tp; fp_all += len(pred) - tp; fn_all += len(true) - tp
        if pred: n_pred_nonempty += 1
        if not true:
            sing_tot += 1; sing_ok += (1 if not pred else 0)
        else:
            nonsing_tot += 1; nonsing_f += sc
    res = dict(
        macro_f05 = tot / max(n, 1), n_entities = n,
        micro_precision = tp_all / max(tp_all + fp_all, 1),
        micro_recall = tp_all / max(tp_all + fn_all, 1),
        singleton_accuracy = sing_ok / max(sing_tot, 1), n_singletons = sing_tot,
        nonsingleton_macro_f05 = nonsing_f / max(nonsing_tot, 1),
        frac_pred_nonempty = n_pred_nonempty / max(n, 1),
    )
    if verbose:
        for k, v in res.items():
            print(f'{k:>24}: {v:.5f}' if isinstance(v, float) else f'{k:>24}: {v}')
    return res

def oracle_ceiling(cand_map, gt_map, ids=None, verbose=True):
    """Exact macro-F0.5 ceiling of a candidate set (perfect matcher picks exactly the retained truths).
    Per S1 with m>0 truths and r retained: 5r/(4r+m); singleton: 1.0 (oracle predicts empty).
    Also returns candidate diagnostics: pair recall, mean per-S1 recall, all-truths-retained rate,
    no-truth-retained rate, mean/p50/p95/p99/max candidates per S1."""
    import statistics
    ids = list(ids) if ids is not None else list(gt_map)
    tot = 0.0; tp = 0; nt = 0; all_ok = 0; none_ok = 0; nonempty = 0; per_s1_recall = []; sizes = []
    for s1 in ids:
        true = gt_map[s1]; cand = cand_map.get(s1, set()); sizes.append(len(cand))
        m = len(true); r = len(cand & true)
        if m == 0:
            tot += 1.0; continue
        nonempty += 1; tp += r; nt += m; per_s1_recall.append(r / m)
        all_ok += (r == m); none_ok += (r == 0)
        tot += 5 * r / (4 * r + m) if r else 0.0
    sizes_sorted = sorted(sizes)
    q = lambda p: sizes_sorted[min(len(sizes_sorted) - 1, int(p * len(sizes_sorted)))] if sizes_sorted else 0
    res = dict(
        oracle_macro_f05 = tot / max(len(ids), 1), n_entities = len(ids),
        pair_recall = tp / max(nt, 1),
        mean_per_s1_recall = (sum(per_s1_recall) / len(per_s1_recall)) if per_s1_recall else 0.0,
        frac_nonempty_all_truths_retained = all_ok / max(nonempty, 1),
        frac_nonempty_no_truth_retained = none_ok / max(nonempty, 1),
        mean_candidates = sum(sizes) / max(len(sizes), 1), p50_candidates = q(0.5), p95_candidates = q(0.95),
        p99_candidates = q(0.99), max_candidates = sizes_sorted[-1] if sizes_sorted else 0,
    )
    if verbose:
        for k, v in res.items():
            print(f'{k:>36}: {v:.5f}' if isinstance(v, float) else f'{k:>36}: {v}')
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--gt', required=True)
    ap.add_argument('--ids', default=None)
    ap.add_argument('--oracle', action='store_true', help='treat --pred as a CANDIDATE file and report the oracle ceiling + candidate diagnostics')
    a = ap.parse_args()
    gt = load_id_lists(a.gt); pred = load_id_lists(a.pred)
    ids = None
    if a.ids:
        ids = [l.strip() for l in open(a.ids) if l.strip()]
    if a.oracle:
        oracle_ceiling(pred, gt, ids)
    else:
        macro_f05(pred, gt, ids)
