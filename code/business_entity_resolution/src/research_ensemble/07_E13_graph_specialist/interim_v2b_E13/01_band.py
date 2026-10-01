#!/opt/conda/bin/python3
"""B1 step 1: TEST band table for the E13 band specialist on v2b's test predictions (read-only inputs; outputs under data/).
band = 0.001 < p2 < 0.999 of prod_v2b/data/preds_test_abc_rob_cv2_all.parquet (83,761,275 rows, natural (s1_id, cand_id) order).
Resumable steps:
  base: e13_block.p2_aggregates over ALL pairs of each S1 in the file's natural row order (p2_rank is ordinal) -> band rows with the
        20 preds columns used as features (p1, 12 stage-2, 6 sibling, p2) + meta + 5 aggregates -> data/band_base.parquet
  s1  : per v2b test feats part (48): streaming scan of (s1_idx, cand_idx + 267 abc_rob) semi-joined to the band keys -> data/band_s1/part-NNN.parquet
  g   : 18 g_* via e13_block.make_features(B_chunk, P_chunk = ALL pairs (s1_idx, cand_idx, p2) of the chunk's S1, R = rec_test_s23 (full),
        R1 = rec_test_s1 (full = frequency universe)) in s1_idx-range chunks of --chunk S1 -> data/band_g/chunk-KK.parquet
usage: 01_band.py [--chunk 200000] [--steps base,s1,g]"""
import b1common as C
import os, sys, glob, json, time, argparse, gc
ap = argparse.ArgumentParser(); ap.add_argument('--chunk', type=int, default=200000); ap.add_argument('--steps', default='base,s1,g'); A = ap.parse_args()
import numpy as np, polars as pl
import pv2b.common  # noqa (sys.path for pv1 etc.)
from pv2b.feats import S1_FEATS
import e13_block as E
log = C.log; K2 = ['s1_idx', 'cand_idx']; ST = A.steps.split(','); RES = {'mode': C.MODE, 'gate_gb': C.GATE, 'threads': C.TH, 'chunk': A.chunk}
log('mode', C.MODE, 'gate', C.GATE, 'threads', C.TH, 'MemAvailable', C.memavail(), 'steps', ST)
fb = f'{C.DD}/band_base.parquet'
if 'base' in ST and not os.path.exists(fb):
    C.gate('base'); t = time.time()
    P = pl.read_parquet(C.PREDS); assert P.height == C.N_PAIRS, P.height
    log('preds read', P.shape, f'RSS {C.rss_gb():.1f} GB')
    P = E.p2_aggregates(P)
    B = P.filter(E.band_expr())
    RES['base'] = dict(pairs=P.height, band_pairs=B.height, band_s1=B['s1_idx'].n_unique(), s1_all=P['s1_idx'].n_unique(),
                       band_by_country=dict(B.group_by('country').len().sort('country').iter_rows()),
                       pairs_by_country=dict(P.group_by('country').len().sort('country').iter_rows()))
    del P; gc.collect()
    B.write_parquet(fb + '.tmp'); os.rename(fb + '.tmp', fb)
    RES['base']['seconds'] = time.time() - t; log('base', RES['base'], f'RSS {C.rss_gb():.1f} GB peak {C.peak_rss_gb():.1f} GB')
    json.dump(RES, open(f'{C.LD}/01_band_base.json', 'w'), indent=1, default=str)
    del B; gc.collect()
keys = pl.read_parquet(fb, columns=K2); log('band keys', keys.height)
if 's1' in ST:
    t = time.time(); S1C = S1_FEATS['abc_rob']; assert len(S1C) == 267
    parts = sorted(glob.glob(f'{C.V2B}/feats/test_[0-9]*.parquet')); assert len(parts) == 48, len(parts)
    os.makedirs(f'{C.DD}/band_s1', exist_ok=True); tot = 0; kl = keys.lazy()
    for i, f in enumerate(parts):
        o = f'{C.DD}/band_s1/part-{i:03d}.parquet'
        if os.path.exists(o):
            tot += pl.scan_parquet(o).select(pl.len()).collect().item(); continue
        C.gate(f'part {i}'); t1 = time.time()
        X = pl.scan_parquet(f).select(K2 + S1C).join(kl, on=K2, how='semi').collect(engine='streaming')
        X.write_parquet(o + '.tmp'); os.rename(o + '.tmp', o); tot += X.height
        log(f'part {i:03d}', X.height, f'{time.time()-t1:.1f}s', f'RSS {C.rss_gb():.1f} GB', 'MemAvail', C.memavail()); del X
    assert tot == keys.height, (tot, keys.height)
    RES['s1'] = dict(rows=tot, seconds=time.time() - t); log('s1 features done', RES['s1'])
    json.dump(RES, open(f'{C.LD}/01_band_s1.json', 'w'), indent=1, default=str)
if 'g' in ST:
    t = time.time(); os.makedirs(f'{C.DD}/band_g', exist_ok=True)
    R = pl.read_parquet(f'{C.V1}/data/rec_test_s23.parquet', columns=E.REC_COLS); R1 = pl.read_parquet(f'{C.V1}/data/rec_test_s1.parquet', columns=E.REC_COLS)
    Pp = pl.read_parquet(C.PREDS, columns=['s1_idx', 'cand_idx', 'p2']); log('records + p2 read', R.shape, R1.shape, Pp.shape, f'RSS {C.rss_gb():.1f} GB')
    tot = 0; RES['g_chunks'] = []
    for k, lo in enumerate(range(0, C.N_S1, A.chunk)):
        o = f'{C.DD}/band_g/chunk-{k:02d}.parquet'; hi = lo + A.chunk
        if os.path.exists(o):
            tot += pl.scan_parquet(o).select(pl.len()).collect().item(); continue
        C.gate(f'g chunk {k}'); t1 = time.time()
        rng = (pl.col('s1_idx') >= lo) & (pl.col('s1_idx') < hi)
        kb = keys.filter(rng); pb = Pp.filter(rng)
        G = E.make_features(kb, pb, R, R1)
        assert G.height == kb.height
        G.write_parquet(o + '.tmp'); os.rename(o + '.tmp', o); tot += G.height
        RES['g_chunks'].append(dict(k=k, lo=lo, rows=G.height, seconds=round(time.time() - t1, 1), rss=round(C.rss_gb(), 1)))
        log(f'g chunk {k:02d} [{lo}, {hi})', G.height, f'{time.time()-t1:.1f}s', f'RSS {C.rss_gb():.1f} GB peak {C.peak_rss_gb():.1f}', 'MemAvail', C.memavail())
        del G, kb, pb; gc.collect()
    assert tot == keys.height, (tot, keys.height)
    RES['g'] = dict(rows=tot, seconds=time.time() - t); log('g features done', RES['g'])
    json.dump(RES, open(f'{C.LD}/01_band_g.json', 'w'), indent=1, default=str)
log('DONE peak RSS GB', round(C.peak_rss_gb(), 1))
