"""ensemble_v1 shared helpers. Import FIRST (before numpy/polars/lightgbm): caps threads, sets paths.
Member matrix (data/members_val.parquet, data/members_test.parquet): one row per union-v2 candidate pair, identical row order to
prod_v2b/data/preds_abc_rob_cv2_all.parquet (train: sample OOF + val + locked, 28,311,866 rows) and preds_test_abc_rob_cv2_all.parquet (83,761,275 rows).
Members (all on the same candidate set; outside their band the band specialists equal v2b's p2):
  m_v2b  prod_v2b abc_rob cv2 p2 (LB 0.98435)                          test: prod_v2b/data/preds_test_abc_rob_cv2_all.parquet
  m_v3   prod_v3 AD cv2 p2 (decoy block + Q14; LB 0.986761 file)      test: prod_v3/data/preds_test_AD_cv2_all_gate.parquet (France-gated pack)
  m_e06  prod_v2c E06 p3b (CE fine-tuned on v2b band + residual LGB)  test: prod_v2c/build/p3_test_E06.parquet (band 0.02<p2<0.99), p2 elsewhere
  m_e13  prod_v2c E13 2-fold refit p3 (Codex g_* specialist, OOF)     test: interim_v2b_E13/data/p3_all_variants_test.parquet p_cv2 (band 0.001<p2<0.999)
"""
import os, sys, time, json, subprocess
sys.dont_write_bytecode = True
TH = int(os.environ.get('ENS_THREADS', '6'))
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = str(TH)
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
W = '/workspace/saumilya/amazon-ml/work'; H = f'{W}/matching/ensemble_v1'
DD = f'{H}/data'; RD = f'{H}/results'; LD = f'{H}/logs'; BD = f'{H}/build'; SRC = f'{H}/src'
V2B = f'{W}/matching/prod_v2b'; V3 = f'{W}/matching/prod_v3'; V2C = f'{W}/matching/prod_v2c'; EB = f'{W}/blocking/embedding/full'
PY = '/opt/conda/bin/python3'
EVAL = f'{V2C}/common/eval_pairs.py'
MEMBERS = ['m_v2b', 'm_v3', 'm_e06', 'm_e13']
VAL_KEEP = ['s1_idx', 'cand_idx', 'label', 'grp', 'fold', 'es', 'country', 'in_dft', 'p1']
TEST_KEEP = ['s1_idx', 'cand_idx', 'country', 'in_dft', 'p1']
MEMBERS_VAL = f'{DD}/members_val.parquet'; MEMBERS_TEST = f'{DD}/members_test.parquet'
T0 = time.time()


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s]', *a, flush=True)


def memavail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable:'): return int(l.split()[1]) // 1048576
    return -1


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def progress(agent, msg):
    with open(f'{RD}/PROGRESS.log', 'a') as fh:
        fh.write(f'[{time.strftime("%H:%M", time.gmtime())}] {agent}: {msg}\n')


def wait_mem(gb=60, what=''):
    n = 0
    while memavail() < gb:
        if n % 5 == 0: log(f'wait_mem {what}: MemAvailable {memavail()} GB < {gb} GB')
        n += 1; time.sleep(30)


def logit(x):
    import numpy as np
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def expit(z):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def write_val_preds(df, name):
    """df: polars frame with VAL_KEEP columns + 'p' (ensemble probability for ALL 28,311,866 rows). -> data/preds_<name>.parquet"""
    import polars as pl
    assert df.height == 28311866, df.height
    assert df['p'].null_count() == 0 and float(df['p'].min()) >= 0 and float(df['p'].max()) <= 1
    out = f'{DD}/preds_{name}.parquet'
    df.select(['s1_idx', 'cand_idx', 'label', 'grp', 'country', 'in_dft', 'p1', pl.col('p').cast(pl.Float32)]).write_parquet(out); return out


def write_test_preds(df, name):
    """df: polars frame with TEST_KEEP columns + 'p' for ALL 83,761,275 test rows. -> data/test_<name>.parquet"""
    import polars as pl
    assert df.height == 83761275, df.height
    assert df['p'].null_count() == 0
    out = f'{DD}/test_{name}.parquet'
    df.select(['s1_idx', 'cand_idx', 'country', 'in_dft', pl.col('p').cast(pl.Float32)]).write_parquet(out); return out


def run_eval(name, extra=()):
    """shared v2c evaluator: R10c m0 decoder refit on the SAMPLE OOF p, decision on val, paired bootstrap vs v2b. ~8-13 min, ~15 GB.
    -> results/eval_ENS_<name>.json, _val_rows.parquet, _val_selected.parquet, _R10c_m0.txt"""
    wait_mem(60, f'eval {name}')
    cmd = [PY, EVAL, '--preds', f'{DD}/preds_{name}.parquet', '--pcol', 'p', '--tag', f'ENS_{name}', '--outdir', RD, '--threads', str(TH), '--no-sweep', *extra]
    log('run_eval', ' '.join(cmd))
    with open(f'{LD}/eval_{name}.log', 'a') as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=V2C)
    if r.returncode != 0: raise RuntimeError(f'eval_pairs failed for {name}, see {LD}/eval_{name}.log')
    return json.load(open(f'{RD}/eval_ENS_{name}.json'))


def run_eval_selected(name, selected_parquet):
    """decision-level ensembles: evaluate a given val selection (s1_idx, cand_idx) with the same protocol (no decoder refit)."""
    cmd = [PY, EVAL, '--selected', selected_parquet, '--tag', f'ENS_{name}', '--outdir', RD, '--threads', str(TH)]
    log('run_eval_selected', ' '.join(cmd))
    with open(f'{LD}/eval_{name}.log', 'a') as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=V2C)
    if r.returncode != 0: raise RuntimeError(f'eval_pairs --selected failed for {name}, see {LD}/eval_{name}.log')
    return json.load(open(f'{RD}/eval_ENS_{name}.json'))


def run_stress(name, val_preds=None, rows=None, test_preds=None, pcol='p'):
    """band-reweighting stress (RANKING.md 'stress' column; prod_v2b/10_density.py method). ~1-2 min.
    -> results/stress_<name>.json with overall_test_mix = the stress number."""
    val_preds = val_preds or f'{DD}/preds_{name}.parquet'; rows = rows or f'{RD}/eval_ENS_{name}_val_rows.parquet'; test_preds = test_preds or f'{DD}/test_{name}.parquet'
    cmd = [PY, f'{SRC}/03_stress.py', name, val_preds, pcol, rows, test_preds, RD]
    log('run_stress', ' '.join(cmd))
    with open(f'{LD}/stress_{name}.log', 'a') as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0: raise RuntimeError(f'stress failed for {name}, see {LD}/stress_{name}.log')
    return json.load(open(f'{RD}/stress_{name}.json'))


def summarize(name):
    """one scoreboard line from the eval + stress jsons -> results/SCOREBOARD.tsv (append)."""
    e = json.load(open(f'{RD}/eval_ENS_{name}.json')); v = e['val']; pv = v.get('paired_vs_v2b', {})
    s = json.load(open(f'{RD}/stress_{name}.json')) if os.path.exists(f'{RD}/stress_{name}.json') else {}
    line = dict(name=name, val=round(v['macro_f05'], 6), locked30k=round(v.get('locked30k', float('nan')), 6), US=round(v['macro_US'], 6), India=round(v['macro_India'], 6),
                singleton=round(v['singleton_acc'], 6), m1=round(v['by_m_1'], 6), d_vs_v2b=round(pv.get('diff', float('nan')), 6),
                ci_lo=round(pv.get('ci95', [float('nan')] * 2)[0], 6), ci_hi=round(pv.get('ci95', [float('nan')] * 2)[1], 6),
                stress=round(s.get('overall_test_mix', float('nan')), 6),
                band_US=round(s.get('test_band_share', {}).get('US', float('nan')), 4), band_India=round(s.get('test_band_share', {}).get('India', float('nan')), 4),
                band_France=round(s.get('test_band_share', {}).get('France', float('nan')), 4), utc=time.strftime('%H:%M', time.gmtime()))
    hdr = not os.path.exists(f'{RD}/SCOREBOARD.tsv')
    with open(f'{RD}/SCOREBOARD.tsv', 'a') as fh:
        if hdr: fh.write('\t'.join(line) + '\n')
        fh.write('\t'.join(str(x) for x in line.values()) + '\n')
    return line
