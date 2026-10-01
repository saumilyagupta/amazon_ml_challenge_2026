"""B1 shared helpers: paths, thread caps, logging, memory gate. Import FIRST (before numpy/polars/lightgbm)."""
import os, sys, time
sys.dont_write_bytecode = True
TH = int(os.environ.get('B1_THREADS', '8'))
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = str(TH)
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['WANDB_MODE'] = 'disabled'; os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
W = '/workspace/saumilya/amazon-ml/work'
H = os.environ.get('B1_HOME', f'{W}/matching/prod_v2c/build/interim_v2b_E13'); DD = f'{H}/data'; LD = f'{H}/logs'; OD = f'{H}/output'
for _d in (DD, LD, OD): os.makedirs(_d, exist_ok=True)
V1 = f'{W}/matching/prod_v1'; V2B = f'{W}/matching/prod_v2b'; LAB = f'{W}/matching/accuracy_lab_20260925'
E13 = f'{W}/matching/prod_v2c/exp/E13_codex_refit'; RESD = f'{W}/matching/prod_v2c/results'; XA = f'{W}/matching/v2a_experiments'
EB = f'{W}/blocking/embedding/full'; FR = f'{W}/research/france'; SR = '/workspace/saumilya/amazon-ml/student_resource'
PREDS = f'{V2B}/data/preds_test_abc_rob_cv2_all.parquet'
FROZEN_DECODER = f'{V2B}/models/abc_rob_cv2_all_p2_R10c_m0.0.txt'
N_S1, N_PAIRS = 1732544, 83761275
MODE = os.environ.get('B1_MODE', 'interactive')
GATE = 100 if MODE == 'direct' else 0
for p in (f'{V2B}', f'{V1}', f'{W}/matching/prod_v2a', f'{W}/common', f'{W}/research/postproc', E13):
    if p not in sys.path: sys.path.insert(0, p)
T0 = time.time()


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s]', *a, flush=True)


def memavail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable:'): return int(l.split()[1]) // 1048576
    return -1


def rss_gb():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS:'): return int(l.split()[1]) / 1048576
    return -1.0


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def gate(what):
    """direct mode: wait until MemAvailable >= 100 GB before a heavy chunk (fitlock mode: the fitlock start guard applies)."""
    if not GATE: return
    n = 0
    while memavail() < GATE:
        if n % 5 == 0: log(f'gate {what}: MemAvailable {memavail()} GB < {GATE} GB -> wait 60 s')
        n += 1; time.sleep(60)


def progress(msg):
    with open(f'{RESD}/PROGRESS.log', 'a') as fh:
        fh.write(f'[{time.strftime("%H:%M", time.gmtime())}] B1: {msg}\n')
