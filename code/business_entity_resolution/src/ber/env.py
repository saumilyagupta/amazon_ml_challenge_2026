"""Process-level environment setup. Import this module FIRST in every entry point (before numpy / polars / torch / lightgbm).

* caps BLAS / OpenMP / polars threads (the reference machine is shared; defaults are 8 OpenMP threads, 16 polars threads),
* forces OMP_WAIT_POLICY=PASSIVE (LightGBM spin-waiting was 7x slower on an oversubscribed box),
* disables every telemetry / sync hook of the HF / wandb stacks (nothing ever leaves the machine),
* hides GPUs unless the caller asked for one (CUDA_VISIBLE_DEVICES="" for CPU-only runs).
"""
import os


def setup(threads=None, gpu=None):
    t = str(threads or os.environ.get('BER_THREADS', '8'))
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(k, t)
    os.environ.setdefault('POLARS_MAX_THREADS', str(max(int(t), 16) if threads is None else int(t) * 2))
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_MODE'] = 'disabled'; os.environ['WANDB_DISABLED'] = 'true'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; os.environ['DO_NOT_TRACK'] = '1'
    if gpu is None or gpu == '' or str(gpu).lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU-only: never auto-detect GPUs
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)   # 'keep' leaves an externally set value untouched
        if str(gpu) == 'keep':
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    return int(t)


def logger(prefix=''):
    import time
    t0 = time.time()
    def log(*a):
        print(f'[{prefix}{time.time() - t0:.0f}s]', *a, flush=True)
    return log


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
