"""Shared helpers for work/internal_eval (paths, id maps, submission I/O, file registry). Everything outside internal_eval is read-only."""
import os, sys, json, time, hashlib

# root of the work tree; on another machine set AMAZON_ML_WORK=<kit>/work (see internal_eval/SETUP.md)
W = os.environ.get('AMAZON_ML_WORK', '/workspace/saumilya/amazon-ml/work')
IE = os.environ.get('AMAZON_ML_IE', f'{W}/internal_eval')
M = f'{W}/matching'
FR = f'{W}/research/france'
EB = f'{W}/blocking/embedding/full'
SPL = f'{W}/splits'
SUB = f'{W}/results_analysis/submitted_tsv'
DATA, OUT, LOGS = f'{IE}/data', f'{IE}/out', f'{IE}/logs'
MIX = {'US': 663106 / 1732544, 'India': 809986 / 1732544, 'France': 259452 / 1732544}
N_TEST = {'US': 663106, 'India': 809986, 'France': 259452}
D = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
D3 = [k for k in D if k > 2]
K2 = ['s1_idx', 'cand_idx']


def envcap(n=8):
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS'):
        os.environ[k] = str(n)
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'; os.environ['CUDA_VISIBLE_DEVICES'] = ''


def logger(name):
    T0 = time.time(); fh = open(f'{LOGS}/{name}.log', 'a')
    def log(*a):
        s = f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True); fh.write(s + '\n'); fh.flush()
    return log


def ids(split, kind):
    import polars as pl
    f = f'{EB}/ids/{split}_{"s1" if kind == "s1" else "s23"}.parquet'
    col = 's1_idx' if kind == 's1' else 'cand_idx'
    return pl.read_parquet(f, columns=['entity_id', 'country']).with_row_index(col).with_columns(pl.col(col).cast(pl.Int32))


def read_sub(path, split='test'):
    """submission TSV -> (s1_idx, cand_idx) Int32 pairs."""
    import polars as pl
    d = pl.read_csv(path, separator='\t', quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True)
    a, b = d.columns[:2]
    d = d.with_columns(pl.col(b).fill_null('').str.split(',')).explode(b).filter(pl.col(b) != '')
    i1 = ids(split, 's1').select(pl.col('entity_id').alias(a), 's1_idx')
    i2 = ids(split, 's23').select(pl.col('entity_id').alias(b), 'cand_idx')
    out = d.join(i1, on=a).join(i2, on=b)
    assert out.height == d.height, (path, out.height, d.height)
    return out.select(K2).unique()


def md5(path, n=12):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): h.update(c)
    return h.hexdigest()[:n]


# label -> (path, LB or None, family, note). family: which line the file belongs to (for reading offsets; not used in any score).
FILES = {
    'g1_v2a':      (f'{SUB}/01_v2a_2026-09-25.tsv', 0.960193, 'v2a', 'graded #1'),
    'g2_v2b':      (f'{SUB}/02_v2b_abc_rob_cv2_R10c_2026-09-25.tsv', 0.98435, 'v2b', 'graded #2'),
    'g4_v3xash':   (f'{SUB}/04_v3x_ash_v2b_plus_specialist_2026-09-25.tsv', 0.984761, 'v2b', 'graded #4'),
    'g5_v3':       (f'{SUB}/05_v3_AD_R10c_own_oo_gatepack_2026-09-25.tsv', 0.986761, 'v3', 'graded #5'),
    'g6_v3anchor': (f'{SUB}/06_v3anchor_ensemble_2026-09-25.tsv', 0.988419, 'v3', 'graded #6'),
    'g7_vr2':      (f'{SUB}/07_vr2_consensus_2026-09-26.tsv', 0.988842, 'v3', 'graded #7'),
    'c_v1':        (f'{M}/prod_v1/output/matching_results.tsv', None, 'v1', 'matcher v1'),
    'c_H':         (f'{M}/prod_v2a/output/variant_H/matching_results.tsv', None, 'v2a', 'hybrid H'),
    'c_shash03':   (f'{SUB}/NOT_UPLOADED_03_shashvat_gbm_e5small_identity_iter005_2026-09-25.tsv', None, 'other', 'withdrawn #3'),
    'c_v2bpp':     (f'{M}/v2b_postpass2/output/V3dALL_V1_FRcl_USlegal_no12/matching_results.tsv', None, 'v2b', 'v2b + post-pass #2'),
    'c_v3xash_rf': (f'{M}/prod_v3x_ash_refit/output/matching_results.tsv', None, 'v3', 'ash specialist refit on v3'),
    'c_v3frveto':  (f'{SUB}/06_v3_plus_fr_ctag_legal_veto_2026-09-25.tsv', None, 'v3', 'v3 + FR veto'),
    'c_v3pp':      (f'{SUB}/07_v3_postpass_sw_FRcl_USl12_2026-09-25.tsv', None, 'v3', 'v3 + post-pass (07)'),
    'c_v31raw':    (f'{M}/prod_v3_1/output/test_AD2/matching_results.tsv', None, 'v3', 'v3.1 AD2 raw'),
    'c_v31pp':     (f'{SUB}/08_CANDIDATE_v3_1_AD2_postpass_2026-09-26.tsv', None, 'v3', 'v3.1 + post-pass (08)'),
    'c_v32bd':     (f'{SUB}/09_CANDIDATE_v3_2_B_decision_augdecoder_postpass_2026-09-26.tsv', None, 'v3', 'v3.2 aug decoder + pp (09)'),
    'c_v3a_v31':   (f'{SUB}/10_CANDIDATE_v3anchor_v31_2026-09-26.tsv', None, 'v3', 'v3anchor_v31 (10)'),
    'c_E06g':      (f'{M}/prod_v2c/build/interim_v2b_E06/output_guarded/matching_results.tsv', None, 'v2b', 'v2b + E06 CE band, guarded'),
    'c_E13g':      (f'{M}/prod_v2c/build/interim_v2b_E13/output_guarded/matching_results.tsv', None, 'v2b', 'v2b + E13 specialist, guarded'),
    'c_v2cA':      (f'{M}/prod_v2c/build/v2cA/output/matching_results.tsv', None, 'v2b', 'v2cA'),
    'c_v2cC':      (f'{M}/prod_v2c/build/v2cB/output/matching_results.tsv', None, 'v3', 'v2cC (v3 feats, density-trained)'),
    'c_vr2all':    (f'{M}/vr2_release_20260926/submissions/residual_all/matching_results.tsv', None, 'v3', 'vr2 residual_all'),
    'c_vr2frfb':   (f'{M}/vr2_release_20260926/submissions/residual_frfallback/matching_results.tsv', None, 'v3', 'vr2 residual, France = v3anchor'),
    'c_v2shash':   (f'{M}/v2shash/round_c/test_release/output/matching_results.tsv', None, 'v3', 'v2shash round_c release (other session)'),
    'c_harshit':   ('/workspace/saumilya/amazon-ml/harshit_matching_results.tsv', None, 'other', 'teammate file (harshit)'),
}
GRADED = [k for k, v in FILES.items() if v[1] is not None]


def sel_path(label):
    return f'{DATA}/sel/{label}.parquet'


def load_sel(label):
    import polars as pl
    return pl.read_parquet(sel_path(label))
