# Cache raw text of the reduced pool (so build_pairs does not re-read the 10.3M-row sources). pyarrow limited to 4 threads.
import pyarrow as pa, pandas as pd, time
pa.set_cpu_count(4); pa.set_io_thread_count(4)
RAW = '/workspace/saumilya/amazon-ml/work/blocking/rl/'; OUT = '/workspace/saumilya/amazon-ml/work/matching/cross_encoder/data/'
t0 = time.time(); ids = set(pd.read_parquet(OUT + 'reduced_pool_ids.parquet').entity_id)
pool = pd.concat([pd.read_parquet(RAW + f) for f in ['s2.parquet', 's3.parquet']], ignore_index=True)
pool = pool[pool.entity_id.isin(ids)].reset_index(drop=True); pool.to_parquet(OUT + 'reduced_pool_text.parquet')
print('pool text', len(pool), f'{time.time()-t0:.0f}s')
