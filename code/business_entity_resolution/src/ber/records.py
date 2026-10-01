"""Record-level preprocessing for one split: every S1 and every S2/S3 record, in the prepared row order (records/{split}_s1.parquet,
records/{split}_s23.parquet), so record row == embedding row == s1_idx / cand_idx everywhere.
Outputs rec_{split}_s1.parquet, rec_{split}_s23.parquet (matcher views + frequencies + fake-name flag + token ids) and idf/voc files."""
import time
import numpy as np, polars as pl
from .text import ascii_fold, raw_view, name_frame, addr_frame


def build(W, split, log=print):
    t0 = time.time()
    R1 = pl.read_parquet(W.records(split, 's1')); R2 = pl.read_parquet(W.records(split, 's23'))
    df = pl.concat([R1, R2]).with_columns(pl.col('src').cast(pl.Int8))
    log(f'[{split}] loaded {df.height} rows {time.time()-t0:.0f}s')
    nf = name_frame(ascii_fold(df['name_en'])); log(f'[{split}] names {time.time()-t0:.0f}s')
    af = addr_frame(ascii_fold(df['addr_en']), df['country']); log(f'[{split}] addresses {time.time()-t0:.0f}s')
    d = pl.concat([df.select('entity_id', 'country', 'src', 'business_name'), nf, af], how='horizontal')
    d = d.with_columns(
        raw_view(pl.col('business_name')).alias('name_raw'),
        raw_view(pl.lit(df['business_address'])).alias('addr_raw'),
        pl.col('business_name').str.contains(r'[^\x00-\x7F]').alias('name_nonascii'),
        pl.lit(df['business_address']).str.contains(r'[^\x00-\x7F]').alias('addr_nonascii'),
        (pl.when(pl.col('business_name').str.contains(r'[\x{0900}-\x{097F}]')).then(2)
         .when(pl.col('business_name').str.contains(r'[\x{0980}-\x{0DFF}]')).then(3)
         .when(pl.col('business_name').str.contains(r'[^\x00-\x{024F}\s\p{P}\p{S}\p{N}]')).then(4)
         .when(pl.col('business_name').str.contains(r'[^\x00-\x7F]')).then(1).otherwise(0)).cast(pl.Int8).alias('script'),
    ).drop('business_name')
    d = d.with_columns((pl.col('addr_n') == '').alias('addr_empty'), (pl.col('nums').list.len() == 0).alias('addr_nodigit'),
                       ~pl.col('addr_n').str.contains('[a-z]').alias('addr_noletter'), ~pl.col('name_n').str.contains('[a-z]').alias('name_noletter'),
                       pl.col('nums').list.first().fill_null(-1).alias('num_first'), pl.col('nums').list.len().cast(pl.Int16).alias('n_nums'),
                       pl.col('name_n').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).alias('_nt'),
                       pl.col('addr_n').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).alias('_at'),
                       pl.col('name_core').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique().alias('_ct'))
    d = d.with_columns(pl.col('_nt').list.len().cast(pl.Int16).alias('ntok_name'), pl.col('_at').list.len().cast(pl.Int16).alias('ntok_addr'),
                       pl.col('_at').list.unique().list.sort().list.join(' ').alias('addr_key'), pl.col('_at').list.unique().alias('_at')).drop('_nt')
    is1 = pl.col('src') == 1
    # frequencies (unlabelled, within split)
    cnt = lambda key, filt, nm: d.filter(filt).group_by(key).agg(pl.len().cast(pl.Int32).alias(nm))
    for key, nm1, nm2 in [('name_core', 'name_freq_s1', 'name_freq_pool'), ('addr_key', 'addr_freq_s1', 'addr_freq_pool')]:
        d = d.join(cnt(key, is1, nm1), on=key, how='left').join(cnt(key, ~is1, nm2), on=key, how='left')
        d = d.with_columns(pl.col(nm1).fill_null(0), pl.col(nm2).fill_null(0))
    d = d.with_columns([pl.when(pl.col('addr_key') == '').then(0).otherwise(pl.col(c)).alias(c) for c in ('addr_freq_s1', 'addr_freq_pool')])
    # fake-name family: single alphabetic token 5-12 chars, absent from the S1 name vocabulary, >= 3 distinct addresses in the pool
    s1voc = d.filter(is1).select(pl.col('name_n').str.split(' ').alias('t')).explode('t').unique('t')['t']
    single = pl.col('name_n').str.contains(r'^[a-z]{5,12}$')
    nd = d.filter(~is1 & single).group_by('name_n').agg(pl.col('addr_key').n_unique().alias('_nd'))
    d = d.join(nd, on='name_n', how='left').with_columns(
        (single & ~is1 & ~pl.col('name_n').is_in(s1voc.implode()) & (pl.col('_nd').fill_null(0) >= 3)).alias('fake')).drop('_nd')
    log(f'[{split}] freqs+fake {time.time()-t0:.0f}s  fake pool rows {int(d["fake"].sum())}')
    # token vocabularies + IDF (docs = all S1 + pool records of the split)
    N = d.height
    for col, nm in [('_ct', 'name'), ('_at', 'addr')]:
        voc = d.select(pl.col(col).alias('t')).explode('t').drop_nulls().group_by('t').agg(pl.len().alias('df')).sort('t').with_row_index('tid')
        np.save(W.idf(split, nm), np.log(N / voc['df'].to_numpy()).astype(np.float32))
        voc.select('t', 'tid', 'df').write_parquet(W.voc(split, nm))
        tid = d.select(pl.int_range(pl.len()).alias('_r'), pl.col(col).alias('t')).explode('t').drop_nulls().join(voc.select('t', 'tid'), on='t') \
               .group_by('_r').agg(pl.col('tid').cast(pl.Int32).alias(f'{nm}_tids'))
        d = d.with_row_index('_r').join(tid, on='_r', how='left').sort('_r').drop('_r')
        d = d.with_columns(pl.col(f'{nm}_tids').fill_null(pl.lit([], pl.List(pl.Int32))))
        log(f'[{split}] vocab {nm}: {voc.height} tokens {time.time()-t0:.0f}s')
    d = d.drop('_ct', '_at')
    for nm, ids in [('s1', R1), ('s23', R2)]:
        part = d.filter(is1 if nm == 's1' else ~is1).join(ids.select('entity_id').with_row_index('erow'), on='entity_id', how='left')
        assert part['erow'].null_count() == 0 and part.height == ids.height, f'{nm} id mismatch'
        part = part.sort('erow')
        assert (part['erow'].to_numpy() == np.arange(part.height)).all()
        part.write_parquet(W.rec(split, nm))
        log(f'[{split}] wrote rec_{split}_{nm}: {part.height} rows {time.time()-t0:.0f}s')
