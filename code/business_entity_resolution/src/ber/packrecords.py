"""France-pack record views for one split (matcher v3, change C), in the prepared row order (row == s1_idx / cand_idx).
Port of work/research/france/featureshift/build_records.py: build() is copied unchanged; the inputs come from the package's prepared
records (entity_id, country, src, name_en, addr_en = the transliteration-dictionary view, as the v1 views use) instead of the research
record cache. Mirrors ber.records.build: name / address views, frequencies, token vocabularies + IDF. IDF modes (both stored):
  global  : v1's, docs = all S1 + pool records of the split (all countries), idf = log(N / df)   <- used by matcher v3
  country : vocabulary entries are (country, token); idf = log(N_c / df_c) + IDF_OFFSET
Outputs WORK/records/pack_{split}_{tag}_{s1,s23}.parquet, ..._idf_{name,addr}_{global,country}.npy, ..._voc_{name,addr}.parquet.
The views cover ALL records of the split (frequencies and IDF are split-level statistics, unlabelled); the pack FEATURES are then computed
only for the France-gated pairs (features.py). Runtime on the full test split (11.7M records): ~15-25 min, dominated by ascii_fold."""
import os, time
import numpy as np, polars as pl
from . import francepack as fp

V3_CFG = dict(lexicon=False)   # admin, saint, suffix, frstreet, ctag, legal on; no learned lexicon (= the research tables, whose lexicon file was absent)


def prefix(W, split, cfg=None):
    cfg = cfg or fp.Cfg(**V3_CFG)
    return W.p('records', f'pack_{split}_{cfg.tag()}')


def build(df: pl.DataFrame, cfg: fp.Cfg, out_prefix: str, ids1: pl.DataFrame, ids23: pl.DataFrame, log=print):
    """df: entity_id, country, src (1/2/3), name_en, addr_en (romanised views, as the v1 views use business_name_en / business_address_en)."""
    t0 = time.time()
    if 'name_fold' not in df.columns:
        df = df.with_columns(fp.ascii_fold(df['name_en']).alias('name_fold'), fp.ascii_fold(df['addr_en']).alias('addr_fold'))
    nf = fp.name_frame(df['name_fold'], df['country'], cfg); log(f'names {time.time()-t0:.0f}s')
    af = fp.addr_frame(df['addr_fold'], df['country'], cfg); log(f'addresses {time.time()-t0:.0f}s')
    d = pl.concat([df.select('entity_id', 'country', 'src'), nf.select('name_n', 'name_core', 'name_ns', 'name_skel', 'f_ctag'),
                   af.select('addr_n', 'nums', 'hn_sfx', 'adm', 'street_key')], how='horizontal')
    d = d.with_columns((pl.col('addr_n') == '').alias('addr_empty'), pl.col('nums').list.first().fill_null(-1).alias('num_first'),
                       pl.col('nums').list.len().cast(pl.Int16).alias('n_nums'),
                       pl.col('name_n').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.len().cast(pl.Int16).alias('ntok_name'),
                       pl.col('addr_n').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).alias('_at'),
                       pl.col('name_core').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique().alias('_ct'))
    d = d.with_columns(pl.col('_at').list.len().cast(pl.Int16).alias('ntok_addr'), pl.col('_at').list.unique().list.sort().list.join(' ').alias('addr_key'),
                       pl.col('_at').list.unique().alias('_at'))
    is1 = pl.col('src') == 1
    cnt = lambda key, filt, nm: d.filter(filt).group_by(key).agg(pl.len().cast(pl.Int32).alias(nm))
    for key, nm1, nm2 in [('name_core', 'name_freq_s1', 'name_freq_pool'), ('addr_key', 'addr_freq_s1', 'addr_freq_pool')]:
        d = d.join(cnt(key, is1, nm1), on=key, how='left').join(cnt(key, ~is1, nm2), on=key, how='left')
        d = d.with_columns(pl.col(nm1).fill_null(0), pl.col(nm2).fill_null(0))
    d = d.with_columns([pl.when(pl.col('addr_key') == '').then(0).otherwise(pl.col(c)).alias(c) for c in ('addr_freq_s1', 'addr_freq_pool')])
    log(f'freqs {time.time()-t0:.0f}s')
    N = d.height
    for col, nm in [('_ct', 'name'), ('_at', 'addr')]:
        # vocabulary over (country, token); global idf aggregates df over countries
        E = d.select(pl.int_range(pl.len()).alias('_r'), 'country', pl.col(col).alias('t')).explode('t').drop_nulls('t')
        voc = E.group_by('country', 't').agg(pl.len().alias('df')).sort('country', 't').with_row_index('tid')
        dfg = voc.group_by('t').agg(pl.col('df').sum().alias('dfg'))
        Nc = d.group_by('country').agg(pl.len().alias('Nc'))
        voc = voc.join(dfg, on='t', how='left').join(Nc, on='country', how='left')
        np.save(f'{out_prefix}_idf_{nm}_global.npy', np.log(N / voc['dfg'].to_numpy()).astype(np.float32))
        np.save(f'{out_prefix}_idf_{nm}_country.npy', (np.log(voc['Nc'].to_numpy() / voc['df'].to_numpy()) + fp.IDF_OFFSET).astype(np.float32))
        voc.select('tid', 'country', 't', 'df', 'dfg').write_parquet(f'{out_prefix}_voc_{nm}.parquet')
        tid = E.join(voc.select('country', 't', 'tid'), on=['country', 't']).group_by('_r').agg(pl.col('tid').cast(pl.Int32).alias(f'{nm}_tids'))
        d = d.with_row_index('_r').join(tid, on='_r', how='left').sort('_r').drop('_r')
        d = d.with_columns(pl.col(f'{nm}_tids').fill_null(pl.lit([], pl.List(pl.Int32))))
        log(f'vocab {nm}: {voc.height} (country,token) entries {time.time()-t0:.0f}s')
    d = d.drop('_ct', '_at')
    for nm, ids in [('s1', ids1), ('s23', ids23)]:
        part = d.filter(is1 if nm == 's1' else ~is1).join(ids.select('entity_id', 'erow'), on='entity_id', how='left')
        assert part['erow'].null_count() == 0 and part.height == ids.height, f'{nm} id mismatch'
        part = part.sort('erow'); assert (part['erow'].to_numpy() == np.arange(part.height)).all()
        part.write_parquet(f'{out_prefix}_{nm}.parquet'); log(f'wrote {out_prefix}_{nm}: {part.height} rows {time.time()-t0:.0f}s')


def records_frame(W, split):
    """package records of a split -> (df, ids1, ids23) in the prepared row order."""
    parts = [pl.read_parquet(W.records(split, p), columns=['entity_id', 'country', 'src', 'name_en', 'addr_en']) for p in ('s1', 's23')]
    ids1 = parts[0].select('entity_id').with_row_index('erow'); ids23 = parts[1].select('entity_id').with_row_index('erow')
    return pl.concat(parts).with_columns(pl.col('src').cast(pl.Int8)), ids1, ids23


def ensure(W, split, cfg=None, log=print):
    """build the pack record tables of a split once (restartable); returns their prefix."""
    cfg = cfg or fp.Cfg(**V3_CFG); pre = prefix(W, split, cfg)
    if not (os.path.exists(f'{pre}_s1.parquet') and os.path.exists(f'{pre}_s23.parquet')):
        df, ids1, ids23 = records_frame(W, split); log(f'France-pack record views for {split}: {df.height} records, cfg {cfg.tag()}')
        build(df, cfg, pre, ids1, ids23, log)
    return pre
