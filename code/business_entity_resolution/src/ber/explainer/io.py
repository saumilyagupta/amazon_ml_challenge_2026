"""Record loading for the explainer, from the package's prepared records (prepare.py): WORK/records/{split}_s1.parquet and
{split}_s23.parquet hold entity_id, business_name, business_address, country, src (1/2/3), name_en, addr_en (the transliteration-
dictionary view, identical to the raw text when there is no Indic script). Research original: work/features/explainer/explainer/io.py
(same columns, built from the same dictionary view; verified identical on the smoke slices)."""
import polars as pl

REC_COLS = ['entity_id', 'name', 'addr', 'country', 'src', 'name_en', 'addr_en']


def read_tsv(path):
    df = pl.read_csv(path, separator='\t', quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True)
    return df.with_columns([pl.col(c).fill_null('') for c in df.columns])


def load_records(W, split):
    """All S1/S2/S3 records of a split from a package work dir: entity_id, name, addr, country, src (1/2/3), name_en, addr_en."""
    parts = [pl.read_parquet(W.records(split, p), columns=['entity_id', 'business_name', 'business_address', 'country', 'src', 'name_en', 'addr_en'])
             for p in ('s1', 's23')]
    return pl.concat(parts).select(pl.col('entity_id'), pl.col('business_name').alias('name'), pl.col('business_address').alias('addr'),
                                   pl.col('country'), pl.col('src').cast(pl.Int8), pl.col('name_en'), pl.col('addr_en'))


def pairs_from_idx(W, cands, split):
    """Map package (s1_idx, cand_idx) record-row indices to (s1_id, cand_id)."""
    s1 = pl.read_parquet(W.records(split, 's1'), columns=['entity_id']).select(pl.col('entity_id').alias('s1_id')).with_row_index('s1_idx')
    s23 = pl.read_parquet(W.records(split, 's23'), columns=['entity_id']).select(pl.col('entity_id').alias('cand_id')).with_row_index('cand_idx')
    c = cands.with_columns(pl.col('s1_idx').cast(pl.UInt32), pl.col('cand_idx').cast(pl.UInt32))
    return c.join(s1, on='s1_idx', how='left').join(s23, on='cand_idx', how='left')
