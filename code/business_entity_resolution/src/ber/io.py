"""TSV / parquet helpers. All challenge files are tab-separated and must be read with quoting disabled (csv.QUOTE_NONE /
polars quote_char=None): business names contain quotes and commas."""
import polars as pl


def read_tsv(path):
    df = pl.read_csv(path, separator='\t', quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True)
    return df.with_columns([pl.col(c).fill_null('') for c in df.columns])


def load_id_lists(path):
    """{source1_entity_id: set(ids)} from a ground-truth / matching / candidate TSV."""
    out = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            s1, _, rest = line.partition('\t')
            out[s1] = set(x for x in rest.split(',') if x) if rest.strip() else set()
    return out


def gt_pairs_frame(path):
    """ground-truth TSV -> polars frame (s1_id, cand_id), one row per true pair."""
    rows = []
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            s1, _, rest = line.rstrip('\n').partition('\t')
            for c in rest.split(','):
                if c: rows.append((s1, c))
    return pl.DataFrame(rows, schema=['s1_id', 'cand_id'], orient='row')


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip()]


def write_id_lists(g: pl.DataFrame, path, col):
    """g: source1_entity_id (str), ids (comma-joined str, '' for empty), already in the desired row order."""
    g.select(pl.col('source1_entity_id'), pl.col('ids').alias(col)).write_csv(path, separator='\t', quote_style='never')
