"""Train-S1 partition: training sample / validation / locked holdout / other, plus the 2-fold hash folds and early-stopping flag.

mode 'files' (production): resources/splits/val_s1_ids.txt (220,730 = 10% of train S1, entity-disjoint), locked_val_s1_ids.txt
(30,000 of them, never used for tuning), train_sample_s1_ids.txt (400,000 train-split S1 used to fit the matcher). Ids absent
from the prepared records are ignored, so the same files work on any subset.
mode 'hash' (smoke tests / new data): deterministic hash of the entity id."""
import os
import polars as pl
from .paths import RES
from .io import read_ids


def assign(W, cfg, split='train'):
    """-> polars frame s1_idx (i32), entity_id, grp in {sample, val, locked, other}, fold (i8), es (bool)."""
    ids = pl.read_parquet(W.records(split, 's1'), columns=['entity_id']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    sc = cfg['split']
    if split != 'train':
        g = ids.with_columns(pl.lit('other').alias('grp'))
    elif sc['mode'] == 'files':
        val = set(read_ids(os.path.join(RES, 'splits', 'val_s1_ids.txt')))
        locked = set(read_ids(os.path.join(RES, 'splits', 'locked_val_s1_ids.txt')))
        samp = set(read_ids(os.path.join(RES, 'splits', 'train_sample_s1_ids.txt')))
        g = ids.with_columns(pl.when(pl.col('entity_id').is_in(list(locked))).then(pl.lit('locked'))
                             .when(pl.col('entity_id').is_in(list(val))).then(pl.lit('val'))
                             .when(pl.col('entity_id').is_in(list(samp))).then(pl.lit('sample')).otherwise(pl.lit('other')).alias('grp'))
    else:
        h = (pl.col('entity_id').hash(seed=11) % 100000).cast(pl.Int64)
        g = ids.with_columns(pl.when(h < 100000 * sc['val_frac'] * sc['locked_frac']).then(pl.lit('locked'))
                             .when(h < 100000 * sc['val_frac']).then(pl.lit('val')).otherwise(pl.lit('sample')).alias('grp'))
        if sc.get('sample_size'):
            r = (pl.col('entity_id').hash(seed=13) % 1000003).rank('ordinal').over('grp')
            g = g.with_columns(pl.when((pl.col('grp') == 'sample') & (r > sc['sample_size'])).then(pl.lit('other')).otherwise(pl.col('grp')).alias('grp'))
    hh = (pl.col('entity_id').hash(seed=7) % 1000)
    return g.with_columns((hh % 2).cast(pl.Int8).alias('fold'), (hh < 150).alias('es'))


def query_mask(W, cfg, split):
    """boolean numpy mask over S1 rows: which S1 get forward lexical / union candidates (train: sample + val + locked; test: all)."""
    g = assign(W, cfg, split)
    return (g['grp'] != 'other').to_numpy() if split == 'train' else (g['grp'] == g['grp']).to_numpy()
