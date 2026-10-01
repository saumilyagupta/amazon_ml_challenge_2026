"""Indic-script -> Latin word dictionary view (resources/translit_dictionary.tsv).

Provenance: the whole challenge dataset contains only 1,534 distinct Indic-script words (9 scripts). The dictionary was built by the
the team's transliteration pipeline (research tree): 1,363 words read off the TRAIN ground-truth alignment (Indic S2/S3 names are word-by-word
renderings of their matched S1 name; address Indic text is always the state name), 165 from AI4Bharat IndicXlit (MIT, ~11M
parameters, run offline once) constrained to the dataset's own Latin vocabulary, 6 manual native-Tamil shop words. No external
data or service was used. Columns: lang, word, count, english, source[train|ai4bharat+vocab|manual], ai4bharat, ai4bharat_top5.
Unknown Indic words (none in the provided data) are left unchanged (anyascii folding happens later in ber.text)."""
import csv, os, re
import polars as pl
from .paths import RES

WORD = re.compile(r'[ऀ-ൿ‌‍]+')   # maximal run of chars from the 9 Indic blocks (+ ZWNJ/ZWJ inside words)
HAS_INDIC = re.compile(r'[ऀ-ൿ]')
DEFAULT_DICT = os.path.join(RES, 'translit_dictionary.tsv')


def load_dictionary(path=DEFAULT_DICT):
    with open(path, encoding='utf-8', newline='') as f:
        return {r['word']: r['english'] for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)}


def translit_series(s: pl.Series, d=None) -> pl.Series:
    """Replace Indic words by their dictionary English. Non-Indic strings are returned unchanged (identity for S1)."""
    d = load_dictionary() if d is None else d
    def sub(m):
        w = m.group(0).replace('‌', '').replace('‍', '')
        return d.get(m.group(0), d.get(w, m.group(0)))
    df = pl.DataFrame({'s': s.fill_null('')})
    ind = df.filter(pl.col('s').str.contains(r'[ऀ-ൿ]')).unique('s')
    if ind.height == 0:
        return df['s']
    ind = ind.with_columns(pl.col('s').map_elements(lambda x: WORD.sub(sub, x), return_dtype=pl.Utf8).alias('en'))
    return df.join(ind, on='s', how='left', maintain_order='left').with_columns(pl.coalesce('en', 's').alias('s'))['s']
