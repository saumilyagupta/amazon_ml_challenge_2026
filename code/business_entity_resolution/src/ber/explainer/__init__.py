"""Noise-inversion feature library ('explainer'): 127 per-pair features that invert the data generator's operations
(copied from the research library work/features/explainer/explainer/, only the resource path and the record loader changed).

    from ber.explainer import load_records, run_parallel, ALL_FEATURES
    rec = load_records(W, 'test')                                         # records of a package work dir (+ transliteration view)
    X = run_parallel(pairs.select('s1_id', 'cand_id'), rec, n_workers=8)  # polars DataFrame [s1_id, cand_id, *ALL_FEATURES], input row order

Per-country word lists (legal forms, filler, junk names, street abbreviations, admin aliases, IDF tables; France lists from unlabelled
TEST anchor pairs) are LEARNED from the challenge files by tools/learn_wordlists.py and shipped in resources/explainer_wordlists/.
-1 = one side missing; a3_num_rel is categorical; there is no country feature (the country only selects the word lists).
"""
from .features import (compute_features, explain_name, explain_addr, pair_features, record_views, number_relation,
                       FEATURES, VEC_FEATURES, ALL_FEATURES, OPS_NAME, OPS_ADDR, LOOSE, NUM_REL)
from .runner import run_parallel, run_to_parquet
from .io import load_records, pairs_from_idx
from .views import name_view, addr_view, fold, hg
from .wordlists import get_tables, all_tables, set_cache
