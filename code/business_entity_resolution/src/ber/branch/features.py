"""Pair features v0, following FEATURE_EXTRACTION.md (ids N*, A*, X*, S1 refer to that document).

Package copy of the team's branch scripts/features.py (github saumilyagupta/KL-convergence @ b14e566; byte-identical in
work/features/branch_port/src); the only change is the package-relative import of explain_pairs.

Not yet included: char TF-IDF (N3, A13), embeddings (N14, A14), name frequency (N15),
graph/competition features (G*), learned France word lists (section 6).
"""
import math
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import OSA

from . import explain_pairs as E

NUM_REL = {"missing": 0, "exact": 1, "dropped": 2, "substituted": 3, "range": 4, "different": 5}


def build_idf(names, addrs):
    """Per-country token IDF over a document collection (unsupervised)."""
    idf = {}
    for field, docs in (("name", names), ("addr", addrs)):
        for country, texts in docs.items():
            df, n = {}, 0
            for t in texts:
                n += 1
                for tok in set(re.findall(r"[a-z0-9]+", E.fold(t))):
                    df[tok] = df.get(tok, 0) + 1
            idf[(field, country)] = ({k: math.log((n + 1) / (v + 1)) + 1 for k, v in df.items()},
                                     math.log(n + 1) + 1)
    return idf


def _idf(idf, field, country, tok):
    table, default = idf.get((field, country), ({}, 10.0))
    return table.get(tok, default)


def numbers(addr):
    s = E.fold(addr)
    s = E.POBOX.sub(" ", s)
    s = E.HALF.sub(" ", s)
    return [m.lstrip("0") or "0" for m in re.findall(r"\d+", s)]


def num_relation(a, b, fb):
    if not a or not b:
        return NUM_REL["missing"]
    x, y = a[0], b
    if x in y:
        return NUM_REL["exact"]
    for z in y:
        if len(z) < len(x) and (x.startswith(z) or x.endswith(z) or any(x[:i] + x[i + 1:] == z for i in range(len(x)))):
            return NUM_REL["dropped"]
        if len(z) == len(x) and OSA.distance(x, z) == 1:
            return NUM_REL["substituted"]
    if re.search(rf"\b0*{x}\s*-\s*\d+", fb):
        return NUM_REL["range"]
    return NUM_REL["different"]


def street_key(addr, country):
    toks, comp_of, _ = E.addr_norm(addr, country)
    for ci in sorted(set(comp_of)):
        comp = [t for t, c in zip(toks, comp_of) if c == ci]
        nums = [t for t in comp if t.isdigit()]
        words = [t for t in comp if not t.isdigit() and t not in E.STREET_FULL and len(t) > 1]
        if nums and words:
            return nums[0], frozenset(words)
    return None, frozenset()


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def contain(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a) if a else 0.0


def pair_features(a_name, a_addr, b_name, b_addr, src, country, ctx):
    nf, af, idf = ctx["name_filler"], ctx["addr_filler"], ctx["idf"]
    f = {}
    # ---- names
    fa, fb = E.fold(a_name), E.fold(b_name)
    ca, cb = E.core(E.name_tokens(a_name)), E.core(E.name_tokens(b_name))
    f["n_core_equal"] = float(sorted(ca) == sorted(cb) and bool(ca))                              # N1
    f["n_jacc"], f["n_contain_a"], f["n_contain_b"] = jacc(ca, cb), contain(ca, cb), contain(cb, ca)  # N2
    f["n_tset"] = fuzz.token_set_ratio(fa, fb)
    f["n_tsort"] = fuzz.token_sort_ratio(fa, fb)
    f["n_partial"] = fuzz.partial_ratio(fa, fb)
    f["n_ratio"] = fuzz.ratio(" ".join(ca), " ".join(cb))
    nops, nleft = E.explain_name(a_name, b_name, nf)
    f["n_left_count"] = len(nleft)                                                                # N5
    f["n_left_max_idf"] = max((_idf(idf, "name", country, t) for t in nleft), default=0.0)
    dropped = set(ca) - set(cb)
    f["n_dropped_max_idf"] = max((_idf(idf, "name", country, t) for t in dropped), default=0.0)    # N6
    f["n_shared_idf"] = sum(_idf(idf, "name", country, t) for t in set(ca) & set(cb))
    f["n_dup"] = float(len(cb) != len(set(cb)))                                                   # N7
    f["n_acronym"] = float("acronym" in nops)                                                     # N8
    f["n_web"] = float(any(o in nops for o in ("domain", "handle")))                              # N9
    f["n_web_unres"] = float(any(o.endswith("_unresolved") for o in nops))
    f["n_alias"] = float("alias_marker" in nops)                                                  # N10
    f["n_junk"] = float("name_replaced" in nops)                                                  # N11
    f["n_upper_b"] = float(b_name == b_name.upper() and any(ch.isalpha() for ch in b_name))       # N12
    f["n_len_ratio"] = len(b_name) / max(1, len(a_name))
    added = set(cb) - set(ca)
    f["n_added_count"] = len(added)
    f["n_added_all_filler"] = float(bool(added) and all(E.filler_match(t, nf) for t in added))
    f["n_added_max_idf"] = max((_idf(idf, "name", country, t) for t in added), default=0.0)
    # ---- addresses
    na, nb = numbers(a_addr), numbers(b_addr)
    fab = E.fold(b_addr)
    f["a_b_empty"] = float(b_addr.strip() == "")                                                  # A12
    f["a_num_both"] = float(bool(na) and bool(nb))                                                # A1
    f["a_num_exact"] = float(bool(na) and bool(nb) and na[0] in nb)                               # A2
    f["a_num_rel"] = num_relation(na, nb, fab)                                                    # A3
    f["a_num_jacc"] = jacc(na, nb)                                                                # A4
    # house-number offset: a neighbour business sits a few numbers away; a typo is one digit away
    f["a_num_absdiff"] = math.log1p(abs(int(na[0]) - int(nb[0]))) if na and nb and len(na[0]) < 10 and len(nb[0]) < 10 else -1.0
    # signed offset: decoy neighbours sit a few numbers ABOVE the S1 (US +1..+25, 99.8% positive)
    f["a_num_offset"] = float(max(-100, min(100, int(nb[0]) - int(na[0])))) if na and nb and len(na[0]) < 10 and len(nb[0]) < 10 else -999.0
    ka, kb = street_key(a_addr, country), street_key(b_addr, country)
    f["a_street_key_eq"] = float(ka[0] is not None and ka == kb)                                  # A7
    f["a_street_words_jacc"] = jacc(ka[1], kb[1])                                                 # A6
    f["a_street_words_sim"] = fuzz.token_set_ratio(" ".join(sorted(ka[1])), " ".join(sorted(kb[1])))
    ta, _, _ = E.addr_norm(a_addr, country)
    tb, _, _ = E.addr_norm(b_addr, country)
    wa, wb = [t for t in ta if not t.isdigit()], [t for t in tb if not t.isdigit()]
    f["a_jacc"], f["a_contain_b"] = jacc(wa, wb), contain(wb, wa)                                 # A8
    f["a_tset"] = fuzz.token_set_ratio(E.fold(a_addr), fab)
    aops, aleft = E.explain_addr(a_addr, b_addr, country, af, set())
    f["a_left_count"] = len(aleft)                                                                # A9
    f["a_left_max_idf"] = max((_idf(idf, "addr", country, t) for t in aleft if not t.isdigit()), default=0.0)
    f["a_num_inserted"] = float("number_inserted" in aops)                                        # A5
    f["a_placeholder"] = float("placeholder" in aops)
    f["a_locality_replaced"] = float("locality_replaced" in aops)
    # ---- explainer (X1-X4)
    loose = E.LOOSE_OPS & (set(nops) | set(aops))
    f["x_name_ok"] = float(not nleft)
    f["x_addr_ok"] = float(not aleft)
    f["x_n_ops_name"] = len(set(nops) - {"case_or_accent"})
    f["x_n_ops_addr"] = len(set(aops) - {"upper"})
    f["x_loose"] = float(bool(loose))
    for op in ("typo", "scramble", "token_corrupted", "word_dropped", "filler_added", "legal_form", "homoglyph"):
        f[f"x_{op}"] = float(op in nops)
    # ---- source
    f["s_is_s3"] = float(src == "S3")                                                             # S1
    return f
