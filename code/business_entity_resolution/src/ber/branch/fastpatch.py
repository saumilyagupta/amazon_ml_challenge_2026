"""(package copy of work/features/branch_port/src/fastpatch.py; imports made package-relative, logic unchanged)
Memoisation of the branch's pure per-record helpers. The branch code (explain_pairs.py / features.py) is NOT modified: this module
re-binds module attributes to cached wrappers that return equal values (copies of the lists, so callers that append to `ops` cannot
corrupt the cache). Logic is identical; `check.py` verifies bit-equality against the unpatched code on random pairs.

Hot spots (profiled on val pairs): addr_norm is called ~6x per pair (street_key x2, addr_norm x2, explain_addr x2, learn), fold ~10x,
filler_match loops over the whole learned filler list (146 address tokens) per leftover token, numbers() twice. Pairs are processed
sorted by s1_id, so the S1 side is a cache hit for its ~48 candidates; candidate records recur across S1 lists.
"""
import functools
from . import explain_pairs as E
from . import features as F

_APPLIED = False
_FM_CACHE = {}          # id(filler set) -> {token: bool}; only sets registered via register() are cached (their ids are stable)


def register(*filler_sets):
    for s in filler_sets:
        _FM_CACHE.setdefault(id(s), {})


def apply(maxsize=1 << 18):
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    E.fold = functools.lru_cache(maxsize)(E.fold)                      # str -> str (immutable)

    _nt = E.name_tokens
    _nt_c = functools.lru_cache(maxsize)(lambda s: tuple(_nt(s)))
    E.name_tokens = lambda s: list(_nt_c(s))

    _an = E.addr_norm
    @functools.lru_cache(maxsize)
    def _an_c(s, country):
        t, co, ops = _an(s, country)
        return tuple(t), tuple(co), tuple(ops)
    def addr_norm(s, country):
        t, co, ops = _an_c(s, country)
        return list(t), list(co), list(ops)
    E.addr_norm = addr_norm

    _num = F.numbers
    _num_c = functools.lru_cache(maxsize)(lambda a: tuple(_num(a)))
    F.numbers = lambda a: list(_num_c(a))

    F.street_key = functools.lru_cache(maxsize)(F.street_key)          # (addr, country) -> (str|None, frozenset): immutable

    _sc = E.segment_close
    @functools.lru_cache(maxsize)
    def _sc_c(stem, tokens, max_dist):
        return _sc(stem, list(tokens), max_dist)
    E.segment_close = lambda stem, tokens, max_dist: _sc_c(stem, tuple(tokens), max_dist)

    _fm = E.filler_match
    import collections
    from rapidfuzz.distance import OSA
    HG = E.HOMOGLYPH
    _PRE = {}   # id(filler) -> precomputed [(f, f_homoglyph, Counter(f), len(f), f[0])] for the tokens with len >= 3

    def _fast(x, filler):
        """Same boolean as explain_pairs.filler_match: x in filler, or for some filler token f with len(f) >= 3:
        homoglyph-equal, or typo_ok(x, f) (OSA <= 1 if min len <= 4 else 2; False if min len < 3), or scramble_ok(f, x)
        (min len >= 4, f[0] == x[0], shared letter multiset >= 75% of len(f) and >= 60% of len(x)). The |len| pre-filter for the
        OSA test cannot change the result (an OSA distance is never below the length difference)."""
        if x in filler:
            return True
        pre = _PRE.get(id(filler))
        if pre is None:
            pre = _PRE[id(filler)] = [(f, f.translate(HG), collections.Counter(f), len(f), f[0]) for f in filler if len(f) >= 3]
        h = x.translate(HG); lx = len(x); cx = None; x0 = x[0] if lx else ''
        for f, hf, cf, lf, f0 in pre:
            if h == hf:
                return True
            mn = lx if lx < lf else lf
            if mn >= 3:
                k = 1 if mn <= 4 else 2
                if (lx - lf if lx > lf else lf - lx) <= k and OSA.distance(x, f) <= k:
                    return True
            if mn >= 4 and f0 == x0:
                if cx is None:
                    cx = collections.Counter(x)
                shared = sum((cf & cx).values())
                if shared / lf >= 0.75 and shared / lx >= 0.6:
                    return True
        return False

    def filler_match(x, filler):
        c = _FM_CACHE.get(id(filler))
        if c is None:
            return _fm(x, filler)          # unregistered (transient) set: original code
        v = c.get(x)
        if v is None:
            v = c[x] = _fast(x, filler)
        return v
    E.filler_match = filler_match
    filler_match.original = _fm; filler_match.fast = _fast
