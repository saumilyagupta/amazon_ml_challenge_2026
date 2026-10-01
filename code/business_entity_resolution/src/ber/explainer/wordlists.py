"""Per-country word lists LEARNED from the provided files (spec section 6), shipped in resources/explainer_wordlists/.

6.1 legal forms      : top final-position S1 name tokens that also show the generator's 'legal form moved to front' signature
6.2 name filler      : content tokens inserted in B vs A in confident pairs (US/India: labelled train-split pairs;
                       France: unlabelled ANCHOR pairs = same house number + same folded street + name-core tsr >= 95)
6.3 address filler   : address tokens inserted in B vs A (generator words: city cdp township door plot no ...)
6.4 junk names       : single-token names absent from every S1 name of the country, seen in >= k S2/S3 records at distinct addresses
6.5 street abbrevs   : B-token -> A-token abbreviation pairs aligned inside the house-number component of anchor pairs
6.6 admin aliases    : S2/S3-only address components whose co-occurring places map (>= 90%) to one S1 admin area
6.7 IDF tables       : per-country document frequencies over S1+S2+S3 (name content tokens, canonical address tokens)
Sources: US/India lists from TRAIN files (pairs from train_split_ground_truth.tsv only -> no val leakage);
France lists from the TEST files (unlabelled). Junk/admin/IDF/S1-frequency use all files of the country (unlabelled).
"""
import bisect
import json
import os
from collections import Counter, defaultdict

import numpy as np

from . import tables as TB
from .views import fold

from ..paths import RES

# learned lists shipped with the package (resources/explainer_wordlists/, built by tools/learn_wordlists.py); override with set_cache()
CACHE = os.path.join(RES, 'explainer_wordlists')


def set_cache(path):
    global CACHE
    CACHE = path
    _CACHE.clear()
COUNTRIES = ('US', 'India', 'France')


def _akey(c):
    import re
    return re.sub(r'\s+', ' ', fold(c).replace('-', ' ').replace("'", ' ').replace('.', ' ')).strip()


def seed_admin(country):
    m = {}
    if country == 'US':
        for code, name in TB.US_STATES.items():
            m[_akey(code)] = code
            m[_akey(name)] = code
    elif country == 'India':
        for name, code in TB.IN_STATES.items():
            m[_akey(name)] = code
            m[_akey(code)] = code
        for k, v in TB.IN_CODE_EXTRA.items():
            m[_akey(k)] = v
    else:
        for reg, depts in TB.FR_REGION_DEPTS.items():
            code = _akey(reg)
            m[code] = code
            for d in depts:
                m.setdefault(_akey(d), code)
    return m


class Tables:
    """All per-country resources used by views/features. Built from seeds + learned cache."""

    def __init__(self, country, learned=None, use_learned_filler=True):
        L = learned or {}
        self.country = country
        self.connectives = set(TB.CONNECTIVES) | (TB.FR_CONNECTIVES if country == 'France' else set())
        self.legal = dict(TB.LEGAL_CANON)
        for t in L.get('legal', []):
            self.legal.setdefault(t, t.upper())
        if use_learned_filler and 'filler' in L:
            self.filler = set(L['filler']) - {'shree', 'shri', 'sri', 'smt', 'nee'}
        elif use_learned_filler:
            self.filler = set(TB.SEED_FILLER.get(country, ()))
        else:
            self.filler = set()
        self.filler -= set(self.legal)
        from .views import hg
        self.legal_hg = {hg(k): v for k, v in self.legal.items() if len(k) >= 2}
        self.filler_hg = {hg(k) for k in self.filler}
        self.conn_hg = {hg(k) for k in self.connectives if len(k) >= 2}
        self.addr_filler = set(TB.SEED_ADDR_FILLER.get(country, ())) | set(L.get('addr_filler', []))
        base = TB.FR_STREET if country == 'France' else (TB.IN_STREET if country == 'India' else TB.US_STREET)
        self.street = dict(base)
        for b, a in L.get('street_abbr', {}).items():
            tgt = self.street.get(a, a)
            self.street.setdefault(b, tgt)
        self.street_types = set(self.street.values())
        self.addr_filler -= set(self.street)
        self.admin = seed_admin(country)
        for k, code in L.get('admin_alias', {}).items():
            self.admin.setdefault(k, code)
        self.ordinal_map = TB.ORDINAL_MAP
        self.junk = set(L.get('junk', []))
        self.s1_vocab = set(L.get('s1_vocab', []))
        self.idf_name = L.get('idf_name', {})
        self.idf_addr = L.get('idf_addr', {})
        self.idf_name_default = L.get('idf_name_default', 12.0)
        self.idf_addr_default = L.get('idf_addr_default', 12.0)
        self.core_freq = L.get('core_freq', {})
        cf = L.get('core_freq_pct', None)  # [counts sorted asc, cumulative share]
        self._pct_x = np.asarray(cf[0], dtype=np.float64) if cf else np.array([0.0])
        self._pct_y = np.asarray(cf[1], dtype=np.float64) if cf else np.array([0.0])
        self._px = self._py = None
        self.city_alias = dict(TB.IN_CITY_ALIASES) if country == 'India' else {}
        self.admin_equiv = TB.IN_ADMIN_EQUIV if country == 'India' else set()
        self.cidf_name = self.cidf_addr = None
        for kind in ('name', 'addr'):
            pth = f'{CACHE}/{country}_cidf_{kind}.npy'
            if os.path.exists(pth):
                setattr(self, 'cidf_' + kind, np.load(pth))

    def freq_pct(self, key):
        """Share of the country's S1 records whose name-core frequency <= this core's S1 frequency (keys seen once are
        not stored: count 1 is assumed for unknown keys)."""
        if self._px is None:
            self._px, self._py = self._pct_x.tolist(), self._pct_y.tolist()
        c = self.core_freq.get(key, 1)
        i = bisect.bisect_right(self._px, c) - 1
        return self._py[i] if i >= 0 else 0.0


_CACHE = {}


def learned_path(country):
    return f'{CACHE}/{country}.json'


def load_learned(country):
    p = learned_path(country)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        L = json.load(f)
    ex = f'{CACHE}/{country}_big.json'  # large tables (IDF, vocab, frequencies)
    if os.path.exists(ex):
        with open(ex) as f:
            L.update(json.load(f))
    return L


def get_tables(country, reload=False):
    if reload or country not in _CACHE:
        _CACHE[country] = Tables(country, load_learned(country))
    return _CACHE[country]


def all_tables(reload=False):
    return {c: get_tables(c, reload) for c in COUNTRIES}


def save_learned(country, small, big=None):
    os.makedirs(CACHE, exist_ok=True)
    with open(learned_path(country), 'w') as f:
        json.dump(small, f, indent=1, ensure_ascii=False, sort_keys=True)
    if big is not None:
        with open(f'{CACHE}/{country}_big.json', 'w') as f:
            json.dump(big, f, ensure_ascii=False)


# ================================================================ learning helpers (used by scripts/01_wordlists.py)
def _soft_eq(a, b):
    from rapidfuzz.distance import OSA
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    k = 1 if max(len(a), len(b)) <= 5 else 2
    return OSA.distance(a, b, score_cutoff=k) <= k


def learn_from_pairs(pairs_views, s1_final_counts, min_ins=None):
    """pairs_views: iterable of (NameView A, NameView B, AddrView A, AddrView B) for confident/anchor pairs.
    Returns dict with candidate stats for legal (6.1), filler (6.2), addr filler (6.3)."""
    ins_name, seen_name = Counter(), Counter()
    ins_addr, seen_addr = Counter(), Counter()
    moved = Counter()
    npairs = 0
    for na, nb, aa, ab in pairs_views:
        npairs += 1
        A, B = na.main, nb.main
        conf = len(set(A.content) & set(B.content)) >= max(1, 0.5 * len(set(A.content)))
        if conf and not nb.translit and not nb.alias and B.web is None and len(B.content) >= 1 and len(A.content) >= 1:
            # legal moved to front: B's first token is A's last token (and not A's first)
            if len(A.tokens) > 1 and B.tokens and B.tokens[0] == A.tokens[-1] and A.tokens[0] != B.tokens[0]:
                moved[B.tokens[0]] += 1
            aset = set(A.tokens)
            for t in set(B.content):
                seen_name[t] += 1
                if t not in aset and not any(_soft_eq(t, x) for x in A.tokens):
                    ins_name[t] += 1
        aconf = (not ab.empty and not aa.empty) and (len(aa.tokset & ab.tokset) >= 0.5 * max(len(aa.tokset), 1) or
                                                     (aa.house is not None and ab.house is not None and aa.house.val == ab.house.val))
        if aconf:
            asurf = set(aa.surf_toks)
            for t in set(ab.surf_toks):
                seen_addr[t] += 1
                if t not in asurf and not any(_soft_eq(t, x) for x in asurf):
                    ins_addr[t] += 1
    return dict(npairs=npairs, ins_name=ins_name, seen_name=seen_name, ins_addr=ins_addr, seen_addr=seen_addr, moved=moved,
                s1_final=s1_final_counts)


def select_lists(st, min_ins_name, min_rate_name, min_ins_addr, min_rate_addr, min_moved, s1_vocab_counts=None):
    """Turn the counters into lists. Filler: inserted often AND mostly-inserted when present in B."""
    legal = sorted(t for t, c in st['moved'].items() if c >= min_moved and t.isalpha()
                   and st['s1_final'].get(t, 0) >= 20)
    filler = []
    for t, c in st['ins_name'].most_common():
        if c < min_ins_name:
            break
        r = c / max(st['seen_name'][t], 1)
        if r >= min_rate_name and t.isalpha() and len(t) >= 3:
            filler.append((t, c, round(r, 3)))
    afill = []
    for t, c in st['ins_addr'].most_common():
        if c < min_ins_addr:
            break
        r = c / max(st['seen_addr'][t], 1)
        if r >= min_rate_addr and t.isalpha():
            afill.append((t, c, round(r, 3)))
    return legal, filler, afill


def learn_street_abbr(pairs_surf, min_count=10, min_share=0.6):
    """pairs_surf: iterable of (A house-component tokens, B house-component tokens) with equal house number.
    Collect b->a where the sequences have equal length, differ in exactly one position, b abbreviates a."""
    cnt = defaultdict(Counter)
    for ta, tb in pairs_surf:
        if len(ta) != len(tb) or not ta:
            continue
        diff = [(x, y) for x, y in zip(ta, tb) if x != y]
        if len(diff) != 1:
            continue
        a, b = diff[0]
        if not (a.isalpha() and b.isalpha()) or len(b) >= len(a) or b[0] != a[0]:
            continue
        it = iter(a)
        if all(ch in it for ch in b):  # b is a subsequence of a
            cnt[b][a] += 1
    out = {}
    stats = {}
    for b, c in cnt.items():
        a, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= min_share:
            out[b] = a
            stats[b] = (a, n, round(n / tot, 3))
    return out, stats
