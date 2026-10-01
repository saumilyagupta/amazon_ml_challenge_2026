"""Pair features: N1-N13, N15 (name), A1-A13 (address), S1-S2 (structural), X1-X4 (explainer).

Entry points
  explain_name(A, B, T) / explain_addr(A, B, T)  -> ExplainResult (explained, ops, leftovers, dropped, ...)
  pair_features(rec_a, rec_b)                     -> list of floats in FEATURES order (rec_* = cached (NameView, AddrView, src, T))
  compute_features(pairs, records)                -> polars.DataFrame[s1_id, cand_id, *FEATURES]   (single process)
  (see runner.py for the chunked 16-worker version)
Missing != agreement: every comparison has a both-present flag and takes the value -1 when one side is missing.
No country one-hot: the country only selects the learned word lists (T).
"""
import itertools
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA

from .views import name_view, addr_view, hg, skeleton, tnorm
from .wordlists import get_tables

# ------------------------------------------------------------------ operation inventory
OPS_NAME = ['translit', 'alias_marker', 'www_tag', 'phone', 'record_tag', 'id_tag', 'junk_prefix', 'honorific', 'bracket',
            'dotted', 'comma_in_word', 'hyphen', 'amp', 'accent', 'caps', 'lower', 'double_space',
            'domain', 'handle', 'acronym', 'junk_name', 'truncate',
            'legal_add', 'legal_drop', 'legal_restyle', 'legal_move', 'country_tag', 'filler_add', 'word_drop', 'permute',
            'word_dup', 'typo', 'homoglyph', 'scramble', 'corrupt', 'translit_fuzzy']
OPS_ADDR = ['addr_empty', 'placeholder', 'pobox', 'half', 'addr_caps', 'reorder', 'comp_drop', 'admin_alias', 'admin_drop',
            'street_abbr', 'ordinal', 'addr_typo', 'locality_replaced', 'locality_added', 'city_alias',
            'num_zero_pad', 'num_decor', 'num_suffix', 'num_range', 'num_hyphen_split', 'num_digit_drop', 'num_digit_sub',
            'num_near', 'num_other_pos', 'num_missing', 'num_inserted', 'num_replaced']
LOOSE = {'junk_name', 'translit', 'locality_replaced', 'locality_added', 'num_inserted', 'num_replaced'}
TEMPLATE = {'caps', 'lower', 'double_space', 'addr_caps'}  # source templates: flagged but not counted in n_ops
MARKER_OP = {'www_tag': 'www_tag', 'phone': 'phone', 'record_tag': 'record_tag', 'id_tag': 'id_tag', 'junk_prefix': 'junk_prefix',
             'bracket': 'bracket', 'dotted': 'dotted', 'comma_in_word': 'comma_in_word', 'hyphen': 'hyphen', 'amp': 'amp',
             'plus': 'amp', 'double_space': 'double_space', 'alias': 'alias_marker'}
NUM_REL = {0: 'missing', 1: 'exact', 2: 'range', 3: 'hyphen_split', 4: 'digit_drop', 5: 'digit_sub', 6: 'near', 7: 'other_pos', 8: 'different'}

FEATURES = (
    ['s1_is_s3', 's2_template_violation',
     'n_bp', 'n1_core_equal', 'n2_core_jacc', 'n2_cont_a', 'n2_cont_b', 'n4_soft_ratio_b', 'n4_soft_ratio_a', 'n4_n_exact',
     'n4_n_hg', 'n4_n_typo', 'n4_n_scramble', 'n5_left_cnt', 'n5_left_max_idf', 'n6_drop_cnt', 'n6_drop_max_idf', 'n7_dup',
     'n8_acronym', 'n9_web_dist', 'n9_web_sim', 'n10_alias', 'n10_alias_side', 'n11_junk_name', 'n12_b_caps', 'n12_b_lower',
     'n12_b_bracket', 'n12_b_dotted', 'n12_b_legal_first', 'n13_script_mismatch', 'n15_a_core_pct', 'n15_b_core_pct',
     'n_core_len_a', 'n_core_len_b',
     'a_bp', 'a1_num_bp', 'a2_num_exact', 'a3_num_rel', 'a4_num_overlap', 'a4_num_jacc', 'a5_num_inserted', 'a6_street_sim',
     'a7_street_key_bp', 'a7_street_key_eq', 'a8_tok_jacc', 'a8_cont_a', 'a8_cont_b', 'a9_left_cnt', 'a9_left_max_idf',
     'a10_loc_sim', 'a11_admin_agree', 'a12_addr_missing_b', 'a12_placeholder_b', 'a12_pobox_b',
     'x1_name_explained', 'x1_addr_explained', 'x1_both_explained', 'x2_n_ops', 'x2_n_ops_name', 'x2_n_ops_addr',
     'x3_loose_ops', 'x3_any_loose']
    + ['op_' + o for o in OPS_NAME] + ['op_' + o for o in OPS_ADDR])
VEC_FEATURES = ['n3_core_tfidf', 'a13_addr_tfidf']  # computed vectorised per chunk (char 3-5gram TF-IDF, per-country IDF)
ALL_FEATURES = FEATURES + VEC_FEATURES
_FIDX = {f: i for i, f in enumerate(FEATURES)}
_OPIDX = {o: _FIDX['op_' + o] for o in OPS_NAME + OPS_ADDR}
NF = len(FEATURES)


class ExplainResult:
    __slots__ = ('explained', 'ops', 'left', 'dropped', 'counts', 'web_dist', 'web_sim', 'acronym', 'side', 'n_match',
                 'junk', 'loose_left', 'num_rel', 'num_inserted', 'admin_agree')

    def __init__(self):
        self.explained, self.ops, self.left, self.dropped = False, [], [], []
        self.counts = [0, 0, 0, 0]  # exact, homoglyph, typo(OSA), scramble
        self.web_dist, self.web_sim, self.acronym, self.side, self.n_match, self.junk = -1, -1.0, 0, 0, 0, 0
        self.loose_left, self.num_rel, self.num_inserted, self.admin_agree = [], 0, 0, -1

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


# ------------------------------------------------------------------ name explainer
def _soft_kind(a, b):
    """0 none, 2 typo (OSA<=1/2 by length), 3 scramble/corrupt (letter bag >= 75% shared, same first letter)."""
    la, lb = len(a), len(b)
    if la < 4 or lb < 4:
        return 0
    k = 1 if max(la, lb) <= 5 else 2
    if abs(la - lb) <= k and OSA.distance(a, b, score_cutoff=k) <= k:
        return 2
    if a[0] == b[0] and abs(la - lb) <= 3:
        pool = list(a)
        shared = 0
        for ch in b:
            if ch in pool:
                pool.remove(ch)
                shared += 1
        if shared >= 0.75 * max(la, lb) or (shared >= la - 1 and 0 <= lb - la <= 3 and la >= 4):
            return 3
    return 0


def _lenient(a, b):
    """Heavily corrupted word (typo + scramble + inserted letters): same first letter, similar length,
    >= 60% of letters shared (multiset), or OSA <= 1 for short words."""
    la, lb = len(a), len(b)
    if la < 3 or lb < 3 or a[0] != b[0]:
        return False
    if OSA.distance(a, b, score_cutoff=1) <= 1:
        return True
    if not (0.6 <= lb / la <= 1.7) or la < 4:
        return False
    pool = list(a)
    shared = 0
    for ch in b:
        if ch in pool:
            pool.remove(ch)
            shared += 1
    return shared >= 0.6 * max(la, lb)


def _web_targets(a):
    ac = a.content
    notleg = [t for t in a.tokens if t not in a.legal_surf and t not in ('and', 'of', 'the', 'de', 'du', 'des', 'la', 'le', 'et')]
    tg = {''.join(ac), ''.join(a.core), ''.join(t for t in a.tokens if t not in ('and', 'of', 'the')), ''.join(notleg)}
    for k in range(1, len(ac)):  # the generator may drop trailing words
        if len(''.join(ac[:k])) >= 5:
            tg.add(''.join(ac[:k]))
    for k in range(1, len(notleg)):
        if len(''.join(notleg[:k])) >= 5:
            tg.add(''.join(notleg[:k]))
    if 1 < len(ac) <= 4:
        for p in itertools.permutations(ac):
            tg.add(''.join(p))
    if a.legal_surf:
        tg.add(''.join(ac) + a.legal_surf[0])
    tg.discard('')
    return tg


def web_match(stem, a, kind):
    """OSA distance of a web stem to concatenations of A's core tokens (any order, legal form optional, truncation)."""
    if not stem:
        return 99, 0.0, False
    hs = hg(stem)
    best = 99
    if kind == 'handle':
        ac = a.content
        targets = {''.join(ac[:k]) for k in range(1, min(3, len(ac)) + 1)} | {''.join(ac)}
    else:
        targets = _web_targets(a)
    for t in targets:
        if t.startswith(stem) and len(stem) >= max(4, len(t) - 3):
            best = 0
            break
        d = OSA.distance(hs, hg(t), score_cutoff=best)
        if d < best:
            best = d
    ini = ''.join(t[0] for t in a.content if t)
    if len(ini) >= 2 and (stem == ini or sorted(stem) == sorted(ini) or (a.legal_surf and stem == ini + a.legal_surf[0][0])):
        best = 0
    if best > 0 and kind != 'handle':  # initials of some words + one full word, any order ('nitowing', 'adenterprises', 'hanand')
        words = [t for t in a.tokens if t not in ('and', 'of', 'the')]
        for i, w in enumerate(words):
            if len(w) < 3 or w not in stem:
                continue
            rest = stem.replace(w, '', 1)
            if not rest or len(rest) > 4:
                continue
            pool = [x[0] for j, x in enumerate(words) if j != i]
            ok = True
            for ch in rest:
                if ch in pool:
                    pool.remove(ch)
                else:
                    ok = False
                    break
            if ok:
                best = 0
                break
    tl = max(len(stem), 1)
    sim = max(0.0, 1.0 - best / tl)
    ok = best <= max(1, int(round(0.12 * tl)))
    return best, sim, ok


def acronym_match(t, a):
    ws = [x for x in a.content if x[:1].isalpha()]
    ws2 = [x for x in a.tokens if x[:1].isalpha() and x not in a.legal_surf and x not in ('and', 'of', 'the', 'de', 'du', 'des', 'la', 'le', 'et')]
    st = sorted(t)
    for w in (ws, ws2):
        ini = ''.join(x[0] for x in w)
        if len(ini) < 2:
            continue
        cands = [ini[:6]]
        if a.legal_surf:
            cands.append((ini + a.legal_surf[0][0])[:6])
        for c in cands:
            if c == t or sorted(c) == st:
                return True
    return False


def _explain_side(a, s, T, allow_junk, translit, hon=None):
    r = ExplainResult()
    if s.web is not None:
        d, sim, ok = web_match(s.web, a, s.web_kind)
        r.web_dist, r.web_sim = d, sim
        if ok:
            r.ops.append(s.web_kind)
            r.explained = True
            r.n_match = 1
        else:
            r.left = [s.web]
        return r
    bc, bh = s.content, s.hg_content
    ac, ah = a.content, a.hg_content
    if not bc:
        return r
    if len(bc) == 1 and len(ac) >= 2 and 2 <= len(bc[0]) <= 6 and bc[0].isalpha() and bc[0] not in ac and acronym_match(bc[0], a):
        r.acronym = 1
        r.ops.append('acronym')
        r.explained = True
        r.n_match = 1
        return r
    na = len(ac)
    used = [False] * na
    if hon is not None and na > 1 and ac[0] == hon:  # honorific stripped from B but part of the S1 name ('Shree X')
        used[0] = True
    mi = [-1] * len(bc)
    cnt = r.counts
    for j, b in enumerate(bc):  # pass 1: exact
        for i in range(na):
            if not used[i] and ac[i] == b:
                used[i] = True; mi[j] = i; cnt[0] += 1
                break
    for j, b in enumerate(bc):  # pass 2: homoglyph
        if mi[j] >= 0:
            continue
        for i in range(na):
            if not used[i] and ah[i] == bh[j]:
                used[i] = True; mi[j] = i; cnt[1] += 1; r.ops.append('homoglyph')
                break
    for j, b in enumerate(bc):  # pass 3: typo / scramble / transliteration skeleton
        if mi[j] >= 0 or len(b) < 4:
            continue
        for i in range(na):
            if used[i]:
                continue
            k = _soft_kind(ah[i], bh[j])
            if k:
                used[i] = True; mi[j] = i; cnt[k] += 1; r.ops.append('typo' if k == 2 else 'scramble')
                break
            if translit and (skeleton(ac[i]) == skeleton(b) or tnorm(ac[i]) == tnorm(b)):
                used[i] = True; mi[j] = i; cnt[2] += 1; r.ops.append('translit_fuzzy')
                break
    if translit:  # short transliterated words: jai~jay, al~all
        for j, b in enumerate(bc):
            if mi[j] >= 0:
                continue
            for i in range(na):
                if not used[i] and ac[i][:1] == b[:1] and (tnorm(ac[i]) == tnorm(b) or OSA.distance(ac[i], b, score_cutoff=1) <= 1):
                    used[i] = True; mi[j] = i; cnt[2] += 1; r.ops.append('translit_fuzzy')
                    break
    legal_left = [x for x in a.legal_surf if x not in s.legal_surf]
    for j, b in enumerate(bc):  # pass 4: lenient 1:1 pairing of a leftover B word with an otherwise-dropped A word
        if mi[j] >= 0 or b in T.filler:
            continue
        hb = bh[j]
        for i in range(na):
            if not used[i] and _lenient(ah[i], hb):
                used[i] = True; mi[j] = i; cnt[3] += 1; r.ops.append('corrupt')
                break
        else:
            for x in legal_left:  # corrupted legal word ('Prsvooet' <- 'Private')
                if _lenient(hg(x), hb):
                    legal_left.remove(x); mi[j] = -2; r.ops.append('corrupt')
                    break
    matched = set(bc[j] for j in range(len(bc)) if mi[j] >= 0)
    for j, b in enumerate(bc):
        if mi[j] >= 0 or mi[j] == -2:
            continue
        if b in matched:
            r.ops.append('word_dup')
        elif b in T.filler:
            r.ops.append('filler_add')
        else:
            r.left.append(b)
    r.n_match = sum(1 for x in mi if x >= 0)
    r.dropped = [ac[i] for i in range(na) if not used[i]]
    # special wholesale paths: truncation to the first k words ('Internal Medicine Optimal Physicians' -> 'Internal Medicine')
    if r.dropped and r.n_match >= 1 and not r.left:
        k = sum(used)
        if all(used[:k]) and not any(used[k:]) and k < na and (len(r.dropped) >= 2 or k == 1):
            r.ops.append('truncate')
            r.dropped = []
    if r.n_match == 0 and len(bc) == 1 and r.left and allow_junk:
        t = bc[0]
        if t in T.junk or bh[0] in T.junk:
            r.junk = 1
            r.ops.append('junk_name')
            r.left = []
            r.dropped = []
            r.explained = True
            return r
    if r.n_match > 0:
        seq = [x for x in mi if x >= 0]
        if any(seq[k] > seq[k + 1] for k in range(len(seq) - 1)):
            r.ops.append('permute')
    r.ops.extend(['word_drop'] * len(r.dropped))
    # legal forms / country tags
    La, Lb = a.legal, s.legal
    if Lb or La:
        sa, sb = set(La), set(Lb)
        if sb - sa:
            r.ops.append('legal_add')
        if sa - sb:
            r.ops.append('legal_drop')
        if sa == sb and sorted(a.legal_surf) != sorted(s.legal_surf):
            r.ops.append('legal_restyle')
        if s.legal_first and not a.legal_first:
            r.ops.append('legal_move')
    if set(a.tags) != set(s.tags):
        r.ops.append('country_tag')
    r.explained = r.n_match > 0 and not r.left and len(r.dropped) <= 1
    return r


def explain_name(A, B, T):
    """A: NameView of the S1 record, B: NameView of the candidate. Returns ExplainResult or None if a name is missing."""
    if A.empty or B.empty:
        return None
    base = []
    for m in B.markers:
        if m not in A.markers:
            if m.startswith('hon_'):
                base.append('honorific')
            else:
                op = MARKER_OP.get(m)
                if op:
                    base.append(op)
    if B.translit:
        base.append('translit')
    if B.accent and not A.accent:
        base.append('accent')
    if B.caps and not A.caps:
        base.append('caps')
    elif B.lower and not A.lower:
        base.append('lower')
    a = A.sides[0]
    hon = None
    for m in B.markers:
        if m.startswith('hon_') and ' ' not in m:
            hon = m[4:].replace('/', '')
    best, bkey = None, None
    for si, s in enumerate(B.sides):
        r = _explain_side(a, s, T, allow_junk=not B.alias, translit=B.translit, hon=hon)
        key = (r.explained, -len(r.left), -len(r.dropped), r.n_match, -len(r.ops))
        if best is None or key > bkey:
            best, bkey = r, key
            best.side = si
    best.ops = base + best.ops
    return best


# ------------------------------------------------------------------ address explainer
def _num_rel(a, b, bset):
    if a.val == b.val:
        return 1
    if (b.rng and b.rng[0] <= a.val <= b.rng[1]) or (a.rng and a.rng[0] <= b.val <= a.rng[1]):
        return 2
    if b.concat == a.val or a.concat == b.val:
        return 3
    sa, sb = str(a.val), str(b.val)
    la, lb = len(sa), len(sb)
    if lb < la and (sa.startswith(sb) or sa.endswith(sb) or (lb == la - 1 and any(sa[:i] + sa[i + 1:] == sb for i in range(la)))):
        return 4
    if la == lb and sum(x != y for x, y in zip(sa, sb)) == 1:
        return 5
    d = abs(a.val - b.val)
    if d <= 2 or d == 10:
        return 6
    if a.val in bset:
        return 7
    return 8


_REL_OP = {2: 'num_range', 3: 'num_hyphen_split', 4: 'num_digit_drop', 5: 'num_digit_sub', 6: 'num_near', 7: 'num_other_pos',
           8: 'num_replaced'}


def number_relation(A, B):
    """Categorical relation between the canonical house numbers (A3) + inserted-prefix detection (A5)."""
    ops = []
    a, b = A.house, B.house
    inserted = 0
    if b is not None and b.ins_kw is not None and b.val not in A.numset:
        rest = [n for n in B.nums if n is not b and n.decor != 'unit']
        nxt = rest[0] if rest else None
        if a is None:
            inserted, b = 1, nxt
        elif nxt is not None and _num_rel(a, nxt, B.numset) <= 6:
            inserted, b = 1, nxt
        elif _num_rel(a, b, B.numset) >= 7:
            inserted, b = 1, nxt
    if inserted:
        ops.append('num_inserted')
    if a is None or b is None:
        if a is not None and b is None and not inserted:
            ops.append('num_missing')
        return 0, ops, inserted
    rel = _num_rel(a, b, B.numset)
    if rel == 8 and b.val in A.numset:
        rel = 7
    if rel in _REL_OP:
        ops.append(_REL_OP[rel])
    elif rel == 1 and b.rng and not a.rng:  # '9004' -> '9004-9006': range whose low end is the house number
        ops.append('num_range')
    if b.zpad and not a.zpad:
        ops.append('num_zero_pad')
    if b.decor and b.decor != 'unit' and not a.decor:
        ops.append('num_decor')
    if (b.suffix or None) != (a.suffix or None):
        ops.append('num_suffix')
    return rel, ops, inserted


def _admin_agree(A, B, T):
    if not A.admin or not B.admin:
        return -1
    if A.admin & B.admin:
        return 1
    for x in A.admin:
        for y in B.admin:
            if (x, y) in T.admin_equiv:
                return 1
    return 0


def _tok_explained(t, A, T):
    if t in A.tokset:
        return 1
    alias = T.city_alias.get(t)
    if alias is not None and alias in A.tokset:
        return 3
    if len(t) >= 4:
        for x in A.tokset:
            if len(x) >= 4 and _soft_kind(x, t):
                return 2
    return 0


def explain_addr(A, B, T):
    r = ExplainResult()
    if A.empty:
        return None
    if B.empty:
        r.explained = True
        r.ops = ['addr_empty'] + (['placeholder'] if B.placeholder else [])
        return r
    ops = r.ops
    if B.placeholder and not A.placeholder:
        ops.append('placeholder')
    if B.pobox and not A.pobox:
        ops.append('pobox')
    if B.half and not A.half:
        ops.append('half')
    if B.caps and not A.caps:
        ops.append('addr_caps')
    ag = _admin_agree(A, B, T)
    r.admin_agree = ag
    if ag == 1 and sorted(A.admin_surf) != sorted(B.admin_surf):
        ops.append('admin_alias')
    elif A.admin and not B.admin:
        ops.append('admin_drop')
    rel, nops, ins = number_relation(A, B)
    r.num_rel, r.num_inserted = rel, ins
    ops.extend(nops)
    # component-level token explanation
    left_by_comp = []
    a_hit = set()
    typo = alias = False
    order = []
    for ci, ct in enumerate(B.comp_toks):
        if not ct:
            continue
        lf = []
        for t in ct:
            k = _tok_explained(t, A, T)
            if k == 0:
                lf.append(t)
            else:
                typo |= k == 2
                alias |= k == 3
        if lf:
            left_by_comp.append((ci, lf))
    if typo:
        ops.append('addr_typo')
    if alias:
        ops.append('city_alias')
    bset = B.tokset
    for ci, ct in enumerate(A.comp_toks):
        if ct and any((t in bset) for t in ct):
            a_hit.add(ci)
            first = min((B.tokens.index(t) for t in ct if t in bset), default=None)
            if first is not None:
                order.append(first)
    if any(order[k] > order[k + 1] for k in range(len(order) - 1)):
        ops.append('reorder')
    a_body = [ci for ci, ct in enumerate(A.comp_toks) if ct]
    missing_comps = [ci for ci in a_body if ci not in a_hit and not (A.house is not None and A.house.comp == ci and B.house is not None)]
    if missing_comps:
        ops.append('comp_drop')
    r.left = [t for _, lf in left_by_comp for t in lf]
    if len(left_by_comp) == 1 and not any(ch.isdigit() for ch in B.comps[left_by_comp[0][0]]):
        ops.append('locality_replaced' if missing_comps else 'locality_added')
        r.loose_left = r.left
        leftover_ok = True
    else:
        leftover_ok = not r.left
    if A.surf_toks and B.surf_toks:
        asurf = set(A.surf_toks)
        for t in B.surf_toks:
            c = T.street.get(t)
            if c is not None and t not in asurf and c in A.tokset and any(T.street.get(x) == c for x in asurf):
                ops.append('street_abbr')
                break
    if A.ordinal and B.ordinal and A.ordword != B.ordword:
        ops.append('ordinal')
    r.explained = leftover_ok and ag != 0
    return r


# ------------------------------------------------------------------ record cache + pair features
def record_views(name, addr, country, src, name_en=None, addr_en=None):
    T = get_tables(country)
    return (name_view(name, src, T, name_en), addr_view(addr, T, addr_en), src, T)


def _jacc(a, b):
    if not a or not b:
        return -1.0, -1.0, -1.0
    i = len(a & b)
    return i / len(a | b), i / len(a), i / len(b)


def _template_violation(nb, ab, src, country):
    if ab.empty:
        return 0
    if src == 2:  # S2 writes the address body (admin component excluded) in upper case in every country
        return int(not ab.caps)
    if src == 3:
        if country == 'US':
            return int(any(len(x.strip()) == 2 for x in ab.admin_surf))
        if country == 'India':
            return int(any(len(x.strip()) != 2 for x in ab.admin_surf))
        return int(ab.caps)
    return 0


def pair_features(ra, rb):
    na, aa, _, T = ra
    nb, ab, srcb, _ = rb
    out = [0.0] * NF
    F = _FIDX
    out[F['s1_is_s3']] = 1.0 if srcb == 3 else 0.0
    out[F['s2_template_violation']] = _template_violation(nb, ab, srcb, T.country)
    # ---------------- name
    en = explain_name(na, nb, T)
    n_ops_name = n_ops_addr = loose = 0
    if en is None:
        for f in ('n1_core_equal', 'n2_core_jacc', 'n2_cont_a', 'n2_cont_b', 'n4_soft_ratio_b', 'n4_soft_ratio_a', 'n5_left_cnt',
                  'n5_left_max_idf', 'n6_drop_cnt', 'n6_drop_max_idf', 'n9_web_dist', 'n9_web_sim', 'x1_name_explained',
                  'x1_both_explained', 'n15_b_core_pct', 'n_core_len_b'):
            out[F[f]] = -1.0
    else:
        out[F['n_bp']] = 1.0
        A, B = na.sides[0], nb.sides[en.side]
        out[F['n1_core_equal']] = 1.0 if sorted(A.core) == sorted(B.core) else 0.0
        j, ca, cb = _jacc(set(A.core), set(B.core))
        out[F['n2_core_jacc']], out[F['n2_cont_a']], out[F['n2_cont_b']] = j, ca, cb
        nm = en.n_match
        out[F['n4_soft_ratio_b']] = nm / max(len(B.content), 1)
        out[F['n4_soft_ratio_a']] = min(1.0, nm / max(len(A.content), 1))
        c = en.counts
        out[F['n4_n_exact']], out[F['n4_n_hg']], out[F['n4_n_typo']], out[F['n4_n_scramble']] = c[0], c[1], c[2], c[3]
        idf, dflt = T.idf_name, T.idf_name_default
        out[F['n5_left_cnt']] = len(en.left)
        out[F['n5_left_max_idf']] = max((idf.get(t, dflt) for t in en.left), default=0.0)
        out[F['n6_drop_cnt']] = len(en.dropped)
        out[F['n6_drop_max_idf']] = max((idf.get(t, dflt) for t in en.dropped), default=0.0)
        out[F['n7_dup']] = 1.0 if B.n_dup > A.n_dup else 0.0
        out[F['n8_acronym']] = en.acronym
        out[F['n9_web_dist']] = en.web_dist
        out[F['n9_web_sim']] = en.web_sim
        out[F['n10_alias_side']] = en.side
        if en.n_match == 0 and not en.acronym and en.web_dist < 0:
            t = B.content[0] if len(B.content) == 1 else None
            if en.junk or (t is not None and (t in T.junk or (t.isalpha() and 5 <= len(t) <= 14 and T.s1_vocab and t not in T.s1_vocab))):
                out[F['n11_junk_name']] = 1.0
        out[F['n15_b_core_pct']] = T.freq_pct(B.core_key)
        out[F['n_core_len_b']] = len(B.core)
        out[F['x1_name_explained']] = 1.0 if en.explained else 0.0
        for o in en.ops:
            out[_OPIDX[o]] = 1.0
            if o not in TEMPLATE:
                n_ops_name += 1
            if o in LOOSE:
                loose += 1
    out[F['n10_alias']] = 1.0 if nb.alias else 0.0
    B0 = nb.sides[0]
    out[F['n12_b_caps']] = 1.0 if nb.caps else 0.0
    out[F['n12_b_lower']] = 1.0 if nb.lower else 0.0
    out[F['n12_b_bracket']] = 1.0 if 'bracket' in nb.markers else 0.0
    out[F['n12_b_dotted']] = 1.0 if 'dotted' in nb.markers else 0.0
    out[F['n12_b_legal_first']] = 1.0 if B0.legal_first else 0.0
    out[F['n13_script_mismatch']] = 1.0 if (nb.indic and not na.indic) else 0.0
    out[F['n15_a_core_pct']] = T.freq_pct(na.sides[0].core_key)
    out[F['n_core_len_a']] = len(na.sides[0].core)
    # ---------------- address
    ea = explain_addr(aa, ab, T)
    both_addr = not aa.empty and not ab.empty
    out[F['a_bp']] = 1.0 if both_addr else 0.0
    out[F['a12_addr_missing_b']] = 1.0 if ab.empty else 0.0
    out[F['a12_placeholder_b']] = 1.0 if ab.placeholder else 0.0
    out[F['a12_pobox_b']] = 1.0 if ab.pobox else 0.0
    if not both_addr:
        for f in ('a2_num_exact', 'a3_num_rel', 'a4_num_overlap', 'a4_num_jacc', 'a6_street_sim', 'a7_street_key_eq', 'a8_tok_jacc',
                  'a8_cont_a', 'a8_cont_b', 'a9_left_cnt', 'a9_left_max_idf', 'a10_loc_sim', 'a11_admin_agree'):
            out[F[f]] = -1.0
    else:
        rel = ea.num_rel
        out[F['a1_num_bp']] = 1.0 if rel > 0 else 0.0
        out[F['a2_num_exact']] = (1.0 if rel == 1 else 0.0) if rel > 0 else -1.0
        out[F['a3_num_rel']] = rel
        if aa.numset and ab.numset:
            i = len(aa.numset & ab.numset)
            out[F['a4_num_overlap']] = 1.0 if i else 0.0
            out[F['a4_num_jacc']] = i / len(aa.numset | ab.numset)
        else:
            out[F['a4_num_overlap']] = out[F['a4_num_jacc']] = -1.0
        out[F['a5_num_inserted']] = ea.num_inserted
        if aa.street_toks and ab.street_toks:
            out[F['a6_street_sim']] = fuzz.ratio(' '.join(sorted(aa.street_toks)), ' '.join(sorted(ab.street_toks))) / 100.0
        else:
            out[F['a6_street_sim']] = -1.0
        if aa.street_key and ab.street_key:
            out[F['a7_street_key_bp']] = 1.0
            out[F['a7_street_key_eq']] = 1.0 if aa.street_key == ab.street_key else 0.0
        else:
            out[F['a7_street_key_eq']] = -1.0
        j, ca, cb = _jacc(aa.tokset, ab.tokset)
        out[F['a8_tok_jacc']], out[F['a8_cont_a']], out[F['a8_cont_b']] = j, ca, cb
        idf, dflt = T.idf_addr, T.idf_addr_default
        out[F['a9_left_cnt']] = len(ea.left)
        out[F['a9_left_max_idf']] = max((idf.get(t, dflt) for t in ea.left), default=0.0)
        if aa.localities and ab.localities:
            best = 0.0
            for x in aa.localities:
                for y in ab.localities:
                    s = fuzz.ratio(x, y)
                    if s > best:
                        best = s
            out[F['a10_loc_sim']] = best / 100.0
        else:
            out[F['a10_loc_sim']] = -1.0
        out[F['a11_admin_agree']] = ea.admin_agree
    if ea is not None:
        out[F['x1_addr_explained']] = 1.0 if ea.explained else 0.0
        for o in ea.ops:
            out[_OPIDX[o]] = 1.0
            if o not in TEMPLATE:
                n_ops_addr += 1
            if o in LOOSE:
                loose += 1
    if en is not None:
        out[F['x1_both_explained']] = 1.0 if (en.explained and ea is not None and ea.explained) else 0.0
    out[F['x2_n_ops_name']] = n_ops_name
    out[F['x2_n_ops_addr']] = n_ops_addr
    out[F['x2_n_ops']] = n_ops_name + n_ops_addr
    out[F['x3_loose_ops']] = loose
    out[F['x3_any_loose']] = 1.0 if loose else 0.0
    return out


# ------------------------------------------------------------------ vectorised char TF-IDF (N3, A13)
_HV = None


def _hv():
    global _HV
    if _HV is None:
        from sklearn.feature_extraction.text import HashingVectorizer
        _HV = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=2 ** 20, alternate_sign=False, norm=None,
                                dtype=np.float32)
    return _HV


def _tfidf_rows(texts, countries, kind):
    from sklearn.preprocessing import normalize
    X = _hv().transform(texts).tocsr()
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    cvec = np.asarray(countries)
    for c in set(countries):
        T = get_tables(c)
        idf = getattr(T, 'cidf_' + kind, None)
        if idf is None:
            continue
        m = cvec[rows] == c
        X.data[m] *= idf[X.indices[m]]
    return normalize(X, norm='l2', copy=False)


def _rowcos(Xa, ia, Xb, ib, empty_a, empty_b):
    out = np.asarray(Xa[ia].multiply(Xb[ib]).sum(axis=1)).ravel().astype(np.float32)
    out[empty_a[ia] | empty_b[ib]] = -1.0
    return out


# ------------------------------------------------------------------ single-process entry point
REC_FIELDS = ('name', 'addr', 'country', 'src', 'name_en', 'addr_en')


def _records_lookup(records, ids):
    """records: polars DataFrame (entity_id, name, addr, country[, src, name_en, addr_en]) or dict id -> tuple/dict."""
    if isinstance(records, dict):
        out = {}
        for i in ids:
            r = records[i]
            if isinstance(r, dict):
                r = tuple(r.get(k) for k in REC_FIELDS)
            out[i] = r
        return out
    df = records.filter(pl.col('entity_id').is_in(list(ids)))
    cols = [c for c in REC_FIELDS if c in df.columns]
    if 'src' not in df.columns:
        df = df.with_columns(pl.col('entity_id').str.slice(1, 1).cast(pl.Int8).alias('src'))
    for c in ('name_en', 'addr_en'):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, pl.Utf8).alias(c))
    df = df.select(['entity_id'] + list(REC_FIELDS))
    return {r[0]: r[1:] for r in df.iter_rows()}


def features_from_texts(a_ids, a_rec, b_ids, b_rec, vec=True):
    """a_rec/b_rec: dict id -> (name, addr, country, src, name_en, addr_en). Returns dict feature -> np.ndarray."""
    cache = {}

    def views(i, rec):
        v = cache.get(i)
        if v is None:
            n, a, c, s, ne, ae = rec[i]
            v = record_views(n or '', a or '', c, int(s), ne, ae)
            cache[i] = v
        return v

    M = np.empty((len(a_ids), NF), dtype=np.float32)
    for k, (x, y) in enumerate(zip(a_ids, b_ids)):
        M[k] = pair_features(views(x, a_rec), views(y, b_rec))
    out = {f: M[:, i] for i, f in enumerate(FEATURES)}
    if vec:
        ua = list(dict.fromkeys(a_ids)); ub = list(dict.fromkeys(b_ids))
        pa = {x: i for i, x in enumerate(ua)}; pb = {x: i for i, x in enumerate(ub)}
        ia = np.fromiter((pa[x] for x in a_ids), dtype=np.int64, count=len(a_ids))
        ib = np.fromiter((pb[x] for x in b_ids), dtype=np.int64, count=len(b_ids))
        va = [cache[x] for x in ua]; vb = [cache[x] for x in ub]
        ca = [v[3].country for v in va]; cb = [v[3].country for v in vb]

        def ntext(v, side0=True):
            nv = v[0]
            if nv.empty:
                return ''
            return ' '.join(nv.sides[0].core)

        na_t = [ntext(v) for v in va]; nb_t = [ntext(v) for v in vb]
        aa_t = [' '.join(v[1].tokens) for v in va]; ab_t = [' '.join(v[1].tokens) for v in vb]
        Xa, Xb = _tfidf_rows(na_t, ca, 'name'), _tfidf_rows(nb_t, cb, 'name')
        out['n3_core_tfidf'] = _rowcos(Xa, ia, Xb, ib, np.array([t == '' for t in na_t]), np.array([t == '' for t in nb_t]))
        Xa, Xb = _tfidf_rows(aa_t, ca, 'addr'), _tfidf_rows(ab_t, cb, 'addr')
        out['a13_addr_tfidf'] = _rowcos(Xa, ia, Xb, ib, np.array([t == '' for t in aa_t]), np.array([t == '' for t in ab_t]))
    return out


def compute_features(pairs, records, vec=True):
    """pairs: polars DataFrame with s1_id, cand_id. records: polars DataFrame or dict (see _records_lookup).
    Returns polars DataFrame [s1_id, cand_id, *ALL_FEATURES] in the input row order (single process)."""
    a_ids = pairs['s1_id'].to_list()
    b_ids = pairs['cand_id'].to_list()
    rec = _records_lookup(records, set(a_ids) | set(b_ids))
    out = features_from_texts(a_ids, rec, b_ids, rec, vec=vec)
    cols = {'s1_id': a_ids, 'cand_id': b_ids}
    cols.update(out)
    return pl.DataFrame(cols)
