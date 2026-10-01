"""Explain each labelled (S1 -> S2/S3) pair as S1 + a set of catalogued noise operations.

Normalise both sides with the current operation set, compare token multisets, and
account for every leftover token with an operation (typo, filler, digit drop, ...).
A pair is "explained" when nothing is left over. Leftovers are the residual that
tells us which operation is still missing from REVERSE_ENGINEERING.md.

Operations are split into two tiers:
  strict - the transformation is reproduced exactly (case, legal form, abbreviation, ...)
  loose  - the op is identified but its random content can't be checked
           (junk name, transliteration, replaced locality, heavy scramble)

Usage: python -m ber.branch.explain_pairs --sample P.parquet --out DIR [--examples N] [--filler-min N]
Input: a labelled pair sample (tools/learn_branch_lists.py writes it and runs the same learning step directly).

Package copy of the team's reverse-engineering branch (github saumilyagupta/KL-convergence, feat/features-reverse-engineering @ b14e566,
scripts/explain_pairs.py, as ported in work/features/branch_port/src): logic unchanged; only the fixed input / output paths were
replaced by the --sample / --out arguments. Accent folding is unicodedata NFKD (standard library).
"""
import argparse
import collections
import functools
import itertools
import json
import pathlib
import re
import unicodedata

import pandas as pd
from rapidfuzz.distance import OSA, Levenshtein

# PACKAGE: no fixed paths; main() takes --sample / --out (tools/learn_branch_lists.py builds the sample and calls learn())
SAMPLE = None
OUT = None

LOOSE_OPS = {"name_replaced", "transliteration", "scramble", "locality_replaced", "native_script",
             "number_inserted", "number_replaced"}

NONLATIN = re.compile(r"[ऀ-෿]")
ALIAS = re.compile(r"\b(?:formerly known as|also known as|doing business as|trading as|known as|formerly"
                   r"|a/k/a|f/k/a|d/b/a|d\.b\.a\.?|dba|aka|fka|t/a|nee)(?=[\s:]|$):?")
RECORD_TAG = re.compile(r"\s*#\d{4,}\s*$")
DOMAIN = re.compile(r"^(?:www\.)?([a-z0-9_\-]+?)\.?(?:com|in|net|org|co|fr)$")
HANDLE = re.compile(r"^[#@]([a-z0-9_]+)$")
URL_SUFFIX = re.compile(r"\s*\|\s*(?:www\.)?\S+\s*$")
ID_TAG = re.compile(r"\(\s*id\s*:\s*|\s*\d{4,}\s*\)\s*$|\s*\)\s*$")
PHONE = re.compile(r"\s*-\s*\d{6,}\s*$")
PLACEHOLDER = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b")
POBOX = re.compile(r"\b(?:po box|p\.o\. box|pmb)\s*\d+")
HALF = re.compile(r"\b1/2\b")
HOMOGLYPH = str.maketrans({"0": "o", "1": "l", "i": "l", "5": "s", "3": "e", "4": "a", "8": "b"})
NUM_PREFIX = re.compile(r"(?:door no|plot no|plot|h\s?no|hn|block [a-z]-|no|#+)\s*#*\s*(?:[a-z]{0,2}\d*[/-])?[a-z]?-?0*(\d+)")

LEGAL = set("""llc inc incorporated corp corporation co company ltd limited pvt private pc llp lp plc pllc public
               the sarl sas sasu eurl sa sci ei""".split())
COUNTRY_TAG = {"india", "france"}
CONNECTIVE = {"and", "of"}
ORDINALS = {w: str(i) for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
    "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}


@functools.lru_cache(maxsize=None)
def is_legal(t):
    """Legal-form token, tolerating homoglyphs ('lnc', 'c0rp', '1imited') and light typos ('pirvate')."""
    if t in LEGAL:
        return True
    h = t.translate(HOMOGLYPH)
    return any(h == w.translate(HOMOGLYPH) or (len(w) >= 5 and typo_ok(t, w)) for w in LEGAL)

STREET_TYPES = {
    "road": "rd", "rd": "rd", "street": "st", "st": "st", "saint": "st", "avenue": "ave", "ave": "ave", "av": "ave",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd",
    "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "circle": "cir", "cir": "cir", "trail": "trl", "trl": "trl",
    "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy", "terrace": "ter", "ter": "ter",
    "square": "sq", "sq": "sq", "point": "pt", "pt": "pt", "port": "pt", "turnpike": "tpke", "tpke": "tpke",
    "fort": "ft", "ft": "ft", "mount": "mt", "mt": "mt", "cove": "cv", "cv": "cv",
    "north": "n", "n": "n", "south": "s", "s": "s", "east": "e", "e": "e", "west": "w", "w": "w",
    "northeast": "ne", "ne": "ne", "northwest": "nw", "nw": "nw", "southeast": "se", "se": "se",
    "southwest": "sw", "sw": "sw",
}
STREET_FULL = collections.defaultdict(set)
for _full, _canon in STREET_TYPES.items():
    STREET_FULL[_canon].add(_full)
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
IN_STATES = {
    "uttar pradesh": "up", "karnataka": "ka", "maharashtra": "mh", "delhi": "dl", "gujarat": "gj", "bihar": "br",
    "west bengal": "wb", "tamil nadu": "tn", "andhra pradesh": "ap", "telangana": "ts", "rajasthan": "rj",
    "haryana": "hr", "madhya pradesh": "mp", "kerala": "kl", "punjab": "pb", "odisha": "od", "orissa": "od",
    "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "himachal pradesh": "hp",
    "goa": "ga", "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
}
STATE_RE = {c: re.compile(r"\b(" + "|".join(sorted(m, key=len, reverse=True)) + r")\b")
            for c, m in [("US", US_STATES), ("India", IN_STATES)]}
STATE_MAP = {"US": US_STATES, "India": IN_STATES}


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def ascii_drop(s):
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s.lower() if ord(c) < 128))


def typo_ok(a, b):
    """Light typo: small edit distance (adjacent transposition = 1) relative to length."""
    if min(len(a), len(b)) < 3:
        return False
    return OSA.distance(a, b) <= (1 if min(len(a), len(b)) <= 4 else 2)


def scramble_ok(a, b):
    """Heavy scramble: most of b's letters come from a (insertions/transpositions), e.g. 'prnoecssnig'."""
    if min(len(a), len(b)) < 4:
        return False
    ca, cb = collections.Counter(a), collections.Counter(b)
    shared = sum((ca & cb).values())
    return shared / len(a) >= 0.75 and shared / len(b) >= 0.6 and a[0] == b[0]


def pair_leftovers(ra, rb, test, op, ops):
    """Greedily pair leftover tokens that satisfy `test`; record `op` per pair."""
    for x in list(rb.elements()):
        if rb[x] <= 0:
            continue
        y = next((y for y in ra.elements() if test(y, x)), None)
        if y is not None:
            ops.append(op)
            rb[x] -= 1
            ra[y] -= 1
            ra += collections.Counter()
            rb += collections.Counter()


def segment_close(stem, tokens, max_dist):
    """Is `stem` within `max_dist` edits of some concatenation of distinct tokens (full or initial, any order)?"""
    tokens = [t for t in tokens if t][:6]
    best = 99
    for k in range(1, len(tokens) + 1):
        for perm in itertools.permutations(tokens, k):
            if len(perm) > 4 and k != len(tokens):
                continue
            for mask in range(2 ** k):
                cand = "".join(t[0] if mask >> i & 1 else t for i, t in enumerate(perm))
                if abs(len(cand) - len(stem)) > max_dist:
                    continue
                best = min(best, Levenshtein.distance(cand, stem, score_cutoff=max_dist + 1))
                if best == 0:
                    return 0
    return best


# ---------------------------------------------------------------- names

def name_tokens(s):
    s = fold(s)
    s = PHONE.sub("", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace(".", "")
    return re.findall(r"[a-z0-9]+", s)


@functools.lru_cache(maxsize=None)
def is_tag(t):
    """Country tag or legal form, possibly corrupted ('inia', 'idnia') or joined ('pvtltd')."""
    if is_legal(t) or t in COUNTRY_TAG:
        return True
    if any(len(t) >= 4 and OSA.distance(t, c) <= 2 for c in COUNTRY_TAG):
        return True
    return any(t.startswith(w) and is_legal(t[len(w):]) for w in LEGAL if len(t) > len(w))


def core(tokens):
    return [t for t in tokens if not is_tag(t) and t not in CONNECTIVE]


def filler_match(x, filler):
    """Inserted filler word, possibly with a typo or homoglyph ('ciyt', 'plto', '5ervices')."""
    if x in filler:
        return True
    h = x.translate(HOMOGLYPH)
    return any(h == f.translate(HOMOGLYPH) or typo_ok(x, f) or scramble_ok(f, x) for f in filler if len(f) >= 3)


def web_stem(fb):
    """Return (op, stem) if the name is a domain / handle, else None."""
    t = fb.strip().split()[-1] if fb.strip() else ""
    m = HANDLE.match(t)
    if m:
        return "handle", m.group(1)
    t = t.strip("()[]")
    m = DOMAIN.match(t)
    if m and (("." in t) or len(t) > 8) and len(fb.split()) <= 2:
        return "domain", m.group(1).replace("-", "").replace("_", "")
    return None


def explain_name(a, b, filler):
    ops = []
    if b.strip() == "":
        return ["name_empty"], []
    if NONLATIN.search(b):
        return ["transliteration"], []
    if b != a and fold(b) == fold(a):
        ops.append("case_or_accent")
    at = name_tokens(a)
    ca = core(at)
    fb = fold(b)
    if URL_SUFFIX.search(fb) and "|" in fb:
        ops.append("url_suffix")
        fb = URL_SUFFIX.sub("", fb)
    if RECORD_TAG.search(fb):
        ops.append("record_tag")
        fb = RECORD_TAG.sub("", fb)
    if re.search(r"\w,\w", fb) and not re.search(r"\w,\w", fold(a)):
        ops.append("comma_inserted")
        fb = re.sub(r"(\w),(\w)", r"", fb)
    if re.search(r"\(\s*id\s*:", fb):
        ops.append("id_tag")
        fb = re.sub(r"\(\s*id\s*:\s*", " ", fb)
        fb = re.sub(r"\s*\d{4,}\s*\)", " ", fb)
    m = ALIAS.search(fb) if not ALIAS.search(fold(a)) else None
    if m:
        ops.append("alias_marker")
        left, right = fb[:m.start()], fb[m.end():]
        fb = max((left, right), key=lambda x: len(set(core(name_tokens(x))) & set(ca)))
    w = web_stem(fb)
    if w:
        op, stem = w
        stems = {re.sub(r"[^a-z0-9]", "", stem), ascii_drop(stem)}
        full = core(at) + [t for t in at if is_legal(t)]
        best = min(segment_close(s, full, 2) for s in stems if s) if any(stems) else 99
        if best == 0:
            return ops + [op], []
        if best <= 2:
            return ops + [op, "typo"], []
        ops.append(op + "_unresolved")
    bt = name_tokens(fb)
    cb = core(bt)
    if [t for t in at if is_legal(t)] != [t for t in bt if is_legal(t)]:
        ops.append("legal_form")
    if len(cb) == 1 and 2 <= len(cb[0]) <= 6 and ca and segment_close(cb[0], [t[0] for t in ca], 0) == 0:
        return ops + ["acronym"], []
    ra, rb = collections.Counter(ca), collections.Counter(cb)
    common = ra & rb
    ra -= common
    rb -= common
    if not ra and not rb and ca != cb:
        ops.append("word_permute")
    # duplicated words ("Pinnacle Pinnacle")
    for x in list(rb.elements()):
        if x in common or x in ca:
            ops.append("word_duplicated")
            rb[x] -= 1
    rb += collections.Counter()
    pair_leftovers(ra, rb, lambda y, x: x.translate(HOMOGLYPH) == y.translate(HOMOGLYPH), "homoglyph", ops)
    pair_leftovers(ra, rb, typo_ok, "typo", ops)
    pair_leftovers(ra, rb, scramble_ok, "scramble", ops)
    for x in list(rb.elements()):
        la = list(ra.elements()) + ca
        if any(x == p + q for p in la for q in la if p != q) or any(len(x) >= 3 and x in y for y in la):
            ops.append("token_join_split")
            rb[x] -= 1
    rb += collections.Counter()
    for x in list(rb.elements()):
        if filler_match(x, filler):
            ops.append("filler_added")
            rb[x] -= 1
    rb += collections.Counter()
    if not common and not ops_has_match(ops) and rb:
        return ops + ["name_replaced"], []
    # one leftover on each side in the same slot -> the word was corrupted beyond typo range
    if len(list(rb.elements())) == 1 and len(list(ra.elements())) == 1:
        x, y = next(rb.elements()), next(ra.elements())
        if x[0] == y[0] or abs(len(x) - len(y)) <= 2:
            ops.append("token_corrupted")
            rb = collections.Counter()
            ra = collections.Counter()
    if ra:
        ops.append("word_dropped")
    return ops, list(rb.elements())


def ops_has_match(ops):
    return any(o in ops for o in ("typo", "scramble", "homoglyph", "token_join_split", "word_duplicated"))


# ---------------------------------------------------------------- addresses

def addr_norm(s, country):
    s = fold(s)
    ops = []
    if PLACEHOLDER.search(s):
        ops.append("placeholder")
        s = PLACEHOLDER.sub(" ", s)
    if POBOX.search(s):
        ops.append("po_box")
        s = POBOX.sub(" ", s)
    if HALF.search(s):
        ops.append("half_number")
        s = HALF.sub(" ", s)
    if country in STATE_RE:
        s = STATE_RE[country].sub(lambda m: STATE_MAP[country][m.group(1)], s)
    s = s.replace(".", " ")
    toks, comp_of = [], []
    for ci, comp in enumerate(s.split(",")):
        for t in re.findall(r"[a-z0-9]+", comp):
            m = re.fullmatch(r"0*(\d+)([a-z]*)", t)
            if m:
                toks.append(m.group(1) or "0")
                comp_of.append(ci)
                if m.group(2) and m.group(2) not in ("st", "nd", "rd", "th", "s", "n", "r", "t"):
                    toks.append(m.group(2))
                    comp_of.append(ci)
                continue
            toks.append(ORDINALS.get(t) or STREET_TYPES.get(t, t))
            comp_of.append(ci)
    return toks, comp_of, ops


def addr_typo(y, x):
    if y.isdigit() or x.isdigit():
        return False
    return any(typo_ok(x, f) for f in STREET_FULL.get(y, {y}) | {y})


def number_link(y, x, a_nums):
    if not (x.isdigit() and y.isdigit()):
        return None
    if len(x) < len(y) and (y.startswith(x) or y.endswith(x) or
                           any(y[:i] + y[i + 1:] == x for i in range(len(y)))):
        return "digit_dropped"
    if len(x) == len(y) and Levenshtein.distance(x, y) == 1:
        return "digit_substituted"
    return None


def explain_addr(a, b, country, filler, subst):
    if b.strip() == "":
        return ["addr_empty"], []
    ta, _, _ = addr_norm(a, country)
    comma_op = []
    if re.search(r"[A-Za-z],[A-Za-z]", b) and not re.search(r"[A-Za-z],[A-Za-z]", a):
        comma_op = ["comma_inserted"]
        b = re.sub(r"([A-Za-z]),([A-Za-z])", r"\1\2", b)
    tb, comp_b, ops = addr_norm(b, country)
    ops += comma_op
    if b == b.upper() and re.search("[A-Z]", b):
        ops.append("upper")
    if NONLATIN.search(b):
        ops.append("native_script")
    fb = fold(b)
    ra, rb = collections.Counter(ta), collections.Counter(tb)
    common = ra & rb
    ra -= common
    rb -= common
    a_nums = [t for t in ta if t.isdigit()]
    # number range "5000-5002": second number close above a matched number
    for x in [x for x in rb.elements() if x.isdigit()]:
        if any(y.isdigit() and 0 < int(x) - int(y) <= 10 and re.search(rf"\b0*{y}\s*-\s*0*{x}\b", fb) for y in a_nums):
            ops.append("number_range")
            rb[x] -= 1
    rb += collections.Counter()
    for x in [x for x in rb.elements() if x.isdigit()]:
        for y in [y for y in ra.elements() if y.isdigit()]:
            op = number_link(y, x, a_nums)
            if op:
                ops.append(op)
                rb[x] -= 1
                ra[y] -= 1
                ra += collections.Counter()
                rb += collections.Counter()
                break
    bn = [x for x in rb.elements() if x.isdigit()]
    for i in range(len(bn) - 1):
        joined = bn[i] + bn[i + 1]
        if joined in a_nums and rb[bn[i]] > 0 and rb[bn[i + 1]] > 0:
            ops.append("number_split")
            rb[bn[i]] -= 1
            rb[bn[i + 1]] -= 1
    rb += collections.Counter()
    for x in [x for x in rb.elements() if not x.isdigit()]:
        o = next((v for w, v in ORDINALS.items() if typo_ok(x, w)), None)
        if o and o in a_nums:
            ops.append("typo")
            rb[x] -= 1
    rb += collections.Counter()
    # inserted fake house number behind a prefix ("DOOR NO 415", "H.NO ##143", "Block D-00751", "#316")
    for x in [x for x in rb.elements() if x.isdigit()]:
        if any(m.group(1) == x for m in NUM_PREFIX.finditer(fb)) or re.match(rf"\s*#*0*{x}\b", fb):
            ops.append("number_inserted")
            rb[x] -= 1
    rb += collections.Counter()
    for x in list(rb.elements()):
        if x in common or x in ta:
            ops.append("component_duplicated")
            rb[x] -= 1
    rb += collections.Counter()
    pair_leftovers(ra, rb, addr_typo, "typo", ops)
    pair_leftovers(ra, rb, lambda y, x: (y, x) in subst, "alias_substitution", ops)
    for x in list(rb.elements()):
        if filler_match(x, filler) or x in CONNECTIVE:
            ops.append("filler_added")
            rb[x] -= 1
    rb += collections.Counter()
    # leftovers confined to comma components that contain no S1 token -> a replaced/added locality
    if rb:
        left_comps = {comp_b[i] for i, t in enumerate(tb) if t in rb}
        a_set = set(ta)
        if all(not any(tb[i] in a_set for i, c in enumerate(comp_b) if c == lc and not tb[i].isdigit())
               or all(tb[i] in rb or tb[i] in filler for i, c in enumerate(comp_b) if c == lc)
               for lc in left_comps) and any(not t.isdigit() for t in rb.elements()):
            ops.append("locality_replaced")
            rb = collections.Counter()
        elif all(t.isdigit() for t in rb.elements()) and a_nums:
            ops.append("number_replaced")
            rb = collections.Counter()
    if any(x.isdigit() for x in ra.elements()):
        ops.append("number_dropped")
    if any(not x.isdigit() for x in ra.elements()):
        ops.append("tokens_dropped")
    return ops, list(rb.elements())


# ---------------------------------------------------------------- driver

def learn(df, filler_min, subst_min):
    """Learn filler words and 1:1 token substitutions from leftovers on a learning split."""
    nf, af, sub = collections.Counter(), collections.Counter(), collections.Counter()
    for r in df.itertuples():
        _, left = explain_name(r.a_name, r.b_name, set())
        nf.update(t for t in left if not t.isdigit())
        ta, _, _ = addr_norm(r.a_addr, r.country)
        tb, _, _ = addr_norm(r.b_addr, r.country)
        ra, rb = collections.Counter(ta) - collections.Counter(tb), collections.Counter(tb) - collections.Counter(ta)
        wa = [t for t in ra if not t.isdigit()]
        wb = [t for t in rb if not t.isdigit()]
        af.update(wb)
        if len(wa) == 1 and len(wb) == 1 and min(len(wa[0]), len(wb[0])) >= 3 and wb[0] not in af:
            sub[(wa[0], wb[0])] += 1
    return ({t for t, n in nf.items() if n >= filler_min},
            {t for t, n in af.items() if n >= filler_min},
            {p for p, n in sub.items() if n >= subst_min})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=40)
    ap.add_argument("--filler-min", type=int, default=100)
    ap.add_argument("--subst-min", type=int, default=8)
    ap.add_argument("--sample", required=True, help="pair sample parquet (written by tools/learn_branch_lists.py)")
    ap.add_argument("--out", required=True, help="output directory for coverage.json / residual_examples.txt")
    args = ap.parse_args()
    OUT = pathlib.Path(args.out)
    df = pd.read_parquet(args.sample)
    OUT.mkdir(parents=True, exist_ok=True)

    # learn on half the sample, report on the other half so learned lists can't memorise the evaluation pairs
    lrn = df.sample(frac=0.5, random_state=1)
    ev = df.drop(lrn.index)
    name_filler, addr_filler, subst = learn(lrn, args.filler_min, args.subst_min)

    rows = []
    for r in ev.itertuples():
        nops, nleft = explain_name(r.a_name, r.b_name, name_filler)
        aops, aleft = explain_addr(r.a_addr, r.b_addr, r.country, addr_filler, subst)
        rows.append((r.src, r.country, not nleft, not aleft, nops, aops, nleft, aleft))
    res = pd.DataFrame(rows, columns=["src", "country", "name_ok", "addr_ok", "nops", "aops", "nleft", "aleft"],
                       index=ev.index)
    res["both_ok"] = res.name_ok & res.addr_ok
    res["strict"] = res.both_ok & ~res.apply(lambda r: bool(LOOSE_OPS & (set(r.nops) | set(r.aops))), axis=1)
    cols = ["name_ok", "addr_ok", "both_ok", "strict"]
    strata = res.groupby(["src", "country"])[cols].mean().round(4)
    overall = res[cols].mean().round(4)
    print("== coverage on held-out half (share of pairs fully explained)\n", strata.to_string(),
          "\noverall", overall.to_dict())

    def op_rates(col):
        c = collections.Counter()
        for (s, k), g in res.groupby(["src", "country"]):
            for ops in g[col]:
                for o in set(ops):
                    c[(o, f"{s}-{k}")] += 1 / len(g)
        return pd.Series(c).unstack().fillna(0).round(4)

    print("\n== name op rates\n", op_rates("nops").sort_values("S2-US", ascending=False).to_string())
    print("\n== address op rates\n", op_rates("aops").sort_values("S2-US", ascending=False).to_string())
    for field, col in [("name", "nleft"), ("addr", "aleft")]:
        c = collections.Counter(t for x in res[col] for t in x)
        print(f"\n== top residual {field} tokens", c.most_common(40))
    print(f"\nlearned name filler ({len(name_filler)}):", sorted(name_filler))
    print(f"learned addr filler ({len(addr_filler)}):", sorted(addr_filler))
    print(f"learned substitutions ({len(subst)}):", sorted(subst)[:60])

    bad_n = ev[~res.name_ok].sample(min(args.examples, int((~res.name_ok).sum())), random_state=0)
    bad_a = ev[~res.addr_ok].sample(min(args.examples, int((~res.addr_ok).sum())), random_state=0)
    with open(OUT / "residual_examples.txt", "w", encoding="utf-8") as f:
        for i, r in bad_n.iterrows():
            f.write(f"NAME [{r.src}-{r.country}] {r.a_name!r} -> {r.b_name!r} | left={res.nleft[i]}\n")
        for i, r in bad_a.iterrows():
            f.write(f"ADDR [{r.src}-{r.country}] {r.a_addr!r} -> {r.b_addr!r} | left={res.aleft[i]}\n")
    json.dump({"overall": overall.to_dict(),
               "strata": {f"{s}-{k}": v for (s, k), v in strata.to_dict("index").items()},
               "name_filler": sorted(name_filler), "addr_filler": sorted(addr_filler),
               "substitutions": sorted(map(list, subst))},
              open(OUT / "coverage.json", "w"), indent=1)


if __name__ == "__main__":
    main()
