"""Canonical per-record views (spec section 'Canonical views per record').

name_view(raw, src, T, name_en=None) -> NameView    (fold / hg / name_tokens / name_core / alias_sides / web_stem)
addr_view(raw, T, addr_en=None)      -> AddrView    (addr_comps / addr_tokens / numbers / street_key / admin)
T is a wordlists.Tables object for the record's country (legal forms, filler, street types, admin aliases, ...).
Everything is pure Python + precompiled regexes; views are computed once per record and reused for all its pairs.
"""
import re
from anyascii import anyascii
from rapidfuzz.distance import OSA

# ------------------------------------------------------------------ folding
_NDEG = re.compile(r'\b([Nn])\s?[°º]')
_PRE = str.maketrans({'°': ' ', 'º': ' ', '’': "'", '‘': "'", '`': "'", '´': "'", ' ': ' '})
_WS = re.compile(r'\s+')
HG = str.maketrans({'0': 'o', '1': 'l', 'i': 'l', '5': 's', '3': 'e', '4': 'a', '8': 'b'})
_INDIC = re.compile(r'[ऀ-෿]')
_ACCENTED = re.compile(r'[À-ÖØ-öø-ɏ]')


def has_indic(s):
    return bool(s) and _INDIC.search(s) is not None


def fold(s):
    """NFKD-like accent strip via anyascii (ISC), lowercase, collapse whitespace. 'N°' -> 'no '."""
    if not s:
        return ''
    if '°' in s or 'º' in s:
        s = _NDEG.sub(r'\1o ', s)
    s = s.translate(_PRE)
    if not s.isascii():
        s = anyascii(s)
    return _WS.sub(' ', s.lower()).strip()


def hg(tok):
    """Homoglyph view: 0->o 1->l i->l 5->s 3->e 4->a 8->b (only for tokens containing a letter)."""
    if tok.isdigit():
        return tok
    return tok.translate(HG)


def tnorm(tok):
    """Transliteration spelling normaliser (laxmi~lakshmi, jai~jay, sree~shree)."""
    return tok.replace('ksh', 'x').replace('ks', 'x').replace('sh', 's').replace('ee', 'i').replace('aa', 'a').replace('oo', 'u') \
        .replace('y', 'i').replace('w', 'v').replace('ph', 'f').replace('th', 't').replace('dh', 'd').replace('bh', 'b')


def skeleton(tok):
    """Consonant skeleton (for transliterated tokens): drop non-initial vowels/h/y, collapse doubles."""
    s = tok.replace('ph', 'f').replace('sh', 's').replace('th', 't').replace('c', 'k').replace('q', 'k').replace('w', 'v').replace('z', 'j')
    out = s[:1] + re.sub(r'[aeiouyh]', '', s[1:])
    return re.sub(r'(.)\1+', r'\1', out)


# ------------------------------------------------------------------ names
RE_WWW_TAG = re.compile(r'\s*\|\s*(?:https?://)?(?:www\.)?([^\s|]*)\s*$', re.I)
RE_PHONE = re.compile(r'\s+-\s*\+?\d[\d ]{5,}\d\s*$')
RE_RECTAG = re.compile(r'\s+#\s?\d{3,}\s*$')
RE_IDTAG = re.compile(r'\s*[\(\[]\s*id\s*:?\s*\d+\s*[\)\]]\s*$', re.I)
RE_ALIAS = re.compile(r'(?:^|\s)(formerly known as|also known as|doing business as|trading as|known as|formerly|'
                      r'a/k/a|f/k/a|d/b/a|d\.b\.a\.?|dba|fka|aka|t/a|n[eé]e)\s*:?(?=\s|$)', re.I)
RE_JUNKPRE = re.compile(r'^\s*(?:--|<<|>>|##|\*\*\*|\.\.\.|~~|==|\*\*|\*|!!|\?\?|//|::|\+\+|__|\^\^)\s*')
RE_HONOR = re.compile(r'^(the\s+the|the|mr|mrs|ms|dr|m/s|sri|shri|smt|shree)\.?\s+(?=\S)', re.I)
RE_DOTTED = re.compile(r'(?<![a-z0-9])((?:[a-z]\.){2,}[a-z]?)(?![a-z0-9])')
RE_COMMA_IN_WORD = re.compile(r'(?<=[a-z]),(?=[a-z])')
RE_HYPHEN_WORD = re.compile(r'[a-z]-[a-z]')
RE_BRACKET_WORD = re.compile(r'[\(\[]\s*([a-z]+)\s*[\)\]]')
RE_WEB = re.compile(r'^(?:https?://)?(www\.)?([a-z0-9][a-z0-9\-]*?)\.(com|c0m|co\.in|in|net|org|co|fr|io|biz|info|us)$')
RE_TOK = re.compile(r'[a-z0-9]+')


class Side:
    __slots__ = ('tokens', 'content', 'core', 'hg_content', 'legal', 'legal_surf', 'tags', 'legal_first', 'web', 'web_kind',
                 'core_key', 'filler', 'n_dup')

    def __init__(self):
        self.tokens, self.content, self.core, self.hg_content = [], [], [], []
        self.legal, self.legal_surf, self.tags, self.filler = [], [], [], []
        self.legal_first, self.web, self.web_kind, self.core_key, self.n_dup = False, None, None, '', 0


class NameView:
    __slots__ = ('raw', 'empty', 'indic', 'translit', 'markers', 'alias', 'sides', 'caps', 'lower', 'accent', 'nonascii')

    @property
    def main(self):
        return self.sides[0]


def _is_country_tag(w):
    return 4 <= len(w) <= 8 and (OSA.distance(w, 'india', score_cutoff=2) <= 2 or OSA.distance(w, 'france', score_cutoff=2) <= 2)


def _side(x, T, markers):
    sd = Side()
    xs = x.strip()
    if not xs:
        return sd
    if ' ' not in xs:
        if xs[0] in '@#' and len(xs) > 2 and xs[1].isalpha():
            stem = ''.join(RE_TOK.findall(fold(xs[1:])))
            sd.web, sd.web_kind = stem, 'handle'
        else:
            f = fold(xs)
            m = RE_WEB.match(f)
            if m:
                sd.web, sd.web_kind = m.group(2).replace('-', ''), 'domain'
            elif f.startswith('www.'):
                sd.web, sd.web_kind = ''.join(RE_TOK.findall(f[4:])), 'domain'
            elif len(f) >= 8 and f.endswith('com') and f.isalnum() and f[:-3] not in T.s1_vocab:
                sd.web, sd.web_kind = f[:-3], 'domain'
        if sd.web is not None:
            sd.tokens = sd.content = sd.core = [sd.web]
            sd.hg_content = [hg(sd.web)]
            sd.core_key = sd.web
            return sd
    f = fold(xs)
    if '&' in xs:
        markers.add('amp')
    if '+' in xs:
        markers.add('plus')
    if '(' in xs or '[' in xs:
        markers.add('bracket')
    if RE_HYPHEN_WORD.search(f):
        markers.add('hyphen')
    if ',' in f and RE_COMMA_IN_WORD.search(f):
        markers.add('comma_in_word')
        f = RE_COMMA_IN_WORD.sub('', f)
    if '.' in f:
        g = RE_DOTTED.sub(lambda m: m.group(1).replace('.', ''), f)
        if g != f:
            markers.add('dotted')
            f = g
    brk = set(RE_BRACKET_WORD.findall(f)) if ('(' in f or '[' in f) else ()
    f = f.replace('&', ' and ').replace('+', ' and ').replace("'", '')
    toks = RE_TOK.findall(f)
    sd.tokens = toks
    legal, content, core = T.legal, [], []
    for t in toks:
        h = hg(t) if (t[0] == 'l' or not t.isalpha()) else None  # homoglyph variants: 'lnc', 'l1c', 'c0', '1td', '5ervices', '0f'
        if t in legal or (h is not None and h in T.legal_hg):
            sd.legal.append(legal[t] if t in legal else T.legal_hg[h])
            sd.legal_surf.append(t)
        elif t in ('india', 'france') or (t in brk and _is_country_tag(t)):
            sd.tags.append(t)
        elif t in T.connectives or (h is not None and h in T.conn_hg):
            continue
        else:
            content.append(t)
            if t in T.filler or (h is not None and h in T.filler_hg):
                sd.filler.append(t)
            else:
                core.append(t)
    if not content:  # name made only of legal/connective words: keep what is there
        content = [t for t in toks if t not in T.connectives] or list(toks)
    if not core:
        core = list(content)
    sd.content, sd.core = content, core
    sd.hg_content = [hg(t) for t in content]
    sd.legal_first = bool(toks) and toks[0] in legal and len(toks) > 1
    sd.core_key = ' '.join(sorted(core))
    sd.n_dup = len(content) - len(set(content))
    return sd


def name_view(raw, src, T, name_en=None):
    v = NameView()
    raw = raw or ''
    v.raw = raw
    s = raw.strip()
    v.markers = set()
    v.alias = False
    v.empty = not s
    v.indic = has_indic(s)
    v.translit = False
    v.nonascii = not s.isascii()
    cased = s.upper() != s.lower()
    v.caps = cased and not v.indic and s == s.upper() and len(s) >= 3
    v.lower = cased and not v.indic and s == s.lower() and len(s) >= 3
    v.accent = bool(_ACCENTED.search(s)) if v.nonascii else False
    if '  ' in s:
        v.markers.add('double_space')
    if v.indic:
        v.translit = True
        s = name_en.strip() if (name_en and not has_indic(name_en)) else anyascii(s)
    if '|' in s:
        m = RE_WWW_TAG.search(s)
        if m:
            v.markers.add('www_tag')
            s = s[:m.start()]
    for _ in range(2):
        m = RE_PHONE.search(s)
        if m:
            v.markers.add('phone'); s = s[:m.start()]; continue
        m = RE_RECTAG.search(s)
        if m:
            v.markers.add('record_tag'); s = s[:m.start()]; continue
        m = RE_IDTAG.search(s)
        if m:
            v.markers.add('id_tag'); s = s[:m.start()]; continue
        break
    parts = [s]
    if src != 1:
        m = RE_ALIAS.search(s)
        if m:
            head, tail = s[:m.start()].strip(), s[m.end():].strip()
            if head or tail:
                v.alias = True
                v.markers.add('alias')
                parts = [p for p in (tail, head) if p]
    sides = []
    for p in parts:
        m = RE_JUNKPRE.match(p)
        if m:
            v.markers.add('junk_prefix')
            p = p[m.end():]
        m = RE_HONOR.match(p) if src != 1 else None  # S1 names are clean: 'Shree Developers' keeps its first word
        if m:
            v.markers.add('hon_' + _WS.sub(' ', m.group(1).lower()))
            p = p[m.end():]
        sides.append(_side(p, T, v.markers))
    if not sides:
        sides = [Side()]
    v.sides = sides
    if not sides[0].tokens:
        v.empty = True
    return v


# ------------------------------------------------------------------ addresses
RE_PLACEHOLDER = re.compile(r'(?<![a-z])(?:<\s*null\s*>|null|n/a|none)(?![a-z])')
RE_POBOX = re.compile(r'\b(?:p\.?\s?o\.?\s?box|pmb|post box)\s*#?\s*\d+')
RE_UNITC = re.compile(r'^(?:unit|apt|apartment|ste|suite|fl|floor|flr|room|rm)\b|^#\s+\w+$')
RE_INS_PREF = re.compile(r'^(?:door\s*no\.?|h\.?\s?no\.?|h\.\s?n\.?|hn|plot(?:\s*no\.?)?|block\s+[a-z]\s*-|#{1,3})\s*[:\-]?\s*(?=[a-z]?-?\d)')
RE_DECOR = re.compile(r'(?:^|\s)(?:no\.?|n|#{1,3}|h\.?\s?no\.?|door\s*no\.?|hn|plot\s*no\.?|plot)\s*[:\-]?\s*(?=\d)')
RE_PAREN_NUM = re.compile(r'\((\d+[a-z]?)\)')
RE_ATOK = re.compile(r'[a-z0-9]+(?:[/\-][a-z0-9]+)*')
RE_NUMTOK = re.compile(r'[a-z]{0,3}-?\d[\da-z/\-]*')
RE_ORD = re.compile(r'^(\d{1,3})(?:st|nd|rd|th|er|e|eme|ere)$')
RE_INT = re.compile(r'\d+')
_DIG = re.compile(r'\d')
_ALPHA = re.compile(r'[a-z]')
RE_RANGE = re.compile(r'^(\d+)-(\d+)$')
RE_SUFFIX = re.compile(r'^(\d+)([a-d])$')
SUFFIX_WORDS = {'bis': 'bis', 'ter': 'ter', 'quater': 'quater', 'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'}
DECOR_WORDS = {'no', 'h', 'hno', 'door', 'hn', 'plot', 'n'}
ADDR_STOP = {'and', 'of', 'the', 'de', 'du', 'des', 'la', 'le', 'les', 'et', 'd', 'l', 'en', 'au', 'aux'}


class NumTok:
    __slots__ = ('raw', 'val', 'ints', 'suffix', 'rng', 'concat', 'zpad', 'decor', 'ins_kw', 'comp')

    def __repr__(self):
        return f'NumTok({self.raw!r}, val={self.val}, sfx={self.suffix}, rng={self.rng}, ins={self.ins_kw})'


class AddrView:
    __slots__ = ('raw', 'empty', 'placeholder', 'pobox', 'comps', 'comp_toks', 'comp_is_admin', 'admin', 'admin_surf',
                 'tokens', 'tokset', 'nums', 'numset', 'house', 'street_toks', 'street_key', 'localities', 'surf_toks',
                 'ordinal', 'half', 'caps', 'n_comps', 'hg_tokset', 'ordword')


def _admin_key(c):
    return _WS.sub(' ', c.replace('-', ' ').replace("'", ' ').replace('.', ' ')).strip()


def addr_view(raw, T, addr_en=None):
    v = AddrView()
    raw = raw or ''
    v.raw = raw
    s = raw
    if has_indic(s):
        s = addr_en if (addr_en and not has_indic(addr_en)) else s
    v.placeholder = False
    v.pobox = False
    v.half = False
    v.ordinal = False
    v.ordword = False
    comps, raw_comps = [], []
    for rc in s.split(','):
        c = fold(rc)
        if 'null' in c or 'n/a' in c or 'none' in c:
            g = RE_PLACEHOLDER.sub(' ', c)
            if g != c:
                v.placeholder = True
                c = g
        if 'box' in c or 'pmb' in c:
            g = RE_POBOX.sub(' ', c)
            if g != c:
                v.pobox = True
                c = g
        if '1/2' in c:
            g = re.sub(r'(?<![\d/])1/2(?![\d/])', ' ', c)
            if g != c:
                v.half = True
                c = g
        if '(' in c:
            c = RE_PAREN_NUM.sub(r' \1 ', c)
        c = _WS.sub(' ', c.strip(' .;:-'))
        if c.strip(' .;:-<>'):
            comps.append(c)
            raw_comps.append(rc)
    v.comps = comps
    v.n_comps = len(comps)
    v.empty = not comps
    admin_map, street, afill = T.admin, T.street, T.addr_filler
    v.admin, v.admin_surf = set(), []
    v.comp_toks, v.comp_is_admin, v.nums, v.surf_toks = [], [], [], []
    v.localities = []
    house_comp = None
    for ci, c in enumerate(comps):
        code = admin_map.get(_admin_key(c))
        if code is not None:
            v.admin.add(code)
            v.admin_surf.append(c)
            v.comp_is_admin.append(True)
            v.comp_toks.append([])
            continue
        v.comp_is_admin.append(False)
        is_unit = RE_UNITC.match(c) is not None
        ins_kw = None
        m = RE_INS_PREF.match(c)
        if m:
            ins_kw = m.group(0).strip()
            c_body = c[m.end():]
        else:
            c_body = c
        decor = bool(m) or RE_DECOR.search(c) is not None
        toks = RE_ATOK.findall(c_body)
        ctoks = []
        prev_num = None
        for t in toks:
            if (t[0].isdigit() or RE_NUMTOK.fullmatch(t)) and RE_NUMTOK.fullmatch(t) and len(_ALPHA.findall(t)) <= 3 and not RE_ORD.match(t):
                n = NumTok()
                n.raw, n.comp, n.decor, n.ins_kw = t, ci, decor, ins_kw
                ins_kw = None
                ints = RE_INT.findall(t)
                n.ints = [int(x) for x in ints]
                n.val = n.ints[0]
                n.zpad = ints[0].startswith('0') and len(ints[0]) > 1
                n.suffix = None
                n.rng = None
                n.concat = None
                ms = RE_SUFFIX.match(t)
                if ms:
                    n.suffix = ms.group(2)
                mr = RE_RANGE.match(t)
                if mr:
                    lo, hi = int(mr.group(1)), int(mr.group(2))
                    if hi > lo and hi - lo <= 50:
                        n.rng = (lo, hi)
                    n.concat = int(mr.group(1) + mr.group(2)) if len(mr.group(1) + mr.group(2)) <= 12 else None
                if is_unit:
                    n.decor = 'unit'
                v.nums.append(n)
                if house_comp is None and not is_unit:
                    house_comp = ci
                prev_num = n
                continue
            if prev_num is not None and prev_num.suffix is None and t in SUFFIX_WORDS:
                prev_num.suffix = SUFFIX_WORDS[t]
                prev_num = None
                continue
            prev_num = None
            mo = RE_ORD.match(t)
            if mo:
                ctoks.append('o' + str(int(mo.group(1))))
                v.ordinal = True
                continue
            if t in T.ordinal_map:
                ctoks.append('o' + T.ordinal_map[t])
                v.ordinal = True
                v.ordword = True
                continue
            for w in t.replace('/', ' ').replace('-', ' ').split():
                if w.isdigit():
                    continue
                v.surf_toks.append(w)
                if w in afill or w in DECOR_WORDS or w in ADDR_STOP:
                    continue
                ctoks.append(street.get(w, w))
        v.comp_toks.append(ctoks)
    body = ''.join(raw_comps[i] for i in range(len(comps)) if not v.comp_is_admin[i])
    v.caps = body == body.upper() and body != body.lower()
    # house number: first number token outside unit components
    v.house = None
    for n in v.nums:
        if n.decor != 'unit':
            v.house = n
            break
    if v.house is None and v.nums:
        v.house = v.nums[0]
    v.numset = set()
    for n in v.nums:
        v.numset.update(n.ints)
    toks = [t for ct in v.comp_toks for t in ct]
    v.tokens = toks
    v.tokset = set(toks)
    v.hg_tokset = None
    # street key: house number + street-name tokens (street types / admin / generator words removed)
    v.street_toks, v.street_key = [], None
    street_ci = -1
    if v.house is not None:
        ci = v.house.comp
        st = [t for t in v.comp_toks[ci] if t not in T.street_types]
        if not st and ci + 1 < len(comps) and not v.comp_is_admin[ci + 1] and not _DIG.search(comps[ci + 1]):
            st = [t for t in v.comp_toks[ci + 1] if t not in T.street_types]
        v.street_toks = st
        if st:
            v.street_key = f'{v.house.val}|' + ' '.join(sorted(st))
    else:
        for ci, ct in enumerate(v.comp_toks):
            if ct and any(t in T.street_types for t in ct):
                v.street_toks = [t for t in ct if t not in T.street_types]
                street_ci = ci
                break
    for ci, c in enumerate(comps):
        if v.comp_is_admin[ci] or ci == street_ci or (v.house is not None and ci == v.house.comp) or _DIG.search(c):
            continue
        if v.comp_toks[ci]:
            v.localities.append(' '.join(v.comp_toks[ci]))
    return v
