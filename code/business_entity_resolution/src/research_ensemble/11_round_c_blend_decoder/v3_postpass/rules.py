"""Label-free neighbour-decoy name/address rules for the v2b post-pass #2 (pure functions, no I/O).

kind 'country' : record legal-free name tokens == S1 legal-free tokens + exactly ONE extra token, and that token is a country tag
                 ('france' for France, also '(France)' since brackets are dropped by the tokenizer).
kind 'legal'   : record legal-free tokens == S1 legal-free tokens (multiset), the canonical legal-form multiset differs, and the
                 record carries at least one legal form (legal form switched or added; a pure legal DROP is not a decoy).
Address       : same explainer street tokens (checked by the caller), house offset k = h2 - h1, and ALL number parts of the two
                 addresses agree except the house number: Counter(nums2) - Counter(nums1) == {h1+k} and Counter(nums1) - Counter(nums2)
                 == {h1} (k != 0); for k == 0 the number multisets must be equal. Numbers are compared as integers (zero padding),
                 digits glued to a preceding letter (street-type artefacts 'COUR2', 'CHEM1', flat ids 'A105') are ignored.
Tokenizer      : NFKD fold, lower, '.' removed (S.A.R.L. -> sarl), tokens [a-z0-9]+ (so '(SARL)', '[SAS]', 'Sàrl' -> 'sarl'/'sas').
"""
import re, unicodedata, collections

D = (1, 2, 3, 4, 5, 7, 9, 11, 13, 21)

LEGAL_FR = {t: t.upper() for t in ('sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'snc', 'ei')}
# English analogue (US/India): explainer LEGAL_CANON without the French forms and without 'cie'
LEGAL_EN = {'private': 'PVT', 'pvt': 'PVT', 'prvt': 'PVT', 'pte': 'PVT', 'limited': 'LTD', 'ltd': 'LTD', 'ltda': 'LTD',
            'llc': 'LLC', 'inc': 'INC', 'incorporated': 'INC', 'corp': 'CORP', 'corporation': 'CORP', 'co': 'CO', 'company': 'CO',
            'llp': 'LLP', 'pllc': 'PLLC', 'pc': 'PC', 'lp': 'LP', 'plc': 'PLC', 'public': 'PUBLIC', 'opc': 'OPC', 'gmbh': 'GMBH', 'pa': 'PA'}
TAGS = {'France': {'france'}, 'US': {'usa', 'us', 'america'}, 'India': {'india'}}
LEGAL = {'France': LEGAL_FR, 'US': LEGAL_EN, 'India': LEGAL_EN}

_TOK = re.compile(r'[a-z0-9]+')
_NUM = re.compile(r'(?<![^\W\d_])\d+')


def fold(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', (s or '').lower()) if not unicodedata.combining(c))


def name_tokens(n):
    return _TOK.findall(fold(n).replace('.', ''))


def classify(n1, n2, canon, tags):
    """-> (kind or None, extra_token or None)"""
    t1, t2 = name_tokens(n1), name_tokens(n2)
    c1 = [t for t in t1 if t not in canon]
    c2 = [t for t in t2 if t not in canon]
    if not c1:
        return None, None
    C1, C2 = collections.Counter(c1), collections.Counter(c2)
    if len(c2) == len(c1) + 1:
        if not (C1 - C2):
            d = C2 - C1
            w = next(iter(d))
            if w in tags:
                return 'country', w
        return None, None
    if C1 == C2:
        L1 = sorted(canon[t] for t in t1 if t in canon)
        L2 = sorted(canon[t] for t in t2 if t in canon)
        if L2 and L1 != L2:
            return 'legal', '+'.join(L1) + '>' + '+'.join(L2)
    return None, None


def nums(a):
    out = []
    for x in _NUM.findall(a or ''):
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out


def allnum_ok(a1, a2, h1, k):
    n1, n2 = collections.Counter(nums(a1)), collections.Counter(nums(a2))
    if k == 0:
        return n1 == n2
    return (n2 - n1) == collections.Counter({h1 + k: 1}) and (n1 - n2) == collections.Counter({h1: 1})


def classify_rows(rows, country):
    """rows: iterable of (n1, n2, a1, a2, h1, k) -> list of (kind, extra, allnum)"""
    canon, tags = LEGAL[country], TAGS[country]
    out = []
    for n1, n2, a1, a2, h1, k in rows:
        kind, extra = classify(n1, n2, canon, tags)
        out.append((kind, extra, allnum_ok(a1, a2, h1, k) if kind else None))
    return out


if __name__ == '__main__':
    # self-test
    assert classify('Salt Sport SARL', 'Salt Sport SAS', LEGAL_FR, TAGS['France']) == ('legal', 'SARL>SAS')
    assert classify('Salt Sport', 'SALT SPORT (FRANCE)', LEGAL_FR, TAGS['France'])[0] == 'country'
    assert classify('Salt Sport SARL', 'Salt Sport France S.A.S.', LEGAL_FR, TAGS['France'])[0] == 'country'
    assert classify('Salt Sport SARL', 'Salt Sport S.A.R.L.', LEGAL_FR, TAGS['France'])[0] is None
    assert classify('Salt Sport SARL', 'Salt Sport', LEGAL_FR, TAGS['France'])[0] is None
    assert classify('Salt Sport', '[Sàrl] Salt Sport', LEGAL_FR, TAGS['France'])[0] == 'legal'
    assert classify('Salt Sport', 'Salt Sport Groupe', LEGAL_FR, TAGS['France'])[0] is None
    assert nums('0070 AVENUE DU 18 JUIN 1940, N°3, COUR2 X') == [70, 18, 1940, 3]
    assert allnum_ok('27 Rue du 8 Mai 1945, Lille', '30 RUE DU 8 MAI 1945, LILLE', 27, 3)
    assert not allnum_ok('27 Rue du 8 Mai 1945, Lille', '30 RUE DU 8 MAI, LILLE', 27, 3)
    assert allnum_ok('5 bis Rue X', 'N° 0005 R X', 5, 0)
    assert classify('Best Infotech Pvt Ltd', 'Best Infotech Private Limited', LEGAL_EN, TAGS['India'])[0] is None
    assert classify('Best Infotech Ltd', 'Best Infotech Pvt Ltd', LEGAL_EN, TAGS['India'])[0] == 'legal'
    print('rules self-test OK')
