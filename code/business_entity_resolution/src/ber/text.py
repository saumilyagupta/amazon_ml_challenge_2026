"""Vectorised (polars) text normalisation. Licence-clean: anyascii (ISC) for accent folding, no GPL transliteration library.
All functions take / return polars expressions or Series; no per-row Python except anyascii on the UNIQUE non-ASCII strings."""
import polars as pl
from anyascii import anyascii

US_STATES = {'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca', 'colorado': 'co', 'connecticut': 'ct',
             'delaware': 'de', 'district of columbia': 'dc', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il',
             'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
             'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt',
             'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
             'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
             'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
             'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy', 'puerto rico': 'pr'}
IN_STATES = {'maharashtra': 'mh', 'delhi': 'dl', 'new delhi': 'dl', 'uttar pradesh': 'up', 'karnataka': 'ka', 'tamil nadu': 'tn', 'tamilnadu': 'tn',
             'west bengal': 'wb', 'westbengal': 'wb', 'gujarat': 'gj', 'telangana': 'tg', 'haryana': 'hr', 'kerala': 'kl', 'keralam': 'kl',
             'rajasthan': 'rj', 'bihar': 'br', 'madhya pradesh': 'mp', 'andhra pradesh': 'ap', 'orissa': 'od', 'odisha': 'od', 'punjab': 'pb',
             'goa': 'ga', 'assam': 'as', 'jharkhand': 'jh', 'chhattisgarh': 'cg', 'uttarakhand': 'uk', 'himachal pradesh': 'hp',
             'jammu and kashmir': 'jk', 'chandigarh': 'ch', 'puducherry': 'py', 'pondicherry': 'py'}
IN_CODE_CANON = {'ts': 'tg', 'or': 'od', 'ct': 'cg', 'ut': 'uk'}          # applied to India rows only
ADDR_ABBR = {'st': 'street', 'str': 'street', 'rd': 'road', 'dr': 'drive', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard', 'bd': 'boulevard',
             'bvd': 'boulevard', 'ln': 'lane', 'ct': 'court', 'hwy': 'highway', 'pl': 'place', 'pkwy': 'parkway', 'cir': 'circle', 'trl': 'trail',
             'ter': 'terrace', 'sq': 'square', 'apt': 'apartment', 'ste': 'suite', 'fl': 'floor', 'flr': 'floor', 'bldg': 'building',
             'nr': 'near', 'opp': 'opposite', 'mt': 'mount', 'ft': 'fort', 'sec': 'sector', 'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
             # French (country-agnostic: harmless elsewhere)
             'r': 'rue', 'all': 'allee', 'imp': 'impasse', 'chem': 'chemin', 'rte': 'route', 'fbg': 'faubourg', 'bld': 'boulevard'}
ADDR_NOISE = ['no', 'po', 'box', 'door', 'hno', 'h', 'pmb', 'number']
LEGAL = ['pvt', 'private', 'ltd', 'limited', 'llc', 'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'llp', 'plc', 'lp', 'pc',
         'pllc', 'the', 'and', 'of', 'pvtltd', 'opc', 'pte', 'gmbh', 'ltda', 'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'snc', 'selarl',
         'scop', 'mr', 'mrs', 'ms', 'shri', 'sri', 'smt', 'des', 'de', 'du', 'la', 'le', 'les', 'et']
_URL_TAIL = r'\s*\|\s*(https?://)?(www\.)?[a-z0-9\-]+(\.[a-z]{2,4})+/?\s*$'
_URL_FULL = r'^(https?://)?(www\.)?[a-z0-9\-]+(\.[a-z]{2,4})+/?$'


def _padded_map(d):
    """patterns for replace_many on strings whose single spaces were doubled and padded (' ' + s.replace(' ', '  ') + ' ')."""
    pats = [' ' + '  '.join(k.split()) + ' ' for k in d]
    reps = [' ' + v + ' ' for v in d.values()]
    return pats, reps


def ascii_fold(s: pl.Series) -> pl.Series:
    """anyascii on unique non-ASCII strings only (joined back); '°'/'№' dropped first so 'N° 6' -> 'N 6'."""
    s = s.fill_null('').str.replace_all(r'(?i)\bn\s*[°º]', ' ').str.replace_all('[°º№]', ' ')
    df = pl.DataFrame({'s': s})
    na = df.filter(pl.col('s').str.contains(r'[^\x00-\x7F]')).unique('s')
    if na.height:
        na = na.with_columns(pl.col('s').map_elements(anyascii, return_dtype=pl.Utf8).alias('f'))
        df = df.join(na, on='s', how='left', maintain_order='left').with_columns(pl.coalesce('f', 's').alias('s'))
    return df['s']


def raw_view(e: pl.Expr) -> pl.Expr:
    """raw Unicode view: NFKC, casefold, keep letters/marks/digits of all scripts."""
    return e.fill_null('').str.normalize('NFKC').str.to_lowercase().str.replace_all(r'[^\p{L}\p{M}\p{N}]+', ' ').str.strip_chars()


def name_frame(s_fold: pl.Series) -> pl.DataFrame:
    """s_fold: ascii-folded business name (romanized view). Returns name_n, name_core, name_ns, name_skel, name_dm + flags."""
    lp, lr = _padded_map({k: '' for k in LEGAL})
    d = pl.DataFrame({'s': s_fold.str.to_lowercase().str.strip_chars()})
    d = d.with_columns(pl.col('s').str.contains(_URL_TAIL).alias('f_urltail'),
                       pl.col('s').str.contains(r'\bformerly\b').alias('f_formerly'))
    d = d.with_columns(pl.col('s').str.replace(_URL_TAIL, '').str.replace(r'^.*\bformerly\b\s*(known\s+as)?\s*:?', '').str.strip_chars())
    d = d.with_columns((pl.col('s').str.contains(_URL_FULL) & ~pl.col('s').str.contains(' ')).alias('f_url'),
                       pl.col('s').str.contains(r'^@\S+$').alias('f_handle'))
    dm = pl.col('s').str.replace(r'^@', '').str.replace(r'^(https?://)?(www\.)?', '').str.replace(r'(\.[a-z]{2,4})+/?$', '').str.replace_all(r'[^a-z0-9]', '')
    d = d.with_columns(pl.when(pl.col('f_url') | pl.col('f_handle')).then(dm).otherwise(pl.col('s')).alias('s'))
    d = d.with_columns(pl.col('s').str.replace_all(r'\bm/s\b', ' ').str.replace_all('&', ' and ').str.replace_all(r'\.', '')
                       .str.replace_all(r'[^a-z0-9]+', ' ').str.strip_chars().alias('name_n'))
    core = (pl.lit(' ') + pl.col('name_n').str.replace_all(' ', '  ') + pl.lit(' ')).str.replace_many(lp, lr).str.replace_all(r'\s+', ' ').str.strip_chars()
    d = d.with_columns(core.alias('name_core'))
    d = d.with_columns(pl.when(pl.col('name_core') == '').then(pl.col('name_n')).otherwise(pl.col('name_core')).alias('name_core'))
    d = d.with_columns(pl.col('name_core').str.replace_all(' ', '').alias('name_ns'))
    d = d.with_columns(skeleton(pl.col('name_core')).alias('name_skel'))
    return d.drop('s')


def skeleton(e: pl.Expr) -> pl.Expr:
    """phonetic consonant skeleton per token (user baseline): ph->f, sh->s, th->t, c/q->k, w->v, z->j, x->ks,
    drop vowels/h/y except token-initial, collapse doubled letters."""
    s = e.str.replace_all('ph', 'f').str.replace_all('sh', 's').str.replace_all('th', 't').str.replace_all('[cq]', 'k') \
         .str.replace_all('w', 'v').str.replace_all('z', 'j').str.replace_all('x', 'ks').str.replace_all(r'\B[aeiouhy]', '')
    pats = [c + c for c in 'abcdefghijklmnopqrstuvwxyz']; reps = list('abcdefghijklmnopqrstuvwxyz')
    return s.str.replace_many(pats, reps).str.replace_many(pats, reps)


def addr_frame(s_fold: pl.Series, country: pl.Series) -> pl.DataFrame:
    """s_fold: ascii-folded address (romanized view). Returns addr_n, nums (list[i64]), flags."""
    ap, ar = _padded_map({**ADDR_ABBR, **US_STATES, **IN_STATES})
    np_, nr_ = _padded_map({k: '' for k in ADDR_NOISE})
    cp, cr = _padded_map(IN_CODE_CANON)
    d = pl.DataFrame({'s': s_fold.str.to_lowercase(), 'country': country})
    d = d.with_columns(pl.col('s').str.contains(r'\b(null|none|n/a)\b').alias('addr_null'),
                       pl.col('s').str.contains(r'(\b(apt|apartment|suite|ste|unit|flat|floor|fl|flr|room|bldg|building)\b|#)').alias('has_unit'))
    d = d.with_columns(pl.col('s').str.replace_all(r'\b(null|none|n/a)\b', ' ').str.replace_all('&', ' and ')
                       .str.replace_all(r'[^a-z0-9]+', ' ').str.replace_all(r'\b0+(\d)', '$1').str.strip_chars().alias('s'))
    d = d.with_columns(pl.col('s').str.extract_all(r'\d+').list.eval(pl.element().str.slice(-9, 9).cast(pl.Int64)).alias('nums'))
    pad = (pl.lit(' ') + pl.col('s').str.replace_all(' ', '  ') + pl.lit(' '))
    t = pad.str.replace_many(ap, ar).str.replace_all(r'\s+', ' ')
    t = (pl.lit(' ') + t.str.strip_chars().str.replace_all(' ', '  ') + pl.lit(' '))
    t = pl.when(pl.col('country') == 'India').then(t.str.replace_many(cp, cr)).otherwise(t).str.replace_many(np_, nr_)
    d = d.with_columns(t.str.replace_all(r'\s+', ' ').str.strip_chars().alias('addr_n'))
    return d.drop('s', 'country')
