"""francepack: country-neutral canonicalisation + replacement features for the France (unseen country) shift.

Package copy of work/research/france/featureshift/francepack.py (logic unchanged; the lexicon path and the API example were adapted).
Licence-clean: accent folding with anyascii (ISC) only, no GPL transliteration library. Pure polars (vectorised) + a thin per-string wrapper.
Matcher v3 uses it France-GATED: only pairs whose S1 country is France take the pack views of the 43 affected v1 features (ber.packfeats),
at test time; US / India rows keep the v1 views (applied to them the pack rewrites 6-7% of India name cores / addresses).
Designed as a DROP-IN replacement for prod_v1's record views (pv1.text.name_frame / addr_frame): the output columns carry the same
names and semantics (name_n, name_core, name_ns, name_skel, addr_n, nums, num_first ...), so every v1 pair feature can be recomputed
on the pack views with the unchanged v1 formulas. Differences from pv1.text (each can be switched off via Cfg for ablations):

  admin   : France department -> region canonicalisation at COMPONENT level ('Nord', 'Pas-de-Calais', 'Hauts-de-France' -> 'frhdf');
            the region/department is one token like a US state code ('tx'), never 3 tokens ('hauts de france') vs 1 ('nord').
            Table = static seed (all 13 regions / 101 departments, offline knowledge; no lookup). An optional learned lexicon
            (LEX_PATH, JSON) could extend it; matcher v3 does NOT use one (Cfg(lexicon=False); in the research run the lexicon file was
            absent, so its lexicon=True was a no-op). US/India state handling is v1's, unchanged.
  saint   : 'St'/'St.'/'St-' FOLLOWED BY A WORD (>=3 letters, not a unit/direction word) -> 'saint'; 'Ste <word>' -> 'sainte';
            any other 'st' keeps pv1's 'street' mapping. Country-neutral (also fixes US 'St Andrews Church Rd').
  suffix  : house-number suffixes are separated from the number: '45BIS', '45 bis', '98B', '3 TER', '59 T R ...', '15 b Chemin'
            -> number '45' / '98' / '3' + suffix column hn_sfx ('b','t','q' or the letter). pv1 kept '45bis' as an address token and
            mapped 'ter' -> 'terrace'.
  frstreet: French street-type abbreviations missing from pv1 (q->quai, ch/che/chem->chemin, crs->cours, res->residence,
            pass->passage, alle->allee, prom->promenade, esp->esplanade, lot->lotissement, sq->square) + learned list.
  ctag    : country tags '(France)', '(India)' and their typos are removed from the name CORE (pv1 kept them as content words).
  legal   : French legal forms missing from pv1's LEGAL list (ei, eirl, cie, sarlu, selas, selafa, scp, scm, sca, scs, gie, sem,
            spa, earl, gaec, sccv, sel) + learned list, removed from the core.
  (IDF)   : per-country IDF with a constant offset (build.py): idf_c(t) = log(N_c / df_c(t)) + IDF_OFFSET. pv1's IDF is per SPLIT, so
            French tokens get +log(N/N_France) ~ +1.9 nats on test vs +0.5/+0.9 for US/India on train.

API
    from ber import francepack as fp
    cfg = fp.Cfg(lexicon=False)                     # = matcher v3 (research tag 'assfcll' with no lexicon file = the same views)
    cfg = fp.Cfg()                                  # all fixes on; fp.Cfg.off() reproduces the v1 views (ber.text) exactly
    NF = fp.name_frame(fp.ascii_fold(names), countries, cfg)   # polars DataFrame: name_n, name_core, name_ns, name_skel, f_url, ...
    AF = fp.addr_frame(fp.ascii_fold(addrs), countries, cfg)   # addr_n, nums, addr_null, has_unit, hn_sfx, adm, street_key
    fp.addr_view_one('45BIS R. DE LA PAIX, ST-NAZAIRE, Loire-Atlantique', 'France')   # dict for one string (debugging)
    fp.name_view_one('Barca (France) S.A.R.L. [Développement]', 'France')
    fp.admin_agree(adm1, adm2)  -> 1 / 0 / -1 (missing)       (polars expressions)
"""
import json
import os
import re
from dataclasses import dataclass, fields

import polars as pl
from anyascii import anyascii

HERE = os.path.dirname(os.path.abspath(__file__))
LEX_PATH = os.environ.get('BER_FRANCEPACK_LEXICON', '')   # optional learned lists (not shipped, not used by v3)
IDF_OFFSET = 0.7   # mean of the train offsets log(N/N_US)=0.51 and log(N/N_India)=0.92: keeps US/India IDF scale ~unchanged

# ---------------------------------------------------------------------------------------------------------------- pv1 copies
# (identical to ber/text.py = the research prod_v1/pv1/text.py, so that Cfg.off() reproduces the v1 views bit for bit)
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
IN_CODE_CANON = {'ts': 'tg', 'or': 'od', 'ct': 'cg', 'ut': 'uk'}
ADDR_ABBR = {'st': 'street', 'str': 'street', 'rd': 'road', 'dr': 'drive', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard', 'bd': 'boulevard',
             'bvd': 'boulevard', 'ln': 'lane', 'ct': 'court', 'hwy': 'highway', 'pl': 'place', 'pkwy': 'parkway', 'cir': 'circle', 'trl': 'trail',
             'ter': 'terrace', 'sq': 'square', 'apt': 'apartment', 'ste': 'suite', 'fl': 'floor', 'flr': 'floor', 'bldg': 'building',
             'nr': 'near', 'opp': 'opposite', 'mt': 'mount', 'ft': 'fort', 'sec': 'sector', 'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
             'r': 'rue', 'all': 'allee', 'imp': 'impasse', 'chem': 'chemin', 'rte': 'route', 'fbg': 'faubourg', 'bld': 'boulevard'}
ADDR_NOISE = ['no', 'po', 'box', 'door', 'hno', 'h', 'pmb', 'number']
LEGAL = ['pvt', 'private', 'ltd', 'limited', 'llc', 'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'llp', 'plc', 'lp', 'pc',
         'pllc', 'the', 'and', 'of', 'pvtltd', 'opc', 'pte', 'gmbh', 'ltda', 'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'snc', 'selarl',
         'scop', 'mr', 'mrs', 'ms', 'shri', 'sri', 'smt', 'des', 'de', 'du', 'la', 'le', 'les', 'et']
_URL_TAIL = r'\s*\|\s*(https?://)?(www\.)?[a-z0-9\-]+(\.[a-z]{2,4})+/?\s*$'
_URL_FULL = r'^(https?://)?(www\.)?[a-z0-9\-]+(\.[a-z]{2,4})+/?$'

# ---------------------------------------------------------------------------------------------------------------- France seeds
# Offline knowledge (no lookup): the 13 metropolitan regions and their departments. Codes are opaque tokens ('fr' + 3 letters).
FR_REGIONS = {
    'hauts de france': 'frhdf', 'nouvelle aquitaine': 'frnaq', 'pays de la loire': 'frpdl', 'ile de france': 'fridf',
    'auvergne rhone alpes': 'frara', 'provence alpes cote d azur': 'frpac', 'occitanie': 'frocc', 'grand est': 'frges',
    'bretagne': 'frbre', 'normandie': 'frnor', 'bourgogne franche comte': 'frbfc', 'centre val de loire': 'frcvl', 'corse': 'frcor',
}
FR_DEPTS = {  # department -> region code
    'frhdf': ['nord', 'pas de calais', 'somme', 'oise', 'aisne'],
    'frnaq': ['gironde', 'pyrenees atlantiques', 'landes', 'charente maritime', 'charente', 'haute vienne', 'vienne', 'dordogne',
              'lot et garonne', 'correze', 'creuse', 'deux sevres'],
    'frpdl': ['loire atlantique', 'maine et loire', 'sarthe', 'vendee', 'mayenne'],
    'fridf': ['paris', 'hauts de seine', 'seine saint denis', 'val de marne', 'yvelines', 'essonne', 'seine et marne', 'val d oise'],
    'frara': ['rhone', 'isere', 'puy de dome', 'loire', 'haute savoie', 'savoie', 'ain', 'allier', 'ardeche', 'cantal', 'drome',
              'haute loire'],
    'frpac': ['bouches du rhone', 'alpes maritimes', 'var', 'vaucluse', 'alpes de haute provence', 'hautes alpes'],
    'frocc': ['haute garonne', 'herault', 'gard', 'pyrenees orientales', 'aude', 'aveyron', 'gers', 'lot', 'lozere', 'hautes pyrenees',
              'tarn', 'tarn et garonne', 'ariege'],
    'frges': ['bas rhin', 'haut rhin', 'moselle', 'marne', 'meurthe et moselle', 'meuse', 'vosges', 'ardennes', 'aube', 'haute marne'],
    'frbre': ['ille et vilaine', 'finistere', 'morbihan', 'cotes d armor'],
    'frnor': ['seine maritime', 'calvados', 'manche', 'eure', 'orne'],
    'frbfc': ['cote d or', 'doubs', 'saone et loire', 'yonne', 'nievre', 'jura', 'haute saone', 'territoire de belfort'],
    'frcvl': ['loiret', 'indre et loire', 'cher', 'indre', 'eure et loir', 'loir et cher'],
    'frcor': ['corse du sud', 'haute corse'],
}
FR_STREET_EXTRA = {'q': 'quai', 'qu': 'quai', 'ch': 'chemin', 'che': 'chemin', 'crs': 'cours', 'res': 'residence', 'pass': 'passage',
                   'alle': 'allee', 'prom': 'promenade', 'esp': 'esplanade', 'lot': 'lotissement', 'sq': 'square', 'allees': 'allee'}
FR_LEGAL_EXTRA = ['ei', 'eirl', 'cie', 'sarlu', 'selas', 'selafa', 'scp', 'scm', 'sca', 'scs', 'gie', 'sem', 'spa', 'earl', 'gaec', 'sccv',
                  'sel', 'selasu']
CTAGS = ['france', 'india']
SAINT_PROTECT = ['apt', 'ste', 'suite', 'unit', 'fl', 'floor', 'bldg', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw', 'north', 'south',
                 'east', 'west', 'rd', 'road', 'ave', 'dr', 'room', 'rm', 'po', 'pmb', 'box', 'null', 'and']
FR_STREET_TYPES_FOR_SFX = ['rue', 'r', 'av', 'ave', 'avenue', 'bd', 'boulevard', 'blvd', 'chemin', 'ch', 'allee', 'all', 'impasse', 'imp',
                           'pl', 'place', 'route', 'rte', 'quai', 'q', 'cours', 'crs', 'square', 'sq', 'residence', 'res', 'passage',
                           'cite', 'lotissement', 'promenade', 'esplanade', 'faubourg', 'fbg']


def load_lexicon(path=LEX_PATH):
    """learned France lists (research mine_lexicon.py; TRANSDUCTIVE: mined from the unlabelled test files). Empty dict when absent."""
    if not path:
        return {}
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


@dataclass
class Cfg:
    admin: bool = True
    saint: bool = True
    suffix: bool = True
    frstreet: bool = True
    ctag: bool = True
    legal: bool = True
    lexicon: bool = True     # merge the learned lists (lexicon_france.json) on top of the seeds
    v2: bool = False         # pack v2: house-number suffix kept as a token ('98B'/'98 BIS' -> '98 sfxb': B==BIS but India '12A' != '12B'),
                             # + mined 'psg'->passage, + homoglyph legal forms (5arl/5as/5asu/5ci -> sarl/sas/sasu/sci)

    @classmethod
    def off(cls):
        return cls(**{f.name: False for f in fields(cls)})

    @classmethod
    def pack2(cls):
        return cls(v2=True)

    def tag(self):
        return ''.join(k[0] if getattr(self, k) else '-' for k in ('admin', 'saint', 'suffix', 'frstreet', 'ctag', 'legal', 'lexicon')) + ('2' if self.v2 else '')


def _padded_map(d):
    pats = [' ' + '  '.join(k.split()) + ' ' for k in d]
    reps = [' ' + v + ' ' for v in d.values()]
    return pats, reps


def ascii_fold(s: pl.Series) -> pl.Series:
    """= pv1.text.ascii_fold: anyascii on unique non-ASCII strings; 'N°'/'Nº'/'№' dropped first so 'N° 6' -> ' 6'."""
    s = s.fill_null('').str.replace_all(r'(?i)\bn\s*[°º]', ' ').str.replace_all('[°º№]', ' ')
    df = pl.DataFrame({'s': s})
    na = df.filter(pl.col('s').str.contains(r'[^\x00-\x7F]')).unique('s')
    if na.height:
        na = na.with_columns(pl.col('s').map_elements(anyascii, return_dtype=pl.Utf8).alias('f'))
        df = df.join(na, on='s', how='left', maintain_order='left').with_columns(pl.coalesce('f', 's').alias('s'))
    return df['s']


def skeleton(e: pl.Expr) -> pl.Expr:
    s = e.str.replace_all('ph', 'f').str.replace_all('sh', 's').str.replace_all('th', 't').str.replace_all('[cq]', 'k') \
         .str.replace_all('w', 'v').str.replace_all('z', 'j').str.replace_all('x', 'ks').str.replace_all(r'\B[aeiouhy]', '')
    pats = [c + c for c in 'abcdefghijklmnopqrstuvwxyz']; reps = list('abcdefghijklmnopqrstuvwxyz')
    return s.str.replace_many(pats, reps).str.replace_many(pats, reps)


def admin_table(cfg: Cfg):
    """folded component key ('pas de calais') -> region code. Seeds + learned aliases (lexicon['admin_alias'])."""
    t = dict(FR_REGIONS)
    for code, deps in FR_DEPTS.items():
        for d in deps:
            t[d] = code
    if cfg.lexicon:
        for k, v in load_lexicon().get('admin_alias', {}).items():
            t.setdefault(k, v)
    return t


def _legal_list(cfg: Cfg):
    L = list(LEGAL)
    if cfg.legal:
        L += [w for w in FR_LEGAL_EXTRA if w not in L]
        if cfg.v2:
            L += [w for w in ('5arl', '5as', '5asu', '5ci', '5a', 'eur1') if w not in L]
        if cfg.lexicon:
            L += [w for w in load_lexicon().get('legal', []) if w not in L]
    if cfg.ctag:
        L += [w for w in CTAGS if w not in L]
        if cfg.lexicon:
            L += [w for w in load_lexicon().get('ctag_typos', []) if w not in L]
    return L


def _abbr_map(cfg: Cfg):
    m = dict(ADDR_ABBR)
    if cfg.frstreet:
        for k, v in FR_STREET_EXTRA.items():
            m.setdefault(k, v)
        if cfg.v2:
            m.setdefault('psg', 'passage')
        if cfg.lexicon:
            for k, v in load_lexicon().get('street_abbr', {}).items():
                m.setdefault(k, v)
    return m


# ---------------------------------------------------------------------------------------------------------------- names
def name_frame(s_fold: pl.Series, country: pl.Series = None, cfg: Cfg = None) -> pl.DataFrame:
    """= pv1.text.name_frame with the pack's legal/country-tag list for the core (cfg.legal / cfg.ctag)."""
    cfg = cfg or Cfg()
    lp, lr = _padded_map({k: '' for k in _legal_list(cfg)})
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
    # extra pack columns (not used by pv1 features): country-tag flag
    d = d.with_columns(pl.col('name_n').str.contains(r'\b(france|india)\b').alias('f_ctag'))
    return d.drop('s')


# ---------------------------------------------------------------------------------------------------------------- addresses
def _admin_components(s: pl.Expr, table: dict) -> pl.Expr:
    """replace every comma-separated component whose folded key is a France region/department by its region code."""
    keys = list(table.keys()); vals = list(table.values())
    comp_key = pl.element().str.replace_all(r"[\-'\.]", ' ').str.replace_all(r'\s+', ' ').str.strip_chars()
    return s.str.split(',').list.eval(
        pl.when(comp_key.is_in(keys)).then(pl.lit(' ') + comp_key.replace_strict(keys, vals, default='') + pl.lit(' ')).otherwise(pl.element())
    ).list.join(',')


def addr_frame(s_fold: pl.Series, country: pl.Series, cfg: Cfg = None) -> pl.DataFrame:
    """= pv1.text.addr_frame + admin/saint/suffix/French-street fixes. Extra columns: hn_sfx (house-number suffix: 'b','t','q', letter
    or ''), adm (canonical admin code found: US 2-letter / India code / France region code, '' if none), street_key ('<num>|<sorted
    street words>' or '')."""
    cfg = cfg or Cfg()
    ap, ar = _padded_map({**_abbr_map(cfg), **US_STATES, **IN_STATES})
    np_, nr_ = _padded_map({k: '' for k in ADDR_NOISE})
    cp, cr = _padded_map(IN_CODE_CANON)
    d = pl.DataFrame({'s': s_fold.str.to_lowercase(), 'country': country})
    d = d.with_columns(pl.col('s').str.contains(r'\b(null|none|n/a)\b').alias('addr_null'),
                       pl.col('s').str.contains(r'(\b(apt|apartment|suite|ste|unit|flat|floor|fl|flr|room|bldg|building)\b|#)').alias('has_unit'))
    s = pl.col('s')
    if cfg.admin:
        s = _admin_components(s, admin_table(cfg))
    if cfg.saint:
        prot = '|'.join(SAINT_PROTECT)
        s = s.str.replace_all(rf'\bst\.?(\s+)({prot})\b', 'street$1$2')
        s = s.str.replace_all(r'\bst\.?[\s\-]+([a-z]{3,})', 'saint $1')
        s = s.str.replace_all(r'\bste\.?[\s\-]+([a-z]{3,})', 'sainte $1')
    sfx = pl.lit('')
    if cfg.suffix:
        types = '|'.join(FR_STREET_TYPES_FOR_SFX)
        # suffix value, first match wins: 'NN bis|ter|quater' / 'NNb' / 'NN b|t <street type>'
        sfx = (pl.when(s.str.contains(r'\b\d+\s*bis\b')).then(pl.lit('b'))
               .when(s.str.contains(r'\b\d+\s*ter\b')).then(pl.lit('t'))
               .when(s.str.contains(r'\b\d+\s*quater\b')).then(pl.lit('q'))
               .when(s.str.contains(r'\b\d+[a-d]\b')).then(s.str.extract(r'\b\d+([a-d])\b', 1))
               .when(s.str.contains(rf'\b\d+\s+[bt]\s+\.?({types})\b')).then(s.str.extract(rf'\b\d+\s+([bt])\s+\.?(?:{types})\b', 1))
               .otherwise(pl.lit('')))
        if cfg.v2:   # keep the suffix as ONE canonical token next to the number
            s = s.str.replace_all(r'\b(\d+)\s*bis\b', '$1 sfxb').str.replace_all(r'\b(\d+)\s*ter\b', '$1 sfxt').str.replace_all(r'\b(\d+)\s*quater\b', '$1 sfxq') \
                 .str.replace_all(r'\b(\d+)([a-d])\b', '$1 sfx$2').str.replace_all(rf'\b(\d+)\s+([bt])\s+(\.?(?:{types}))\b', '$1 sfx$2 $3')
        else:
            s = s.str.replace_all(r'\b(\d+)\s*(bis|ter|quater)\b', '$1').str.replace_all(r'\b(\d+)([a-d])\b', '$1') \
                 .str.replace_all(rf'\b(\d+)\s+[bt]\s+(\.?(?:{types}))\b', '$1 $2')
    d = d.with_columns(s.alias('s'), sfx.alias('hn_sfx'))
    # house-number component (first comma-separated component containing a digit), canonicalised like addr_n, for the street key
    hc = pl.col('s').str.split(',').list.eval(pl.element().filter(pl.element().str.contains(r'\d'))).list.first().fill_null('')
    hc = hc.str.replace_all('&', ' and ').str.replace_all(r'[^a-z0-9]+', ' ').str.replace_all(r'\b0+(\d)', '$1').str.strip_chars()
    hc = (pl.lit(' ') + hc.str.replace_all(' ', '  ') + pl.lit(' ')).str.replace_many(ap, ar).str.replace_all(r'\s+', ' ').str.strip_chars()
    stop = sorted(set(_abbr_map(cfg).values()) | set(ADDR_NOISE) | {'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'et', 'and', 'of', 'the',
                                                                     'saint', 'sainte', 'a', 'au', 'aux', 'en', 'bis', 'ter'})
    hnum = hc.str.extract(r'(\d+)', 1)
    words = hc.str.split(' ').list.eval(pl.element().filter(pl.element().str.contains(r'^[a-z]{2,}$') & ~pl.element().is_in(stop) & ~pl.element().str.starts_with('sfx')
                                                            & ~pl.element().str.contains(r'^fr[a-z]{3}$'))).list.head(4).list.sort().list.join(' ')
    d = d.with_columns(pl.when(hnum.is_not_null() & (words != '')).then(hnum.str.slice(-9, 9).cast(pl.Int64).cast(pl.Utf8) + pl.lit('|') + words)
                       .otherwise(pl.lit('')).alias('street_key'))
    d = d.with_columns(pl.col('s').str.replace_all(r'\b(null|none|n/a)\b', ' ').str.replace_all('&', ' and ')
                       .str.replace_all(r'[^a-z0-9]+', ' ').str.replace_all(r'\b0+(\d)', '$1').str.strip_chars().alias('s'))
    d = d.with_columns(pl.col('s').str.extract_all(r'\d+').list.eval(pl.element().str.slice(-9, 9).cast(pl.Int64)).alias('nums'))
    pad = (pl.lit(' ') + pl.col('s').str.replace_all(' ', '  ') + pl.lit(' '))
    t = pad.str.replace_many(ap, ar).str.replace_all(r'\s+', ' ')
    t = (pl.lit(' ') + t.str.strip_chars().str.replace_all(' ', '  ') + pl.lit(' '))
    t = pl.when(pl.col('country') == 'India').then(t.str.replace_many(cp, cr)).otherwise(t).str.replace_many(np_, nr_)
    d = d.with_columns(t.str.replace_all(r'\s+', ' ').str.strip_chars().alias('addr_n'))
    # admin code present in the canonical address (for admin_agree); France region codes are unique 'frxxx' tokens
    usc = sorted(set(US_STATES.values())); inc = sorted(set(IN_STATES.values()))
    tok = pl.col('addr_n').str.split(' ')
    d = d.with_columns(
        pl.when(pl.col('country') == 'France').then(tok.list.eval(pl.element().filter(pl.element().str.contains(r'^fr[a-z]{3}$'))).list.last())
        .when(pl.col('country') == 'US').then(tok.list.eval(pl.element().filter(pl.element().is_in(usc))).list.last())
        .otherwise(tok.list.eval(pl.element().filter(pl.element().is_in(inc))).list.last()).fill_null('').alias('adm'))
    return d.drop('s', 'country')


def street_key_expr(addr_n: pl.Expr, nums: pl.Expr, cfg: Cfg = None) -> pl.Expr:
    """'<first house number>|<sorted street words>' on the canonical address: words between the first number and the next number,
    minus street types, particles, admin codes and the city is NOT removed (S1 France has ~15 communes; city words rarely follow
    the street in the same comma-free string, so we keep only the 4 tokens after the number)."""
    cfg = cfg or Cfg()
    stop = set(_abbr_map(cfg).values()) | {'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'et', 'and', 'of', 'the', 'saint', 'sainte',
                                             'a', 'au', 'aux', 'en'}
    stop = sorted(stop)
    after = addr_n.str.extract(r'(?:^|\s)\d+\s+((?:[a-z][a-z0-9]*\s*){1,6})', 1).fill_null('')
    words = after.str.split(' ').list.eval(pl.element().filter((pl.element() != '') & ~pl.element().is_in(stop) & ~pl.element().str.contains(r'^fr[a-z]{3}$'))).list.head(3).list.sort().list.join(' ')
    num = nums.list.first().cast(pl.Utf8)
    return pl.when(num.is_not_null() & (words != '')).then(num + pl.lit('|') + words).otherwise(pl.lit(''))


def admin_agree(a1: pl.Expr, a2: pl.Expr) -> pl.Expr:
    """1 agree / 0 conflict / -1 a side has no admin area (never counted as disagreement)."""
    return pl.when((a1 == '') | (a2 == '')).then(-1).when(a1 == a2).then(1).otherwise(0).cast(pl.Float32)


def sfx_agree(s1: pl.Expr, s2: pl.Expr) -> pl.Expr:
    """house-number suffix: 1 both equal & present / 0 both present & different / -1 at least one absent (S3 drops 'bis')."""
    return pl.when((s1 == '') | (s2 == '')).then(-1).when(s1 == s2).then(1).otherwise(0).cast(pl.Float32)


# ---------------------------------------------------------------------------------------------------------------- one-string helpers
def addr_view_one(addr, country='France', cfg: Cfg = None):
    cfg = cfg or Cfg()
    AF = addr_frame(ascii_fold(pl.Series([addr])), pl.Series([country]), cfg)
    return AF.row(0, named=True)


def name_view_one(name, country='France', cfg: Cfg = None):
    return name_frame(ascii_fold(pl.Series([name])), pl.Series([country]), cfg or Cfg()).row(0, named=True)


if __name__ == '__main__':
    for a in ['45BIS R. DE LA PAIX, ST-NAZAIRE, Loire-Atlantique', '8 RUE DUCAU, BORDEAUX, Nouvelle-Aquitaine', 'N° 14 Rue Du Nord, Lille, Nord',
              '59 T R DE LA CONTRIE, NANTES, Loire-Atlantique', '15 b Chemin du Moulin, La Teste-de-Buch, Nouvelle-Aquitaine',
              '64 Q. Des Queyries, Bordeaux', 'Pays de la Loire, 70 Rue De Trignac, St.-nazaire', '1130 St Andrews Church Road, Sanford, NC',
              '123 Main St, Apt 4, Springfield, IL', '4520 Main St Ste 100, Dallas, Texas', '719A Oak Ave, Austin, TX']:
        for c in (Cfg.off(), Cfg()):
            v = addr_view_one(a, 'US' if a.endswith(('NC', 'IL', 'Texas', 'TX')) else 'France', c)
            print(c.tag(), '|', a, '->', v['addr_n'], v['nums'], repr(v['hn_sfx']), repr(v['adm']), repr(v['street_key']))
    for n in ['Barca (France) S.A.R.L. [Développement]', 'Petit & Cie International', 'Ets Adrienne Sarl', 'Nis (India) Motors Pvt Ltd']:
        for c in (Cfg.off(), Cfg()):
            v = name_view_one(n, 'France', c)
            print(c.tag(), '|', n, '->', v['name_n'], '|', v['name_core'], '|', v['name_skel'])


# ---------------------------------------------------------------------------------------------------------------- dense-scale calibration
# The e5 embedding space is COMPRESSED for French text (dense-pair medians, test: record top1-top2 gap France 0.023 vs India 0.040 /
# US 0.067; the same compression appears in synthetic France), so cosine levels/gaps learned on US/India mean something else in France.
# Label-free per-country quantile mapping of the cosine-derived features onto the US+India distribution (TRANSDUCTIVE when fitted on
# test pairs). Exact zeros of the 'relative-to-best' features (= this pair IS the best) and NaN (channel absent) are kept.
DENSE_QN = ['cos', 's1_top1', 'rel_s1', 's1_gap12', 'rev_top1', 'rel_rec', 'rev_gap12', 'rel_rec_tab', 'u_fs', 'u_rs', 'u_rd']
KEEP_ZERO = {'rel_s1', 'rel_rec', 'rel_rec_tab'}


def qmap_fit(src, ref, n=2001, keep_zero=False):
    """quantile tables (src_q, ref_q) mapping the src marginal onto the ref marginal. NaNs ignored; with keep_zero, exact zeros excluded."""
    import numpy as np
    src = np.asarray(src, np.float64); ref = np.asarray(ref, np.float64)
    src = src[np.isfinite(src)]; ref = ref[np.isfinite(ref)]
    if keep_zero:
        src = src[src != 0]; ref = ref[ref != 0]
    q = np.linspace(0, 1, n)
    return np.quantile(src, q), np.quantile(ref, q)


def qmap_apply(x, tab, keep_zero=False):
    import numpy as np
    sq, rq = tab; x = np.asarray(x, np.float64)
    y = np.interp(x, sq, rq)
    y = np.where(np.isfinite(x), y, x)
    if keep_zero:
        y = np.where(x == 0, 0.0, y)
    return y.astype(np.float32)
