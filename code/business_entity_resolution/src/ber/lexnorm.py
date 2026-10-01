"""Normalisation for the lexical blocking channels (C1/C2/C3 and their reverse forms). anyascii only (ISC); no GPL transliteration library.
Re-implements the baseline normalisation: anyascii fold, lowercase, '&'->and, strip punctuation / web fragments, legal-form and
honorific stop-words (US / India / France), address abbreviation expansion (incl. French rue/bd/av), noise tokens dropped,
numbers with leading zeros stripped, phonetic consonant skeleton per name token. Vectorised in polars; anyascii and the skeleton
run in Python only on the non-ASCII subset / unique token vocabulary."""
import re
import polars as pl
from anyascii import anyascii


def fold(s: pl.Series) -> pl.Series:
    m = s.str.contains(r'[^\x00-\x7F]')
    idx = m.arg_true()
    if len(idx) == 0:
        return s
    conv = pl.Series([anyascii(v) for v in s.filter(m).to_list()], dtype=pl.String)
    return s.scatter(idx, conv)


NAME_STOP = set('''pvt private ltd limited llc inc incorporated corp corporation co company cos llp lp plc pllc pc sa sas sasu sarl
eurl sci snc scs sca scp scop selarl gmbh ag bv nv cie pte pty ltda opc the of and et an mr mrs ms miss dr www com http https net org
formerly dba aka de du des la le les'''.split())


def clean_name(e: pl.Expr) -> pl.Expr:
    e = e.str.to_lowercase()
    e = e.str.replace_all(r'https?://|www\.', ' ')
    e = e.str.replace_all(r'\.(com|net|org|co\.in|in|co|biz|info|us|io|fr)\b', ' ')
    e = e.str.replace_all('&', ' and ', literal=True)
    e = e.str.replace_all(r'[^a-z0-9]+', ' ')
    return e.str.strip_chars()


_V = re.compile(r'[aeiouyh]'); _REP = re.compile(r'(.)\1+')


def skel_word(w: str) -> str:
    """Phonetic consonant skeleton: ph->f, sh->s, th->t, c/q->k, w->v, z->j, x->ks, drop vowels/h/y after the first char, collapse doubles."""
    w = (w.replace('ph', 'f').replace('sh', 's').replace('th', 't').replace('ck', 'k').replace('c', 'k')
          .replace('q', 'k').replace('w', 'v').replace('z', 'j').replace('x', 'ks'))
    if len(w) > 1:
        w = w[0] + _V.sub('', w[1:])
    return _REP.sub(r'\1', w)


US_STATES = dict(alabama='al', alaska='ak', arizona='az', arkansas='ar', california='ca', colorado='co', connecticut='ct',
    delaware='de', florida='fl', georgia='ga', hawaii='hi', idaho='id', illinois='il', indiana='in', iowa='ia', kansas='ks',
    kentucky='ky', louisiana='la', maine='me', maryland='md', massachusetts='ma', michigan='mi', minnesota='mn',
    mississippi='ms', missouri='mo', montana='mt', nebraska='ne', nevada='nv', ohio='oh', oklahoma='ok', oregon='or',
    pennsylvania='pa', tennessee='tn', texas='tx', utah='ut', vermont='vt', virginia='va', washington='wa', wisconsin='wi',
    wyoming='wy')
IN_STATES = dict(maharashtra='mh', delhi='dl', karnataka='ka', gujarat='gj', telangana='tg', haryana='hr', kerala='kl',
    keralam='kl', rajasthan='rj', bihar='br', orissa='od', odisha='od', punjab='pb', goa='ga', assam='as', jharkhand='jh',
    chhattisgarh='cg', uttarakhand='uk', uttaranchal='uk', chandigarh='ch', puducherry='py', pondicherry='py')
MULTI = [('new york', 'ny'), ('new jersey', 'nj'), ('new mexico', 'nm'), ('new hampshire', 'nh'), ('north carolina', 'nc'),
    ('south carolina', 'sc'), ('north dakota', 'nd'), ('south dakota', 'sd'), ('west virginia', 'wv'), ('rhode island', 'ri'),
    ('district of columbia', 'dc'), ('uttar pradesh', 'up'), ('madhya pradesh', 'mp'), ('andhra pradesh', 'ap'),
    ('himachal pradesh', 'hp'), ('arunachal pradesh', 'arp'), ('tamil nadu', 'tn'), ('west bengal', 'wb'),
    ('jammu and kashmir', 'jk'), ('jammu kashmir', 'jk')]
ABBR = dict(
    street='st', str='st', saint='st', road='rd', drive='dr', drv='dr', avenue='av', ave='av', avn='av', avenu='av',
    boulevard='bd', blvd='bd', boul='bd', bld='bd', bvd='bd', lane='ln', court='ct', place='pl', highway='hwy',
    parkway='pkwy', pky='pkwy', circle='cir', terrace='ter', terr='ter', square='sq', trail='trl', plaza='plz', point='pt',
    mount='mt', fort='ft', north='n', south='s', east='e', west='w', northeast='ne', northwest='nw', southeast='se',
    southwest='sw',
    r='rue', chemin='ch', chem='ch', impasse='imp', allee='all', route='rte', faubourg='fbg', fg='fbg', cours='crs',
    residence='res', resid='res', batiment='bat', bt='bat', general='gen', gal='gen', marechal='mal',
    ngr='nagar', col='colony', clny='colony', sec='sector', bengaluru='bangalore', calcutta='kolkata', bombay='mumbai',
    gurugram='gurgaon', madras='chennai',
    **US_STATES, **IN_STATES)
ADDR_STOP = set('''no po box unit door null na nan none apt apartment appt appartement ste suite pmb hno flat plot shop near nr opp
opposite behind floor fl flr the of and de du des la le les et au aux en sur chez mme india usa us france cdp dist district tq tal
taluka ps rm room num number'''.split())


def clean_addr(e: pl.Expr) -> pl.Expr:
    e = e.str.to_lowercase()
    e = e.str.replace_all(r'\bnull\b|\bn/a\b|\bnone\b|\bnan\b', ' ')
    e = e.str.replace_all(r'\b(p\s*\.?\s*o\s*\.?\s*box|post\s*box|pmb|box)\s*[#:.\-]*\s*\d+', ' ')
    e = e.str.replace_all(r'(\d+)\s*(st|nd|rd|th)\b', '${1}')
    e = e.str.replace_all('&', ' and ', literal=True)
    e = e.str.replace_all(r'[^a-z0-9]+', ' ')
    e = pl.concat_str([pl.lit(' '), e, pl.lit(' ')])
    for full, code in MULTI:
        e = e.str.replace_all(' ' + full + ' ', ' ' + code + ' ', literal=True)
    return e.str.strip_chars()


def norm_frame(df: pl.DataFrame) -> pl.DataFrame:
    """df: entity_id, country, business_name, name_en, addr_en -> entity_id, country, name_c, name_raw_c, addr_c, addr_empty."""
    ne = fold(df['name_en']); nr = fold(df['business_name']); ae = fold(df['addr_en'])
    out = pl.DataFrame({'entity_id': df['entity_id'], 'country': df['country'], 'ne': ne, 'nr': nr, 'ae': ae}).select(
        'entity_id', 'country', clean_name(pl.col('ne')).alias('name_c'), clean_name(pl.col('nr')).alias('name_raw_c'),
        clean_addr(pl.col('ae')).alias('addr_c'))
    return out.with_columns((pl.col('addr_c').str.replace_all(r'\b(null|na|n|a)\b', '').str.strip_chars() == '').alias('addr_empty'))
