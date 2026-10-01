#!/opt/conda/bin/python3
"""Structural archetypes of (s1_idx, cand_idx) pairs from the forensics per-record views (label-free).

classify(pairs, split) -> pairs + columns
  house_rel   equal | plus12 | plusD3 | minusD | other_offset | s1_missing | rec_missing | both_missing   (record house - S1 house)
  street_rel  same_key | same_toks_diff_number | different | missing                                   (street tokens, house-free)
  name_rel    identical_core | legal_switch | legal_add | legal_drop | country_tag | rec_plus1_decoyword | rec_plus1_other | rec_plus2 |
              rec_shorter | swap1_abbr | swap1_generic | swap1_typo | swap1_other | no_shared_token | partial_other | empty_content
              (set relation of the CONTENT tokens = name minus legal forms / country tags, the same token view the forensics decoy
               families ADDW/ADDO/EQ use; legal/tag relations are only tested when the content sets are identical)
  rec_addr_empty (bool), s1_core_shared_by ('1' | '2-5' | '6-50' | '>50' S1 of the country with the same core_key), n_core_shared (int)
  arch_full = '<addr>|<name_rel>' with addr in
              sameKey | plus12 | plusD3 | minusD | otherOff | sameToksNoNum | diffStreetSameNum | diffStreet | noStreetSameNum | noAddrKey | noStreet | addrEmpty
assign_archetype(df, keep) -> 'archetype' = arch_full if kept else 'rare|<name group>'.

Views: work/research/france/forensics/data/views_{test_fr,test_usin,train}.parquet (src 1 = S1, 2/3 = records).
Decoy list: forensics out/decoys.json, rule npos >= 500 and npos/(nneg+1) >= 20 (as IE/src/03_pairclass.py).
Generic words: tokens with >= 1000 occurrences in the country's S1 core_key (same split; TEST-INPUT-DERIVED for split='test'), as
work/research/france/pseudolabels/scripts/07_k0_swap.py. Abbreviation rule as 07b_swap_classes.py (same first letter, <= 3 chars, shorter).
Typo rule: Levenshtein <= 1 (max token length >= 3) or <= 2 (max length >= 6).
CLI: archetype.py build test|train   -> caches under agents/B_archetypes/cache/."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import polars as pl

VIEW_COLS = ['entity_id', 'a_empty', 'house', 'street_toks', 'core_key', 'content', 'legal', 'tags', 'country', 'src']
COUNTRIES = ('US', 'India', 'France')
NAME_GROUP = {'identical_core': 'sameName', 'legal_switch': 'sameName', 'legal_add': 'sameName', 'legal_drop': 'sameName', 'country_tag': 'sameName',
              'rec_plus1_decoyword': 'plusWords', 'rec_plus1_other': 'plusWords', 'rec_plus2': 'plusWords', 'rec_shorter': 'shorter',
              'swap1_abbr': 'swap', 'swap1_generic': 'swap', 'swap1_typo': 'swap', 'swap1_other': 'swap',
              'no_shared_token': 'unrelated', 'partial_other': 'unrelated', 'empty_content': 'unrelated'}


def decoy_words():
    d = json.load(open(f'{FO}/out/decoys.json'))
    return {c: sorted(w for w, npos, nneg, n0 in d[c]['added_words'] if npos >= 500 and npos / (nneg + 1) >= 20) for c in COUNTRIES}


def _norm_set(col):
    """'PVT|LTD|LTD' -> 'LTD|PVT' (sorted unique)."""
    return pl.col(col).fill_null('').str.split('|').list.eval(pl.element().filter(pl.element() != '')).list.unique().list.sort().list.join('|')


def _views_raw(split):
    if split == 'test':
        return pl.concat([pl.read_parquet(f'{FO}/data/views_test_fr.parquet', columns=VIEW_COLS),
                          pl.read_parquet(f'{FO}/data/views_test_usin.parquet', columns=VIEW_COLS)])
    return pl.read_parquet(f'{FO}/data/views_train.parquet', columns=VIEW_COLS)


def build_views(split, log=print):
    os.makedirs(CACHE, exist_ok=True)
    V = _views_raw(split).with_columns(pl.col('street_toks').fill_null(''), pl.col('content').fill_null(''), pl.col('core_key').fill_null(''),
                                       _norm_set('legal').alias('legal'), _norm_set('tags').alias('tags'), pl.col('house').cast(pl.Int64))
    log(split, 'views', V.height)
    S1 = V.filter(pl.col('src') == 1).join(ids(split, 's1').select('entity_id', 's1_idx'), on='entity_id', how='inner')
    R = V.filter(pl.col('src') != 1).join(ids(split, 's23').select('entity_id', 'cand_idx'), on='entity_id', how='inner')
    assert S1.height == ids(split, 's1').height, (S1.height, ids(split, 's1').height)
    assert R.height == ids(split, 's23').height, (R.height, ids(split, 's23').height)
    shared = S1.filter(pl.col('core_key') != '').group_by('country', 'core_key').len('n_core_shared')
    S1 = S1.join(shared, on=['country', 'core_key'], how='left').with_columns(pl.col('n_core_shared').fill_null(1).cast(pl.Int32))
    S1.select('s1_idx', 'country', 'house', 'street_toks', 'core_key', 'content', 'legal', 'tags', 'a_empty', 'n_core_shared').sort('s1_idx').write_parquet(f'{CACHE}/views_{split}_s1.parquet')
    R.select('cand_idx', 'house', 'street_toks', 'core_key', 'content', 'legal', 'tags', 'a_empty').sort('cand_idx').write_parquet(f'{CACHE}/views_{split}_rec.parquet')
    # generic words: S1 core tokens with >= 1000 occurrences per country
    tf = S1.select('country', pl.col('core_key').str.split(' ').alias('t')).explode('t').filter(pl.col('t') != '').group_by('country', 't').len('n')
    tf.filter(pl.col('n') >= 1000).rename({'t': 'word'}).sort('country', 'n', descending=[False, True]).write_parquet(f'{CACHE}/generic_{split}.parquet')
    log(split, 'S1', S1.height, 'records', R.height, 'generic words', tf.filter(pl.col('n') >= 1000).group_by('country').len().to_dicts())


def load_views(split, log=print):
    p = f'{CACHE}/views_{split}_s1.parquet'
    if not os.path.exists(p):
        build_views(split, log)
    S1 = pl.read_parquet(p); R = pl.read_parquet(f'{CACHE}/views_{split}_rec.parquet'); G = pl.read_parquet(f'{CACHE}/generic_{split}.parquet')
    return S1, R, G


def _lev(a, b):
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_typo(a, b):
    if a is None or b is None: return False
    m = max(len(a), len(b)); d = _lev(a, b)
    return (d <= 1 and m >= 3) or (d <= 2 and m >= 6)


def classify(pairs, split, log=print):
    """pairs: DataFrame with s1_idx, cand_idx (Int32, index space of `split`). Returns pairs (unique) + archetype columns."""
    S1, R, G = load_views(split, log)
    DW = decoy_words()
    dw = pl.DataFrame({'country': [c for c in COUNTRIES for _ in DW[c]], 'word': [w for c in COUNTRIES for w in DW[c]]}).with_columns(pl.lit(True).alias('is_decoy'))
    G = G.select('country', 'word').with_columns(pl.lit(True).alias('is_generic'))
    P = pairs.select(K2).unique().with_columns(pl.col('s1_idx').cast(pl.Int32), pl.col('cand_idx').cast(pl.Int32))
    n0 = P.height
    P = P.join(S1.rename({'house': 'h1', 'street_toks': 'st1', 'content': 'ca', 'core_key': 'ck1', 'legal': 'l1', 'tags': 'tg1', 'a_empty': 's1_addr_empty'}), on='s1_idx', how='left')
    P = P.join(R.rename({'house': 'h2', 'street_toks': 'st2', 'content': 'cb', 'core_key': 'ck2', 'legal': 'l2', 'tags': 'tg2', 'a_empty': 'rec_addr_empty'}), on='cand_idx', how='left')
    assert P.height == n0 and P['country'].null_count() == 0 and P['cb'].null_count() == 0, 'pairs outside the views'
    d = pl.col('h2') - pl.col('h1')
    house_rel = (pl.when(pl.col('h1').is_null() & pl.col('h2').is_null()).then(pl.lit('both_missing'))
                   .when(pl.col('h1').is_null()).then(pl.lit('s1_missing')).when(pl.col('h2').is_null()).then(pl.lit('rec_missing'))
                   .when(d == 0).then(pl.lit('equal')).when(d.is_in(D12)).then(pl.lit('plus12')).when(d.is_in(D3)).then(pl.lit('plusD3'))
                   .when(d.is_in(MINUS_D)).then(pl.lit('minusD')).otherwise(pl.lit('other_offset')))
    P = P.with_columns(house_rel.alias('house_rel'), d.alias('house_offset'))
    st_missing = (pl.col('st1') == '') | (pl.col('st2') == '')
    same_toks = ~st_missing & (pl.col('st1') == pl.col('st2'))
    P = P.with_columns(pl.when(st_missing).then(pl.lit('missing')).when(same_toks & (pl.col('house_rel') == 'equal')).then(pl.lit('same_key'))
                         .when(same_toks).then(pl.lit('same_toks_diff_number')).otherwise(pl.lit('different')).alias('street_rel'))
    tok = lambda c: pl.col(c).str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique()
    P = P.with_columns(tok('ca').alias('t1'), tok('cb').alias('t2'))
    P = P.with_columns(pl.col('t1').list.len().alias('n1'), pl.col('t2').list.len().alias('n2'), pl.col('t1').list.set_intersection('t2').list.len().alias('ni'),
                       pl.col('t2').list.set_difference('t1').list.sort().list.first().alias('w_in'), pl.col('t1').list.set_difference('t2').list.sort().list.first().alias('w_out'))
    P = P.join(dw.rename({'word': 'w_in'}), on=['country', 'w_in'], how='left').join(G.rename({'word': 'w_in'}), on=['country', 'w_in'], how='left') \
         .with_columns(pl.col('is_decoy').fill_null(False), pl.col('is_generic').fill_null(False))
    n1, n2, ni = pl.col('n1'), pl.col('n2'), pl.col('ni')
    ident = (ni == n1) & (ni == n2)
    swap = (n1 == n2) & (ni == n1 - 1) & (n1 >= 2)
    single = (n1 == 1) & (n2 == 1) & (ni == 0)
    abbr = (pl.col('w_in').str.slice(0, 1) == pl.col('w_out').str.slice(0, 1)) & (pl.col('w_in').str.len_chars() <= 3) & (pl.col('w_in').str.len_chars() < pl.col('w_out').str.len_chars())
    # typo test (python) only where it can matter: swap1 not abbr/generic, and single-token names
    need = P.filter((swap & ~abbr & ~pl.col('is_generic')) | single).select(*K2, 'w_in', 'w_out')
    if need.height:
        typo = [_is_typo(a, b) for a, b in zip(need['w_in'].to_list(), need['w_out'].to_list())]
        P = P.join(need.select(K2).with_columns(pl.Series('is_typo', typo)), on=K2, how='left').with_columns(pl.col('is_typo').fill_null(False))
    else:
        P = P.with_columns(pl.lit(False).alias('is_typo'))
    name_rel = (pl.when((n1 == 0) | (n2 == 0)).then(pl.lit('empty_content'))
                  .when(ident & (pl.col('l1') == pl.col('l2')) & (pl.col('tg1') == pl.col('tg2'))).then(pl.lit('identical_core'))
                  .when(ident & (pl.col('l1') != pl.col('l2')) & (pl.col('l1') != '') & (pl.col('l2') != '')).then(pl.lit('legal_switch'))
                  .when(ident & (pl.col('l1') == '') & (pl.col('l2') != '')).then(pl.lit('legal_add'))
                  .when(ident & (pl.col('l1') != '') & (pl.col('l2') == '')).then(pl.lit('legal_drop'))
                  .when(ident).then(pl.lit('country_tag'))
                  .when((ni == n1) & (n2 == n1 + 1) & pl.col('is_decoy')).then(pl.lit('rec_plus1_decoyword'))
                  .when((ni == n1) & (n2 == n1 + 1)).then(pl.lit('rec_plus1_other'))
                  .when((ni == n1) & (n2 >= n1 + 2)).then(pl.lit('rec_plus2'))
                  .when((ni == n2) & (n2 < n1)).then(pl.lit('rec_shorter'))
                  .when(swap & abbr).then(pl.lit('swap1_abbr'))
                  .when(swap & pl.col('is_generic')).then(pl.lit('swap1_generic'))
                  .when(swap & pl.col('is_typo')).then(pl.lit('swap1_typo'))
                  .when(swap).then(pl.lit('swap1_other'))
                  .when(single & pl.col('is_typo')).then(pl.lit('swap1_typo'))
                  .when(ni == 0).then(pl.lit('no_shared_token'))
                  .otherwise(pl.lit('partial_other')))
    P = P.with_columns(name_rel.alias('name_rel'))
    hr, sr = pl.col('house_rel'), pl.col('street_rel')
    addr = (pl.when(pl.col('rec_addr_empty')).then(pl.lit('addrEmpty'))
              .when(sr == 'same_key').then(pl.lit('sameKey'))
              .when((sr == 'same_toks_diff_number') & (hr == 'plus12')).then(pl.lit('plus12'))
              .when((sr == 'same_toks_diff_number') & (hr == 'plusD3')).then(pl.lit('plusD3'))
              .when((sr == 'same_toks_diff_number') & (hr == 'minusD')).then(pl.lit('minusD'))
              .when((sr == 'same_toks_diff_number') & (hr == 'other_offset')).then(pl.lit('otherOff'))
              .when(sr == 'same_toks_diff_number').then(pl.lit('sameToksNoNum'))
              .when((sr == 'different') & (hr == 'equal')).then(pl.lit('diffStreetSameNum'))
              .when(sr == 'different').then(pl.lit('diffStreet'))
              .when(hr == 'equal').then(pl.lit('noStreetSameNum'))
              .when(hr == 'both_missing').then(pl.lit('noAddrKey'))
              .otherwise(pl.lit('noStreet')))
    P = P.with_columns(addr.alias('addr_rel'))
    P = P.with_columns((pl.col('addr_rel') + '|' + pl.col('name_rel')).alias('arch_full'),
                       pl.when(pl.col('n_core_shared') <= 1).then(pl.lit('1')).when(pl.col('n_core_shared') <= 5).then(pl.lit('2-5'))
                         .when(pl.col('n_core_shared') <= 50).then(pl.lit('6-50')).otherwise(pl.lit('>50')).alias('s1_core_shared_by'))
    keep = K2 + ['country', 'house_rel', 'house_offset', 'street_rel', 'name_rel', 'addr_rel', 'arch_full', 'rec_addr_empty', 's1_core_shared_by', 'n_core_shared', 'w_in', 'w_out']
    return P.select(keep)


# ---- the archetype vocabulary: a designed 2-level grid (56 cells, full coverage, no tail bucket)
# group A (same content, the exact house offset is the decoy signal): 4 names x 8 address levels
# group B (edited names): 6 names x 4 address levels (offsets collapse to nearKey, no-address cases to weakAddr)
NAME_A = {'identical_core': 'sameName', 'legal_drop': 'sameName', 'legal_add': 'formChange', 'legal_switch': 'formChange', 'country_tag': 'formChange',
          'rec_plus1_decoyword': 'plus1Decoy', 'rec_plus1_other': 'plusOther', 'rec_plus2': 'plusOther'}
NAME_B = {'swap1_abbr': 'swapMinor', 'swap1_typo': 'swapMinor', 'swap1_generic': 'swapGeneric', 'swap1_other': 'swapOther', 'rec_shorter': 'shorter',
          'no_shared_token': 'noShared', 'partial_other': 'partialOther', 'empty_content': 'noShared'}
ADDR_A = {'sameKey': 'sameKey', 'plus12': 'plus12', 'plusD3': 'plusD3', 'minusD': 'minusD', 'otherOff': 'otherOff', 'diffStreetSameNum': 'partialAddr',
          'noStreetSameNum': 'partialAddr', 'sameToksNoNum': 'partialAddr', 'diffStreet': 'weakAddr', 'noStreet': 'weakAddr', 'noAddrKey': 'weakAddr', 'addrEmpty': 'addrEmpty'}
ADDR_B = {'sameKey': 'sameKey', 'plus12': 'nearKey', 'plusD3': 'nearKey', 'minusD': 'nearKey', 'otherOff': 'nearKey', 'diffStreetSameNum': 'partialAddr',
          'noStreetSameNum': 'partialAddr', 'sameToksNoNum': 'partialAddr', 'diffStreet': 'weakAddr', 'noStreet': 'weakAddr', 'noAddrKey': 'weakAddr', 'addrEmpty': 'weakAddr'}
ARCH_NAME_GROUP = {'sameName': 'sameName', 'formChange': 'sameName', 'plus1Decoy': 'plusWords', 'plusOther': 'plusWords', 'shorter': 'shorter',
                   'swapMinor': 'swap', 'swapGeneric': 'swap', 'swapOther': 'swap', 'noShared': 'unrelated', 'partialOther': 'unrelated'}
ALL_ARCHETYPES = sorted({f'{a}|{n}' for a in set(ADDR_A.values()) for n in set(NAME_A.values())} | {f'{a}|{n}' for a in set(ADDR_B.values()) for n in set(NAME_B.values())})


def assign_archetype(df, keep=None):
    """archetype = '<addr level>|<name level>' from the designed grid (keep is ignored; kept for call compatibility)."""
    inA = pl.col('name_rel').is_in(list(NAME_A))
    name = pl.when(inA).then(pl.col('name_rel').replace_strict(NAME_A, default=None)).otherwise(pl.col('name_rel').replace_strict(NAME_B, default='noShared'))
    addr = pl.when(inA).then(pl.col('addr_rel').replace_strict(ADDR_A, default='weakAddr')).otherwise(pl.col('addr_rel').replace_strict(ADDR_B, default='weakAddr'))
    return df.with_columns((addr + '|' + name).alias('archetype'))


def load_keep():
    return ALL_ARCHETYPES


if __name__ == '__main__':
    envcap(8)
    log = logger('archetype_build')
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        for sp in sys.argv[2:] or ['test', 'train']:
            build_views(sp, log)
        log('DONE')
