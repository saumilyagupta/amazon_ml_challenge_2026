"""Unit tests of the explainer on the noise-spec examples (copied from work/features/explainer/tests/test_views.py).
Run: cd src && /opt/conda/bin/python3 -m pytest -q tools/test_explainer_views.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import pytest
from ber.explainer.wordlists import get_tables
from ber.explainer.views import name_view, addr_view, hg, fold
from ber.explainer.features import explain_name, explain_addr, number_relation, record_views, compute_features, ALL_FEATURES, web_match

US, IN, FR = get_tables('US'), get_tables('India'), get_tables('France')


def nv(s, T=US, src=2):
    return name_view(s, src, T)


def test_dotted_legal_llc():
    v = nv('Star Clean Valley L.L.C.')
    assert v.main.tokens == ['star', 'clean', 'valley', 'llc']
    assert v.main.legal == ['LLC'] and 'dotted' in v.markers
    assert v.main.core == ['star', 'clean', 'valley']


def test_phone_suffix_strip():
    v = nv('Best Infotech - 9219770714', IN)
    assert v.main.tokens == ['best', 'infotech'] and 'phone' in v.markers


def test_record_and_id_tags():
    assert nv('Piedmont [Charities] #29588').main.tokens == ['piedmont', 'charities']
    assert 'id_tag' in nv('Acme Tools (ID: 48213)').markers


def test_comma_in_word_rejoin():
    v = nv('DOMCI,E Services')
    assert v.main.tokens[0] == 'domcie' and 'comma_in_word' in v.markers
    assert nv('Westerman, Blair and Brickey').main.tokens[:2] == ['westerman', 'blair']  # comma + space is not rejoined


def test_homoglyph():
    assert hg('5ervices') == hg('services')
    A, B = record_views('Global Services Inc', '1 Main St, Troy, NY', 'US', 1), record_views('Global 5ervices Inc', '1 Main St, Troy, NY', 'US', 2)
    r = explain_name(A[0], B[0], US)
    assert r.explained and 'homoglyph' in r.ops


def test_web_stem_domain():
    v = nv('developmentdex.com', FR, 3)
    assert v.main.web == 'developmentdex' and v.main.web_kind == 'domain'
    A = name_view('Dex Development SARL', 1, FR)
    r = explain_name(A, v, FR)
    assert r.explained and 'domain' in r.ops and r.web_dist == 0
    assert nv('www.orthopedicsafehealth.com').main.web == 'orthopedicsafehealth'
    assert nv('@bestsumitomo', US, 3).main.web_kind == 'handle'


def test_acronym_cmv():
    A = name_view('Centre Medical Vauban SAS', 1, FR)
    r = explain_name(A, name_view('CMV', 2, FR), FR)
    assert r.explained and r.acronym == 1 and 'acronym' in r.ops
    r2 = explain_name(A, name_view('MVC', 2, FR), FR)  # any order
    assert r2.acronym == 1


def test_alias_split_dba():
    v = name_view('Lumxylo D.B.A. Greene, Cardone and Schumacher PC', 3, US)
    assert v.alias and 'alias' in v.markers
    assert v.sides[0].content == ['greene', 'cardone', 'schumacher'] and v.sides[1].content == ['lumxylo']
    A = name_view('Greene, Cardone and Schumacher PC', 1, US)
    r = explain_name(A, v, US)
    assert r.explained and 'alias_marker' in r.ops and r.side == 0


def _rel(a1, a2, T=US):
    A, B = addr_view(a1, T), addr_view(a2, T)
    return number_relation(A, B), A, B


def test_number_digit_dropped():
    (rel, ops, ins), _, _ = _rel('136 Main Street, Troy, NY', '36 MAIN ST, TROY, NY')
    assert rel == 4 and 'num_digit_drop' in ops


def test_number_hyphen_split():
    (rel, ops, ins), _, _ = _rel('254 Oak Avenue, Troy, NY', '2-54 Oak Ave, Troy, New York')
    assert rel == 3 and 'num_hyphen_split' in ops


def test_number_range_and_zero_pad():
    (rel, ops, _), _, _ = _rel('9004 Meadow Hills Drive, Scottsdale, AZ', '9004-9006 Meadow Hills Dr, Scottsdale, Arizona')
    assert rel in (1, 2)
    (rel, ops, _), A, B = _rel('635 Concord Street, Billings, MT', '00635 CONCORD ST, BILLINGS, MT')
    assert rel == 1 and 'num_zero_pad' in ops and B.house.val == 635


def test_bis_suffix():
    (rel, ops, _), A, B = _rel('98B Rue de Dieppe, Lille, Hauts-de-France', '98 BIS R. DE DIEPPE, LILLE, Nord', FR)
    assert A.house.val == 98 and B.house.val == 98 and A.house.suffix == 'b' and B.house.suffix == 'bis'
    assert rel == 1 and 'num_suffix' in ops
    assert addr_view('5 bis Rue Pierre Dignac, La Teste-de-Buch, Nouvelle-Aquitaine', FR).street_key == '5|dignac pierre'


def test_inserted_prefix_door_no():
    (rel, ops, ins), A, B = _rel('12 Gandhi Road, Salem, Tamil Nadu', 'DOOR NO 187 12 GANDHI ROAD, SALEM, TN', IN)
    assert ins == 1 and rel == 1 and 'num_inserted' in ops
    (rel, ops, ins), A, B = _rel('Sree Balaji Complex, Salem, Tamil Nadu', 'DOOR NO 187 Sree Balaji Complex, Salem, TN', IN)
    assert ins == 1 and 'num_inserted' in ops
    (rel, ops, ins), _, _ = _rel('19 Dda Flats, New Delhi, Delhi', 'H.NO 59 DDA FLATS, NEW DELHI, Delhi', IN)
    assert ins == 0 and rel == 5  # prefix + one digit substituted, not an insertion


def test_french_street_type_r_dot():
    a = addr_view('63 R. DE DIEPPE, LILLE, Hauts-de-France', FR)
    b = addr_view('63 Rue de Dieppe, Lille, Hauts-de-France', FR)
    assert 'rue' in a.tokens and a.tokens == b.tokens and a.street_key == b.street_key
    assert addr_view('77 AV LEON JOUHAUX, LILLE', FR).tokens == addr_view('77 Avenue Leon Jouhaux, Lille', FR).tokens


def test_department_region_alias():
    a = addr_view('30 Rue Lachassaigne, Bordeaux, Gironde', FR)
    b = addr_view('30 Rue Lachassaigne, Bordeaux, Nouvelle-Aquitaine', FR)
    assert a.admin == b.admin == {'nouvelle aquitaine'}
    r = explain_addr(b, a, FR)
    assert r.explained and 'admin_alias' in r.ops and r.admin_agree == 1


def test_us_state_code_vs_full_and_ordinals():
    a, b = addr_view('2653 6th Place, Portland, OR', US), addr_view('2653 Sixth Pl, Portland, Oregon', US)
    assert a.admin == b.admin == {'OR'} and a.tokens == b.tokens


def test_placeholders_pobox_missing_flags():
    b = addr_view('17408 SADDLE CREEK, null, PO Box 55, NEW CANEY, TX', US)
    assert b.placeholder and b.pobox and b.house.val == 17408
    df = compute_features(pl.DataFrame({'s1_id': ['S1-1', 'S1-1'], 'cand_id': ['S2-1', 'S3-2']}),
                          {'S1-1': ('Clyeus Consumer PC', '9178 Burlingame Court, Bainbridge Island, WA', 'US', 1, None, None),
                           'S2-1': ('CLYEUS CONSUMER', '', 'US', 2, None, None),
                           'S3-2': ('Clyeus Consumer', '9178 Burlingame Ct, Bainbridge Island, Washington', 'US', 3, None, None)})
    assert df.columns[2:] == ALL_FEATURES
    r0, r1 = df.row(0, named=True), df.row(1, named=True)
    assert r0['a_bp'] == 0 and r0['a12_addr_missing_b'] == 1 and r0['a7_street_key_eq'] == -1 and r0['a11_admin_agree'] == -1
    assert r1['a_bp'] == 1 and r1['a7_street_key_eq'] == 1 and r1['x1_name_explained'] == 1 and r1['x1_addr_explained'] == 1
    assert r1['s1_is_s3'] == 1 and r0['n12_b_caps'] == 1


def test_fold_accents_and_ndeg():
    assert fold('Fócus Léarning') == 'focus learning'
    assert fold('N° 12 Rue') == 'no 12 rue'
