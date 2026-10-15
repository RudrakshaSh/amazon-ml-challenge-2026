"""Normalisation rules, checked on real noise examples from eda/FINDINGS.md."""

import pytest

from ber.normalize.address import AddressNormalizer
from ber.normalize.text import NameNormalizer, fold, join_single_letters, ocr_fix
from ber.normalize.translit import Transliterator, learn_dictionary, rule_translit

N = NameNormalizer()
A = AddressNormalizer()


@pytest.mark.parametrize("a, b", [
    # true pairs from the EDA: must normalise to the same core
    ("UW Runners Private Limited", "UW  RUNNERS PRlVATE LTD"),     # l/i typo in legal part
    ("Vision Health P.C.", "VISION HEALTH P.C."),
    ("Ear Nose & Throat Peak Specialists Inc.", "EAR NOSE + THROAT PEAK SPECIALISTS INCORPORATED"),
    ("Prairie Grand Opportunity", "Prairie Grand ópportunity"),
    ("YX Training Private Limited", "YX Training  Private Ltd"),
    ("Deleves & Cie SARL", "Deleves +-Cie S.A.R.L."),
    ("MI Loisirs EI", "MI Loisirs E.I."),
    ("Bujinkan Comite SARL", "BUJINKAN C0MITE SARL"),
    ("Lecrit Institut SAS", "SAS LECRIT INSTITUT"),  # legal form moved; core order kept
    ("Health Services", "Hea1th 5ervices"),
])
def test_true_pairs_share_core(a, b):
    assert N(a).core == N(b).core


@pytest.mark.parametrize("a, b", [
    # lookalike distractors from the EDA: cores must stay different
    ("Verrify America LLC", "Verrufy America LLC"),
    ("Clo T. Hudson, DMD", "Clo T. Hudson, DMMD Corp"),
    ("Juana L. Adams, MD, PC", "Juana L. Ádams, JD, PC"),
    ("Desert League II", "DESERT LEAGUE IC LLC"),
])
def test_distractors_keep_differences(a, b):
    assert N(a).core != N(b).core


def test_legal_and_core():
    r = N("Rajkiran Technology Exports Pvt ((Ltd))")
    assert r.core == "rajkiran technology exports" and r.legal == "ltd pvt"
    assert N("The Company").core  # never empty


def test_alias_and_domain():
    r = N("Caloquo aka Doryphores Parents SARL")
    assert r.has_alias and r.pre_core == "caloquo" and r.alt_core == "doryphores parents"
    assert N("Soldelta Labs F/K/A Club des Choeur").alt_core == "club des choeur"
    assert N("Korveo Co DBA: Hari Ventures Limited").alt_core == "hari ventures"
    d = N("www.unitedimpex.com")
    assert d.is_domain and d.core == "unitedimpex"
    assert N("biodanzafranceparents.com").compact == "biodanzafranceparents"
    assert N("Biodanza (France) Parents SARL").compact == "biodanzafranceparents"


def test_single_letters_and_ocr():
    assert join_single_letters("l l c".split()) == ["llc"]
    assert join_single_letters("a p bannister".split()) == ["ap", "bannister"]
    assert join_single_letters("gray reta d o m d p c".split()) == ["gray", "reta", "do", "md", "pc"]
    assert ocr_fix("c0mite") == "comite" and ocr_fix("1st") == "1st" and ocr_fix("2877") == "2877"
    assert fold("Établissements Côte") == "etablissements cote"


@pytest.mark.parametrize("raw, country, norm, region, numbers", [
    ("1795 Westchester Drive, High Point, NC", "US", "1795 westchester dr high point", "NC", "1795"),
    ("MA, WORCESTER, 22 LANCASTER STREET", "US", "worcester 22 lancaster st", "MA", "22"),
    ("00132 Lemonwood Lane, Hollister, Missouri", "US", "132 lemonwood ln hollister", "MO", "00132"),
    ("71 Ottawa St, # 102, Joliet, Illinois", "US", "71 ottawa st 102 joliet", "IL", "71 102"),
    ("Hemlock Trl, null, Coventry, Connecticut", "US", "hemlock trl coventry", "CT", ""),
    ("H.NO #52, BANGALORE RURAL, KA", "India", "52 bangalore rural", "KA", "52"),
    ("Mumbai, A-18, Goregaon East, महाराष्ट्र", "India", "mumbai a 18 goregaon east", "MH", "18"),
    ("41 R PIERRE, BORDEAUX, Nouvelle-Aquitaine", "France", "41 rue pierre bordeaux", "NAQ", "41"),
    ("N°23 R. D'arras, Lille, Nord", "France", "23 rue d arras lille", "HDF", "23"),
    ("", "US", "", "", ""),
])
def test_address(raw, country, norm, region, numbers):
    r = A(raw, country)
    assert (r.norm, r.region, r.numbers) == (norm, region, numbers)


def test_address_reorder_gives_same_bag():
    a = A("Phoenix, 3013 27th Street, AZ", "US")
    b = A("3013 27TH ST, PHOENIX, AZ", "US")
    assert a.bag == b.bag and a.region == b.region == "AZ"


def test_box_flag():
    assert A("41 CLAYTON PL, PMB 521, YONKERS, NY", "US").has_box
    assert not A("41 Clayton Place, Yonkers, NY", "US").has_box


def test_translit_dictionary_and_rules():
    indic = ["विजय इन्वेस्टमेंट एलएलपी", "विजय प्राइवेट लिमिटेड", "मॉडर्न प्राइवेट लिमिटेड"]
    latin = ["Vijay Investment LLP", "Vijay Private Limited", "Modern Private Limited"]
    d = learn_dictionary(indic, latin, min_count=2)
    lut = dict(zip(d.indic, d.latin))
    assert lut["प्राइवेट"] == "private" and lut["विजय"] == "vijay"
    tr = Transliterator(lut)
    assert tr("लिमिटेड") == ("limited", True)
    assert tr("इन्वेस्टमेंट")[1] is False  # count 1 < min_count -> rules
    assert rule_translit("लिमिटेड") == "limited"
    n = NameNormalizer(Transliterator(lut))("विजय प्राइवेट लिमिटेड")
    assert n.core == "vijay" and n.legal == "ltd pvt" and n.indic and n.n_indic_oov == 0


def test_normalize_frame_columns():
    import pandas as pd

    from ber.normalize import stage
    stage._init_worker({"प्राइवेट": "private"})
    df = pd.DataFrame({"entity_id": ["S2-1", "S2-2"], "source": [2, 2],
                       "business_name": ["राम प्राइवेट लिमिटेड", "Hea1th 5ervices LLC"],
                       "business_address": ["KH NO. 570, NEW DELHI, Delhi", ""],
                       "country": ["India", "US"]})
    out = stage.normalize_frame(df)
    assert out.name_core.tolist()[1] == "health services"
    assert out.region.tolist() == ["DL", ""]
    assert "name_raw" in out and "addr_bag" in out and out.name_indic.tolist() == [True, False]


def test_skeleton_vocab_fallback():
    from ber.normalize.translit import build_vocab, skeleton
    assert skeleton("limatid") == skeleton("limited")
    assert skeleton("praivet") == skeleton("private")
    assert skeleton("sisatamaj") == skeleton("systems")
    vocab = build_vocab(["Private Limited", "Private Limited", "Global Systems", "Global Systems"])
    tr = Transliterator({}, vocab)
    assert tr("ਲਿਮਟਿਡ") == ("limited", True)      # Gurmukhi, never in any dictionary
    assert tr("ਪ੍ਰਾਈਵੇਟ") == ("private", True)
    assert tr("ਪ੍ਰਾ") == ("pvt", True)              # short form via overrides table


@pytest.mark.parametrize("a, b", [
    # v2: injected ids / phone numbers, doubled words, swapped letters in legal words
    ("Happy Yoga", "Happy Yoga (ID: 76971)"),
    ("Arti Project", "Arti Project - 8778657304"),
    ("Green Infotech Private Ltd", "Green Infotech Private Ltd (ID: [43998)]"),
    ("Dominquez Lake Co", "Dominquez Lake Co [#53957]"),
    ("Royal Biomedical Company", "ROYAL BIOMEDICAL BIOMEDICAL COMPANY"),
    ("Tech Software Private Limited", "Tech Software Priavte Limited"),
    ("Bnp Group of Companies", "Bnp Group Companies"),
])
def test_v2_true_pairs_share_core(a, b):
    assert N(a).core == N(b).core


def test_v2_keeps_real_numbers():
    assert N("Electricians Local 786 LLC").core == "electricians local 786"
    assert N("3543 Seventh Blvd Apartments Partners").core.startswith("3543 ")
