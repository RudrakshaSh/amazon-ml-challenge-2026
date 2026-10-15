import pytest

from ber.normalize.geo import detect_region


@pytest.mark.parametrize("addr, country, expected", [
    ("1795 Westchester Drive, High Point, NC", "US", "NC"),
    ("IA, Iowa City, 1064 Newton Rd, Unit 11", "US", "IA"),
    ("1709 Georgine Court, Imperial, Missouri", "US", "MO"),
    ("District of Columbia, Unit 713, 1011 4st Street, Washington", "US", "DC"),
    ("123 Main St, Seattle, Washington", "US", "WA"),
    ("Upper Allen Township, 360-364 Melbourne Lane, Pennsylvania", "US", "PA"),
    ("5212 Tinkers Creek Place, Clinton, In", "US", ""),          # code must be exact upper
    ("Liliya Road, Amreli, GJ", "India", "GJ"),
    ("Narnaul, Kailash Nagar, Haryana, Mahendragarh", "India", "HR"),
    ("#525 1/7D, Kolkata, Calcutta, পশ্চিমবঙ্গ", "India", "WB"),
    ("Office No 1, Hadapsar, Pune, महाराष्ट्र", "India", "MH"),
    ("P No 145, Beeramguda, Orissa", "India", "OD"),
    ("Door 6 4Rd Floor, Thrissur, KL", "India", "KL"),
    ("C-199, First Floor, New Delhi, Delhi", "India", "DL"),
    ("17 Rue de Valmy, Lille, Hauts-de-France", "France", "HDF"),
    ("13 RUE ALPHONSE MERCIER, LILLE, Nord", "France", "HDF"),
    ("Pessac, Gironde, 60 AVENUE DU PARC", "France", "NAQ"),
    ("", "US", ""),
    ("MH, Mumbai", "US", ""),                                    # codes are per country
])
def test_detect_region(addr, country, expected):
    assert detect_region(addr, country) == expected
