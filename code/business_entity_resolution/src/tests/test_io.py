from ber.io import read_tsv


def test_read_tsv_keeps_quotes_and_empty(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                 'S1-1\t"Joe\'s" "Bar\t\tUS\n')
    df = read_tsv(p)
    assert df.business_name[0] == '"Joe\'s" "Bar' and df.business_address[0] == ""
