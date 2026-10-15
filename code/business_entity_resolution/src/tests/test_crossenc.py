import pandas as pd
import pytest


def test_with_text_builds_record_texts(tmp_path):
    from ber.crossenc.stage import with_text
    rec = pd.DataFrame({"entity_id": ["S1-1", "S2-9"], "name_core": ["smart healthcare", "smart health"],
                        "name_legal": ["ltd pvt", ""], "addr_norm": ["303 ashram rd ahmedabad", ""],
                        "region": ["GJ", ""]})
    d = tmp_path / "records_train"
    d.mkdir()
    rec.to_parquet(d / "part-0.parquet", index=False)
    out = with_text(pd.DataFrame({"q_id": ["S2-9"], "s1_id": ["S1-1"], "y": [1]}), d)
    assert out.q_text.iloc[0] == "smart health ; "
    assert out.s_text.iloc[0] == "smart healthcare ltd pvt ; 303 ashram rd ahmedabad GJ"
    with pytest.raises(RuntimeError):
        with_text(pd.DataFrame({"q_id": ["S2-404"], "s1_id": ["S1-1"], "y": [0]}), d)
