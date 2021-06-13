
def test_norm(char_tknzr):
    # Case normalization.
    if char_tknzr.is_uncased:
        assert char_tknzr.norm('ABC') == 'abc'
    else:
        assert char_tknzr.norm('ABC') == 'ABC'

    # Full-width to half-width.
    assert char_tknzr.norm('０') == '0'
    # NFKD to NFKC.
    assert char_tknzr.norm('é') == 'é'
