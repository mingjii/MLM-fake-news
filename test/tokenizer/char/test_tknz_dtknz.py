def test_tknz(char_tknzr, txt_tks_pair):
    txt = txt_tks_pair['txt']
    tks = txt_tks_pair['tks']

    assert char_tknzr.tknz(txt) == tks


def test_dtknz(char_tknzr, txt_tks_pair):
    txt = txt_tks_pair['txt']
    tks = txt_tks_pair['tks']

    assert char_tknzr.dtknz(tks) == txt


def test_empty(char_tknzr):
    assert char_tknzr.tknz('') == []
    assert char_tknzr.dtknz([]) == ''
