
def test_build_vocab(char_tknzr, batch_txt):

    char_tknzr.build_vocab(batch_txt)

    # Ensure vocabulary size.
    assert char_tknzr.vocab_size() <= char_tknzr.max_vocab
    # Ensure model special tokens were covered.
    assert '<cls>' in char_tknzr.tk2id
    assert '<sep>' in char_tknzr.tk2id
    assert '<unk>' in char_tknzr.tk2id
    assert '<mask>' in char_tknzr.tk2id
    assert '<pad>' in char_tknzr.tk2id
    # Ensure ordinary tokens were covered.
    assert '你' in char_tknzr.tk2id
    assert '好' in char_tknzr.tk2id
    assert '嗎' in char_tknzr.tk2id
    assert '<en>' in char_tknzr.tk2id
    assert '<num>' in char_tknzr.tk2id
    # Ensure min_count constraint is activated.
    if char_tknzr.min_count >= 5:
        assert '<per0>' not in char_tknzr.tk2id
        assert '<loc0>' not in char_tknzr.tk2id
        assert '<org0>' not in char_tknzr.tk2id
    else:
        assert '<per0>' in char_tknzr.tk2id
        assert '<loc0>' in char_tknzr.tk2id
        assert '<org0>' in char_tknzr.tk2id
