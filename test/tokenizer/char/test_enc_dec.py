from tokenizer.char import Tknzr_char


def test_1_input(txt_enc_dec_pair):

    txt = txt_enc_dec_pair['txt']
    txt_pair = txt_enc_dec_pair['txt_pair']
    max_seq_len = txt_enc_dec_pair['max_seq_len']
    rm_sp_tks = txt_enc_dec_pair['rm_sp_tks']
    enc = txt_enc_dec_pair['enc']
    dec = txt_enc_dec_pair['dec']

    char_tknzr = Tknzr_char(is_uncased=True, max_vocab=100, min_count=1)
    char_tknzr.build_vocab([txt, txt_pair])

    enc_out = char_tknzr.enc(
        txt=txt,
        txt_pair=txt_pair,
        max_seq_len=max_seq_len
    )

    assert enc_out == enc
    assert char_tknzr.dec(enc_out, rm_sp_tks=rm_sp_tks) == dec


def test_batch_input(batch_txt_enc_dec_pair):

    batch_txt = batch_txt_enc_dec_pair['txt']
    batch_txt_pair = batch_txt_enc_dec_pair['txt_pair']
    max_seq_len = batch_txt_enc_dec_pair['max_seq_len']
    rm_sp_tks = batch_txt_enc_dec_pair['rm_sp_tks']
    batch_enc = batch_txt_enc_dec_pair['enc']
    batch_dec = batch_txt_enc_dec_pair['dec']

    char_tknzr = Tknzr_char(is_uncased=True, max_vocab=100, min_count=1)
    char_tknzr.build_vocab(batch_txt + batch_txt_pair)

    batch_enc_out = char_tknzr.batch_enc(
        batch_txt=batch_txt,
        batch_txt_pair=batch_txt_pair,
        max_seq_len=max_seq_len
    )

    assert batch_enc_out == batch_enc
    assert char_tknzr.batch_dec(
        batch_enc_out, rm_sp_tks=rm_sp_tks) == batch_dec
