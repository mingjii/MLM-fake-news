import os

from tokenizer.char import Tknzr_char


def test_save_load(exp_name, file_path, char_tknzr, batch_txt):

    char_tknzr.build_vocab(batch_txt)
    char_tknzr.save(exp_name=exp_name)

    # Ensure saved file exists.
    assert os.path.exists(file_path)

    # Load saved file.
    load_tknzr = Tknzr_char.load(exp_name=exp_name)

    # Ensure load and save consistency.
    assert load_tknzr.is_uncased == char_tknzr.is_uncased
    assert load_tknzr.min_count == char_tknzr.min_count
    assert load_tknzr.max_vocab == char_tknzr.max_vocab
    assert load_tknzr.tk2id == char_tknzr.tk2id
    assert load_tknzr.id2tk == char_tknzr.id2tk
