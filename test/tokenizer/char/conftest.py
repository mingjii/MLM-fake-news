import os

import pytest

import util
from tokenizer.char import Tknzr_char


@pytest.fixture(
    params=[
        Tknzr_char(is_uncased=True, max_vocab=100, min_count=1),
        Tknzr_char(is_uncased=False, max_vocab=10, min_count=5),
    ],
)
def char_tknzr(request):
    return request.param


@pytest.fixture
def batch_txt():
    return [
        '你好嗎',
        '你好嗎',
        '你好嗎',
        '你好嗎',
        '你好嗎',
        '<en><unk><num>',
        '<en><unk><num>',
        '<en><unk><num>',
        '<en><unk><num>',
        '<en><unk><num>',
        '<per0><loc0><org0>',
        '<per0><loc0><org0>',
        '<per0><loc0><org0>',
        '<per0><loc0><org0>',
        '<cls><sep><mask><mask><mask><pad><pad>',
        '<cls><sep><mask><mask><mask><pad><pad>',
        '<cls><sep><mask><mask><mask><pad><pad>',
        '<cls><sep><mask><mask><mask><pad><pad>',
        '<cls><sep><mask><mask><mask><pad><pad>',
    ]


@pytest.fixture
def file_path(request, exp_name: str) -> str:
    r"""Tokenizer configuration output file path.
    After testing, clean up files and directories create during test.
    """
    abs_dir_path = os.path.join(util.path.EXP_PATH, exp_name)
    abs_file_path = os.path.join(abs_dir_path, Tknzr_char.file_name)

    def fin():
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)
        if os.path.exists(abs_dir_path):
            os.removedirs(abs_dir_path)

    request.addfinalizer(fin)
    return abs_file_path


@pytest.fixture(params=[
    # Number only.
    {
        'txt': '123',
        'tks': ['1', '2', '3', ],
    },
    # English only.
    {
        'txt': 'abc',
        'tks': ['a', 'b', 'c', ],
    },
    # Chinese only.
    {
        'txt': '你好嗎',
        'tks': ['你', '好', '嗎'],
    },
    # Number + English + Chinese.
    {
        'txt': '123abc你好嗎',
        'tks': ['1', '2', '3', 'a', 'b', 'c', '你', '好', '嗎'],
    },
    # Preprocessed special tokens.
    {
        'txt': '<en><unk><num><per0><loc1><org234>',
        'tks': ['<en>', '<unk>', '<num>', '<per0>', '<loc1>', '<org234>'],
    },
    # Model special tokens.
    {
        'txt': '<cls><sep><mask><mask><mask><pad><pad>',
        'tks': [
            '<cls>', '<sep>', '<mask>', '<mask>', '<mask>', '<pad>', '<pad>'
        ],
    },
    # Ensure token consistency.
    {
        'txt': '<en1><unk2><num3><per><loc><org><cls4><sep5><mask6><pad7>',
        'tks': list(
            '<en1><unk2><num3><per><loc><org><cls4><sep5><mask6><pad7>'
        ),
    },
    {
        'txt': ''.join([
            '<cls>123<num>abc<en><mask><unk><mask><sep>你好嗎<per0>?<loc1>',
            '<org2><sep><pad><pad>'
        ]),
        'tks': [
            '<cls>', '1', '2', '3', '<num>', 'a', 'b', 'c', '<en>', '<mask>',
            '<unk>', '<mask>', '<sep>', '你', '好', '嗎', '<per0>', '?',
            '<loc1>', '<org2>', '<sep>', '<pad>', '<pad>'
        ],
    },
])
def txt_tks_pair(request):
    return request.param


@pytest.fixture(params=[
    # Ensure consistency between encode and decode.
    {
        'txt': '你好嗎',
        'txt_pair': '',
        'max_seq_len': -1,
        'rm_sp_tks': False,
        'enc': [0, 5, 6, 7, 1],
        'dec': '<cls>你好嗎<sep>',
    },
    {
        'txt': '你好嗎',
        'txt_pair': '好',
        'max_seq_len': 10,
        'rm_sp_tks': True,
        'enc': [0, 6, 5, 7, 1, 5, 1, 2, 2, 2],
        'dec': '你好嗎 好 ',
    },
    # Truncate to max_seq_len.
    {
        'txt': '<en><num><unk><unk>',
        'txt_pair': '<mask><mask><mask><mask>',
        'max_seq_len': 8,
        'rm_sp_tks': False,
        'enc': [0, 5, 6, 3, 3, 1, 4, 4],
        'dec': '<cls><en><num><unk><unk><sep><mask><mask>',
    },
    # Pad to max_seq_len.
    {
        'txt': '<en>',
        'txt_pair': '<org0>',
        'max_seq_len': 10,
        'rm_sp_tks': False,
        'enc': [0, 5, 1, 6, 1, 2, 2, 2, 2, 2],
        'dec': '<cls><en><sep><org0><sep><pad><pad><pad><pad><pad>',
    },
])
def txt_enc_dec_pair(request):
    return request.param


@pytest.fixture(params=[
    # Ensure consistency between encode and decode.
    # Pad to max_seq_len.
    {
        'txt': [
            '你好嗎',
            '<per0><loc1><org2><en><num>',
        ],
        'txt_pair': [
            '我很好',
            '假新聞',
        ],
        'max_seq_len': -1,
        'rm_sp_tks': False,
        'enc': [
            [0, 6, 5, 7, 1, 13, 14, 5, 1, 2, 2],
            [0, 8, 9, 10, 11, 12, 1, 15, 16, 17, 1],
        ],
        'dec': [
            '<cls>你好嗎<sep>我很好<sep><pad><pad>',
            '<cls><per0><loc1><org2><en><num><sep>假新聞<sep>',
        ],
    },
    # Truncate to max_seq_len.
    {
        'txt': [
            '你好嗎',
            '<per0><loc1><org2><en><num>',
        ],
        'txt_pair': [
            '我很好',
            '假新聞',
        ],
        'max_seq_len': 5,
        'rm_sp_tks': False,
        'enc': [
            [0, 6, 5, 7, 1],
            [0, 8, 9, 10, 11],
        ],
        'dec': [
            '<cls>你好嗎<sep>',
            '<cls><per0><loc1><org2><en>',
        ],
    },
])
def batch_txt_enc_dec_pair(request):
    return request.param
