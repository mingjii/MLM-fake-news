r""":term:`Tokenizer` base class."""
import json
import os
import re
import typing
import unicodedata
import sentencepiece as spm
from collections import Counter
from typing import ClassVar, Dict, List, Optional, Sequence
import util.path
import util.cfg


def trunc_to_max(seq, *, max_seq_len=-1):
    if max_seq_len == -1:
        return seq
    # Truncate sequence to maximum sequence length.
    return seq[:max_seq_len]


def pad_to_max(seq, pad, *, max_seq_len=-1):
    if max_seq_len == -1:
        return seq

    # Calculate padding length.
    pad_len = max(0, max_seq_len - len(seq))

    # Pad to maximum sequence length.
    return seq + [pad] * pad_len


class Tknzr_sentPiece:
    file_name: ClassVar[str] = 'tknzr'
    tknzr_name: ClassVar[str] = 'sentencePiece'
    cls_tk: ClassVar[str] = '[cls]'
    cls_tkid: ClassVar[int] = 0
    sep_tk: ClassVar[str] = '[sep]'
    sep_tkid: ClassVar[int] = 1
    pad_tk: ClassVar[str] = '[pad]'
    pad_tkid: ClassVar[int] = 2
    unk_tk: ClassVar[str] = '[unk]'
    unk_tkid: ClassVar[int] = 3
    mask_tk: ClassVar[str] = '<mask>'
    mask_tkid: ClassVar[int] = 4

    def __init__(
            self,
            *,
            is_uncased: bool,
            vocab_size: int,
            char_coverage: float=0.9995,
            exp_name: Optional[str] = None,
            **kwargs: Optional[Dict],
    ):
        if not isinstance(is_uncased, bool):
            raise TypeError('`is_uncased` must be an instance of `bool`.')

        self.is_uncased = is_uncased
        self.num_vocab = vocab_size
        self.char_coverage = char_coverage
        self.user_defined_unk_tk = '<unk>'
        self.whitespace_tk = '▁'
        self.user_defined_unk_tkid = None
        self.whitespace_tkid = None
        self.user_defined_symbols = [
            self.mask_tk,
            self.user_defined_unk_tk,
            '<en>',
            '<num>'
        ]
        for i in range(20):
            self.user_defined_symbols.append(f'<per{i}>')
            self.user_defined_symbols.append(f'<org{i}>')
            self.user_defined_symbols.append(f'<loc{i}>')

        model_path = os.path.join(util.path.EXP_PATH, exp_name)
        model_file = os.path.join(model_path, self.file_name+'.model')
        if os.path.exists(model_file):
            if os.path.isdir(model_file):
                raise FileExistsError(
                    f'Tokenizer file path {model_file} is a directory.'
                )
            self.processor = spm.SentencePieceProcessor()
            self.processor.load(model_file)
            self.user_defined_unk_tkid = self.processor.piece_to_id(
                self.user_defined_unk_tk)
            self.whitespace_tkid = self.processor.piece_to_id(
                self.whitespace_tk)

    @classmethod
    def load(cls, exp_name: str):
        if not isinstance(exp_name, str):
            raise TypeError('`exp_name` must be an instance of `str`.')

        if not exp_name:
            raise ValueError('`exp_name` must be non-empty.')

        cfg_path = os.path.join(util.path.EXP_PATH, exp_name)
        cfg_file = os.path.join(cfg_path, 'cfg.json')

        if not os.path.exists(cfg_file):
            raise FileNotFoundError(
                f'Tokenizer file path {cfg_file} does not exist.'
            )

        if os.path.isdir(cfg_file):
            raise FileExistsError(
                f'Tokenizer file path {cfg_file} is a directory.'
            )

        with open(cfg_file, 'r', encoding='utf-8') as input_file:
            cfg = json.load(input_file)
            return cls(**cfg)

    def norm(self, txt: str) -> str:
        # NFKC normalization.
        # txt = unicodedata.normalize('NFKC', txt)
        # Strip both end.
        txt = txt.strip()
        # Collapse multiple whitespace.
        txt = ' '.join(re.split(r'\s+', txt))
        # Case normalization.
        if self.is_uncased:
            txt = txt.lower()

        return txt

    def tknz(self, txt: str) -> List[str]:
        txt = self.norm(txt)

        return self.processor.encode_as_pieces(txt)

    def dtknz(self, tks: Sequence[str]) -> str:
        return self.processor.decode_pieces(tks).strip()

    def enc(
            self,
            txt: str,
            *,
            txt_pair: Optional[str] = '',
            max_seq_len: Optional[int] = -1) -> List[int]:
        # Prepend `[cls]` token id.
        tkids = [self.cls_tkid]

        # Convert tokens into token ids.
        temp = self.processor.encode_as_ids(self.norm(txt))
        if len(temp) > 0 and temp[0] == self.whitespace_tkid:
            temp = temp[1:]
        tkids += temp

        # Append `[SEP]` token id.
        tkids.append(self.sep_tkid)

        if txt_pair != '':
            # Convert tokens into token ids.
            temp = self.processor.encode_as_ids(self.norm(txt_pair))
            if len(temp) > 0 and temp[0] == self.whitespace_tkid:
                temp = temp[1:]
            tkids += temp
            # Append `[SEP]` token id.
            tkids.append(self.sep_tkid)

        # Replace user_defined unkown token(<unk>) to default unkown token([unk])
        tkids = [
            self.unk_tkid if x == self.user_defined_unk_tkid else x for x in tkids]

        # First truncate sequence to maximum sequence length, then pad sequence
        # to maximum sequence length.
        return trunc_to_max(
            pad_to_max(
                tkids,
                self.pad_tkid,
                max_seq_len=max_seq_len,
            ),
            max_seq_len=max_seq_len
        )

    def dec(
            self,
            tkids: Sequence[int],
            *,
            rm_sp_tks: Optional[bool] = False,
    ) -> str:
        # Remove special token ids.
        if rm_sp_tks:
            sp_tkids = [
                self.__class__.cls_tkid,
                self.__class__.pad_tkid,
            ]
            # Filter out <cls> and <pad>.
            tkids = list(filter(lambda tkid: tkid not in sp_tkids, tkids))
            # Replace [sep] to whitespace
            tkids = [self.whitespace_tkid if x ==
                     self.sep_tkid else x for x in tkids]

        # text = self.processor.decode_ids(tkids)
        text = ''.join([self.processor.id_to_piece(x) if x !=
                       self.whitespace_tkid else ' ' for x in tkids])

        return text.strip()

    def batch_enc(
            self,
            batch_txt: Sequence[str],
            *,
            batch_txt_pair: Optional[Sequence[str]] = None,
            max_seq_len: Optional[int] = -1,
    ) -> List[List[int]]:
        if batch_txt_pair is None:
            batch_txt_pair = [''] * len(batch_txt)

        batch_tkids = [
            self.enc(txt=txt, txt_pair=txt_pair, max_seq_len=-1)
            for txt, txt_pair in zip(batch_txt, batch_txt_pair)
        ]

        # If `max_seq_len == -1`, then `max_seq_len` is the longest sequence
        # length in the batch.
        if max_seq_len == -1:
            max_seq_len = max(map(len, batch_tkids))

        # Truncate each token ids sequence in batch to maximum sequence length.
        batch_tkids = [
            trunc_to_max(tkids, max_seq_len=max_seq_len)
            for tkids in batch_tkids
        ]

        # Pad each token ids sequence in batch to maximum sequence length.
        return [
            pad_to_max(
                tkids,
                self.__class__.pad_tkid,
                max_seq_len=max_seq_len
            )
            for tkids in batch_tkids
        ]

    def batch_dec(
            self,
            batch_tkids: Sequence[Sequence[int]],
            *,
            rm_sp_tks: bool = False,
    ) -> List[str]:
        # Decode each sequence of token ids in the batch.
        return [self.dec(tkids, rm_sp_tks=rm_sp_tks) for tkids in batch_tkids]

    def train(
        self,
        data_file: str,
        exp_name: str,
    ) -> None:
        data_file = os.path.join(util.path.DATA_PATH, data_file)
        model_name = os.path.join(util.path.EXP_PATH, exp_name, self.file_name)

        spm.SentencePieceTrainer.train(
            f"--input={data_file} \
            --model_prefix={model_name} \
            --vocab_size={self.num_vocab} \
            --character_coverage={self.char_coverage} \
            --user_defined_symbols={','.join(self.user_defined_symbols)} \
            --bos_id={self.cls_tkid} \
            --eos_id={self.sep_tkid} \
            --pad_id={self.pad_tkid} \
            --unk_id={self.unk_tkid} \
            --bos_piece={self.cls_tk} \
            --eos_piece={self.sep_tk} \
            --pad_piece={self.pad_tk} \
            --unk_piece={self.unk_tk} \
            --unk_surface={self.unk_tk}")
        self.processor = spm.SentencePieceProcessor()
        self.processor.load(model_name+'.model')
        self.user_defined_unk_tkid = self.processor.piece_to_id(
            self.user_defined_unk_tk)
        self.whitespace_tkid = self.processor.piece_to_id(
            self.whitespace_tk)

    def vocab_size(self) -> int:
        return self.processor.get_piece_size()
