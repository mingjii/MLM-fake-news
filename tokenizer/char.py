r""":term:`Tokenizer` base class."""

import json
import os
import re
import typing
import unicodedata
from collections import Counter
from typing import ClassVar, Dict, List, Optional, Sequence
import util.path

# Pattern for all preprocessed special tokens.
pre_pattern = re.compile(r'<(en|num|unk|(loc|org|per)\d+)>')

# Pattern for model training special tokens.
model_pattern = re.compile(r'<(cls|sep|pad|unk|mask)>')


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


class Tknzr_char:
    file_name: ClassVar[str] = 'tknzr.json'
    tknzr_name: ClassVar[str] = 'char'
    cls_tk: ClassVar[str] = '<cls>'
    cls_tkid: ClassVar[int] = 0
    sep_tk: ClassVar[str] = '<sep>'
    sep_tkid: ClassVar[int] = 1
    pad_tk: ClassVar[str] = '<pad>'
    pad_tkid: ClassVar[int] = 2
    unk_tk: ClassVar[str] = '<unk>'
    unk_tkid: ClassVar[int] = 3
    msk_tk: ClassVar[str] = '<mask>'
    msk_tkid: ClassVar[int] = 4

    def __init__(
            self,
            is_uncased: bool,
            max_vocab: int,
            min_count: int,
            *,
            tk2id: Optional[Dict[str, int]] = None,
            **kwargs: Optional[Dict],
    ):
        if not isinstance(is_uncased, bool):
            raise TypeError('`is_uncased` must be an instance of `bool`.')

        self.is_uncased = is_uncased
        self.max_vocab = max_vocab
        self.min_count = min_count

        # Load pre-trained vocabulary.
        self.tk2id: Dict[str, int] = {}
        self.id2tk: Dict[int, str] = {}
        if tk2id is not None:
            self.tk2id = tk2id
            self.id2tk = {v: k for k, v in tk2id.items()}
        # Initialize vocabulary with special tokens.
        else:
            for tk, tkid in [
                (self.__class__.cls_tk, self.__class__.cls_tkid),
                (self.__class__.sep_tk, self.__class__.sep_tkid),
                (self.__class__.pad_tk, self.__class__.pad_tkid),
                (self.__class__.unk_tk, self.__class__.unk_tkid),
                (self.__class__.msk_tk, self.__class__.msk_tkid),
            ]:
                self.tk2id[tk] = tkid
                self.id2tk[tkid] = tk

    def save(self, exp_name: str) -> None:
        file_dir = os.path.join(util.path.EXP_PATH, exp_name)
        file_path = os.path.join(file_dir, self.__class__.file_name)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        elif not os.path.isdir(file_dir):
            raise FileExistsError(f'{file_dir} is not a directory.')

        elif os.path.isdir(file_path):
            raise FileExistsError(f'{file_path} is a directory.')

        with open(file_path, 'w', encoding='utf8') as output_file:
            json.dump(
                {
                    'is_uncased': self.is_uncased,
                    'max_vocab': self.max_vocab,
                    'min_count': self.min_count,
                    'tk2id': self.tk2id,
                },
                output_file,
                ensure_ascii=False
            )

    @classmethod
    def load(cls, exp_name: str):
        if not isinstance(exp_name, str):
            raise TypeError('`exp_name` must be an instance of `str`.')

        if not exp_name:
            raise ValueError('`exp_name` must be non-empty.')

        file_path = os.path.join(util.path.EXP_PATH, exp_name, cls.file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'Tokenizer file path {file_path} does not exist.'
            )

        if os.path.isdir(file_path):
            raise FileExistsError(
                f'Tokenizer file path {file_path} is a directory.'
            )

        with open(file_path, 'r', encoding='utf-8') as input_file:
            return cls(**json.load(input_file))

    def norm(self, txt: str) -> str:
        # NFKC normalization.
        txt = unicodedata.normalize('NFKC', txt)
        # Strip both end.
        txt = txt.strip()
        # Collapse multiple whitespace.
        txt = ' '.join(re.split(r'\s+', txt))
        # Case normalization.
        if self.is_uncased:
            txt = txt.lower()

        return txt

    def tknz(self, txt: str) -> List[str]:
        """Tokenize based on character and special tokens."""
        out = []
        txt = self.norm(txt)

        while txt:
            pre_match = pre_pattern.match(txt)
            model_match = model_pattern.match(txt)

            if pre_match:
                sp_tk = txt[pre_match.start():pre_match.end()]
                out.append(sp_tk)
                txt = txt[pre_match.end():]
            elif model_match:
                sp_tk = txt[model_match.start():model_match.end()]
                out.append(sp_tk)
                txt = txt[model_match.end():]
            else:
                out.append(txt[0])
                txt = txt[1:]

        return out

    def dtknz(self, tks: Sequence[str]) -> str:
        return ''.join(tks)

    def enc(
            self,
            txt: str,
            *,
            txt_pair: Optional[str] = '',
            max_seq_len: Optional[int] = -1) -> List[int]:
        # Prepend `[cls]` token id.
        tkids = [self.cls_tkid]

        # Convert tokens into token ids.
        for tk in self.tknz(txt):
            try:
                tkids.append(self.tk2id[tk])
            # Convert unknown tokens into `[unk]` token id.
            except KeyError:
                tkids.append(self.unk_tkid)

        # Append `[SEP]` token id.
        tkids.append(self.sep_tkid)

        if txt_pair != '':
            # Convert tokens into token ids.
            for tk in self.tknz(txt_pair):
                try:
                    tkids.append(self.tk2id[tk])
                # Convert unknown tokens into `[unk]` token id.
                except KeyError:
                    tkids.append(self.unk_tkid)

            # Append `[SEP]` token id.
            tkids.append(self.sep_tkid)

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

        tks = []
        # Convert token ids into tokens.
        for tkid in tkids:
            # Change <sep> to space.
            # Only trigger if rm_sp_tks == True.
            if rm_sp_tks and tkid == self.__class__.sep_tkid:
                tks.append(' ')
                continue

            tks.append(self.id2tk[tkid])

        return self.dtknz(tks)

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

    def build_vocab(
        self,
        batch_txt: Sequence[str],
    ) -> None:
        # Count each token's frequency.
        c: typing.Counter[str] = Counter()

        for txt in batch_txt:
            c.update(self.tknz(self.norm(txt)))

        max_id = max(self.tk2id.values()) + 1

        for tk, tk_count in c.most_common():
            # Stop adding tokens when pass vocabulary size limit.
            # If `self.max_vocab == -1`, then add as many tokens as possible.
            if self.max_vocab != -1 and max_id >= self.max_vocab:
                break

            # Stop adding the token when the token frequency is low.
            # Since we sort token by frequency, the rest of tokens will not
            # have frequency higher than `self.min_count` and thus we can
            # break loop savely.
            if tk_count < self.min_count:
                break

            # Skip the token if already exists.
            if tk in self.tk2id:
                continue

            # Add token to vocabulary.
            self.tk2id[tk] = max_id
            self.id2tk[max_id] = tk
            max_id += 1

    def vocab_size(self) -> int:
        return len(self.tk2id)
