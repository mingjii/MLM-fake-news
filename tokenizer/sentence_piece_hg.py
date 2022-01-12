r""":term:`Tokenizer` base class."""
import json
import os
import re
import typing
import unicodedata
from tokenizers import Tokenizer, AddedToken, decoders, trainers, normalizers, pre_tokenizers, Regex
from tokenizers.models import Unigram
from tokenizers.processors import BertProcessing
from collections import Counter
from typing import ClassVar, Dict, List, Optional, Sequence
import util.path
import util.cfg



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
        exp_name: str,
        **kwargs: Optional[Dict],
    ):
        tknzr_path = os.path.join(util.path.EXP_PATH, exp_name, 'tknzr.json')
        try:
            print(tknzr_path)
            self.processor = Tokenizer.from_file(tknzr_path)
        except: 
            Exception(
            "Wrong filename! Or the tokenizer hasn't trained yet!"
        )


    def tknz(self, txt: str) -> List[str]:
        out = self.processor.encode(txt)
        return out.tokens

    def enc(
        self,
        txt: str,
        *,
        txt_pair: Optional[str] = '',
        max_seq_len: Optional[int] = -1
    ) -> List[int]:
        if max_seq_len != -1:
            self.processor.enable_truncation(max_length=max_seq_len)
            self.processor.enable_padding(
                length=max_seq_len,
                pad_id=self.__class__.pad_tkid,
                pad_token=self.__class__.pad_tk,
            )
        if txt_pair:
            out = self.processor.encode(txt, txt_pair)
        else:
            out = self.processor.encode(txt)
        return out.ids

    def dec(
            self,
            tkids: Sequence[int],
            *,
            rm_sp_tks: Optional[bool] = False,
    ) -> str:

        out = self.processor.decode(tkids, skip_special_tokens=False)
        if rm_sp_tks:
            out.replace(self.__class__.cls_tk, "")
            out.replace(self.__class__.pad_tk, "")
            out.replace(self.__class__.sep_tk, " ")
        return out

    def batch_enc(
            self,
            batch_paired_txt: Sequence[str],
            *,
            max_seq_len: int = -1,
    ) -> List[List[int]]:
        if max_seq_len != -1:
            self.processor.enable_truncation(max_length=max_seq_len)
            self.processor.enable_padding(
                length=max_seq_len,
                pad_id=self.__class__.pad_tkid,
                pad_token=self.__class__.pad_tk,
            )
        out = self.processor.encode_batch(batch_paired_txt)
        return [x.ids for x in out]

    def batch_dec(
            self,
            batch_tkids: Sequence[Sequence[int]],
            *,
            rm_sp_tks: bool = False,
    ) -> List[str]:

        # Decode each sequence of token ids in the batch.
        return [self.dec(tkids, rm_sp_tks=rm_sp_tks) for tkids in batch_tkids]

    @classmethod
    def train(
        cls,
        exp_name: str,
        files: List[str],
        vocab_size: int,
        add_prefix_space: bool = False,
        replacement: str = "_",
    ) -> None:
        processor = Tokenizer(Unigram())
        processor.normalizer = normalizers.Sequence(
            [normalizers.Nmt(), normalizers.NFKC(), 
            normalizers.Replace(Regex(" {2,}"), " ")]
        )
        processor.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )
        processor.post_processor = BertProcessing(
            (str(cls.sep_tk), cls.sep_tkid),
            (str(cls.cls_tk), cls.cls_tkid)
        )
        processor.decoder = decoders.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )

        special_tokens = [
            cls.cls_tk,
            cls.sep_tk,
            cls.pad_tk,
            cls.unk_tk,
            cls.mask_tk,
            '<en>', '<num>', '，', ',', '。', '：',
            ':', '；', ';', '！', '!', '？', '?'
        ]
        # add_tokens = ['<en>', '<num>', '，', ',', '。', '：',
        #                     ':', '；', ';', '！', '!', '？', '?']
        for i in range(20):
            special_tokens.append(f'<per{i}>')
            special_tokens.append(f'<org{i}>')
            special_tokens.append(f'<loc{i}>')

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=cls.unk_tk,
            max_piece_length=10,
        )
        processor.train(files=files, trainer=trainer)
        # processor.add_tokens(add_tokens)
        save_filename = os.path.join(util.path.EXP_PATH, exp_name, 'tknzr.json')
        processor.save(save_filename,pretty=True)

    @property
    def vocab_size(self) -> int:
        return self.processor.get_vocab_size()
