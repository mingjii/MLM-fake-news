r""":term:`Tokenizer` base class."""

import json
import os
from typing import ClassVar, Dict, List, Optional, Sequence

from transformers import BertTokenizerFast

import util.path
from tokenizer.char import Tknzr_char


class pretrained_Tknzr(Tknzr_char):
    file_name: ClassVar[str] = 'tknzr.json'
    tknzr_name: ClassVar[str] = 'bert_base_chinese'

    def __init__(
        self,
        additional_tokens: List[str],
        **kwargs: Optional[Dict],
    ):
        super().__init__(
            is_uncased=True,
            max_vocab=0,
            min_count=0,
        )
        self.pretrained_tknzr = BertTokenizerFast.from_pretrained(
            'bert-base-chinese',
            additional_special_tokens=additional_tokens,
        )
        # Load pre-trained vocabulary.
        self.tk2id: Dict[str, int] = self.pretrained_tknzr.get_vocab()
        self.id2tk: Dict[int, str] = {v: k for k, v in self.tk2id.items()}
        self.additional_tokens = additional_tokens

        self.pad_tk = self.pretrained_tknzr.pad_token
        # print(f"pad:{self.pretrained_tknzr.pad_token}")
        self.pad_tkid = self.pretrained_tknzr.pad_token_id
        self.sep_tk = self.pretrained_tknzr.sep_token
        # print(f"SEP:{self.pretrained_tknzr.sep_token}")
        self.sep_tkid = self.pretrained_tknzr.sep_token_id
        self.cls_tk = self.pretrained_tknzr.cls_token
        self.cls_tkid = self.pretrained_tknzr.cls_token_id
        self.unk_tk = self.pretrained_tknzr.unk_token
        self.unk_tkid = self.pretrained_tknzr.unk_token_id
        self.mask_tk = self.pretrained_tknzr.mask_token
        self.mask_tkid = self.pretrained_tknzr.mask_token_id

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
                    'max_vocab': self.vocab_size(),
                    "tknzr_type": self.tknzr_name,
                    'additional_tokens': self.additional_tokens,
                    'tk2id': self.tk2id,
                },
                output_file,
                ensure_ascii=False
            )

    def tknz(self, txt: str) -> List[str]:
        norm_text = self.norm(txt)
        return self.pretrained_tknzr.tokenize(norm_text)

    def dtknz(self, tks: Sequence[str]) -> str:
        return self.norm(
            self.pretrained_tknzr.convert_tokens_to_string(tks)).replace(
            " ", "")

    def enc(self, txt: str, *, max_seq_len: Optional[int] = -1) -> List[int]:
        norm_text = self.norm(txt)
        # Shape: (1, max_seq_len)
        out = self.pretrained_tknzr.encode(
            norm_text,
            max_length=max_seq_len,
            padding='max_length',
            truncation=True,
        )

        # Shape: (max_seq_len)
        return out

    def dec(
            self,
            tkids: Sequence[int],
            *,
            rm_sp_tks: Optional[bool] = False,
    ) -> str:
        out = self.pretrained_tknzr.decode(
            tkids,
            skip_special_tokens=rm_sp_tks,
            clean_up_tokenization_spaces=True,
        )
        return out.replace(" ", "")

    def batch_enc(
            self,
            batch_txt: Sequence[str],
            *,
            max_seq_len: int = -1,
    ) -> List[List[int]]:
        batch_txt = [self.norm(x) for x in batch_txt]
        out = self.pretrained_tknzr(
            batch_txt,
            max_length=max_seq_len,
            padding='max_length',
            truncation=True,
        )
        return out["input_ids"]

    def batch_dec(
            self,
            batch_tkids: Sequence[Sequence[int]],
            *,
            rm_sp_tks: bool = False,
    ) -> List[str]:
        out = self.pretrained_tknzr.batch_decode(
            list(batch_tkids),
            skip_special_tokens=rm_sp_tks,
            clean_up_tokenization_spaces=True,
        )
        return [x.replace(" ", "") for x in out]

    def build_vocab() -> None:
        pass

    def vocab_size(self) -> int:
        return len(self.tk2id)


if __name__ == "__main__":
    from dataset import DSET_OPT
    dset = DSET_OPT["news"]("news_v2.2.db", n_sample=-1)
    tknzr = pretrained_Tknzr.load("pretknzr_news_v2.2")
    text = "紐西蘭外交部長<per0><en>上周在該國政府成立的「<org0>」發表演講時闡述了紐西蘭的對華政策"
    print(tknzr.vocab_size())
    print("origin text:")
    print(text)

    # tokensize
    tokens = tknzr.tknz(text)
    print("tokens:")
    print(tokens)

    # detokenize
    text = tknzr.dtknz(tokens)
    print("detokenize text:")
    print(text)

    text = "紐西蘭外交部長<per0><en>上周在該國政府成立的「<org0>」發表演講時闡述了紐西蘭的對華政策"
    # encode
    tkids = tknzr.enc(text, max_seq_len=50)
    print("token ids:")
    print(tkids)

    # decode
    text = tknzr.dec(tkids, rm_sp_tks=False)
    print("decode text:")
    print(text)

    batch_text = [dset[234], dset[23453], dset[939]]
    batch_text = [a + b for a, b in batch_text]
    # encode
    batch_tkids = tknzr.batch_enc(batch_text[0], batch_text[1], max_seq_len=50)
    print("batch_token ids:")
    print(batch_tkids)

    print(batch_text)
    # decode
    batch_text = tknzr.batch_dec(batch_text[0], batch_text[1], rm_sp_tks=False)
    print("decode batch_text:")
    print(batch_text)
'''
import pandas as pd
from s2s.tknzr import Tknzr
df = pd.read_pickle("data/sub_num_data.pkl")
t = Tknzr(True, -1, 1)
t.tknz(df["content"][10])
df["title"][10]
t.tknz(df["title"][10])
'''
