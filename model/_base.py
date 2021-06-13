import os
import re
import abc
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.path
import tokenizer.char


class baseModel(nn.Module):
    def __init__(
            self,
            *,
            d_emb: int,
            d_hid: int,
            p_emb: float,
            p_hid: float,
            tknzr: tokenizer.char.Tknzr_char,
            **kwargs: Optional[Dict],
    ):
        super().__init__()
        self.msk_id = tknzr.msk_tkid

        self.emb_lyr = nn.Embedding(
            num_embeddings=tknzr.vocab_size(),
            embedding_dim=d_emb,
            padding_idx=tknzr.pad_tkid,
        )

        self.emb_to_hid = nn.Sequential(
            nn.Dropout(p=p_emb),
            nn.Linear(
                in_features=d_emb,
                out_features=d_hid,
            ),
            nn.ReLU(),
            nn.Dropout(p=p_hid),
        )

        self.hid_to_emb = nn.Sequential(
            nn.Dropout(p=p_hid),
            nn.Linear(
                in_features=d_hid,
                out_features=d_emb,
            ),
        )

    @staticmethod
    @abc.abstractmethod
    def forward(
            self,
            batch_msk_seq: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_fn(
            self,
            batch_msk_seq: torch.Tensor,
            batch_seq: torch.Tensor,
    ) -> torch.Tensor:
        msk_mask = self.get_mask(batch_msk_seq)

        logits = self(
            batch_msk_seq=batch_msk_seq,
        )

        logits = logits[msk_mask]

        batch_seq = batch_seq[msk_mask]

        # Cross entropy loss.
        # out shape: (1)
        return F.cross_entropy(logits, batch_seq)

    def pred(
            self,
            batch_msk_seq: torch.Tensor,
    ) -> torch.Tensor:
        # batch_src shape : (B, src_seq_len)
        # batch_tgt shape : (B, tgt_seq_len)
        # out shape: (B, tgt_seq_len, vocab_size)
        logits = self(
            batch_msk_seq=batch_msk_seq,
        )

        # Convert logits to probability.
        # in shape : (B, tgt_seq_len, vocab_size)
        # out shape: (B, tgt_seq_len, vocab_size)
        return F.softmax(logits, dim=-1)

    def get_mask(self, x: torch.Tensor):
        seq_len = x.size(-1)
        # Create [MASK] masks.
        # Output shape: `(B, S)`.
        # Output dtype: `torch.bool`.
        msk_mask = x == self.msk_id
        msk_mask = msk_mask.to(x.device)

        return msk_mask

    def save(self, ckpt: int, exp_name: str) -> None:
        file_dir = os.path.join(util.path.EXP_PATH, exp_name)
        file_path = os.path.join(file_dir, f'model-{ckpt}.pt')

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        elif not os.path.isdir(file_dir):
            raise FileExistsError(f'{file_dir} is not a directory.')

        elif os.path.isdir(file_path):
            raise FileExistsError(f'{file_path} is a directory.')

        # Save model parameters in zip compressed pickle.
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load(cls, ckpt: int, exp_name: str, **kwargs: Optional[Dict]):
        if not isinstance(ckpt, int):
            raise TypeError('`ckpt` must be an instance of `int`.')
        if not isinstance(exp_name, str):
            raise TypeError('`exp_name` must be an instance of `str`.')

        if ckpt < -1:
            raise ValueError('`ckpt` must satisfy `ckpt >= -1`.')
        if not exp_name:
            raise ValueError('`exp_name` must be non-empty.')

        file_dir = os.path.join(util.path.EXP_PATH, exp_name)
        if not os.path.exists(file_dir):
            raise FileNotFoundError(
                f'Experiment file path {file_dir} does not exist.'
            )

        # Load latest checkpoint.
        if ckpt == -1:
            ckpt_files = []
            for ckpt_f in os.listdir(file_dir):
                match = re.match(r'model-(\d+).pt', ckpt_f)
                if match is None:
                    continue
                ckpt_files.append(int(match.group(1)))
            ckpt = max(ckpt_files)

        # Format file name with checkpoint step.
        file_path = os.path.join(file_dir, f'model-{ckpt}.pt')

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'Checkpoint file path {file_path} does not exist.'
            )

        if os.path.isdir(file_path):
            raise FileExistsError(
                f'Checkpoint file path {file_path} is a directory.'
            )

        # Construct new model.
        self = cls(**kwargs)

        # Load pre-trained parameters.
        self.load_state_dict(torch.load(file_path))

        return self
