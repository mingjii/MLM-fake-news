from typing import Dict, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import tokenizer.char
import os
import re
import util.path


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        *,
        d_hid: int,
        p_hid: float,
        max_seq_len: int,
        **kwargs: Optional[Dict],
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=p_hid)

        # Create positional encoding table.
        # Shape : `(S, H)`
        self.pe = torch.zeros(max_seq_len, d_hid)

        # Position order from `0` to `S - 1`.
        # Shape: `(S, H)`
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        # Compute the positional encodings once in log space.
        # Shape : `(1, S, H)`
        div_term = torch.exp(torch.arange(0, d_hid, 2) *
                             -(math.log(10000.0) / d_hid))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to each sequences.
        # Input shape : `(B, S, H)`
        # Output shape : `(B, S, H)`
        pe = self.pe.detach().to(src.device)
        output = src + pe[:, :src.size(1)]
        return self.dropout(output)


class TransformerModel(nn.Module):
    model_name = "transformer"

    def __init__(
            self,
            *,
            d_ff: int,
            d_hid: int,
            n_hid_lyr: int,
            max_seq_len: int,
            n_head: int = 8,
            p_hid: float,
            tknzr: tokenizer.char.Tknzr_char,
            **kwargs: Optional[Dict],
    ):
        super().__init__()

        self.pad_tkid = tknzr.pad_tkid
        self.pe = PositionalEncoding(
            d_hid=d_hid,
            p_hid=p_hid,
            max_seq_len=max_seq_len
        )

        self.emb = nn.Embedding(
            num_embeddings=tknzr.vocab_size(),
            embedding_dim=d_hid,
            padding_idx=tknzr.pad_tkid,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_hid,
            dim_feedforward=d_ff,
            dropout=p_hid,
            nhead=n_head,
        )

        self.hid = nn.TransformerEncoder(
            layer,
            num_layers=n_hid_lyr
        )

    def forward(
            self,
            batch_mask_tkids: torch.Tensor,
    ) -> torch.Tensor:
        # Vocab embeddings.
        # in shape : (B, S)
        # out shape: (B, S, d_hid)
        x = self.emb(batch_mask_tkids)

        # Pos embeddings.
        # in shape : (B, S, d_hid)
        # out shape: (B, S, d_hid)
        x = self.pe(x)

        # Create attention mask.
        # pad_mask Shape: (B, S)
        pad_mask = self.create_attn_mask(batch_mask_tkids)

        # Hidden model with Transformer
        # out shape: (B, S, H)
        out = self.hid(
            src=x.transpose(0, 1),
            src_key_padding_mask=pad_mask,
        ).transpose(0, 1).contiguous()

        # Prediction.
        # in shape : (B, S, H)
        # out shape: (B, S, V)
        return out @ self.emb.weight.transpose(0, 1)

    def create_attn_mask(self, x: torch.Tensor):
        # Create padding self attention masks.
        # Output shape: `(B, S)`.
        # Output dtype: `torch.bool`.
        pad_mask = x == self.pad_tkid
        pad_mask = pad_mask.to(x.device)

        return pad_mask

    def loss_fn(
            self,
            batch_mask_tkids: torch.Tensor,
            batch_target_tkids: torch.Tensor,
            batch_is_mask: torch.Tensor
    ) -> torch.Tensor:
        # (B, S, V)
        logits = self(batch_mask_tkids=batch_mask_tkids)

        # Cross entropy loss.
        # Only calculate loss from masked position.
        batch_loss = F.cross_entropy(
            # (B, S, V) -> (BxS, V)
            logits.view(-1, self.emb.num_embeddings),
            # (B, S) -> (BxS)
            batch_target_tkids.view(-1),
            ignore_index=self.emb.padding_idx,
            reduction='none',
        )
        # (B, S) -> (BxS)
        batch_loss = batch_loss * batch_is_mask.view(-1)

        # Average loss only for masked position.
        # (BxS) -> (1)
        return batch_loss.sum() / batch_is_mask.sum()

    def pred(
            self,
            batch_mask_tkids: torch.Tensor,
    ) -> torch.Tensor:
        # batch_mask_tkids shape : (B, S)
        # out shape: (B, S, V)
        logits = self(batch_mask_tkids=batch_mask_tkids)

        # Convert logits to probability.
        # in shape : (B, S, V)
        # out shape: (B, S, V)
        return F.softmax(logits, dim=-1)

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
