from typing import Dict, Optional
import math

import torch
import torch.nn as nn
from model._base import baseModel
import tokenizer.char


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        *,
        d_emb: int,
        p_hid: float,
        max_seq_len: int,
        **kwargs: Optional[Dict],
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=p_hid)

        # Create positional encoding table.
        # Shape : `(S, E)`
        self.pe = torch.zeros(max_seq_len, d_emb)

        # Position order from `0` to `S - 1`.
        # Shape: `(S, 1)`
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        # Compute the positional encodings once in log space.
        # Shape : `(1, S, E)`
        div_term = torch.exp(torch.arange(0, d_emb, 2) *
                             -(math.log(10000.0) / d_emb))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to each sequences.
        # Input shape : `(B, S, E)`
        # Output shape : `(B, S, E)`
        pe = self.pe.detach().to(src.device)
        output = src + pe[:, :src.size(1)]
        return self.dropout(output)


class TransformerModel(baseModel):
    model_name = "Transformer"

    def __init__(
            self,
            *,
            d_emb: int,
            d_hid: int,
            n_hid_lyr: int,
            max_seq_len: int,
            n_head: int = 8,
            p_emb: float,
            p_hid: float,
            tknzr: tokenizer.char.Tknzr_char,
            **kwargs: Optional[Dict],
    ):
        super().__init__(
            d_emb=d_emb,
            d_hid=d_hid,
            p_emb=p_emb,
            p_hid=p_hid,
            tknzr=tknzr,
        )

        self.pad_tkid = tknzr.pad_tkid
        self.pe = PositionalEncoding(
            d_emb=d_hid,
            p_hid=p_hid,
            max_seq_len=max_seq_len
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_hid,
            nhead=n_head,
            dropout=p_hid,
        )

        norm = nn.LayerNorm(d_hid)

        self.hid = nn.TransformerEncoder(
            layer,
            num_layers=n_hid_lyr
        )

    def forward(
            self,
            batch_msk_seq: torch.Tensor,
    ) -> torch.Tensor:
        # Embeddings.
        # in shape : (B, src_seq_len)
        # out shape: (B, src_seq_len, d_emb)
        x = self.emb_lyr(batch_msk_seq)

        # Embedding to hidden layers.
        # in shape : (B, src_seq_len, d_emb)
        # out shape: (B, src_seq_len, d_enc_hid)
        x = self.pe(self.emb_to_hid(x))

        # Create mask
        # reg_mask Shape: (seq_len, seq_len)
        # pad_mask Shape: (B, seq_len)
        reg_mask, pad_mask = self.create_mask(batch_msk_seq)

        # Hidden model with Transformer
        # out shape: (B, tgt_seq_len, d_dec_hid)
        x = x.transpose(0, 1)
        out = self.hid(
            src=x,
            mask=reg_mask,
            src_key_padding_mask=pad_mask,
        )

        out = out.transpose(0, 1).contiguous()

        # Hidden to embedding layers.
        # in shape : (B, tgt_seq_len, d_dec_hid)
        # out shape: (B, tgt_seq_len, d_emb)
        out = self.hid_to_emb(out)

        # Prediction.
        # in shape : (B, tgt_seq_len, d_dec_hid)
        # out shape: (B, tgt_seq_len, vocab_size)
        return out @ self.emb_lyr.weight.transpose(0, 1)

    def create_mask(self, x: torch.Tensor):
        seq_len = x.size(-1)
        # Create auto-regressive self attention masks.
        # Need to move tensor to model running device.
        # Output shape: `(S, S)`.
        # Output dtype: `torch.bool`.
        reg_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        reg_mask = torch.triu(reg_mask, diagonal=1)
        reg_mask = reg_mask.to(x.device)

        # Create padding self attention masks.
        # Output shape: `(B, S)`.
        # Output dtype: `torch.bool`.
        pad_mask = x == self.pad_tkid
        pad_mask = pad_mask.to(x.device)

        return reg_mask, pad_mask
