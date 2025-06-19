import math

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["Transformer", "transformer_base", "transformer_big"]


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = lucid.zeros(max_len, d_model)
        position = lucid.arange(0, max_len).unsqueeze(axis=1)
        div_term = lucid.exp(lucid.arange(0, d_model, 2) * (-math.log(1e4) / d_model))

        pe[:, 0::2] = lucid.sin(position * div_term)
        pe[:, 1::2] = lucid.cos(position * div_term)

        pe = pe.unsqueeze(axis=1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x += self.pe[: x.shape[0]]
        x = self.dropout(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = _PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        mem_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        mem_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)

        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            mem_mask=mem_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            mem_key_padding_mask=mem_key_padding_mask,
        )
        output = self.fc_out(output)

        return output


@register_model
def transformer_base(
    src_vocab_size: int = 12000, tgt_vocab_size: int = 12000
) -> Transformer:
    return Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )


@register_model
def transformer_big(
    src_vocab_size: int = 12000, tgt_vocab_size: int = 12000
) -> Transformer:
    return Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model=1024,
        num_heads=16,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=4096,
        dropout=0.3,
    )
