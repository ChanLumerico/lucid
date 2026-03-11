from dataclasses import dataclass
import math

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

__all__ = ["Transformer", "TransformerConfig", "transformer_base", "transformer_big"]


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float = 0.1
    max_len: int = 5000

    def __post_init__(self) -> None:
        if self.src_vocab_size <= 0:
            raise ValueError("src_vocab_size must be greater than 0")
        if self.tgt_vocab_size <= 0:
            raise ValueError("tgt_vocab_size must be greater than 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be greater than 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be greater than 0")
        if self.num_decoder_layers <= 0:
            raise ValueError("num_decoder_layers must be greater than 0")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be greater than 0")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.max_len <= 0:
            raise ValueError("max_len must be greater than 0")


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
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.src_tok_emb = nn.Embedding(config.src_vocab_size, config.d_model)
        self.tgt_tok_emb = nn.Embedding(config.tgt_vocab_size, config.d_model)

        self.positional_encoding = _PositionalEncoding(
            config.d_model, config.dropout, config.max_len
        )

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        self.fc_out = nn.Linear(config.d_model, config.tgt_vocab_size)

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
        kv_cache: nn.KVCache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        cache_start_layer_idx: int = 0,
        encoder_kv_cache: nn.KVCache | None = None,
        use_encoder_cache: bool = False,
        encoder_cache_position: Tensor | None = None,
        encoder_cache_start_layer_idx: int = 0,
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
            kv_cache=kv_cache,
            use_cache=use_cache,
            cache_position=cache_position,
            cache_start_layer_idx=cache_start_layer_idx,
            encoder_kv_cache=encoder_kv_cache,
            use_encoder_cache=use_encoder_cache,
            encoder_cache_position=encoder_cache_position,
            encoder_cache_start_layer_idx=encoder_cache_start_layer_idx,
        )
        output = self.fc_out(output)

        return output


@register_model
def transformer_base(
    src_vocab_size: int = 12000, tgt_vocab_size: int = 12000, **kwargs
) -> Transformer:
    config_kwargs = {
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
    }
    config_kwargs.update(kwargs)
    return Transformer(TransformerConfig(**config_kwargs))


@register_model
def transformer_big(
    src_vocab_size: int = 12000, tgt_vocab_size: int = 12000, **kwargs
) -> Transformer:
    config_kwargs = {
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "d_model": 1024,
        "num_heads": 16,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 4096,
        "dropout": 0.3,
    }
    config_kwargs.update(kwargs)
    return Transformer(TransformerConfig(**config_kwargs))
