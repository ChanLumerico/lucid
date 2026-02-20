import math
from typing import Callable
from copy import deepcopy

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = [
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
    "SinusoidalPosEmbedding",
    "LearnedPosEmbedding",
    "RotaryPosEmbedding",
]


@nn.auto_repr(
    "d_model",
    "num_heads",
    "dim_feedforward",
    "dropout",
    "activation",
    "layer_norm_eps",
    "norm_first",
    "bias",
)
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(
            d_model, num_heads, dropout=dropout, bias=bias
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = activation
        self.norm_first = norm_first

    def _sa_block(
        self,
        x: Tensor,
        src_mask: Tensor | None,
        src_key_padding_mask: Tensor | None,
        is_causal: bool,
        kv_cache: nn.KVCache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        cache_layer_idx: int | None = None,
    ) -> Tensor:
        attn_output = self.self_attn(
            x,
            x,
            x,
            src_key_padding_mask,
            src_mask,
            is_causal,
            kv_cache=kv_cache,
            use_cache=use_cache,
            cache_position=cache_position,
            cache_layer_idx=cache_layer_idx,
        )
        attn_output = self.dropout1(attn_output)

        return attn_output

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        kv_cache: nn.KVCache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        cache_layer_idx: int | None = None,
    ) -> Tensor:
        if self.norm_first:
            x = src + self._sa_block(
                self.norm1(src),
                src_mask,
                src_key_padding_mask,
                is_causal,
                kv_cache=kv_cache,
                use_cache=use_cache,
                cache_position=cache_position,
                cache_layer_idx=cache_layer_idx,
            )
            x += self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                src
                + self._sa_block(
                    src,
                    src_mask,
                    src_key_padding_mask,
                    is_causal,
                    kv_cache=kv_cache,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    cache_layer_idx=cache_layer_idx,
                )
            )
            x = self.norm2(x + self._ff_block(x))

        return x


@nn.auto_repr(
    "d_model",
    "num_heads",
    "dim_feedforward",
    "dropout",
    "activation",
    "layer_norm_eps",
    "norm_first",
    "bias",
)
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(
            d_model, num_heads, dropout=dropout, bias=bias
        )
        self.multihead_attn = nn.MultiHeadAttention(
            d_model, num_heads, dropout=dropout, bias=bias
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = activation
        self.norm_first = norm_first

    def _sa_block(
        self,
        x: Tensor,
        tgt_mask: Tensor | None,
        tgt_key_padding_mask: Tensor | None,
        is_causal: bool,
        kv_cache: nn.KVCache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        cache_layer_idx: int | None = None,
    ) -> Tensor:
        attn_output = self.self_attn(
            x,
            x,
            x,
            tgt_key_padding_mask,
            tgt_mask,
            is_causal,
            kv_cache=kv_cache,
            use_cache=use_cache,
            cache_position=cache_position,
            cache_layer_idx=cache_layer_idx,
        )
        attn_output = self.dropout1(attn_output)

        return attn_output

    def _mha_block(
        self,
        x: Tensor,
        memory: Tensor,
        mem_mask: Tensor | None,
        mem_key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        attn_output = self.multihead_attn(
            x, memory, memory, mem_key_padding_mask, mem_mask, is_causal
        )
        attn_output = self.dropout2(attn_output)

        return attn_output

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout3(x)

        return x

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        mem_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        mem_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        mem_is_causal: bool = False,
        kv_cache: nn.KVCache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        cache_layer_idx: int | None = None,
    ) -> Tensor:
        if self.norm_first:
            x = tgt + self._sa_block(
                self.norm1(tgt),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
                kv_cache=kv_cache,
                use_cache=use_cache,
                cache_position=cache_position,
                cache_layer_idx=cache_layer_idx,
            )
            x += self._mha_block(
                self.norm2(x), memory, mem_mask, mem_key_padding_mask, mem_is_causal
            )
            x += self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                tgt
                + self._sa_block(
                    tgt,
                    tgt_mask,
                    tgt_key_padding_mask,
                    tgt_is_causal,
                    kv_cache=kv_cache,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    cache_layer_idx=cache_layer_idx,
                )
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, mem_mask, mem_key_padding_mask, mem_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer | nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        kv_cache: nn.KVCache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        cache_start_layer_idx: int = 0,
    ) -> Tensor:
        output = src
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                kv_cache=kv_cache,
                use_cache=use_cache,
                cache_position=cache_position,
                cache_layer_idx=cache_start_layer_idx + i,
            )
        if self.norm is not None:
            output = self.norm(output)

        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, norm={self.norm is not None}"


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer | nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        mem_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        mem_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        mem_is_causal: bool = False,
        kv_cache: nn.KVCache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
        cache_start_layer_idx: int = 0,
    ) -> Tensor:
        output = tgt
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                mem_mask=mem_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                mem_key_padding_mask=mem_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                mem_is_causal=mem_is_causal,
                kv_cache=kv_cache,
                use_cache=use_cache,
                cache_position=cache_position,
                cache_layer_idx=cache_start_layer_idx + i,
            )
        if self.norm is not None:
            output = self.norm(output)

        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, norm={self.norm is not None}"


@nn.auto_repr(
    "d_model",
    "num_heads",
    "num_encoder_layers",
    "num_decoder_layers",
    "dim_feedforward",
    "dropout",
    "activation",
    "layer_norm_eps",
    "norm_first",
    "bias",
)
class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
        custom_encoder: nn.Module | None = None,
        custom_decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if custom_encoder is None:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                num_heads,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                norm_first,
                bias,
            )
            self.encoder = TransformerEncoder(
                encoder_layer,
                num_encoder_layers,
                norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
            )
        else:
            self.encoder = custom_encoder

        if custom_decoder is None:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                num_heads,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                norm_first,
                bias,
            )
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_decoder_layers,
                norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
            )
        else:
            self.decoder = custom_decoder

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
        memory = self.encoder(
            src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
            kv_cache=encoder_kv_cache,
            use_cache=use_encoder_cache,
            cache_position=encoder_cache_position,
            cache_start_layer_idx=encoder_cache_start_layer_idx,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            mem_mask=mem_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            mem_key_padding_mask=mem_key_padding_mask,
            tgt_is_causal=False,
            mem_is_causal=False,
            kv_cache=kv_cache,
            use_cache=use_cache,
            cache_position=cache_position,
            cache_start_layer_idx=cache_start_layer_idx,
        )
        return output


class SinusoidalPosEmbedding(nn.Module):
    def __init__(
        self, seq_len: int | None = None, embed_dim: int | None = None
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, input_: lucid.FloatTensor) -> lucid.FloatTensor:
        if input_.ndim not in {2, 3}:
            raise ValueError(f"{type(self).__name__} expects 2D and 3D Tensor inputs.")

        if input_.dtype.base_dtype is not float:
            raise TypeError(f"'{type(self).__name__}' expects a float Tensor.")

        _is_batched = True if input_.ndim == 3 else False
        x = input_ if _is_batched else input_.unsqueeze(axis=0)

        seq_len = x.shape[1] if self.seq_len is None else self.seq_len
        embed_dim = x.shape[2] if self.embed_dim is None else self.embed_dim

        if seq_len != x.shape[1]:
            raise ValueError(
                f"seq_len does not match: '{self.seq_len}', '{x.shape[1]}'. "
                f"Use 'seq_len=None' for dynamic length inference."
            )
        if embed_dim != x.shape[2]:
            raise ValueError(
                f"embed_dim does not match: '{self.embed_dim}', '{x.shape[2]}'. "
                f"Use 'embed_dim=None' for dynamic length inference."
            )

        posemb = F.sinusoidal_pos_embedding(seq_len, embed_dim, x.device, x.dtype)
        embed = x + posemb
        if not _is_batched:
            embed = embed.squeeze(axis=0)

        return embed


class LearnedPosEmbedding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, input_: lucid.FloatTensor, offset: int = 0) -> lucid.FloatTensor:
        if input_.ndim not in {2, 3}:
            raise ValueError(f"{type(self).__name__} expects 2D and 3D Tensor inputs.")

        if input_.dtype.base_dtype is not float:
            raise TypeError(f"'{type(self).__name__}' expects a float Tensor.")

        seq_len, embed_dim = input_.shape[-2:]
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"embed_dim does not match: '{self.embed_dim}', '{embed_dim}'."
            )

        if offset < 0 or seq_len + offset > self.max_len:
            raise ValueError(f"Invalid position.")

        pos_ids = lucid.arange(
            offset, offset + seq_len, device=input_.device, dtype=lucid.Long
        )
        pos_emb = self.pos_emb(pos_ids)
        if input_.ndim == 3:
            pos_emb = pos_emb.unsqueeze(axis=0)

        return input_ + pos_emb


class RotaryPosEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int | None = None,
        max_seq_len: int | None = None,
        interleaved: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.interleaved = interleaved
        self._cache_seq_len = 0

        self.cos_cached: nn.Buffer | None
        self.sin_cached: nn.Buffer | None
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)

    def _build_cache(self, seq_len: int, embed_dim: int, device: str) -> None:
        theta = lucid.exp(
            -2
            * lucid.arange(embed_dim // 2, device=device, dtype=lucid.Double)
            * (math.log(10000.0) / embed_dim)
        )
        indices = lucid.arange(seq_len, device=device, dtype=lucid.Double)
        freq_half = indices.unsqueeze(-1) @ theta.unsqueeze(0)

        if self.interleaved:
            freq = freq_half.repeat(2, axis=-1)
        else:
            freq = lucid.concatenate([freq_half, freq_half], axis=-1)

        self.cos_cached = nn.Buffer(lucid.cos(freq))
        self.sin_cached = nn.Buffer(lucid.sin(freq))
        self._cache_seq_len = seq_len

    def forward(
        self, input_: lucid.FloatTensor, position_ids: lucid.LongTensor | None = None
    ) -> lucid.FloatTensor:
        if input_.ndim < 2:
            raise ValueError(f"{type(self).__name__} expects input ndim >= 2.")

        if input_.dtype.base_dtype is not float:
            raise TypeError(f"'{type(self).__name__}' expects a float Tensor.")

        seq_len = input_.shape[-2]
        embed_dim = input_.shape[-1]

        if self.embed_dim is not None:
            if embed_dim != self.embed_dim:
                raise ValueError(
                    f"Expected embed_dim of '{self.embed_dim}', got '{embed_dim}'"
                )

        if embed_dim % 2 != 0:
            raise ValueError(
                f"Expected even input embedding dimension, got '{embed_dim}'."
            )

        if position_ids is not None:
            if position_ids.ndim != 1 or position_ids.shape[0] != seq_len:
                raise ValueError(
                    "position_ids must be 1-D with length equal to input_.shape[-2]."
                )
            max_pos = int(lucid.max(position_ids).item())
            if max_pos < 0:
                raise ValueError("position_ids must be non-negative.")
            required_len = max_pos + 1
        else:
            required_len = seq_len

        if self.max_seq_len is not None and required_len > self.max_seq_len:
            raise ValueError(
                f"Required sequence length '{required_len}' exceeds max_seq_len "
                f"'{self.max_seq_len}'."
            )

        if (
            self.cos_cached is None
            or self.sin_cached is None
            or self._cache_seq_len < required_len
            or self.cos_cached.shape[-1] != embed_dim
            or self.cos_cached.device != input_.device
        ):
            build_len = required_len if self.max_seq_len is None else self.max_seq_len
            self._build_cache(build_len, embed_dim, input_.device)

        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            ids = position_ids.to(input_.device).astype(lucid.Long)
            cos = self.cos_cached[ids]
            sin = self.sin_cached[ids]

        x = input_.astype(lucid.Double)
        cos = cos.astype(lucid.Double)
        sin = sin.astype(lucid.Double)

        x_rot = lucid.zeros_like(x)
        if self.interleaved:
            x_rot[..., 0::2] = -x[..., 1::2]
            x_rot[..., 1::2] = x[..., 0::2]
        else:
            half = embed_dim // 2
            x_rot[..., :half] = -x[..., half:]
            x_rot[..., half:] = x[..., :half]

        out = x * cos + x_rot * sin
        return out.astype(input_.dtype)
