"""
Transformer modules: TransformerEncoderLayer, TransformerEncoder,
TransformerDecoderLayer, TransformerDecoder, Transformer.
"""

from typing import cast

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike

from lucid.nn.module import Module
from lucid.nn.modules.attention import MultiheadAttention
from lucid.nn.modules.normalization import LayerNorm

from lucid.nn.modules.linear import Linear
from lucid.nn.modules.dropout import Dropout
from lucid.nn.functional.activations import gelu, relu


class TransformerEncoderLayer(Module):
    """Single transformer encoder layer: self-attention + FFN with LayerNorm."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.linear1 = Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def _ff(self, x: Tensor) -> Tensor:
        act = gelu if self.activation == "gelu" else relu
        h = cast(Tensor, self.linear1(x))
        return self.linear2(cast(Tensor, self.dropout2(act(h))))  # type: ignore[return-value]

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            normed: Tensor = cast(Tensor, self.norm1(src))
            src2, _ = self.self_attn(
                normed,
                normed,
                normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            src = src + cast(Tensor, self.dropout1(src2))
            src = src + cast(
                Tensor, self.dropout3(self._ff(cast(Tensor, self.norm2(src))))
            )
        else:
            src2, _ = self.self_attn(
                src,
                src,
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            src = cast(Tensor, self.norm1(src + cast(Tensor, self.dropout1(src2))))
            src = cast(
                Tensor, self.norm2(src + cast(Tensor, self.dropout3(self._ff(src))))
            )
        return src

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout_val}"
        )


class TransformerEncoder(Module):
    """Stack of N TransformerEncoderLayers."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(
                encoder_layer.d_model,
                encoder_layer.nhead,
                encoder_layer.dim_feedforward,
                encoder_layer.dropout_val,
                encoder_layer.activation,
                encoder_layer.batch_first,
                encoder_layer.norm_first,
            )
            for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)
        self.norm = norm
        if norm is not None:
            self.add_module("norm", norm)
        self.num_layers = num_layers

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = cast(
                Tensor,
                layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask),
            )
        if self.norm is not None:
            output = cast(Tensor, self.norm(output))
        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}"


class TransformerDecoderLayer(Module):
    """Single transformer decoder layer: masked self-attention + cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.linear1 = Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = LayerNorm(d_model, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

    def _ff(self, x: Tensor) -> Tensor:
        act = gelu if self.activation == "gelu" else relu
        h = cast(Tensor, self.linear1(x))
        return self.linear2(cast(Tensor, self.dropout2(act(h))))  # type: ignore[return-value]

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            normed: Tensor = cast(Tensor, self.norm1(tgt))
            tgt2, _ = self.self_attn(
                normed,
                normed,
                normed,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            tgt = tgt + cast(Tensor, self.dropout1(tgt2))
            tgt2, _ = self.multihead_attn(
                cast(Tensor, self.norm2(tgt)),
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            tgt = tgt + cast(Tensor, self.dropout3(tgt2))
            tgt = tgt + cast(
                Tensor, self.dropout4(self._ff(cast(Tensor, self.norm3(tgt))))
            )
        else:
            tgt2, _ = self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            tgt = cast(Tensor, self.norm1(tgt + cast(Tensor, self.dropout1(tgt2))))
            tgt2, _ = self.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            tgt = cast(Tensor, self.norm2(tgt + cast(Tensor, self.dropout3(tgt2))))
            tgt = cast(
                Tensor, self.norm3(tgt + cast(Tensor, self.dropout4(self._ff(tgt))))
            )
        return tgt

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout_val}"
        )


class TransformerDecoder(Module):
    """Stack of N TransformerDecoderLayers."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerDecoderLayer(
                decoder_layer.d_model,
                decoder_layer.nhead,
                decoder_layer.dim_feedforward,
                decoder_layer.dropout_val,
                decoder_layer.activation,
                decoder_layer.batch_first,
                decoder_layer.norm_first,
            )
            for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)
        self.norm = norm
        if norm is not None:
            self.add_module("norm", norm)
        self.num_layers = num_layers

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = cast(
                Tensor,
                layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask),
            )
        if self.norm is not None:
            output = cast(Tensor, self.norm(output))
        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}"


class Transformer(Module):
    """Full encoder-decoder Transformer."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        enc_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            device=device,
            dtype=dtype,
        )
        dec_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            device=device,
            dtype=dtype,
        )
        enc_norm = LayerNorm(d_model, device=device, dtype=dtype)
        dec_norm = LayerNorm(d_model, device=device, dtype=dtype)
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers, norm=enc_norm)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers, norm=dec_norm)
        self.d_model = d_model
        self.nhead = nhead

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        memory = cast(
            Tensor,
            self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask),
        )
        return cast(
            Tensor,
            self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask),
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, nhead={self.nhead}"
