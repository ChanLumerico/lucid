"""
Transformer modules: TransformerEncoderLayer, TransformerEncoder,
TransformerDecoderLayer, TransformerDecoder, Transformer.
"""

import math
from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init


class TransformerEncoderLayer(Module):
    """Single transformer encoder layer: self-attention + FFN with LayerNorm.

    Args:
        d_model:         Total embedding dimension.
        nhead:           Number of attention heads.
        dim_feedforward: Hidden size of the FFN (default: 2048).
        dropout:         Dropout probability (default: 0.1).
        activation:      Activation in FFN ('relu' or 'gelu').
        batch_first:     If True, input shape is (B, T, d_model).
        norm_first:      If True, use pre-LN (residual after norm).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        from lucid.nn.modules.attention import MultiheadAttention
        from lucid.nn.modules.normalization import LayerNorm
        from lucid.nn.modules.linear import Linear
        from lucid.nn.modules.dropout import Dropout

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            batch_first=batch_first,
                                            device=device, dtype=dtype)
        self.linear1 = Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def _ff(self, x: Any) -> Any:
        from lucid.nn import functional as F
        if self.activation == "gelu":
            return self.linear2(self.dropout2(F.gelu(self.linear1(x))))
        return self.linear2(self.dropout2(F.relu(self.linear1(x))))

    def forward(
        self,
        src: Any,
        src_mask: Any = None,
        src_key_padding_mask: Any = None,
    ) -> Any:
        if self.norm_first:
            src2, _ = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src),
                                     attn_mask=src_mask)
            src = src + self.dropout1(src2)
            src = src + self.dropout3(self._ff(self.norm2(src)))
        else:
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
            src = self.norm1(src + self.dropout1(src2))
            src = self.norm2(src + self.dropout3(self._ff(src)))
        return src

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, nhead={self.nhead}, "
                f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout_val}")


class TransformerEncoder(Module):
    """Stack of N TransformerEncoderLayers.

    Args:
        encoder_layer: A TransformerEncoderLayer instance (cloned N times).
        num_layers:    Number of sub-encoder layers.
        norm:          Optional normalization module applied to output.
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Any = None,
    ) -> None:
        super().__init__()
        # Create fresh layers with the same config (deepcopy doesn't work with TensorImpl)
        self.layers = [
            TransformerEncoderLayer(
                encoder_layer.d_model, encoder_layer.nhead,
                encoder_layer.dim_feedforward, encoder_layer.dropout_val,
                encoder_layer.activation, encoder_layer.batch_first,
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

    def forward(
        self,
        src: Any,
        mask: Any = None,
        src_key_padding_mask: Any = None,
    ) -> Any:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}"


class TransformerDecoderLayer(Module):
    """Single transformer decoder layer: masked self-attention + cross-attention + FFN.

    Args:
        d_model:         Total embedding dimension.
        nhead:           Number of attention heads.
        dim_feedforward: Hidden size of the FFN.
        dropout:         Dropout probability.
        activation:      'relu' or 'gelu'.
        batch_first:     If True, input shape is (B, T, d_model).
        norm_first:      Pre-LN if True.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        from lucid.nn.modules.attention import MultiheadAttention
        from lucid.nn.modules.normalization import LayerNorm
        from lucid.nn.modules.linear import Linear
        from lucid.nn.modules.dropout import Dropout

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            batch_first=batch_first,
                                            device=device, dtype=dtype)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                                  batch_first=batch_first,
                                                  device=device, dtype=dtype)
        self.linear1 = Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = LayerNorm(d_model, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

    def _ff(self, x: Any) -> Any:
        from lucid.nn import functional as F
        if self.activation == "gelu":
            return self.linear2(self.dropout2(F.gelu(self.linear1(x))))
        return self.linear2(self.dropout2(F.relu(self.linear1(x))))

    def forward(
        self,
        tgt: Any,
        memory: Any,
        tgt_mask: Any = None,
        memory_mask: Any = None,
        tgt_key_padding_mask: Any = None,
        memory_key_padding_mask: Any = None,
    ) -> Any:
        if self.norm_first:
            tgt2, _ = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt),
                                     attn_mask=tgt_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2, _ = self.multihead_attn(self.norm2(tgt), memory, memory,
                                          attn_mask=memory_mask)
            tgt = tgt + self.dropout3(tgt2)
            tgt = tgt + self.dropout4(self._ff(self.norm3(tgt)))
        else:
            tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
            tgt = self.norm2(tgt + self.dropout3(tgt2))
            tgt = self.norm3(tgt + self.dropout4(self._ff(tgt)))
        return tgt

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, nhead={self.nhead}, "
                f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout_val}")


class TransformerDecoder(Module):
    """Stack of N TransformerDecoderLayers.

    Args:
        decoder_layer: A TransformerDecoderLayer instance.
        num_layers:    Number of sub-decoder layers.
        norm:          Optional normalization module applied to output.
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Any = None,
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerDecoderLayer(
                decoder_layer.d_model, decoder_layer.nhead,
                decoder_layer.dim_feedforward, decoder_layer.dropout_val,
                decoder_layer.activation, decoder_layer.batch_first,
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

    def forward(
        self,
        tgt: Any,
        memory: Any,
        tgt_mask: Any = None,
        memory_mask: Any = None,
        tgt_key_padding_mask: Any = None,
        memory_key_padding_mask: Any = None,
    ) -> Any:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}"


class Transformer(Module):
    """Full encoder-decoder Transformer.

    Args:
        d_model:              Total embedding dimension (default: 512).
        nhead:                Number of attention heads (default: 8).
        num_encoder_layers:   Number of encoder layers (default: 6).
        num_decoder_layers:   Number of decoder layers (default: 6).
        dim_feedforward:      FFN hidden size (default: 2048).
        dropout:              Dropout probability (default: 0.1).
        activation:           'relu' or 'gelu'.
        batch_first:          If True, I/O shape is (B, T, d_model).
    """

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
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        enc_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first,
            device=device, dtype=dtype,
        )
        dec_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first,
            device=device, dtype=dtype,
        )
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers)
        self.d_model = d_model
        self.nhead = nhead

    def forward(
        self,
        src: Any,
        tgt: Any,
        src_mask: Any = None,
        tgt_mask: Any = None,
        memory_mask: Any = None,
        src_key_padding_mask: Any = None,
        tgt_key_padding_mask: Any = None,
        memory_key_padding_mask: Any = None,
    ) -> Any:
        memory = self.encoder(src, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, nhead={self.nhead}"
