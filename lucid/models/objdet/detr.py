import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid import register_model


class _MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class _SpatialPosEncoding(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, mask: Tensor) -> Tensor:
        if mask.ndim != 3:
            raise ValueError("Mask must have shape (B, H, W)")

        not_mask = 1.0 - mask.astype(lucid.Float)
        y_embed = lucid.cumsum(not_mask, axis=1)
        x_embed = lucid.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = lucid.arange(self.num_pos_feats, dtype=lucid.Float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = lucid.stack(
            [lucid.sin(pos_x[..., 0::2]), lucid.cos(pos_x[..., 1::2])], axis=4
        )
        pos_x = pos_x.reshape(*mask.shape[:3], -1)
        pos_y = lucid.stack(
            [lucid.sin(pos_y[..., 0::2]), lucid.cos(pos_y[..., 1::2])], axis=4
        )
        pos_y = pos_y.reshape(*mask.shape[:3], -1)

        pos = lucid.concatenate([pos_y, pos_x], axis=3)
        return pos.transpose((0, 3, 1, 2))


class _TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

        self._config = dict(
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

    @staticmethod
    def _with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        q = k = self._with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.drop(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: _TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer])
        for _ in range(num_layers - 1):
            self.layers.append(type(encoder_layer)(**encoder_layer._config))

        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> None:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class _TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiHeadAttention(d_model, n_head, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

        self._config = dict(
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

    @staticmethod
    def _with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        q = k = self._with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        q = self._with_pos_embed(tgt, query_pos)
        k = self._with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn(
            q,
            k,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.drop(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class _TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: _TransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer])
        for _ in range(num_layers - 1):
            self.layers.append(type(decoder_layer)(**decoder_layer._config))

        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        output = tgt
        intermediate: list[Tensor] = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm else output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        if self.return_intermediate:
            return lucid.stack(intermediate)

        return output.unsqueeze(axis=0)


class _Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
        normalize_before: bool = False,
        return_intermediate_dec: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        enc_layer = _TransformerEncoderLayer(
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        self.encoder = _TransformerEncoder(
            enc_layer, num_encoder_layers, norm=nn.LayerNorm(d_model)
        )

        dec_layer = _TransformerDecoderLayer(
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        self.decoder = _TransformerDecoder(
            dec_layer,
            num_decoder_layers,
            norm=nn.LayerNorm(d_model),
            return_intermediate=return_intermediate_dec,
        )

    def forward(
        self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor
    ) -> tuple[Tensor, Tensor]:
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        bs, n_queries, _ = query_embed.shape
        tgt = lucid.zeros(
            (bs, n_queries, self.d_model), dtype=src.dtype, device=src.device
        )
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )

        return hs, memory


# TODO: Continue implementation of DETR model
NotImplemented
