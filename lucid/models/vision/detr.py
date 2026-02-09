import math
from typing import Sequence

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid import register_model

from lucid.models.objdet.util import DetectionDict
from lucid.models.imgclf.resnet import ResNet, resnet_50, resnet_101


__all__ = ["DETR", "detr_r50", "detr_r101"]


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
        d_model: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = d_model // 2
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

        dim_t = lucid.arange(
            self.num_pos_feats, dtype=lucid.Float32, device=mask.device
        )
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
    ) -> Tensor:
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


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(axis=-1)
    b_x1 = cx - 0.5 * w
    b_y1 = cy - 0.5 * h
    b_x2 = cx + 0.5 * w
    b_y2 = cy + 0.5 * h
    return lucid.stack([b_x1, b_y1, b_x2, b_y2], axis=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(axis=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return lucid.stack([cx, cy, w, h], axis=-1)


def box_area(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(axis=-1)
    return (x2 - x1).clip(min_value=0) * (y2 - y1).clip(min_value=0)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    x1 = lucid.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = lucid.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = lucid.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = lucid.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (x2 - x1).clip(min_value=0) * (y2 - y1).clip(min_value=0)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)

    x1_c = lucid.minimum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1_c = lucid.minimum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2_c = lucid.maximum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2_c = lucid.maximum(boxes1[:, None, 3], boxes2[None, :, 3])

    area_c = (x2_c - x1_c).clip(min_value=0) * (y2_c - y1_c).clip(min_value=0)
    return iou - (area_c - union) / (area_c + 1e-7)


class _HungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @staticmethod
    @lucid.no_grad()
    def _linear_sum_assignment(cost_matrix: Tensor) -> tuple[Tensor, Tensor]:
        cost = Tensor(Tensor.copy_data(cost_matrix.data))
        n_rows, n_cols = cost.shape

        size = max(n_rows, n_cols)
        padded = lucid.zeros((size, size), dtype=cost.dtype)
        padded[:n_rows, :n_cols] = cost

        mask = lucid.zeros((size, size), dtype=lucid.Int8)
        row_cover = lucid.zeros(size, dtype=bool)
        col_cover = lucid.zeros(size, dtype=bool)

        path = lucid.zeros((size * 2, 2), dtype=lucid.Int32)

        def _step1() -> None:
            padded -= padded.min(axis=1, keepdims=True)

        def _step2() -> None:
            padded -= padded.min(axis=0)

        def _step3() -> None:
            for i in range(size):
                for j in range(size):
                    if (
                        lucid.abs(padded[i, j]) < 1e-9
                        and not col_cover[j]
                        and not row_cover[i]
                    ):
                        mask[i, j] = 1
                        col_cover[j] = True
                        row_cover[i] = True

            row_cover[:] = False
            col_cover[:] = False

        def _step4() -> bool:
            for j in range(size):
                if lucid.any(mask[:, j] == 1):
                    col_cover[j] = True
            return int(col_cover.sum().item()) >= n_rows

        def _find_zero() -> tuple[int, int]:
            for i in range(size):
                if row_cover[i]:
                    continue
                for j in range(size):
                    if lucid.abs(padded[i, j]) < 1e-9 and not col_cover[j]:
                        return i, j
            return -1, -1

        def _find_star_in_row(row: int) -> int:
            cols = lucid.nonzero(mask[row] == 1)
            return cols[0].item() if cols.size else -1

        def _find_star_in_col(col: int) -> int:
            rows = lucid.nonzero(mask[:, col] == 1)
            return rows[0].item() if rows.size else -1

        def _find_prime_in_row(row: int) -> int:
            cols = lucid.nonzero(mask[row] == 2)
            return cols[0].item() if cols.size else -1

        def _augment_path(count: int) -> None:
            for i in range(count + 1):
                r, c = path[i]
                mask[r, c] = 0 if mask[r, c] == 1 else 1

        def _clear_covers() -> None:
            row_cover[:] = False
            col_cover[:] = False

        def _erase_primes() -> None:
            mask[mask == 2] = 0

        def _step6() -> None:
            uncovered_rows = ~row_cover
            uncovered_cols = ~col_cover
            if not lucid.any(uncovered_rows) or not lucid.any(uncovered_cols):
                return

            min_val = lucid.min(padded[uncovered_rows][:, uncovered_cols])
            padded[row_cover] += min_val
            padded[:, ~col_cover] -= min_val

        _step1()
        _step2()
        _step3()

        max_iters = mask.size * 4
        for _ in range(max_iters):
            if _step4():
                break
            row, col = _find_zero()
            if row == -1:
                _step6()
                continue

            mask[row, col] = 2
            star_col = _find_star_in_row(row)
            if star_col != -1:
                row_cover[row] = True
                col_cover[star_col] = False
                continue

            path_count = 0
            path[path_count] = (row, col)
            for _ in range(mask.shape[0] * 2):
                star_row = _find_star_in_col(path[path_count, 1])
                if star_row == -1:
                    break

                path_count += 1
                path[path_count] = (star_row, path[path_count - 1, 1])

                prime_col = _find_prime_in_row(star_row)
                path_count += 1
                path[path_count] = (star_row, prime_col)

            _augment_path(path_count)
            _clear_covers()
            _erase_primes()

        row_ind, col_ind = [], []
        for i in range(n_rows):
            j = lucid.nonzero(mask[i] == 1)
            if j.size:
                col = j[0].item()
                if col < n_cols:
                    row_ind.append(i)
                    col_ind.append(col)

        return Tensor(row_ind, dtype=lucid.Int32), Tensor(col_ind, dtype=lucid.Int32)

    def forward(
        self, outputs: tuple[Tensor, Tensor], targets: Sequence[dict[str, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        pred_logits, pred_boxes = outputs

        prob = F.softmax(pred_logits, axis=-1)
        out_bbox = pred_boxes

        results: list[tuple[Tensor, Tensor]] = []
        for b, target in enumerate(targets):
            tgt_ids = target["class_id"]
            tgt_bbox = target["box"]

            if tgt_ids.size == 0:
                empty = lucid.empty(0, dtype=lucid.Int32, device=pred_logits.device)
                results.append((empty, empty))
                continue

            cost_class = -prob[b][:, tgt_ids]
            bbox_cost = lucid.sum(
                lucid.abs(out_bbox[b][:, None, :] - tgt_bbox[None, :, :]), axis=2
            )
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox)
            )

            total_cost = (
                self.cost_bbox * bbox_cost
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            total_cost_cpu = Tensor(Tensor.copy_data(total_cost.data), device="cpu")

            row_ind, col_ind = self._linear_sum_assignment(total_cost_cpu)
            results.append(
                (
                    Tensor(row_ind, dtype=lucid.Int32, device=pred_logits.device),
                    Tensor(col_ind, dtype=lucid.Int32, device=pred_logits.device),
                )
            )

        return results


class _BackboneBase(nn.Module):
    def __init__(self, backbone: ResNet) -> None:
        super().__init__()
        self.body = nn.ModuleDict(
            {
                "stem": backbone.stem,
                "maxpool": backbone.maxpool,
                "layer1": backbone.layer1,
                "layer2": backbone.layer2,
                "layer3": backbone.layer3,
                "layer4": backbone.layer4,
            }
        )
        self.num_channels = backbone.layer4[-1].conv3[0].out_channels

    def forward(self, x: Tensor) -> Tensor:
        for key in ["stem", "maxpool", "layer1", "layer2", "layer3", "layer4"]:
            x = self.body[key](x)
        return x


class DETR(nn.Module):
    def __init__(
        self,
        backbone: _BackboneBase,
        transformer: _Transformer,
        num_classes: int,
        num_queries: int = 100,
        aux_loss: bool = True,
        matcher: _HungarianMatcher | None = None,
        class_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.position_embedding = _SpatialPosEncoding(transformer.d_model)

        self.transformer = transformer
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss

        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = _MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.matcher = (
            matcher
            if matcher is not None
            else _HungarianMatcher(
                cost_class=class_loss_coef,
                cost_bbox=bbox_loss_coef,
                cost_giou=giou_loss_coef,
            )
        )

        self.class_loss_coef = class_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.eos_coef = eos_coef

    def _get_src_permutation_idx(
        self, indices: Sequence[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        batch_idx = []
        src_idx = []
        for i, (src, _) in enumerate(indices):
            if src.size == 0:
                continue
            batch_idx.append(
                lucid.full(src.shape, i, dtype=lucid.Int32, device=self.device)
            )
            src_idx.append(src)

        if not src_idx:
            return (
                lucid.empty(0, dtype=lucid.Int32, device=self.device),
                lucid.empty(0, dtype=lucid.Int32, device=self.device),
            )

        return lucid.concatenate(batch_idx), lucid.concatenate(src_idx)

    def _get_tgt_permutation_idx(
        self, indices: Sequence[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        batch_idx = []
        tgt_idx = []
        for i, (_, tgt) in enumerate(indices):
            if tgt.size == 0:
                continue
            batch_idx.append(
                lucid.full(tgt.shape, i, dtype=lucid.Int32, device=self.device)
            )
            tgt_idx.append(tgt)

        if not tgt_idx:
            return (
                lucid.empty(0, dtype=lucid.Int32, device=self.device),
                lucid.empty(0, dtype=lucid.Int32, device=self.device),
            )

        return lucid.concatenate(batch_idx), lucid.concatenate(tgt_idx)

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor]]:
        features = self.backbone(x)
        B, _, H, W = features.shape

        if mask is None:
            mask = lucid.zeros(B, H, W, dtype=lucid.Float32, device=x.device)
        pos_embed = self.position_embedding(mask)

        src = self.input_proj(features)
        src_flat = src.flatten(2).transpose((0, 2, 1))
        pos_flat = pos_embed.flatten(2).transpose((0, 2, 1))
        mask_flat = mask.reshape(B, -1).astype(bool)

        query_embed = self.query_embed.weight.unsqueeze(axis=0).repeat(B, axis=0)
        hs, _ = self.transformer(src_flat, mask_flat, query_embed, pos_flat)

        outputs_class = self.class_embed(hs[-1])
        outputs_coord = F.sigmoid(self.bbox_embed(hs[-1]))

        out: list[tuple[Tensor, Tensor]] = [(outputs_class, outputs_coord)]
        if self.aux_loss:
            aux_outputs: list[tuple[Tensor, Tensor]] = []
            for layer_out in hs[:-1]:
                aux_outputs.append(
                    (self.class_embed(layer_out), F.sigmoid(self.bbox_embed(layer_out)))
                )

            return aux_outputs + out

        final_out = out[0]
        return final_out

    @lucid.no_grad()
    def predict(self, x: Tensor, k: int = 100) -> list[list[DetectionDict]]:
        outputs = self.forward(x)
        if isinstance(outputs, list):
            outputs = outputs[-1]

        logits, boxes = outputs
        probs = F.softmax(logits, axis=-1)[..., :-1]
        boxes = boxes.clip(min_value=0.0, max_value=1.0)

        scores = lucid.max(probs, axis=-1)
        labels = lucid.argmax(probs, axis=-1)

        results: list[list[DetectionDict]] = []
        B, _, H, W = x.shape
        for b in range(B):
            _, keep = lucid.topk(scores[b], k=k)

            image_boxes = boxes[b][keep]
            image_scores = scores[b][keep]
            image_labels = labels[b][keep]

            abs_boxes = box_cxcywh_to_xyxy(image_boxes)
            abs_boxes = abs_boxes * Tensor(
                [W, H, W, H], dtype=abs_boxes.dtype, device=abs_boxes.device
            )
            abs_boxes[..., 0::2] = abs_boxes[..., 0::2].clip(0, W - 1)
            abs_boxes[..., 1::2] = abs_boxes[..., 1::2].clip(0, H - 1)

            detections: list[DetectionDict] = []
            for box, score, label in zip(abs_boxes, image_scores, image_labels):
                detections.append(
                    {
                        "box": box.tolist(),
                        "score": float(score.item()),
                        "class_id": int(label.item()),
                    }
                )
            results.append(detections)

        return results

    def _compute_losses(
        self, outputs: tuple[Tensor, Tensor], targets: Sequence[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        pred_logits, pred_boxes = outputs

        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)

        num_boxes = sum(t["class_id"].shape[0] for t in targets)
        num_boxes = max(num_boxes, 1)

        target_classes = lucid.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=lucid.Int32,
            device=pred_logits.device,
        )
        if idx[0].size > 0:
            target_classes[idx] = lucid.concatenate(
                [t["class_id"][j] for t, (_, j) in zip(targets, indices)]
            )

        empty_weight = lucid.ones(
            pred_logits.shape[-1], dtype=lucid.Float32, device=pred_logits.device
        )
        empty_weight[-1] = self.eos_coef

        loss_ce = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target_classes.reshape(-1),
            weight=empty_weight,
        )

        if idx[0].size > 0:
            src_boxes = pred_boxes[idx]
            target_boxes = lucid.concatenate(
                [t["box"][j] for t, (_, j) in zip(targets, indices)]
            )
            loss_bbox = lucid.sum(lucid.abs(src_boxes - target_boxes), axis=1)
            loss_bbox = loss_bbox.sum() / num_boxes

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
            loss_giou = (1 - giou.diagonal()).sum() / num_boxes

        else:
            loss_bbox = lucid.zeros(
                (), dtype=pred_boxes.dtype, device=pred_boxes.device
            )
            loss_giou = lucid.zeros(
                (), dtype=pred_boxes.dtype, device=pred_boxes.device
            )

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "indices": indices,
        }

    def get_loss(
        self,
        x: Tensor,
        targets: Sequence[dict[str, Tensor]],
        mask: Tensor | None = None,
    ) -> Tensor:
        outputs = self.forward(x, mask)
        aux_outputs = outputs[:-1] if isinstance(outputs, list) else []
        main_outputs = outputs[-1] if isinstance(outputs, list) else outputs

        losses = self._compute_losses(main_outputs, targets)
        total_loss = (
            self.class_loss_coef * losses["loss_ce"]
            + self.bbox_loss_coef * losses["loss_bbox"]
            + self.giou_loss_coef * losses["loss_giou"]
        )

        if self.aux_loss:
            for aux_output in aux_outputs:
                aux_losses = self._compute_losses(aux_output, targets)
                total_loss += (
                    self.class_loss_coef * aux_losses["loss_ce"]
                    + self.bbox_loss_coef * aux_losses["loss_bbox"]
                    + self.giou_loss_coef * aux_losses["loss_giou"]
                )

        return total_loss


def _build_backbone(name: str, pretrained: bool) -> _BackboneBase:
    import lucid.weights as W

    weight_prefix = {
        "resnet_50": "ResNet_50_Weights",
        "resnet_101": "ResNet_101_Weights",
    }

    if name == "resnet_50":
        weights = getattr(W, weight_prefix[name]).DEFAULT if pretrained else None
        backbone = resnet_50(weights=weights)
    elif name == "resnet_101":
        weights = getattr(W, weight_prefix[name]).DEFAULT if pretrained else None
        backbone = resnet_101(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone '{name}'")

    back = _BackboneBase(backbone)
    if pretrained:
        for m in back.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return back


@register_model
def detr_r50(
    num_classes: int = 91,
    num_queries: int = 100,
    pretrained_backbone: bool = False,
    **kwargs,
) -> DETR:
    backbone = _build_backbone("resnet_50", pretrained_backbone)
    transformer = _Transformer(
        d_model=256,
        n_head=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )
    return DETR(backbone, transformer, num_classes, num_queries, **kwargs)


@register_model
def detr_r101(
    num_classes: int = 91,
    num_queries: int = 100,
    pretrained_backbone: bool = False,
    **kwargs,
) -> DETR:
    backbone = _build_backbone("resnet_101", pretrained_backbone)
    transformer = _Transformer(
        d_model=256,
        n_head=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )
    return DETR(backbone, transformer, num_classes, num_queries, **kwargs)
