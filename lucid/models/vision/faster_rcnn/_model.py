"""Faster R-CNN object detector (Ren et al., NeurIPS 2015).

Paper: "Faster R-CNN: Towards Real-Time Object Detection with
        Region Proposal Networks"

This module implements the **ResNet-50-FPN** two-stage detector — the
modern reference configuration shipped with the COCO ``box AP 37.0``
checkpoint — rather than the original VGG16 single-scale design.  The
submodule layout mirrors the reference detector verbatim so the COCO
checkpoint loads strict:

  Image (B, C, H, W)
    ↓  ResNet-50 backbone (frozen BN, eps=0) → C2, C3, C4, C5
    ↓  FPN: 1×1 lateral (``inner_blocks``) + top-down nearest add +
           3×3 output (``layer_blocks``) + LastLevelMaxPool ("pool")
  [P2, P3, P4, P5, pool]
    ├─ RPN head (3×3 conv + ReLU → cls_logits + bbox_pred), 3 anchors/cell
    │    sizes ((32,),(64,),(128,),(256,),(512,)), ratios (0.5,1,2)
    │    → per-level top-k → decode → clip → NMS 0.7 → top-1000 proposals
    │
    └─ MultiScale RoI Align (7×7, sampling_ratio=2, aligned) over P2-P5
         ↓  FPN level-assignment k = floor(4 + log2(sqrt(wh)/224)), k∈[2,5]
       TwoMLPHead: fc6 (256·7·7 → 1024) → fc7 (1024 → 1024)
         ↓
       FastRCNNPredictor: cls_score (1024 → 91) + bbox_pred (1024 → 91·4)
         ↓  softmax, per-class decode (10,10,5,5), clip, NMS 0.5, top-100

Faithfulness notes
------------------
* Backbone batch-norm is **frozen** (eval-only affine + running-stat math,
  no ``num_batches_tracked``) with ``eps = 0`` matching the reference
  detection ``FrozenBatchNorm2d``.
* FPN adds a parameter-free ``LastLevelMaxPool`` 5th level fed only to RPN.
* RPN box-coder weights are ``(1, 1, 1, 1)``; RoI box-coder weights are
  ``(10, 10, 5, 5)``.
* The detector accepts an already-resized + normalised image batch (the
  reference ``GeneralizedRCNNTransform`` normalisation / resize is a
  :class:`~lucid.utils.transforms.Detection` preset that runs outside the
  model).  ``forward`` returns raw RoI-head outputs and ``postprocess``
  produces the final per-image detections.
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    _FeaturePyramidNetwork,
    _ReferenceAnchorGenerator,
    _ResNetBody,
    batched_nms,
    clip_boxes_to_image,
    decode_boxes,
    multiscale_roi_align,
    nms,
    remove_small_boxes,
)
from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig

# ---------------------------------------------------------------------------
# Backbone with FPN  (key prefix: ``backbone.body.*`` / ``backbone.fpn.*``)
# ---------------------------------------------------------------------------


class _BackboneWithFPN(nn.Module):
    """ResNet trunk (``body``) feeding a Feature Pyramid Network (``fpn``).

    Submodule names mirror the reference so the converter is a pure
    identity map for every backbone key.
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
        fpn_out_channels: int,
        bn_eps: float,
    ) -> None:
        super().__init__()
        self.body = _ResNetBody(in_channels, layers, bn_eps=bn_eps)
        self.fpn = _FeaturePyramidNetwork(self.body.out_channels_list, fpn_out_channels)
        self.out_channels = fpn_out_channels

    def forward(self, x: Tensor) -> list[Tensor]:  # type: ignore[override]
        c_feats = cast(list[Tensor], self.body(x))  # [C2, C3, C4, C5]
        return self.fpn.forward(c_feats)  # [P2, P3, P4, P5, pool]


# ---------------------------------------------------------------------------
# RPN  (key prefix: ``rpn.head.conv.0.0`` / ``rpn.head.cls_logits`` / ...)
# ---------------------------------------------------------------------------


class _RPNHead(nn.Module):
    """RPN head: shared 3×3 conv + ReLU, then cls / bbox 1×1 sibling convs.

    The conv is wrapped in a ``Sequential`` of a ``Sequential`` so the
    state-dict keys read ``conv.0.0.weight`` (reference
    ``Conv2dNormActivation`` with no norm), matching exactly.
    """

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1))
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(  # type: ignore[override]
        self, features: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        logits: list[Tensor] = []
        bbox: list[Tensor] = []
        for feat in features:
            t = F.relu(cast(Tensor, self.conv(feat)))
            logits.append(cast(Tensor, self.cls_logits(t)))
            bbox.append(cast(Tensor, self.bbox_pred(t)))
        return logits, bbox


class _RegionProposalNetwork(nn.Module):
    """Region Proposal Network — ``head`` + anchor-based proposal layer.

    Holds only the parametric ``head``; the anchor generator and proposal
    filtering are stateless helpers driven by the parent config.
    """

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.head = _RPNHead(in_channels, num_anchors)


# ---------------------------------------------------------------------------
# RoI heads  (key prefix: ``roi_heads.box_head.*`` / ``roi_heads.box_predictor.*``)
# ---------------------------------------------------------------------------


class _TwoMLPHead(nn.Module):
    """Reference ``TwoMLPHead``: flatten → fc6 → ReLU → fc7 → ReLU."""

    def __init__(self, in_channels: int, representation_size: int) -> None:
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x.flatten(1)
        x = F.relu(cast(Tensor, self.fc6(x)))
        x = F.relu(cast(Tensor, self.fc7(x)))
        return x


class _FastRCNNPredictor(nn.Module):
    """Reference ``FastRCNNPredictor``: sibling cls_score + bbox_pred."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return cast(Tensor, self.cls_score(x)), cast(Tensor, self.bbox_pred(x))


class _RoIHeads(nn.Module):
    """RoI heads container — ``box_head`` (TwoMLPHead) + ``box_predictor``."""

    def __init__(
        self,
        in_channels: int,
        roi_size: int,
        representation_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.box_head = _TwoMLPHead(
            in_channels * roi_size * roi_size, representation_size
        )
        self.box_predictor = _FastRCNNPredictor(representation_size, num_classes)

    def forward(self, roi_feats: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        feats = cast(Tensor, self.box_head(roi_feats))
        return cast(tuple[Tensor, Tensor], self.box_predictor(feats))


# ---------------------------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------------------------


class FasterRCNNForObjectDetection(PretrainedModel):
    r"""Faster R-CNN with a ResNet-50-FPN backbone (Ren et al., NeurIPS 2015).

    The two-stage anchor-based detector in its modern reference
    configuration: a ResNet-50 trunk with frozen batch-norm feeds a
    Feature Pyramid Network, a Region Proposal Network emits per-image
    proposals from five pyramid levels, and a Fast R-CNN-style RoI head
    classifies + refines RoI-aligned crops.  The submodule layout mirrors
    the reference detector so the COCO ``box AP 37.0`` checkpoint loads
    strict and reproduces inference.

    Parameters
    ----------
    config : FasterRCNNConfig
        Frozen architecture spec.  Use the :func:`faster_rcnn_resnet50_fpn`
        factory for the COCO-pretrained configuration (``num_classes = 91``).

    Attributes
    ----------
    config : FasterRCNNConfig
        Stored copy of the config that built this model.
    backbone : _BackboneWithFPN
        ResNet-50 ``body`` + ``fpn`` producing five feature maps
        ``[P2, P3, P4, P5, pool]`` at strides ``4/8/16/32/64``.
    rpn : _RegionProposalNetwork
        Proposal head shared across all pyramid levels.
    roi_heads : _RoIHeads
        ``box_head`` (TwoMLPHead) + ``box_predictor`` (FastRCNNPredictor).

    Notes
    -----
    See Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks", NeurIPS 2015 (arXiv:1506.01497) and Lin et
    al., "Feature Pyramid Networks for Object Detection", CVPR 2017.  The
    model expects an already resized + normalised image batch; final
    detections come from :meth:`postprocess`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.faster_rcnn import faster_rcnn_resnet50_fpn
    >>> model = faster_rcnn_resnet50_fpn()
    >>> model.eval()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape[-1]   # num_classes
    91
    >>> dets = model.postprocess(out, image_sizes=[(224, 224)])
    >>> sorted(dets[0].keys())
    ['boxes', 'labels', 'scores']
    """

    config_class: ClassVar[type[FasterRCNNConfig]] = FasterRCNNConfig
    base_model_prefix: ClassVar[str] = "faster_rcnn"

    # FPN level strides: P2..P5 then the pool level.
    _strides: ClassVar[tuple[int, ...]] = (4, 8, 16, 32, 64)

    def __init__(self, config: FasterRCNNConfig) -> None:
        super().__init__(config)
        self._cfg = config

        self.backbone = _BackboneWithFPN(
            in_channels=config.in_channels,
            layers=config.backbone_layers,
            fpn_out_channels=config.fpn_out_channels,
            bn_eps=config.backbone_bn_eps,
        )
        C = self.backbone.out_channels

        num_anchors = len(config.rpn_anchor_ratios)  # 1 scale per level x ratios
        self._num_anchors = num_anchors
        self.rpn = _RegionProposalNetwork(C, num_anchors)

        # One anchor size per FPN level (5 levels including pool).
        sizes: tuple[tuple[int, ...], ...] = tuple(
            (s,) for s in config.rpn_anchor_sizes
        )
        ratios: tuple[tuple[float, ...], ...] = tuple(
            tuple(config.rpn_anchor_ratios) for _ in config.rpn_anchor_sizes
        )
        self._anchor_gen = _ReferenceAnchorGenerator(sizes, ratios)

        self.roi_heads = _RoIHeads(
            in_channels=C,
            roi_size=config.roi_size,
            representation_size=config.roi_representation_size,
            num_classes=config.num_classes,
        )

    # ------------------------------------------------------------------
    # RPN proposal generation (inference)
    # ------------------------------------------------------------------

    def _rpn_proposals(
        self,
        logits: list[Tensor],
        deltas: list[Tensor],
        anchors: list[Tensor],
        image_size: tuple[int, int],
    ) -> list[Tensor]:
        """Decode + filter RPN predictions into per-image proposals.

        Mirrors the reference ``filter_proposals``: per-level top-k pre-NMS
        selection, decode, clip, small-box removal, score threshold, then
        per-level (class-offset) NMS and a post-NMS top-k.
        """
        B = int(logits[0].shape[0])
        cfg = self._cfg
        dev = logits[0].device.type

        # Flatten each level to (B, H*W*A, C) in spatial-major / anchor-minor
        # order so it lines up with the anchor grid.
        per_level_scores: list[Tensor] = []
        per_level_deltas: list[Tensor] = []
        level_sizes: list[int] = []
        for lg, dl in zip(logits, deltas):
            A = int(lg.shape[1])
            fH = int(lg.shape[2])
            fW = int(lg.shape[3])
            sc = lg.permute(0, 2, 3, 1).reshape(B, fH * fW * A)  # (B, N_l)
            dlf = (
                dl.reshape(B, A, 4, fH, fW)
                .permute(0, 3, 4, 1, 2)
                .reshape(B, fH * fW * A, 4)
            )
            per_level_scores.append(sc)
            per_level_deltas.append(dlf)
            level_sizes.append(fH * fW * A)

        proposals: list[Tensor] = []
        for b in range(B):
            boxes_parts: list[Tensor] = []
            scores_parts: list[Tensor] = []
            levels_parts: list[Tensor] = []
            for lvl in range(len(logits)):
                sc_b = per_level_scores[lvl][b]  # (N_l,)
                dl_b = per_level_deltas[lvl][b]  # (N_l, 4)
                anc = anchors[lvl]  # (N_l, 4)

                K = min(cfg.rpn_pre_nms_top_n, int(sc_b.shape[0]))
                top_idx = lucid.argsort(-sc_b)[:K]
                sc_t = sc_b[top_idx]
                dl_t = dl_b[top_idx]
                anc_t = anc[top_idx]

                props = decode_boxes(dl_t, anc_t, (1.0, 1.0, 1.0, 1.0))
                props = clip_boxes_to_image(props, image_size)
                boxes_parts.append(props)
                scores_parts.append(F.sigmoid(sc_t))
                levels_parts.append(
                    lucid.full((int(sc_t.shape[0]),), float(lvl), device=dev)
                )

            boxes_b = lucid.cat(boxes_parts, dim=0)
            scores_b = lucid.cat(scores_parts, dim=0)
            lvls_b = lucid.cat(levels_parts, dim=0)

            keep_small = remove_small_boxes(boxes_b, cfg.rpn_min_size)
            if int(keep_small.shape[0]) == 0:
                proposals.append(lucid.zeros((0, 4), device=dev))
                continue
            boxes_b = boxes_b[keep_small]
            scores_b = scores_b[keep_small]
            lvls_b = lvls_b[keep_small]

            if cfg.rpn_score_thresh > 0.0:
                mask = [
                    i
                    for i in range(int(scores_b.shape[0]))
                    if float(scores_b[i].item()) >= cfg.rpn_score_thresh
                ]
                if not mask:
                    proposals.append(lucid.zeros((0, 4), device=dev))
                    continue
                m_t = lucid.tensor(mask, device=dev).long()
                boxes_b = boxes_b[m_t]
                scores_b = scores_b[m_t]
                lvls_b = lvls_b[m_t]

            keep = batched_nms(boxes_b, scores_b, lvls_b, cfg.rpn_nms_thresh)
            keep = keep[: cfg.rpn_post_nms_top_n]
            proposals.append(boxes_b[keep])

        return proposals

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        proposals: list[Tensor] | None = None,
    ) -> ObjectDetectionOutput:
        """Run Faster R-CNN on a (pre-processed) image batch.

        Args:
            x:         (B, C, H, W) resized + normalised image batch.
            proposals: Optional precomputed per-image proposals.  When
                       ``None`` the RPN generates them.

        Returns:
            ``ObjectDetectionOutput`` with raw RoI-head outputs:
              ``logits``     : (Σ proposals, num_classes) class logits.
              ``pred_boxes`` : (Σ proposals, num_classes, 4) per-class boxes.
              ``proposals``  : per-image proposal tensors.
        """
        iH = int(x.shape[2])
        iW = int(x.shape[3])
        dev = x.device.type

        # 1. Backbone + FPN → [P2, P3, P4, P5, pool]
        features = cast(list[Tensor], self.backbone(x))

        # 2. RPN → per-image proposals (when not supplied)
        if proposals is None:
            logits, deltas = self.rpn.head.forward(features)
            anchors = self._anchor_gen.forward(features, list(self._strides))
            proposals = self._rpn_proposals(logits, deltas, anchors, (iH, iW))

        # 3. MultiScale RoI Align over the four FPN detection levels (P2-P5).
        det_feats = features[:4]
        det_scales = [1.0 / float(s) for s in self._strides[:4]]
        roi_feats = multiscale_roi_align(
            det_feats,
            proposals,
            output_size=self._cfg.roi_size,
            spatial_scales=det_scales,
            sampling_ratio=self._cfg.roi_sampling_ratio,
            canonical_scale=self._cfg.canonical_scale,
            canonical_level=self._cfg.canonical_level,
        )

        # 4. RoI head → class logits + per-class box deltas
        K = self._cfg.num_classes
        if int(roi_feats.shape[0]) > 0:
            class_logits, box_deltas = self.roi_heads(roi_feats)
        else:
            class_logits = lucid.zeros((0, K), device=dev)
            box_deltas = lucid.zeros((0, K * 4), device=dev)

        # 5. Decode per-class boxes → (N, K, 4)
        pred_boxes = self._decode_per_class(proposals, box_deltas, (iH, iW))

        return ObjectDetectionOutput(
            logits=class_logits,
            pred_boxes=pred_boxes,
            loss=None,
            proposals=tuple(proposals),
        )

    def _decode_per_class(
        self,
        proposals: list[Tensor],
        box_deltas: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Decode ``(N, K*4)`` deltas against proposals → ``(N, K, 4)`` boxes."""
        dev = box_deltas.device.type
        K = self._cfg.num_classes
        N = int(box_deltas.shape[0])
        if N == 0:
            return lucid.zeros((0, K, 4), device=dev)
        flat_props = lucid.cat([p for p in proposals if int(p.shape[0]) > 0], dim=0)
        deltas_3d = box_deltas.reshape(N, K, 4)
        per_class: list[Tensor] = []
        for c in range(K):
            boxes_c = decode_boxes(
                deltas_3d[:, c, :], flat_props, self._cfg.bbox_reg_weights
            )
            per_class.append(clip_boxes_to_image(boxes_c, image_size))
        return lucid.stack(per_class, dim=1)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_sizes: list[tuple[int, int]] | None = None,
        proposals: list[Tensor] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Softmax → per-class score filter → per-class NMS → top-k.

        Mirrors the reference ``postprocess_detections``: drops the
        background class (slot 0), score-thresholds, removes empty boxes,
        runs per-class NMS, and keeps the top ``max_detections`` scores.

        Parameters
        ----------
        output : ObjectDetectionOutput
            Raw RoI-head outputs from :meth:`forward`.
        image_sizes : list of (H, W), optional
            Unused (boxes are already clipped); accepted for API symmetry
            with other detectors.
        proposals : list of Tensor, optional
            Per-image proposals; falls back to ``output.proposals``.

        Returns
        -------
        list of dict
            One dict per image with ``"boxes"`` ``(D, 4)``, ``"scores"``
            ``(D,)``, ``"labels"`` ``(D,)`` int64.
        """
        if proposals is None:
            if output.proposals is None:
                raise ValueError(
                    "postprocess() needs proposals — pass them explicitly or "
                    "call forward() first so the output carries them."
                )
            proposals = list(output.proposals)

        logits = output.logits
        pred_boxes = output.pred_boxes  # (N, K, 4)
        cfg = self._cfg
        dev = logits.device.type
        results: list[dict[str, Tensor]] = []
        offset = 0

        for props in proposals:
            N_i = int(props.shape[0])
            lg_i = logits[offset : offset + N_i]
            bx_i = pred_boxes[offset : offset + N_i]
            offset += N_i

            if N_i == 0:
                results.append(self._empty_det(dev))
                continue

            scores_i = F.softmax(lg_i, dim=-1)  # (N_i, K)
            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(1, cfg.num_classes):  # skip background slot 0
                sc_c_all = scores_i[:, c]
                bx_class = bx_i[:, c, :]  # (N_i, 4)
                mask = [
                    i
                    for i in range(N_i)
                    if float(sc_c_all[i].item()) > cfg.score_thresh
                ]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask, device=dev).long()
                sc_c = sc_c_all[mask_t]
                bx_c = bx_class[mask_t]
                keep = nms(bx_c, sc_c, cfg.nms_thresh)
                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(
                    lucid.full((int(keep.shape[0]),), float(c), device=dev)
                )

            if not keep_boxes:
                results.append(self._empty_det(dev))
                continue

            all_b = lucid.cat(keep_boxes, dim=0)
            all_s = lucid.cat(keep_scores, dim=0)
            all_l = lucid.cat(keep_labels, dim=0)
            order = lucid.argsort(-all_s)[: cfg.max_detections]
            results.append(
                {
                    "boxes": all_b[order],
                    "scores": all_s[order],
                    "labels": all_l[order].long(),
                }
            )
        return results

    @staticmethod
    def _empty_det(dev: str) -> dict[str, Tensor]:
        return {
            "boxes": lucid.zeros((0, 4), device=dev),
            "scores": lucid.zeros((0,), device=dev),
            "labels": lucid.zeros((0,), device=dev).long(),
        }
