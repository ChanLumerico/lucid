"""Mask R-CNN instance segmentation model (He et al., ICCV 2017).

Paper: "Mask R-CNN"

This module implements the **ResNet-50-FPN** instance-segmentation detector
— the modern reference configuration shipped with the COCO ``box AP 37.9 /
mask AP 34.6`` checkpoint.  Mask R-CNN is Faster R-CNN plus a parallel mask
branch, so the entire backbone + FPN + RPN + box-head stack is **reused
verbatim** from :mod:`lucid.models.vision.faster_rcnn`; only the mask branch
on ``roi_heads`` is new.  The submodule layout mirrors the reference detector
so the COCO checkpoint loads strict (307 keys = Faster R-CNN's 295 + 12
mask-branch keys):

  Image (B, C, H, W)
    ↓  ResNet-50 backbone (frozen BN, eps=0) → C2, C3, C4, C5
    ↓  FPN: 1×1 lateral + top-down nearest add + 3×3 output + LastLevelMaxPool
  [P2, P3, P4, P5, pool]
    ├─ RPN head → per-level top-k → decode → clip → NMS 0.7 → 1000 proposals
    │
    ├─ MultiScale RoI Align (7×7) → TwoMLPHead → FastRCNNPredictor
    │    → softmax, per-class decode, clip, NMS 0.5, top-100 detections
    │
    └─ (on the kept detections)
       MultiScale RoI Align (14×14) over P2-P5 (same FPN level assignment)
         ↓  MaskRCNNHeads: 4 × (Conv3×3(256→256, pad 1) + ReLU)
         ↓  MaskRCNNPredictor: ConvTranspose2d 2×2 s2 (256→256) + ReLU
         ↓                     → Conv1×1 (256→num_classes)
       per-detection mask logits (N, num_classes, 28, 28) — gather the
       predicted class channel and sigmoid for the final per-instance mask.

Faithfulness notes
------------------
* Backbone / FPN / RPN / box-head are byte-identical to the shipped
  Faster R-CNN — the shared 295 keys map the same way.
* The mask branch's RoI Align uses ``output_size = 14`` (vs 7 for the box
  head), ``sampling_ratio = 2``, ``aligned = False``, and the **same**
  canonical FPN level assignment as the box head.
* ``mask_head`` is a ``MaskRCNNHeads`` — four blocks, each a
  ``Sequential(Conv2d 3×3, ReLU)`` so the keys read
  ``roi_heads.mask_head.{i}.0.weight``.
* ``mask_predictor`` is a ``MaskRCNNPredictor`` — ``conv5_mask``
  (ConvTranspose2d 2×2 stride 2) + ReLU + ``mask_fcn_logits`` (Conv1×1).
* The detector accepts an already-resized + normalised image batch (the
  reference ``GeneralizedRCNNTransform`` normalisation / resize is a
  :class:`~lucid.utils.transforms.Detection` preset that runs outside the
  model).
"""

from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import InstanceSegmentationOutput
from lucid.models._utils._detection import (
    _ReferenceAnchorGenerator,
    multiscale_roi_align,
    nms,
)
from lucid.models.vision.faster_rcnn._model import (
    _BackboneWithFPN,
    _FastRCNNPredictor,
    _RegionProposalNetwork,
    _TwoMLPHead,
)
from lucid.models.vision.faster_rcnn._model import (
    FasterRCNNForObjectDetection as _FasterRCNNForObjectDetection,
)
from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig

# ---------------------------------------------------------------------------
# Mask branch building blocks
# ---------------------------------------------------------------------------
# Key prefixes mirror the reference detector verbatim:
#   roi_heads.mask_head.{i}.0.{weight,bias}        (MaskRCNNHeads)
#   roi_heads.mask_predictor.conv5_mask.{weight,bias}
#   roi_heads.mask_predictor.mask_fcn_logits.{weight,bias}


@final
class _MaskRCNNHeads(nn.Sequential):
    """Reference ``MaskRCNNHeads``: four ``Conv3×3 + ReLU`` blocks.

    Subclasses ``Sequential`` (like the reference) so each block is a
    direct integer child; every block is itself a
    ``Sequential(Conv2d, ReLU)`` (the reference ``Conv2dNormActivation``
    with no norm), so the conv is index ``.0`` and the state-dict keys
    read ``mask_head.{i}.0.weight`` / ``mask_head.{i}.0.bias`` (the ReLU
    at ``.1`` is parameter-free and contributes no key).
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_convs: int) -> None:
        blocks: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(num_convs):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            ch_in = hidden_channels
        super().__init__(*blocks)


@final
class _MaskRCNNPredictor(nn.Module):
    """Reference ``MaskRCNNPredictor``: deconv-upsample then 1×1 logits.

    ``conv5_mask`` upsamples ``14×14 → 28×28`` (ConvTranspose2d 2×2
    stride 2), a parameter-free ReLU follows, then ``mask_fcn_logits``
    (Conv1×1) emits one logit map per class.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, num_classes: int
    ) -> None:
        super().__init__()
        self.conv5_mask = nn.ConvTranspose2d(in_channels, hidden_channels, 2, stride=2)
        self.mask_fcn_logits = nn.Conv2d(hidden_channels, num_classes, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = F.relu(cast(Tensor, self.conv5_mask(x)))
        return cast(Tensor, self.mask_fcn_logits(x))


# ---------------------------------------------------------------------------
# RoI heads — box branch (reused) + mask branch (new)
# ---------------------------------------------------------------------------


@final
class _MaskRoIHeads(nn.Module):
    """RoI heads container: ``box_head`` + ``box_predictor`` (reused from
    Faster R-CNN) plus the new ``mask_head`` + ``mask_predictor``.

    The box-branch submodule names (``box_head`` = TwoMLPHead,
    ``box_predictor`` = FastRCNNPredictor) are byte-identical to
    :class:`~lucid.models.vision.faster_rcnn._model._RoIHeads`, so the 295
    shared keys map the same way.  The mask branch adds the 12 keys
    described in the module docstring.
    """

    def __init__(
        self,
        in_channels: int,
        roi_size: int,
        representation_size: int,
        num_classes: int,
        mask_hidden_channels: int,
        mask_num_convs: int,
        mask_predictor_hidden: int,
    ) -> None:
        super().__init__()
        # Box branch — identical to the Faster R-CNN RoI heads.
        self.box_head = _TwoMLPHead(
            in_channels * roi_size * roi_size, representation_size
        )
        self.box_predictor = _FastRCNNPredictor(representation_size, num_classes)
        # Mask branch.
        self.mask_head = _MaskRCNNHeads(
            in_channels, mask_hidden_channels, mask_num_convs
        )
        self.mask_predictor = _MaskRCNNPredictor(
            mask_hidden_channels, mask_predictor_hidden, num_classes
        )

    @override
    def forward(self, roi_feats: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        feats = cast(Tensor, self.box_head(roi_feats))
        return cast(tuple[Tensor, Tensor], self.box_predictor(feats))

    def predict_masks(self, mask_feats: Tensor) -> Tensor:
        """Run the mask branch on RoI-aligned crops → ``(N, K, 28, 28)``."""
        x = cast(Tensor, self.mask_head(mask_feats))
        return cast(Tensor, self.mask_predictor(x))


# ---------------------------------------------------------------------------
# Mask R-CNN
# ---------------------------------------------------------------------------


class MaskRCNNForObjectDetection(PretrainedModel):
    r"""Mask R-CNN with a ResNet-50-FPN backbone (He et al., ICCV 2017).

    The two-stage instance-segmentation detector in its modern reference
    configuration: Faster R-CNN's ResNet-50-FPN backbone, RPN, and Fast
    R-CNN box head, plus a parallel FCN mask branch on the RoI heads.  The
    submodule layout mirrors the reference detector so the COCO ``box AP
    37.9 / mask AP 34.6`` checkpoint loads strict (307 keys) and reproduces
    inference.

    Parameters
    ----------
    config : MaskRCNNConfig
        Frozen architecture spec.  Use the
        :func:`mask_rcnn_resnet50_fpn` factory for the COCO-pretrained
        configuration (``num_classes = 91``).

    Attributes
    ----------
    config : MaskRCNNConfig
        Stored copy of the config that built this model.
    backbone : _BackboneWithFPN
        ResNet-50 ``body`` + ``fpn`` producing five feature maps
        ``[P2, P3, P4, P5, pool]`` at strides ``4/8/16/32/64`` (reused
        from Faster R-CNN).
    rpn : _RegionProposalNetwork
        Proposal head shared across all pyramid levels (reused).
    roi_heads : _MaskRoIHeads
        ``box_head`` + ``box_predictor`` (reused) plus ``mask_head``
        (MaskRCNNHeads) + ``mask_predictor`` (MaskRCNNPredictor).

    Notes
    -----
    See He et al., "Mask R-CNN", ICCV 2017 (arXiv:1703.06870), Ren et al.,
    "Faster R-CNN", NeurIPS 2015, and Lin et al., "Feature Pyramid Networks
    for Object Detection", CVPR 2017.  The model expects an already resized
    + normalised image batch; final per-instance detections + masks come
    from :meth:`postprocess`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask_rcnn import mask_rcnn_resnet50_fpn
    >>> model = mask_rcnn_resnet50_fpn()
    >>> model.eval()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape[-1]   # num_classes
    91
    >>> out.pred_masks.shape[-2:]
    (28, 28)
    """

    config_class: ClassVar[type[MaskRCNNConfig]] = MaskRCNNConfig
    base_model_prefix: ClassVar[str] = "mask_rcnn"

    # FPN level strides: P2..P5 then the pool level.
    _strides: ClassVar[tuple[int, ...]] = (4, 8, 16, 32, 64)

    def __init__(self, config: MaskRCNNConfig) -> None:
        super().__init__(config)
        self._cfg = config

        self.backbone = _BackboneWithFPN(
            in_channels=config.in_channels,
            layers=config.backbone_layers,
            fpn_out_channels=config.fpn_out_channels,
            bn_eps=config.backbone_bn_eps,
        )
        C = self.backbone.out_channels

        num_anchors = len(config.rpn_anchor_ratios)
        self._num_anchors = num_anchors
        self.rpn = _RegionProposalNetwork(C, num_anchors)

        sizes: tuple[tuple[int, ...], ...] = tuple(
            (s,) for s in config.rpn_anchor_sizes
        )
        ratios: tuple[tuple[float, ...], ...] = tuple(
            tuple(config.rpn_anchor_ratios) for _ in config.rpn_anchor_sizes
        )
        # Reuse the Faster R-CNN anchor generator verbatim.
        self._anchor_gen = _ReferenceAnchorGenerator(sizes, ratios)

        self.roi_heads = _MaskRoIHeads(
            in_channels=C,
            roi_size=config.roi_det_size,
            representation_size=config.roi_representation,
            num_classes=config.num_classes,
            mask_hidden_channels=config.mask_hidden_channels,
            mask_num_convs=config.mask_num_convs,
            mask_predictor_hidden=config.mask_predictor_hidden,
        )

    # ------------------------------------------------------------------
    # RPN proposal generation (inference) — delegate to the shared impl
    # ------------------------------------------------------------------

    def _rpn_proposals(
        self,
        logits: list[Tensor],
        deltas: list[Tensor],
        anchors: list[Tensor],
        image_size: tuple[int, int],
    ) -> list[Tensor]:
        """Decode + filter RPN predictions into per-image proposals.

        Reuses Faster R-CNN's proposal layer verbatim — the box-branch +
        RPN behaviour must be byte-identical so the shared keys produce
        identical proposals.
        """
        return _FasterRCNNForObjectDetection._rpn_proposals(
            cast(_FasterRCNNForObjectDetection, self),
            logits,
            deltas,
            anchors,
            image_size,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        proposals: list[Tensor] | None = None,
    ) -> InstanceSegmentationOutput:
        """Run Mask R-CNN on a (pre-processed) image batch.

        Args:
            x:         (B, C, H, W) resized + normalised image batch.
            proposals: Optional precomputed per-image proposals.  When
                       ``None`` the RPN generates them.

        Returns:
            ``InstanceSegmentationOutput`` with raw RoI-head outputs:
              ``logits``     : (Σ proposals, num_classes) class logits.
              ``pred_boxes`` : (Σ proposals, num_classes, 4) per-class boxes.
              ``pred_masks`` : (Σ proposals, num_classes, 28, 28) mask logits.
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
            output_size=self._cfg.roi_det_size,
            spatial_scales=det_scales,
            sampling_ratio=self._cfg.roi_sampling_ratio,
            canonical_scale=self._cfg.canonical_scale,
            canonical_level=self._cfg.canonical_level,
        )

        # 4. Box RoI head → class logits + per-class box deltas
        K = self._cfg.num_classes
        if int(roi_feats.shape[0]) > 0:
            class_logits, box_deltas = self.roi_heads(roi_feats)
        else:
            class_logits = lucid.zeros((0, K), device=dev)
            box_deltas = lucid.zeros((0, K * 4), device=dev)

        # 5. Decode per-class boxes → (N, K, 4)
        pred_boxes = self._decode_per_class(proposals, box_deltas, (iH, iW))

        # 6. Mask RoI Align (14×14) over the SAME proposals + level assignment.
        mask_feats = multiscale_roi_align(
            det_feats,
            proposals,
            output_size=self._cfg.roi_mask_size,
            spatial_scales=det_scales,
            sampling_ratio=self._cfg.roi_sampling_ratio,
            canonical_scale=self._cfg.canonical_scale,
            canonical_level=self._cfg.canonical_level,
        )
        mH = self._cfg.roi_mask_size * 2  # deconv stride-2 upsample (14 → 28)
        if int(mask_feats.shape[0]) > 0:
            mask_logits = self.roi_heads.predict_masks(mask_feats)
        else:
            mask_logits = lucid.zeros((0, K, mH, mH), device=dev)

        return InstanceSegmentationOutput(
            logits=class_logits,
            pred_boxes=pred_boxes,
            pred_masks=mask_logits,
        )

    def _decode_per_class(
        self,
        proposals: list[Tensor],
        box_deltas: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Decode ``(N, K*4)`` deltas against proposals → ``(N, K, 4)`` boxes."""
        return _FasterRCNNForObjectDetection._decode_per_class(
            cast(_FasterRCNNForObjectDetection, self),
            proposals,
            box_deltas,
            image_size,
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: InstanceSegmentationOutput,
        image_sizes: list[tuple[int, int]] | None = None,
        proposals: list[Tensor] | None = None,
        features: list[Tensor] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Box post-process → per-detection mask gather.

        Mirrors the reference inference flow: run the box branch
        post-processing (softmax → per-class score filter → per-class NMS →
        top-``max_detections``), then sigmoid the per-RoI mask logits and
        gather the channel for each detection's predicted class.

        Parameters
        ----------
        output : InstanceSegmentationOutput
            Raw RoI-head outputs from :meth:`forward`.
        image_sizes : list of (H, W), optional
            Unused (boxes are already clipped); accepted for API symmetry.
        proposals : list of Tensor, optional
            Per-image proposals the RoI features were sampled from
            (required — pass the same list :meth:`forward` used).
        features : list of Tensor, optional
            Multi-scale FPN feature maps; accepted for API symmetry with
            the detector-stage post-processor but not consumed here (mask
            logits are already gathered onto the proposals).

        Returns
        -------
        list of dict
            One dict per image with ``"boxes"`` ``(D, 4)``, ``"scores"``
            ``(D,)``, ``"labels"`` ``(D,)`` int64, and ``"masks"``
            ``(D, 1, 28, 28)`` sigmoid mask probabilities (the channel of
            each detection's predicted class).
        """
        if proposals is None:
            raise ValueError(
                "postprocess() needs the per-image proposals used in forward()."
            )

        logits = output.logits
        pred_boxes = output.pred_boxes  # (N, K, 4)
        pred_masks = output.pred_masks  # (N, K, 28, 28)
        cfg = self._cfg
        dev = logits.device.type
        results: list[dict[str, Tensor]] = []
        offset = 0

        for props in proposals:
            N_i = int(props.shape[0])
            lg_i = logits[offset : offset + N_i]
            bx_i = pred_boxes[offset : offset + N_i]
            mk_i = pred_masks[offset : offset + N_i]  # (N_i, K, 28, 28)
            offset += N_i

            if N_i == 0:
                results.append(self._empty_det(dev))
                continue

            scores_i = F.softmax(lg_i, dim=-1)  # (N_i, K)
            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []
            keep_masks: list[Tensor] = []

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
                # Mask channel for class c (sigmoid → probability).
                mk_c = F.sigmoid(mk_i[mask_t][:, c, :, :])  # (k, 28, 28)
                keep = nms(bx_c, sc_c, cfg.nms_thresh)
                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(
                    lucid.full((int(keep.shape[0]),), float(c), device=dev)
                )
                keep_masks.append(mk_c[keep].unsqueeze(1))  # (k, 1, 28, 28)

            if not keep_boxes:
                results.append(self._empty_det(dev))
                continue

            all_b = lucid.cat(keep_boxes, dim=0)
            all_s = lucid.cat(keep_scores, dim=0)
            all_l = lucid.cat(keep_labels, dim=0)
            all_m = lucid.cat(keep_masks, dim=0)
            order = lucid.argsort(-all_s)[: cfg.max_detections]
            results.append(
                {
                    "boxes": all_b[order],
                    "scores": all_s[order],
                    "labels": all_l[order].long(),
                    "masks": all_m[order],
                }
            )
        return results

    @staticmethod
    def _empty_det(dev: str) -> dict[str, Tensor]:
        return {
            "boxes": lucid.zeros((0, 4), device=dev),
            "scores": lucid.zeros((0,), device=dev),
            "labels": lucid.zeros((0,), device=dev).long(),
            "masks": lucid.zeros((0, 1, 28, 28), device=dev),
        }
