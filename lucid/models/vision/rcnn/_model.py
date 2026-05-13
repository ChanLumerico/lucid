"""R-CNN backbone and object detector (Girshick et al., 2014).

Paper: "Rich feature hierarchies for accurate object detection
        and semantic segmentation" (CVPR 2014)

Key idea:
  1. Generate ~2 000 category-independent region proposals (selective search,
     external — proposals are given as model input).
  2. Warp each proposal to a fixed roi_size × roi_size crop.
  3. Forward each crop independently through an AlexNet-style CNN to get a
     fixed-length feature vector (pool5 → 9 216-dim for 227 × 227 input).
  4. Classify with a linear softmax head; refine box coordinates with a
     separate linear regression head.

Faithfulness notes
------------------
* The original paper trains SVMs on top of frozen CNN features.  Modern
  reproductions (and the Caffe reference code) use softmax end-to-end, which
  is what we implement here.
* ``proposals`` must be supplied externally (e.g. from selective search or
  any other proposal method).  This module covers the CNN + head portion.
* Bounding-box regression targets are *class-specific* (num_classes × 4
  deltas), matching the paper's formulation.
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    batched_nms,
    box_cxcywh_to_xyxy,
    clip_boxes_to_image,
    decode_boxes,
)
from lucid.models.vision.rcnn._config import RCNNConfig


# ---------------------------------------------------------------------------
# CNN backbone (AlexNet-style, applied per RoI crop)
# ---------------------------------------------------------------------------


class _ConvFeatures(nn.Module):
    """AlexNet convolutional trunk applied to warped RoI crops.

    Input  : (N, C, roi_size, roi_size)  — N warped proposal crops
    Output : (N, 9216)                   — flattened pool5 features
                                           (for roi_size = 227)

    Architecture (matches original AlexNet / RCNN paper):
      Conv1  : 11×11, s=4, p=2 → 96ch  → 55×55
      Pool1  : 3×3, s=2         →        27×27
      Conv2  : 5×5,  p=2 → 256ch →       27×27
      Pool2  : 3×3, s=2         →        13×13
      Conv3  : 3×3,  p=1 → 384ch →       13×13
      Conv4  : 3×3,  p=1 → 384ch →       13×13
      Conv5  : 3×3,  p=1 → 256ch →       13×13
      Pool5  : 3×3, s=2         →         6× 6
      Flatten:               → 256*6*6 = 9 216
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.out_dim: int = 256 * 6 * 6  # = 9 216 for 227 × 227 input

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.features(x))
        return x.flatten(1)  # (N, 9216)


# ---------------------------------------------------------------------------
# FC head (classification + bbox regression)
# ---------------------------------------------------------------------------


class _FCHead(nn.Module):
    """Two-layer FC trunk followed by dual prediction heads.

    Input  : (N, feat_dim)   — pool5 feature per RoI
    Output : (class_logits, bbox_deltas)
               class_logits : (N, num_classes + 1)
               bbox_deltas  : (N, num_classes * 4)  — class-specific deltas

    Architecture:
      fc6   : feat_dim → 4 096, ReLU, Dropout
      fc7   : 4 096    → 4 096, ReLU, Dropout
      cls   : 4 096    → num_classes + 1   (linear)
      bbox  : 4 096    → num_classes × 4   (linear)
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc6  = nn.Linear(feat_dim, 4096)
        self.fc7  = nn.Linear(4096, 4096)
        self.drop = nn.Dropout(p=dropout)

        self.cls_head  = nn.Linear(4096, num_classes + 1)
        self.bbox_head = nn.Linear(4096, num_classes * 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        x = F.relu(cast(Tensor, self.fc6(x)))
        x = cast(Tensor, self.drop(x))
        x = F.relu(cast(Tensor, self.fc7(x)))
        x = cast(Tensor, self.drop(x))
        return cast(Tensor, self.cls_head(x)), cast(Tensor, self.bbox_head(x))


# ---------------------------------------------------------------------------
# R-CNN for Object Detection
# ---------------------------------------------------------------------------


class RCNNForObjectDetection(PretrainedModel):
    """R-CNN object detector (Girshick et al., CVPR 2014).

    Applies an AlexNet-style CNN to each warped region proposal independently,
    then predicts a class label and a bounding-box refinement for each region.

    Input contract
    --------------
    ``x``         : (B, C, H, W) batch of images.
    ``proposals`` : list of B tensors, each (N_i, 4) xyxy pixel coordinates
                    of region proposals for image i.  At inference time these
                    typically come from selective search or similar.
                    If ``None`` or empty for an image, that image contributes
                    zero detections.

    Output contract
    ---------------
    ``ObjectDetectionOutput``:
      ``logits``    : (Σ N_i, num_classes + 1) class logits (raw, pre-softmax).
      ``pred_boxes``: (Σ N_i, 4) decoded xyxy boxes after applying the
                      top-class bbox delta to each proposal.
    """

    config_class: ClassVar[type[RCNNConfig]] = RCNNConfig
    base_model_prefix: ClassVar[str] = "rcnn"

    def __init__(self, config: RCNNConfig) -> None:
        super().__init__(config)
        self._num_classes  = config.num_classes
        self._roi_size     = config.roi_size
        self._score_thresh = config.score_thresh
        self._nms_thresh   = config.nms_thresh
        self._max_det      = config.max_detections

        self.conv_features = _ConvFeatures(config.in_channels)
        self.fc_head = _FCHead(
            feat_dim=self.conv_features.out_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warp_proposals(
        self,
        img: Tensor,
        proposals: Tensor,
    ) -> Tensor:
        """Warp all proposals from one image into roi_size × roi_size crops.

        Args:
            img:       (1, C, H, W) single image.
            proposals: (N, 4) xyxy proposals in pixel coordinates.

        Returns:
            (N, C, roi_size, roi_size) warped crops.
        """
        H = int(img.shape[2])
        W = int(img.shape[3])
        S = self._roi_size
        crops: list[Tensor] = []

        for n in range(int(proposals.shape[0])):
            x1 = max(0, int(proposals[n, 0].item()))
            y1 = max(0, int(proposals[n, 1].item()))
            x2 = min(W, int(proposals[n, 2].item()))
            y2 = min(H, int(proposals[n, 3].item()))

            # Ensure positive size
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)

            crop: Tensor = img[:, :, y1:y2, x1:x2]  # (1, C, h, w)
            warped = F.interpolate(crop, size=(S, S), mode="bilinear")
            crops.append(warped)

        if not crops:
            C = int(img.shape[1])
            return lucid.zeros((0, C, S, S))
        return lucid.cat(crops, dim=0)

    def _decode_top_boxes(
        self,
        proposals: Tensor,
        bbox_deltas: Tensor,
        top_class: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Decode class-specific bbox deltas for each proposal's top class.

        Args:
            proposals:   (N, 4) xyxy input proposals.
            bbox_deltas: (N, num_classes * 4) raw regression output.
            top_class:   (N,) int index of predicted foreground class
                         (0 = background, 1..K = foreground).
            image_size:  (H, W) used for clipping.

        Returns:
            (N, 4) decoded xyxy boxes.
        """
        N = int(proposals.shape[0])
        K = self._num_classes
        decoded: list[Tensor] = []

        # bbox_deltas: (N, K*4) → (N, K, 4)
        deltas = bbox_deltas.reshape(N, K, 4)

        for n in range(N):
            cls_idx = int(top_class[n].item())
            if cls_idx == 0:
                # Background — return the original proposal
                decoded.append(proposals[n:n + 1])
            else:
                c = cls_idx - 1  # foreground class index (0-based)
                c_clamped = max(0, min(c, K - 1))
                delta_n = deltas[n:n + 1, c_clamped, :]  # (1, 4)
                box_n = decode_boxes(delta_n, proposals[n:n + 1])  # (1, 4)
                decoded.append(box_n)

        if not decoded:
            return lucid.zeros((0, 4))

        boxes = lucid.cat(decoded, dim=0)
        return clip_boxes_to_image(boxes, image_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        proposals: list[Tensor] | None = None,
    ) -> ObjectDetectionOutput:
        """Run R-CNN on a batch of images.

        Args:
            x:         (B, C, H, W) image batch.
            proposals: list of B tensors, each (N_i, 4) xyxy region proposals.
                       Pass ``None`` to return empty detections.

        Returns:
            ``ObjectDetectionOutput`` with:
              ``logits``     : (Σ N_i, num_classes + 1) raw class logits.
              ``pred_boxes`` : (Σ N_i, 4) decoded xyxy boxes (top class).
        """
        B = int(x.shape[0])
        H = int(x.shape[2])
        W = int(x.shape[3])

        if proposals is None:
            proposals = [lucid.zeros((0, 4)) for _ in range(B)]

        all_logits: list[Tensor]     = []
        all_boxes:  list[Tensor]     = []

        for b in range(B):
            props = proposals[b]          # (N_i, 4)
            N_i   = int(props.shape[0])

            if N_i == 0:
                all_logits.append(lucid.zeros((0, self._num_classes + 1)))
                all_boxes.append(lucid.zeros((0, 4)))
                continue

            img = x[b:b + 1]             # (1, C, H, W)

            # 1. Warp proposals → fixed-size crops
            crops = self._warp_proposals(img, props)  # (N_i, C, S, S)

            # 2. Extract CNN features
            feats = cast(Tensor, self.conv_features(crops))  # (N_i, 9216)

            # 3. FC head → class logits + bbox deltas
            logits_i, deltas_i = self.fc_head(feats)
            # logits_i : (N_i, num_classes + 1)
            # deltas_i : (N_i, num_classes * 4)

            # 4. Decode top-class bounding boxes
            scores_i = F.softmax(logits_i, dim=-1)  # (N_i, K+1)
            top_cls_i = lucid.argsort(-scores_i, dim=-1)[:, 0]  # (N_i,)

            boxes_i = self._decode_top_boxes(
                props, deltas_i, top_cls_i, (H, W)
            )

            all_logits.append(logits_i)
            all_boxes.append(boxes_i)

        # Concatenate across batch (flat)
        if all_logits:
            final_logits = lucid.cat(all_logits, dim=0)
            final_boxes = lucid.cat(all_boxes, dim=0)
        else:
            final_logits = lucid.zeros((0, self._num_classes + 1))
            final_boxes  = lucid.zeros((0, 4))

        return ObjectDetectionOutput(logits=final_logits, pred_boxes=final_boxes)

    # ------------------------------------------------------------------
    # Post-processing helper (call separately after forward)
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        proposals: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        """Apply score threshold + per-class NMS to raw R-CNN output.

        Args:
            output:    ``ObjectDetectionOutput`` from ``forward()``.
            proposals: Same proposal list passed to ``forward()``.

        Returns:
            List of per-image result dicts, each with:
              ``"boxes"``  : (K, 4) xyxy kept detections
              ``"scores"`` : (K,)   class confidence scores
              ``"labels"`` : (K,)   integer class indices (1-based, 0=bg)
        """
        logits    = output.logits     # (Σ N_i, num_classes + 1)
        pred_boxes = output.pred_boxes  # (Σ N_i, 4)

        results: list[dict[str, Tensor]] = []
        offset = 0

        for props in proposals:
            N_i = int(props.shape[0])
            lg_i = logits[offset: offset + N_i]      # (N_i, K+1)
            bx_i = pred_boxes[offset: offset + N_i]  # (N_i, 4)
            offset += N_i

            scores_i = F.softmax(lg_i, dim=-1)

            keep_boxes:  list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            # Per-class NMS (skip background class 0)
            for c in range(1, self._num_classes + 1):
                cls_scores = scores_i[:, c]  # (N_i,)

                # Score threshold
                mask: list[int] = [
                    i for i in range(N_i)
                    if float(cls_scores[i].item()) >= self._score_thresh
                ]
                if not mask:
                    continue

                mask_t = lucid.tensor(mask)
                sc_c    = cls_scores[mask_t]
                bx_c    = bx_i[mask_t]

                # NMS
                keep = batched_nms(
                        bx_c, sc_c,
                        lucid.zeros(int(sc_c.shape[0])),  # single class → no offset
                        self._nms_thresh,
                )
                keep = keep[:self._max_det]

                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(
                    lucid.full((int(keep.shape[0]),), float(c))
                )

            if keep_boxes:
                results.append({
                    "boxes":  lucid.cat(keep_boxes,  dim=0),
                    "scores": lucid.cat(keep_scores, dim=0),
                    "labels": lucid.cat(keep_labels, dim=0),
                })
            else:
                results.append({
                    "boxes":  lucid.zeros((0, 4)),
                    "scores": lucid.zeros((0,)),
                    "labels": lucid.zeros((0,)),
                })

        return results
