"""Unit tests for Wave 3d detection models — CPU + Metal parametrized.

Covers:
  R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN, DETR,
  EfficientDet D0, YOLO v1/v2/v3/v4

For each model we run **one forward pass per device** and check shape /
type / loss / deterministic self-consistency in a single test method so
heavy models (Mask R-CNN, EfficientDet, DETR) don't get re-instantiated
for every assertion.

Tests are parametrized over the ``device`` fixture so they run on both
the CPU (Accelerate) and Metal (MLX) compute streams.
"""

import os
import unittest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import InstanceSegmentationOutput, ObjectDetectionOutput

# ─────────────────────────────────────────────────────────────────────────────
# Tiny inputs to keep tests fast
# ─────────────────────────────────────────────────────────────────────────────
_B = 1
_H = 128  # multiple of 32 for stride-32 backbones
_W = 128
_C = 3
_H_ED = 256  # EfficientDet 5-level BiFPN needs ≥256 (P7 = H/128 ≥ 2)
_W_ED = 256


def _img(device: str, h: int = _H, w: int = _W, ch: int = _C) -> Tensor:
    lucid.manual_seed(0)
    return lucid.randn((_B, ch, h, w), device=device)


def _build(factory, device: str):
    """Instantiate `factory`, switch to eval, move to device."""
    m = factory()
    m.eval()
    return m.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.rcnn import rcnn, RCNNForObjectDetection

        m = _build(rcnn, device)
        assert isinstance(m, RCNNForObjectDetection)

        proposals = [
            lucid.tensor(
                [[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]],
                device=device,
            )
        ]
        x = _img(device)
        out = m(x, proposals)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 2
        assert int(out.logits.shape[0]) == 2
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        # Self-consistency
        out2 = m(x, proposals)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Fast R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestFastRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fast_rcnn import (
            fast_rcnn,
            FastRCNNForObjectDetection,
        )

        m = _build(fast_rcnn, device)
        assert isinstance(m, FastRCNNForObjectDetection)

        proposals = [
            lucid.tensor(
                [[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]],
                device=device,
            )
        ]
        x = _img(device)
        out = m(x, proposals)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.shape[0]) == 2
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        out2 = m(x, proposals)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestFasterRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.faster_rcnn import (
            faster_rcnn_resnet50_fpn,
            FasterRCNNForObjectDetection,
        )

        # Reduce RPN top-N to keep the python NMS loop tractable on metal.
        m = _build(
            lambda: faster_rcnn_resnet50_fpn(
                num_classes=91, rpn_pre_nms_top_n=200, rpn_post_nms_top_n=100
            ),
            device,
        )
        assert isinstance(m, FasterRCNNForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        # ResNet-50-FPN: per-RoI logits (N, num_classes), per-class boxes
        # (N, num_classes, 4).
        assert int(out.logits.ndim) == 2
        assert int(out.logits.shape[-1]) == 91
        assert int(out.pred_boxes.ndim) == 3
        assert int(out.pred_boxes.shape[-2]) == 91
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.proposals is not None
        assert out.loss is None

        out2 = m(x)
        assert out.logits.shape == out2.logits.shape
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5

        # postprocess returns one dict per image with boxes/scores/labels.
        dets = m.postprocess(out, image_sizes=[(_H, _W)])
        assert len(dets) == _B
        assert set(dets[0].keys()) == {"boxes", "scores", "labels"}


class TestFasterRCNNTopology:
    """The rebuilt detector mirrors the reference ResNet-50-FPN key layout."""

    def test_reference_key_layout(self) -> None:
        from lucid.models.vision.faster_rcnn import faster_rcnn_resnet50_fpn

        m = faster_rcnn_resnet50_fpn(num_classes=91)
        keys = set(m.state_dict().keys())
        assert len(keys) == 295
        # Backbone body (ResNet) + FPN.
        assert "backbone.body.conv1.weight" in keys
        assert "backbone.body.layer4.2.bn3.running_mean" in keys
        assert "backbone.fpn.inner_blocks.0.0.weight" in keys
        assert "backbone.fpn.layer_blocks.3.0.weight" in keys
        # RPN head.
        assert "rpn.head.conv.0.0.weight" in keys
        assert "rpn.head.cls_logits.weight" in keys
        assert "rpn.head.bbox_pred.weight" in keys
        # RoI heads.
        assert "roi_heads.box_head.fc6.weight" in keys
        assert "roi_heads.box_predictor.cls_score.weight" in keys
        assert "roi_heads.box_predictor.bbox_pred.weight" in keys
        # Frozen BN: no num_batches_tracked anywhere.
        assert not any(k.endswith("num_batches_tracked") for k in keys)

    def test_frozen_bn_eps_zero(self) -> None:
        from lucid.models.vision.faster_rcnn import faster_rcnn_resnet50_fpn

        m = faster_rcnn_resnet50_fpn(num_classes=91)
        # Reference detection FrozenBatchNorm2d uses eps = 0.
        assert float(m.backbone.body.bn1.eps) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Mask R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.mask_rcnn import (
            mask_rcnn,
            MaskRCNNForObjectDetection,
        )

        # Aggressively shrink RPN proposals to keep python-NMS tractable
        m = _build(
            lambda: mask_rcnn(rpn_pre_nms_top_n=100, rpn_post_nms_top_n=50),
            device,
        )
        assert isinstance(m, MaskRCNNForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, InstanceSegmentationOutput)
        # Raw RoI-head outputs: per-proposal class logits (N, num_classes),
        # per-class boxes (N, num_classes, 4), per-class masks (N, K, 28, 28).
        assert int(out.logits.ndim) == 2
        assert int(out.logits.shape[-1]) == 91  # COCO num_classes (incl. bg)
        assert int(out.pred_boxes.ndim) == 3
        assert int(out.pred_boxes.shape[-1]) == 4
        assert int(out.pred_masks.ndim) == 4
        assert int(out.pred_masks.shape[1]) == 91  # one mask per class
        assert int(out.pred_masks.shape[-1]) == 28
        assert int(out.pred_masks.shape[-2]) == 28
        assert int(out.logits.shape[0]) == int(out.pred_masks.shape[0])
        assert int(out.logits.shape[0]) == int(out.pred_boxes.shape[0])
        assert out.loss is None


class TestMaskRCNNTopology:
    """The rebuilt detector mirrors the reference ResNet-50-FPN key layout."""

    def test_reference_key_layout(self) -> None:
        from lucid.models.vision.mask_rcnn import mask_rcnn_resnet50_fpn

        m = mask_rcnn_resnet50_fpn(num_classes=91)
        keys = set(m.state_dict().keys())
        # 295 shared Faster R-CNN keys + 12 mask-branch keys.
        assert len(keys) == 307
        # Shared backbone / FPN / RPN / box-head keys (identity-mapped).
        assert "backbone.body.conv1.weight" in keys
        assert "backbone.fpn.inner_blocks.0.0.weight" in keys
        assert "rpn.head.conv.0.0.weight" in keys
        assert "roi_heads.box_head.fc6.weight" in keys
        assert "roi_heads.box_predictor.cls_score.weight" in keys
        # Mask branch keys.
        for i in range(4):
            assert f"roi_heads.mask_head.{i}.0.weight" in keys
            assert f"roi_heads.mask_head.{i}.0.bias" in keys
        assert "roi_heads.mask_predictor.conv5_mask.weight" in keys
        assert "roi_heads.mask_predictor.mask_fcn_logits.weight" in keys
        # Frozen BN: no num_batches_tracked anywhere.
        assert not any(k.endswith("num_batches_tracked") for k in keys)

    def test_frozen_bn_eps_zero(self) -> None:
        from lucid.models.vision.mask_rcnn import mask_rcnn_resnet50_fpn

        m = mask_rcnn_resnet50_fpn(num_classes=91)
        # Reference detection FrozenBatchNorm2d uses eps = 0.
        assert float(m.backbone.body.bn1.eps) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DETR
# ─────────────────────────────────────────────────────────────────────────────


class TestDETR:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.detr import detr_resnet50, DETRForObjectDetection

        # Slim transformer for test speed (paper default: 6/6 layers, d=256)
        m = _build(
            lambda: detr_resnet50(
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_queries=20,
            ),
            device,
        )
        assert isinstance(m, DETRForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        cfg = m._cfg
        assert tuple(out.logits.shape) == (_B, cfg.num_queries, cfg.num_classes + 1)
        assert tuple(out.pred_boxes.shape) == (_B, cfg.num_queries, 4)
        # Box coords in [0, 1]
        assert float(out.pred_boxes.min().item()) >= 0.0
        assert float(out.pred_boxes.max().item()) <= 1.0
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestDETRResNet101:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.detr import detr_resnet101

        m = _build(
            lambda: detr_resnet101(
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_queries=20,
            ),
            device,
        )
        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        cfg = m._cfg
        assert tuple(out.logits.shape) == (_B, cfg.num_queries, cfg.num_classes + 1)


# ─────────────────────────────────────────────────────────────────────────────
# EfficientDet D0
# ─────────────────────────────────────────────────────────────────────────────


class TestEfficientDetD0:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.efficientdet import (
            efficientdet_d0,
            EfficientDetForObjectDetection,
        )

        m = _build(efficientdet_d0, device)
        assert isinstance(m, EfficientDetForObjectDetection)

        x = _img(device, h=_H_ED, w=_W_ED)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert int(out.logits.shape[1]) > 0
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv1
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV1:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v1, YOLOV1ForObjectDetection

        m = _build(yolo_v1, device)
        assert isinstance(m, YOLOV1ForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        S = m.config.split_size
        B_boxes = m.config.num_boxes
        C = m.config.num_classes
        expected = S * S * B_boxes
        assert tuple(out.logits.shape) == (_B, expected, C)
        assert tuple(out.pred_boxes.shape) == (_B, expected, 4)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5

    def test_tiny_variant(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v1_tiny

        m = _build(yolo_v1_tiny, device)
        out = m(_img(device))
        assert isinstance(out, ObjectDetectionOutput)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv2
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV2:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v2, YOLOV2ForObjectDetection

        m = _build(yolo_v2, device)
        assert isinstance(m, YOLOV2ForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv3
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV3:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v3, YOLOV3ForObjectDetection

        m = _build(yolo_v3, device)
        assert isinstance(m, YOLOV3ForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert int(out.logits.shape[1]) == int(out.pred_boxes.shape[1])
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5

    def test_tiny_variant(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v3_tiny

        m = _build(yolo_v3_tiny, device)
        out = m(_img(device))
        assert isinstance(out, ObjectDetectionOutput)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv4
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV4:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v4, YOLOV4ForObjectDetection

        m = _build(yolo_v4, device)
        assert isinstance(m, YOLOV4ForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Registry smoke-tests (device-independent)
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectionRegistry:
    def test_detection_models_registered(self) -> None:
        import lucid.models as M

        det_models = M.list_models(task="object-detection")
        expected = [
            "rcnn",
            "fast_rcnn",
            "faster_rcnn",
            "faster_rcnn_resnet50_fpn",
            "mask_rcnn",
            "mask_rcnn_resnet50_fpn",
            "detr_resnet50",
            "detr_resnet101",
            "efficientdet_d0",
            "efficientdet_d7",
            "yolo_v1",
            "yolo_v1_tiny",
            "yolo_v2",
            "yolo_v2_tiny",
            "yolo_v3",
            "yolo_v3_tiny",
            "yolo_v4",
        ]
        for name in expected:
            assert name in det_models, f"{name!r} missing from registry"

    def test_create_model_api(self) -> None:
        import lucid.models as M

        m = M.create_model("detr_resnet50")
        assert m is not None


class TestRoIAlign:
    """RoIAlign correctness — sub-bin sampling_ratio averaging + boundary
    clamp must match the reference op (regression guard for the
    grid_sample bilinear fix this builds on)."""

    def test_constant_feature_constant_output(self) -> None:
        import lucid
        from lucid.models._utils._detection import roi_align

        feat = lucid.ones(1, 4, 16, 16) * 3.0
        boxes = [lucid.tensor([[2.0, 3.0, 11.0, 13.0], [0.0, 0.0, 15.0, 15.0]])]
        for ratio in (1, 2, -1):
            out = roi_align(
                feat, boxes, output_size=7, spatial_scale=1.0, sampling_ratio=ratio
            )
            assert tuple(out.shape) == (2, 4, 7, 7)
            # A constant feature must sample to that constant everywhere.
            assert abs(float(out.max().item()) - 3.0) < 1e-5
            assert abs(float(out.min().item()) - 3.0) < 1e-5

    def test_linear_ramp_ratio_invariant(self) -> None:
        import lucid
        from lucid.models._utils._detection import roi_align

        # On a linear (horizontal) ramp, averaging symmetric sub-bin samples
        # equals the single centre sample, so RoIAlign is sampling_ratio-
        # invariant — a correctness property of bilinear sub-bin averaging.
        ramp = lucid.tensor([[[[float(c) for c in range(8)] for _ in range(8)]]])
        boxes = [lucid.tensor([[0.0, 0.0, 7.0, 7.0]])]
        r1 = float(roi_align(ramp, boxes, 1, sampling_ratio=1).item())
        r4 = float(roi_align(ramp, boxes, 1, sampling_ratio=4).item())
        assert abs(r1 - r4) < 1e-5

    def test_subbin_averaging_runs_for_2d_grid(self) -> None:
        import lucid
        from lucid.models._utils._detection import roi_align

        # Exercise the (out_h*ry, out_w*rx) sub-bin reshape/mean path on a
        # multi-bin output with ratio>1 — shape + finiteness guard.
        feat = lucid.randn(1, 2, 20, 20)
        boxes = [lucid.tensor([[1.0, 2.0, 17.0, 18.0]])]
        out = roi_align(feat, boxes, output_size=(5, 5), sampling_ratio=2)
        assert tuple(out.shape) == (1, 2, 5, 5)
        assert bool(lucid.isfinite(out).all().item())


class TestMSDeformAttn:
    """Multi-scale deformable attention (Deformable DETR / Mask2Former) —
    composite over the fixed grid_sample; reproduces the reference op."""

    def test_output_shape(self) -> None:
        import lucid
        from lucid.models._utils._detection import multi_scale_deformable_attention

        bs, nh, hd, nq, nl, npt = 1, 8, 32, 25, 3, 4
        shapes = [(8, 8), (4, 4), (2, 2)]
        s = sum(h * w for h, w in shapes)
        value = lucid.randn(bs, s, nh, hd)
        loc = lucid.rand(bs, nq, nh, nl, npt, 2)
        aw = lucid.rand(bs, nq, nh, nl, npt)
        out = multi_scale_deformable_attention(value, shapes, loc, aw)
        assert tuple(out.shape) == (bs, nq, nh * hd)
        assert bool(lucid.isfinite(out).all().item())

    def test_constant_value_with_unit_weights(self) -> None:
        import lucid
        from lucid.models._utils._detection import multi_scale_deformable_attention

        # Constant value C + attention weights that sum to 1 over (nl*npt)
        # per (query, head) → output must be C everywhere (interior samples,
        # no boundary zero-padding).
        bs, nh, hd, nq, nl, npt = 1, 2, 4, 3, 2, 2
        shapes = [(8, 8), (4, 4)]
        s = sum(h * w for h, w in shapes)
        value = lucid.ones(bs, s, nh, hd) * 2.5
        # interior sampling locations (0.5 = centre) avoid edge zero-pad
        loc = lucid.ones(bs, nq, nh, nl, npt, 2) * 0.5
        aw = lucid.ones(bs, nq, nh, nl, npt) / float(nl * npt)
        out = multi_scale_deformable_attention(value, shapes, loc, aw)
        assert abs(float(out.max().item()) - 2.5) < 1e-5
        assert abs(float(out.min().item()) - 2.5) < 1e-5


class TestFasterRCNNWeightsEnums:
    """Static contract of the Faster R-CNN ResNet-50-FPN Weights enum."""

    def test_default_aliases_coco(self) -> None:
        from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights

        assert (
            FasterRCNNResNet50FPNWeights.DEFAULT is FasterRCNNResNet50FPNWeights.COCO_V1
        )

    def test_entry_fields(self) -> None:
        from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights

        e = FasterRCNNResNet50FPNWeights.COCO_V1.entry
        assert e.num_classes == 91
        # sha256 is either a real 64-hex digest or the upload placeholder.
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert "lucid-dl/faster-rcnn-resnet-50-fpn" in e.url
        assert "/COCO_V1/" in e.url
        meta = FasterRCNNResNet50FPNWeights.COCO_V1.meta
        assert meta["source"] == ("torchvision/FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
        assert meta["license"] == "bsd-3-clause"
        assert meta["num_params"] == 41_755_286
        assert meta["metrics"]["COCO"]["box mAP"] == 37.0

    def test_transforms_detection_preset(self) -> None:
        from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights

        tf = FasterRCNNResNet50FPNWeights.COCO_V1.transforms()
        assert tf.to_dict()["preprocessor_type"] == "Detection"
        assert tf.max_size == 1333

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        assert "COCO_V1" in list_pretrained("faster_rcnn")
        assert "COCO_V1" in list_pretrained("faster_rcnn_resnet50_fpn")


@unittest.skipUnless(
    os.environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestFasterRCNNPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_default(self) -> None:
        import lucid.models as M

        m = M.create_model("faster_rcnn_resnet50_fpn", pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 256, 256))
        assert int(out.logits.shape[-1]) == 91


class TestMaskRCNNWeightsEnums:
    """Static contract of the Mask R-CNN ResNet-50-FPN Weights enum."""

    def test_default_aliases_coco(self) -> None:
        from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights

        assert (
            MaskRCNNResNet50FPNWeights.DEFAULT is MaskRCNNResNet50FPNWeights.COCO_V1
        )

    def test_entry_fields(self) -> None:
        from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights

        e = MaskRCNNResNet50FPNWeights.COCO_V1.entry
        assert e.num_classes == 91
        # sha256 is either a real 64-hex digest or the upload placeholder.
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert "lucid-dl/mask-rcnn-resnet-50-fpn" in e.url
        assert "/COCO_V1/" in e.url
        meta = MaskRCNNResNet50FPNWeights.COCO_V1.meta
        assert meta["source"] == ("torchvision/MaskRCNN_ResNet50_FPN_Weights.COCO_V1")
        assert meta["license"] == "bsd-3-clause"
        assert meta["num_params"] == 44_401_393
        assert meta["metrics"]["COCO"]["box mAP"] == 37.9
        assert meta["metrics"]["COCO"]["mask mAP"] == 34.6

    def test_transforms_detection_preset(self) -> None:
        from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights

        tf = MaskRCNNResNet50FPNWeights.COCO_V1.transforms()
        assert tf.to_dict()["preprocessor_type"] == "Detection"
        assert tf.max_size == 1333

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        assert "COCO_V1" in list_pretrained("mask_rcnn")
        assert "COCO_V1" in list_pretrained("mask_rcnn_resnet50_fpn")


@unittest.skipUnless(
    os.environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestMaskRCNNPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_default(self) -> None:
        import lucid.models as M

        m = M.create_model("mask_rcnn_resnet50_fpn", pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 256, 256))
        assert int(out.logits.shape[-1]) == 91
        assert int(out.pred_masks.shape[-1]) == 28
