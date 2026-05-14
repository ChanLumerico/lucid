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
            faster_rcnn,
            FasterRCNNForObjectDetection,
        )

        # Reduce RPN top-N to keep NMS python-loop tractable on metal
        m = _build(
            lambda: faster_rcnn(rpn_pre_nms_top_n=200, rpn_post_nms_top_n=100),
            device,
        )
        assert isinstance(m, FasterRCNNForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 2
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        out2 = m(x)
        assert out.logits.shape == out2.logits.shape
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


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
        assert int(out.logits.ndim) == 2
        assert int(out.pred_boxes.shape[-1]) == 4
        assert int(out.pred_masks.ndim) == 4
        assert int(out.pred_masks.shape[-1]) == 28
        assert int(out.pred_masks.shape[-2]) == 28
        assert int(out.logits.shape[0]) == int(out.pred_masks.shape[0])
        assert out.loss is None


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
                num_encoder_layers=2, num_decoder_layers=2, num_queries=20,
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
            "mask_rcnn",
            "detr_resnet50", "detr_resnet101",
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
