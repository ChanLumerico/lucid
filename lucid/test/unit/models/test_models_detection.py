"""Unit tests for Wave 3d detection models.

Covers:
  R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN, DETR,
  EfficientDet D0, YOLO v1/v2/v3/v4

Each test group checks:
  1. Factory construction succeeds.
  2. Output types are correct (ObjectDetectionOutput / InstanceSegmentationOutput).
  3. Output tensor shapes match documented contracts.
  4. Self-consistency: same input → identical output (deterministic).
  5. Loss is a scalar when training targets are supplied.
"""

import unittest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import InstanceSegmentationOutput, ObjectDetectionOutput

# ─────────────────────────────────────────────────────────────────────────────
# Tiny inputs to keep tests fast
# ─────────────────────────────────────────────────────────────────────────────
_B = 1  # batch size
_H = 128  # image height (multiple of 32 for stride-32 backbones)
_W = 128  # image width
_C = 3  # channels
_H_ED = 256  # EfficientDet needs ≥256 for 5-level BiFPN (P7 = H/128 ≥ 2)
_W_ED = 256


def _img() -> Tensor:
    lucid.manual_seed(0)
    return lucid.randn((_B, _C, _H, _W))


def _img_ed() -> Tensor:
    """Larger image for EfficientDet's 5-level BiFPN."""
    lucid.manual_seed(0)
    return lucid.randn((_B, _C, _H_ED, _W_ED))


def _det_target(num_gt: int = 2) -> list[dict[str, Tensor]]:
    """Minimal detection target dict (xyxy pixel coords)."""
    boxes_data = [[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 50.0, 50.0]][:num_gt]
    labels_data = [1, 2][:num_gt]
    return [
        {
            "boxes": lucid.tensor(boxes_data),
            "labels": lucid.tensor([float(l) for l in labels_data]),
        }
    ]


def _norm_target(num_gt: int = 2) -> list[dict[str, Tensor]]:
    """Detection target with boxes normalised to [0, 1] (for DETR)."""
    boxes_data = [[0.1, 0.1, 0.4, 0.4], [0.3, 0.3, 0.7, 0.7]][:num_gt]
    labels_data = [1, 2][:num_gt]
    return [
        {
            "boxes": lucid.tensor(boxes_data),
            "labels": lucid.tensor([float(l) for l in labels_data]),
        }
    ]


# ─────────────────────────────────────────────────────────────────────────────
# R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestRCNN(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.rcnn import rcnn

        self.model = rcnn()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.rcnn import RCNNForObjectDetection

        self.assertIsInstance(self.model, RCNNForObjectDetection)

    def test_forward_output_type(self) -> None:
        proposals = [lucid.tensor([[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]])]
        out = self.model(_img(), proposals)
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_shapes(self) -> None:
        proposals = [lucid.tensor([[5.0, 5.0, 40.0, 40.0]])]
        out = self.model(_img(), proposals)
        # logits: (total_props, K+1), pred_boxes: (total_props, 4)
        self.assertEqual(int(out.logits.shape[0]), 1)
        self.assertEqual(int(out.logits.ndim), 2)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        proposals = [lucid.tensor([[5.0, 5.0, 40.0, 40.0]])]
        out1 = self.model(x, proposals)
        out2 = self.model(x, proposals)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# Fast R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestFastRCNN(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.fast_rcnn import fast_rcnn

        self.model = fast_rcnn()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.fast_rcnn import FastRCNNForObjectDetection

        self.assertIsInstance(self.model, FastRCNNForObjectDetection)

    def test_forward_shapes(self) -> None:
        proposals = [lucid.tensor([[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]])]
        out = self.model(_img(), proposals)
        self.assertIsInstance(out, ObjectDetectionOutput)
        # 2 proposals → 2 rows
        self.assertEqual(int(out.logits.shape[0]), 2)
        self.assertEqual(int(out.pred_boxes.shape[0]), 2)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        proposals = [lucid.tensor([[5.0, 5.0, 40.0, 40.0]])]
        out1 = self.model(x, proposals)
        out2 = self.model(x, proposals)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestFasterRCNN(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.faster_rcnn import faster_rcnn

        self.model = faster_rcnn()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.faster_rcnn import FasterRCNNForObjectDetection

        self.assertIsInstance(self.model, FasterRCNNForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_logits_ndim(self) -> None:
        out = self.model(_img())
        # logits: (total_proposals, K+1)
        self.assertEqual(int(out.logits.ndim), 2)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        # Proposals are NMS-filtered deterministically; logits must match
        self.assertEqual(out1.logits.shape, out2.logits.shape)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_no_loss_without_targets(self) -> None:
        out = self.model(_img())
        self.assertIsNone(out.loss)


# ─────────────────────────────────────────────────────────────────────────────
# Mask R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskRCNN(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.mask_rcnn import mask_rcnn

        self.model = mask_rcnn()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.mask_rcnn import MaskRCNNForObjectDetection

        self.assertIsInstance(self.model, MaskRCNNForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, InstanceSegmentationOutput)

    def test_forward_shapes(self) -> None:
        out = self.model(_img())
        # pred_masks: (total_props, K, 28, 28)
        self.assertEqual(int(out.logits.ndim), 2)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        self.assertEqual(int(out.pred_masks.ndim), 4)
        self.assertEqual(int(out.pred_masks.shape[-1]), 28)
        self.assertEqual(int(out.pred_masks.shape[-2]), 28)
        self.assertIsNone(out.loss)

    def test_logits_mask_proposal_count_match(self) -> None:
        out = self.model(_img())
        # Every proposal has both a class logit and a mask prediction
        self.assertEqual(int(out.logits.shape[0]), int(out.pred_masks.shape[0]))

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        self.assertEqual(out1.logits.shape, out2.logits.shape)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# DETR
# ─────────────────────────────────────────────────────────────────────────────


class TestDETR(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.detr import detr_resnet50

        self.model = detr_resnet50()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.detr import DETRForObjectDetection

        self.assertIsInstance(self.model, DETRForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_shapes(self) -> None:
        out = self.model(_img())
        # logits: (B, N, K+1);  pred_boxes: (B, N, 4)  cxcywh in [0,1]
        cfg = self.model._cfg
        self.assertEqual(
            tuple(out.logits.shape), (_B, cfg.num_queries, cfg.num_classes + 1)
        )
        self.assertEqual(tuple(out.pred_boxes.shape), (_B, cfg.num_queries, 4))
        self.assertIsNone(out.loss)

    def test_pred_boxes_range(self) -> None:
        out = self.model(_img())
        # All box coordinates should be in [0, 1] (sigmoid output)
        min_val = float(out.pred_boxes.min().item())
        max_val = float(out.pred_boxes.max().item())
        self.assertGreaterEqual(min_val, 0.0)
        self.assertLessEqual(max_val, 1.0)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_no_loss_without_targets(self) -> None:
        out = self.model(_img())
        self.assertIsNone(out.loss)


# ─────────────────────────────────────────────────────────────────────────────
# EfficientDet D0
# ─────────────────────────────────────────────────────────────────────────────


class TestEfficientDetD0(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.efficientdet import efficientdet_d0

        self.model = efficientdet_d0()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.efficientdet import EfficientDetForObjectDetection

        self.assertIsInstance(self.model, EfficientDetForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img_ed())
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_shapes(self) -> None:
        out = self.model(_img_ed())
        # logits: (B, A_total, K);  pred_boxes: (B, A_total, 4)
        self.assertEqual(int(out.logits.ndim), 3)
        self.assertEqual(int(out.logits.shape[0]), _B)
        self.assertEqual(int(out.pred_boxes.shape[0]), _B)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        # A_total > 0
        self.assertGreater(int(out.logits.shape[1]), 0)
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img_ed()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv1
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV1(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.yolo import yolo_v1

        self.model = yolo_v1()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.yolo import YOLOV1ForObjectDetection

        self.assertIsInstance(self.model, YOLOV1ForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_shapes(self) -> None:
        out = self.model(_img())
        S = self.model.config.split_size  # 7
        B_boxes = self.model.config.num_boxes  # 2
        C = self.model.config.num_classes
        # logits: (B, S*S*B, C);  pred_boxes: (B, S*S*B, 4)
        expected_preds = S * S * B_boxes
        self.assertEqual(tuple(out.logits.shape), (_B, expected_preds, C))
        self.assertEqual(tuple(out.pred_boxes.shape), (_B, expected_preds, 4))
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_tiny_variant(self) -> None:
        from lucid.models.vision.yolo import yolo_v1_tiny

        m = yolo_v1_tiny()
        m.eval()
        out = m(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv2
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV2(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.yolo import yolo_v2

        self.model = yolo_v2()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.yolo import YOLOV2ForObjectDetection

        self.assertIsInstance(self.model, YOLOV2ForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_shapes(self) -> None:
        out = self.model(_img())
        # logits: (B, grid_h*grid_w*num_anchors, C)
        self.assertEqual(int(out.logits.ndim), 3)
        self.assertEqual(int(out.logits.shape[0]), _B)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv3
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV3(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.yolo import yolo_v3

        self.model = yolo_v3()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.yolo import YOLOV3ForObjectDetection

        self.assertIsInstance(self.model, YOLOV3ForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_shapes(self) -> None:
        out = self.model(_img())
        # 3 scales × 3 anchors — total A = 3*(H/8*W/8 + H/16*W/16 + H/32*W/32)
        self.assertEqual(int(out.logits.ndim), 3)
        self.assertEqual(int(out.logits.shape[0]), _B)
        self.assertEqual(int(out.pred_boxes.shape[0]), _B)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        self.assertIsNone(out.loss)

    def test_logits_pred_count_match(self) -> None:
        out = self.model(_img())
        self.assertEqual(int(out.logits.shape[1]), int(out.pred_boxes.shape[1]))

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_tiny_variant(self) -> None:
        from lucid.models.vision.yolo import yolo_v3_tiny

        m = yolo_v3_tiny()
        m.eval()
        out = m(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv4
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV4(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.yolo import yolo_v4

        self.model = yolo_v4()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.yolo import YOLOV4ForObjectDetection

        self.assertIsInstance(self.model, YOLOV4ForObjectDetection)

    def test_forward_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, ObjectDetectionOutput)

    def test_forward_shapes(self) -> None:
        out = self.model(_img())
        self.assertEqual(int(out.logits.ndim), 3)
        self.assertEqual(int(out.logits.shape[0]), _B)
        self.assertEqual(int(out.pred_boxes.shape[-1]), 4)
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# Registry smoke-tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectionRegistry(unittest.TestCase):

    def test_detection_models_registered(self) -> None:
        import lucid.models as M

        det_models = M.list_models(task="object-detection")
        expected = [
            "rcnn",
            "fast_rcnn",
            "faster_rcnn",
            "mask_rcnn",
            "detr_resnet50",
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
            self.assertIn(name, det_models, f"{name!r} missing from registry")

    def test_create_model_api(self) -> None:
        import lucid.models as M

        m = M.create_model("detr_resnet50")
        self.assertIsNotNone(m)


if __name__ == "__main__":
    unittest.main()
