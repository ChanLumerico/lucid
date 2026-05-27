"""Phase 3 — multi-target transforms (Image / Mask / BoundingBoxes).

Verifies typed dispatch: a single transform call moves an image, its
mask, and its boxes consistently; photometric transforms leave
non-image targets untouched; plain tensors still pass through as images.
"""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms._datatypes import to_xyxy


def _sample(h: int = 100, w: int = 120) -> dict[str, object]:
    return {
        "image": T.Image(lucid.rand(3, h, w)),
        "mask": T.Mask(lucid.rand(1, h, w)),
        "boxes": T.BoundingBoxes(
            lucid.tensor([[10.0, 20.0, 60.0, 80.0]]), "xyxy", (h, w)
        ),
    }


def _box(out: dict[str, object]) -> list[float]:
    bb = out["boxes"]
    assert isinstance(bb, T.BoundingBoxes)
    return to_xyxy(bb).numpy().reshape(-1).tolist()


# ── datatypes ───────────────────────────────────────────────────────


class TestDataTypes:
    def test_bbox_format_validation(self) -> None:
        with pytest.raises(ValueError, match="unknown format"):
            T.BoundingBoxes(lucid.tensor([[0.0, 0.0, 1.0, 1.0]]), "nope", (10, 10))

    def test_xywh_roundtrip(self) -> None:
        b = T.BoundingBoxes(lucid.tensor([[10.0, 20.0, 30.0, 40.0]]), "xywh", (100, 100))
        xy = to_xyxy(b).numpy().reshape(-1).tolist()
        assert xy == [10.0, 20.0, 40.0, 60.0]  # x,y,x+w,y+h


# ── geometric consistency across targets ────────────────────────────


class TestResizeMultiTarget:
    def test_image_mask_boxes(self) -> None:
        out = T.Resize((50, 60))(_sample())
        assert tuple(out["image"].data.shape) == (3, 50, 60)
        assert tuple(out["mask"].data.shape) == (1, 50, 60)
        # 100->50 (x0.5 in y), 120->60 (x0.5 in x)
        assert _box(out) == [5.0, 10.0, 30.0, 40.0]
        assert out["boxes"].canvas_size == (50, 60)


class TestFlipMultiTarget:
    def test_hflip_boxes(self) -> None:
        out = T.RandomHorizontalFlip(p=1.0)(_sample())
        # x mirrored within W=120: [120-60, 20, 120-10, 80]
        assert _box(out) == [60.0, 20.0, 110.0, 80.0]

    def test_vflip_boxes(self) -> None:
        out = T.RandomVerticalFlip(p=1.0)(_sample())
        # y mirrored within H=100: [10, 100-80, 60, 100-20]
        assert _box(out) == [10.0, 20.0, 60.0, 80.0]


class TestCropMultiTarget:
    def test_center_crop(self) -> None:
        out = T.CenterCrop(80)(_sample())
        assert tuple(out["image"].data.shape) == (3, 80, 80)
        # top=(100-80)//2=10, left=(120-80)//2=20; shift box by (-20,-10),
        # clip into [0,80]: [max(0,-10),10,40,70] -> [0,10,40,70]
        assert _box(out) == [0.0, 10.0, 40.0, 70.0]
        assert out["boxes"].canvas_size == (80, 80)

    def test_pad_boxes(self) -> None:
        out = T.Pad(5)(_sample())
        # +5 offset both axes; canvas grows by 10
        assert _box(out) == [15.0, 25.0, 65.0, 85.0]
        assert out["boxes"].canvas_size == (110, 130)


class TestRandomResizedCropMultiTarget:
    def test_shapes_and_canvas(self) -> None:
        lucid.manual_seed(0)
        out = T.RandomResizedCrop(64)(_sample())
        assert tuple(out["image"].data.shape) == (3, 64, 64)
        assert tuple(out["mask"].data.shape) == (1, 64, 64)
        assert out["boxes"].canvas_size == (64, 64)


# ── photometric leaves non-image targets alone ──────────────────────


class TestPhotometricUntouched:
    def test_normalize(self) -> None:
        s = _sample()
        out = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(s)
        assert float((out["mask"].data - s["mask"].data).abs().max().item()) == 0.0
        assert _box(out) == [10.0, 20.0, 60.0, 80.0]

    def test_colorjitter(self) -> None:
        s = _sample()
        out = T.ColorJitter(0.4, 0.4)(s)
        assert _box(out) == [10.0, 20.0, 60.0, 80.0]


# ── plain tensor + Compose ──────────────────────────────────────────


class TestPlainAndCompose:
    def test_plain_tensor_image(self) -> None:
        out = T.Resize((50, 60))(lucid.rand(3, 100, 120))
        assert tuple(out.shape) == (3, 50, 60)

    def test_compose_multitarget(self) -> None:
        lucid.manual_seed(1)
        tf = T.Compose([T.Resize((80, 80)), T.RandomHorizontalFlip(p=1.0)])
        out = tf(_sample())
        assert tuple(out["image"].data.shape) == (3, 80, 80)
        assert out["boxes"].canvas_size == (80, 80)

    def test_no_image_raises(self) -> None:
        boxes_only = {"boxes": T.BoundingBoxes(
            lucid.tensor([[0.0, 0.0, 1.0, 1.0]]), "xyxy", (10, 10))}
        with pytest.raises(ValueError, match="no image"):
            T.Resize((5, 5))(boxes_only)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
