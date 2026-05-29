"""Unit tests for Wave 3d segmentation models — CPU + Metal parametrized.

Covers:
  FCN (resnet50/101), UNet (base/small/bilinear),
  Attention U-Net, MaskFormer (resnet50/101),
  Mask2Former (swin tiny/small/base/large)

Each test combines factory check, output type, shape, deterministic
self-consistency, and loss=None checks in ONE forward pass per device.

Tests are parametrized over the ``device`` fixture so they run on both
the CPU (Accelerate) and Metal (MLX) streams.
"""

import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import SemanticSegmentationOutput

_B = 1
_C = 3
_H = 128
_W = 128


def _img(device: str, ch: int = _C) -> Tensor:
    lucid.manual_seed(0)
    return lucid.randn((_B, ch, _H, _W), device=device)


def _build(factory, device: str):
    m = factory()
    m.eval()
    return m.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# FCN
# ─────────────────────────────────────────────────────────────────────────────


class TestFCNResNet50:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fcn import fcn_resnet50, FCNForSemanticSegmentation

        m = _build(fcn_resnet50, device)
        assert isinstance(m, FCNForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestFCNResNet101:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fcn import fcn_resnet101, FCNForSemanticSegmentation

        m = _build(fcn_resnet101, device)
        assert isinstance(m, FCNForSemanticSegmentation)
        out = m(_img(device))
        assert isinstance(out, SemanticSegmentationOutput)
        assert int(out.logits.shape[0]) == _B
        assert int(out.logits.shape[-1]) == _W


# ─────────────────────────────────────────────────────────────────────────────
# UNet variants
# ─────────────────────────────────────────────────────────────────────────────


class TestUNet:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import unet, UNetForSemanticSegmentation

        m = _build(unet, device)
        assert isinstance(m, UNetForSemanticSegmentation)

        x = _img(device, ch=1)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert int(out.logits.shape[1]) == K
        assert int(out.logits.shape[-2]) == _H
        assert int(out.logits.shape[-1]) == _W
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestResUNet2d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import res_unet_2d

        m = _build(
            lambda: res_unet_2d(num_classes=4, base_channels=8, depth=2),
            device,
        )
        out = m(_img(device, ch=1))
        assert isinstance(out, SemanticSegmentationOutput)
        assert tuple(out.logits.shape) == (_B, 4, _H, _W)


class TestUNet3d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import unet_3d
        import lucid

        m = _build(
            lambda: unet_3d(num_classes=3, base_channels=8, depth=2),
            device,
        )
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, 16, 16, 16), device=device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        # (B, K, D, H, W) — same spatial size as input
        assert tuple(out.logits.shape) == (_B, 3, 16, 16, 16)


class TestResUNet3d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import res_unet_3d
        import lucid

        m = _build(
            lambda: res_unet_3d(num_classes=3, base_channels=8, depth=2),
            device,
        )
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, 16, 16, 16), device=device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        assert tuple(out.logits.shape) == (_B, 3, 16, 16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionUNet:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.attention_unet import (
            attention_unet,
            AttentionUNetForSemanticSegmentation,
        )

        m = _build(attention_unet, device)
        assert isinstance(m, AttentionUNetForSemanticSegmentation)

        x = _img(device, ch=1)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        assert int(out.logits.shape[-2]) == _H
        assert int(out.logits.shape[-1]) == _W

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# MaskFormer
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskFormer:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.maskformer import (
            maskformer_resnet50,
            MaskFormerForSemanticSegmentation,
        )

        # Slim transformer + queries for test speed
        m = _build(
            lambda: maskformer_resnet50(
                num_queries=20,
                num_decoder_layers=2,
            ),
            device,
        )
        assert isinstance(m, MaskFormerForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        # Semantic output drops the no-object slot (reference post-processing):
        # exactly num_classes channels, not num_classes + 1.
        assert tuple(out.logits.shape) == (_B, K, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former
# ─────────────────────────────────────────────────────────────────────────────


class TestMask2FormerSwinTiny:
    def test_factory_and_forward(self, device: str) -> None:
        # Use a small config override (fewer queries / decoder layers) so the
        # heavy deformable + masked-attention stack stays cheap on CPU/Metal.
        # The Swin backbone pads internally so any input size works.
        from lucid.models.vision.mask2former import (
            mask2former_swin_tiny,
            Mask2FormerForSemanticSegmentation,
        )

        m = _build(
            lambda: mask2former_swin_tiny(
                num_queries=20,
                num_decoder_layers=3,
            ),
            device,
        )
        assert isinstance(m, Mask2FormerForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        # Semantic output drops the no-object slot → K channels (matches the
        # reference post_process_semantic_segmentation).
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Registry smoke-test (device-independent)
# ─────────────────────────────────────────────────────────────────────────────


class TestSegmentationRegistry:
    def test_segmentation_models_registered(self) -> None:
        import lucid.models as M

        seg_models = M.list_models(task="semantic-segmentation")
        expected = [
            "fcn_resnet50",
            "fcn_resnet101",
            "unet",
            "res_unet_2d",
            "unet_3d",
            "res_unet_3d",
            "attention_unet",
            "maskformer_resnet50",
            "maskformer_resnet101",
            "mask2former_swin_tiny",
            "mask2former_swin_small",
            "mask2former_swin_base",
            "mask2former_swin_large",
        ]
        for name in expected:
            assert name in seg_models, f"{name!r} missing from registry"


# ─────────────────────────────────────────────────────────────────────────────
# FCN pretrained weights — static enum contract (no network)
# ─────────────────────────────────────────────────────────────────────────────

_FCN_SHIPPED = (
    ("fcn_resnet50", "fcn-resnet-50", "FCN_ResNet50_Weights", 35_322_218, 60.5),
    ("fcn_resnet101", "fcn-resnet-101", "FCN_ResNet101_Weights", 54_314_346, 63.7),
)


def _fcn_enums() -> tuple[type, ...]:
    from lucid.models.weights import FCNResNet50Weights, FCNResNet101Weights

    return (FCNResNet50Weights, FCNResNet101Weights)


def test_fcn_weights_default_aliases() -> None:
    for cls in _fcn_enums():
        assert cls.DEFAULT is cls.COCO_WITH_VOC_LABELS_V1


def test_fcn_weights_entry_fields() -> None:
    for cls, (_fac, slug, src, nparams, miou) in zip(_fcn_enums(), _FCN_SHIPPED):
        e = cls.COCO_WITH_VOC_LABELS_V1.entry
        assert e.num_classes == 21
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert f"lucid-dl/{slug}" in e.url
        assert "/COCO_WITH_VOC_LABELS_V1/" in e.url
        meta = cls.COCO_WITH_VOC_LABELS_V1.meta
        assert meta["source"] == f"torchvision/{src}.COCO_WITH_VOC_LABELS_V1"
        assert meta["license"] == "bsd-3-clause"
        assert meta["num_params"] == nparams
        assert meta["metrics"]["COCO-val2017-VOC-labels"]["mIoU"] == miou


def test_fcn_weights_segmentation_preset() -> None:
    for cls in _fcn_enums():
        tf = cls.COCO_WITH_VOC_LABELS_V1.transforms()
        d = tf.to_dict()
        assert d["preprocessor_type"] == "Segmentation"
        assert d["init_kwargs"]["crop_size"] == 520


def test_fcn_weights_registry_discoverable() -> None:
    from lucid.weights import list_pretrained

    for fac, *_ in _FCN_SHIPPED:
        assert "COCO_WITH_VOC_LABELS_V1" in list_pretrained(fac)


@pytest.mark.skipif(
    __import__("os").environ.get("LUCID_TEST_NETWORK") != "1",
    reason="set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
def test_fcn_pretrained_load() -> None:
    import lucid.models as models

    m = models.fcn_resnet50(pretrained=True)
    m.eval()
    out = m(lucid.randn(1, 3, 256, 256))
    assert out.logits.shape == (1, 21, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# MaskFormer pretrained weights — static enum contract (no network)
# ─────────────────────────────────────────────────────────────────────────────

_MASKFORMER_SHIPPED = (
    (
        "maskformer_resnet50",
        "maskformer-resnet-50",
        "facebook/maskformer-resnet50-ade",
        41_307_863,
        44.5,
    ),
    (
        "maskformer_resnet101",
        "maskformer-resnet-101",
        "facebook/maskformer-resnet101-ade",
        60_299_991,
        45.5,
    ),
)


def _maskformer_enums() -> tuple[type, ...]:
    from lucid.models.vision.maskformer import (
        MaskFormerResNet50Weights,
        MaskFormerResNet101Weights,
    )

    return (MaskFormerResNet50Weights, MaskFormerResNet101Weights)


def test_maskformer_weights_default_aliases() -> None:
    for cls in _maskformer_enums():
        assert cls.DEFAULT is cls.ADE20K


def test_maskformer_weights_entry_fields() -> None:
    for cls, (_fac, slug, src, nparams, miou) in zip(
        _maskformer_enums(), _MASKFORMER_SHIPPED
    ):
        e = cls.ADE20K.entry
        assert e.num_classes == 150
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert f"lucid-dl/{slug}" in e.url
        assert "/ADE20K/" in e.url
        meta = cls.ADE20K.meta
        assert meta["source"] == src
        assert meta["license"] == "other"
        assert meta["num_params"] == nparams
        assert meta["metrics"]["ADE20K"]["mIoU"] == miou


def test_maskformer_weights_segmentation_preset() -> None:
    for cls in _maskformer_enums():
        tf = cls.ADE20K.transforms()
        d = tf.to_dict()
        assert d["preprocessor_type"] == "Segmentation"
        assert d["init_kwargs"]["crop_size"] == 512


def test_maskformer_weights_registry_discoverable() -> None:
    from lucid.weights import list_pretrained

    for fac, *_ in _MASKFORMER_SHIPPED:
        assert "ADE20K" in list_pretrained(fac)


@pytest.mark.skipif(
    __import__("os").environ.get("LUCID_TEST_NETWORK") != "1",
    reason="set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
def test_maskformer_pretrained_load() -> None:
    import lucid.models as models

    m = models.maskformer_resnet50(pretrained=True)
    m.eval()
    out = m(lucid.randn(1, 3, 256, 256))
    # Semantic output: num_classes channels (no-object slot dropped).
    assert out.logits.shape == (1, 150, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former pretrained weights — static enum contract (no network)
# ─────────────────────────────────────────────────────────────────────────────

_MASK2FORMER_SHIPPED = (
    (
        "mask2former_swin_tiny",
        "mask2former-swin-tiny-ade",
        "facebook/mask2former-swin-tiny-ade-semantic",
        47_441_169,
        47.7,
    ),
    (
        "mask2former_swin_small",
        "mask2former-swin-small-ade",
        "facebook/mask2former-swin-small-ade-semantic",
        68_815_312,
        51.3,
    ),
    (
        "mask2former_swin_base",
        "mask2former-swin-base-ade",
        "facebook/mask2former-swin-base-ade-semantic",
        107_420_006,
        53.9,
    ),
    (
        "mask2former_swin_large",
        "mask2former-swin-large-ade",
        "facebook/mask2former-swin-large-ade-semantic",
        215_986_594,
        56.1,
    ),
)


def _mask2former_enums() -> tuple[type, ...]:
    from lucid.models.vision.mask2former import (
        Mask2FormerSwinTinyWeights,
        Mask2FormerSwinSmallWeights,
        Mask2FormerSwinBaseWeights,
        Mask2FormerSwinLargeWeights,
    )

    return (
        Mask2FormerSwinTinyWeights,
        Mask2FormerSwinSmallWeights,
        Mask2FormerSwinBaseWeights,
        Mask2FormerSwinLargeWeights,
    )


class TestMask2FormerWeightsEnums:
    def test_default_aliases(self) -> None:
        for cls in _mask2former_enums():
            assert cls.DEFAULT is cls.ADE20K

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams, miou) in zip(
            _mask2former_enums(), _MASK2FORMER_SHIPPED
        ):
            e = cls.ADE20K.entry
            assert e.num_classes == 150
            assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
            assert f"lucid-dl/{slug}" in e.url
            assert "/ADE20K/" in e.url
            meta = cls.ADE20K.meta
            assert meta["source"] == src
            assert meta["license"] == "other"
            assert meta["num_params"] == nparams
            assert meta["metrics"]["ADE20K"]["mIoU"] == miou

    def test_segmentation_preset(self) -> None:
        for cls in _mask2former_enums():
            tf = cls.ADE20K.transforms()
            d = tf.to_dict()
            assert d["preprocessor_type"] == "Segmentation"
            assert d["init_kwargs"]["crop_size"] == 384

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _MASK2FORMER_SHIPPED:
            assert "ADE20K" in list_pretrained(fac)


@pytest.mark.skipif(
    __import__("os").environ.get("LUCID_TEST_NETWORK") != "1",
    reason="set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
def test_mask2former_pretrained_load() -> None:
    import lucid.models as models

    m = models.mask2former_swin_tiny(pretrained=True)
    m.eval()
    out = m(lucid.randn(1, 3, 384, 384))
    # Semantic output: num_classes channels (no-object slot dropped).
    assert out.logits.shape == (1, 150, 384, 384)
