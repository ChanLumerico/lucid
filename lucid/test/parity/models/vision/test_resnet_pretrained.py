"""Pretrained-weights integration parity — ResNet-18 IMAGENET1K_V1.

Exercises the full :mod:`lucid.weights` path end to end: a
``resnet_18_cls(pretrained=True)`` call downloads the hosted checkpoint
from the Hub, verifies its SHA-256, and loads it — then we assert the
resulting logits match the upstream torchvision source the weights were
converted from.

Network- and source-dependent, so it auto-skips when torchvision is
absent or the download is unreachable.  Marked ``parity`` + ``slow``.
"""

import numpy as np
import pytest

import lucid

pytestmark = [pytest.mark.parity, pytest.mark.slow]

_tv = pytest.importorskip("torchvision", reason="torchvision (source) not installed")


def _load_lucid_pretrained() -> object:
    """Build resnet_18_cls(pretrained=True), skipping on network failure."""
    from lucid.models.vision.resnet import resnet_18_cls

    try:
        model = resnet_18_cls(pretrained=True)
    except (OSError, RuntimeError) as exc:  # network down / Hub unreachable
        pytest.skip(f"pretrained download unavailable: {exc}")
    model.eval()
    return model


def test_resnet18_pretrained_matches_source(ref) -> None:
    """Lucid pretrained logits match the torchvision ResNet18 source."""
    from torchvision.models import ResNet18_Weights, resnet18  # source

    lucid_model = _load_lucid_pretrained()

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 224, 224)).astype("float32")

    ref_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    with ref.no_grad():
        ref_logits = ref_model(ref.from_numpy(x)).numpy()

    out_logits = lucid_model(lucid.tensor(x)).logits.numpy()

    assert out_logits.shape == ref_logits.shape == (1, 1000)
    assert int(out_logits.argmax()) == int(ref_logits.argmax())
    np.testing.assert_allclose(out_logits, ref_logits, atol=1e-4)


def test_resnet18_pretrained_sha_pins_artifact() -> None:
    """The enum's SHA-256 matches the hosted file (download verifies it)."""
    from lucid.models.vision.resnet import ResNet18Weights

    # Loading would have raised on a SHA mismatch; here we just assert the
    # entry is fully specified (non-empty pinned hash + Hub URL).
    w = ResNet18Weights.IMAGENET1K_V1
    assert len(w.sha256) == 64
    assert w.url.startswith("https://huggingface.co/lucid-dl/resnet-18/")
    _load_lucid_pretrained()  # exercises download + SHA verify (or skips)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "parity or slow"])
