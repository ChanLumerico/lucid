"""Model parity tests — classic CNNs without timm equivalents.

LeNet, AlexNet, ZFNet, GoogLeNet: no direct timm counterpart with an
identical architecture, so we run self-consistency checks instead:
  - deterministic output given the same input
  - correct output shape
  - logits are finite (no NaN / Inf)

AlexNet / ZFNet (61 M / 62 M params) are marked ``slow`` because their
large FC layers take several seconds to initialise.  VGG is in a
separate file because it is ``heavy`` (> 100 M params).
"""

import numpy as np
import pytest

import lucid
import lucid.models as M
from lucid.test.parity.models._utils import run_self_consistency


# ── LeNet-5 ───────────────────────────────────────────────────────────────────


class TestLeNetParity:
    def test_lenet5_deterministic(self) -> None:
        m = M.lenet_5_cls()
        run_self_consistency(m, input_shape=(1, 1, 32, 32))

    def test_lenet5_relu_deterministic(self) -> None:
        m = M.lenet_5_relu_cls()
        run_self_consistency(m, input_shape=(1, 1, 32, 32))

    def test_lenet5_output_finite(self) -> None:
        m = M.lenet_5_cls()
        m.eval()
        x = lucid.from_numpy(np.random.randn(2, 1, 32, 32).astype(np.float32))
        out = m(x)
        assert np.isfinite(out.logits.numpy()).all()


# ── AlexNet (61 M — slow) ─────────────────────────────────────────────────────


@pytest.mark.slow
class TestAlexNetParity:
    def test_alexnet_deterministic(self) -> None:
        m = M.alexnet_cls()
        run_self_consistency(m, input_shape=(1, 3, 224, 224))

    def test_alexnet_output_finite(self) -> None:
        m = M.alexnet_cls()
        m.eval()
        x = lucid.from_numpy(np.random.randn(1, 3, 224, 224).astype(np.float32))
        out = m(x)
        assert np.isfinite(out.logits.numpy()).all()
        assert out.logits.shape == (1, 1000)


# ── ZFNet (62 M — slow) ───────────────────────────────────────────────────────


@pytest.mark.slow
class TestZFNetParity:
    def test_zfnet_deterministic(self) -> None:
        m = M.zfnet_cls()
        run_self_consistency(m, input_shape=(1, 3, 224, 224))

    def test_zfnet_output_shape(self) -> None:
        m = M.zfnet_cls()
        m.eval()
        x = lucid.from_numpy(np.random.randn(1, 3, 224, 224).astype(np.float32))
        assert m(x).logits.shape == (1, 1000)


# ── GoogLeNet (13 M) ──────────────────────────────────────────────────────────


class TestGoogLeNetParity:
    def test_googlenet_deterministic(self) -> None:
        m = M.googlenet_cls()
        run_self_consistency(m, input_shape=(1, 3, 224, 224))

    def test_googlenet_output_finite(self) -> None:
        m = M.googlenet_cls()
        m.eval()
        x = lucid.from_numpy(np.random.randn(1, 3, 224, 224).astype(np.float32))
        out = m(x)
        assert out.logits.shape == (1, 1000)
        assert np.isfinite(out.logits.numpy()).all()
