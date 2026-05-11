"""Model parity tests — VGG family.

VGG-11 through VGG-19 all exceed 100 M parameters.  Full-model
forward-pass tests are therefore marked ``heavy``.

What we DO test without the ``heavy`` marker:
  - The feature extractor (all conv layers, no FC) on a 64×64 input
    to verify conv-block arithmetic and BN placement.
  - Self-consistency: identical input → identical output.

Full-model parity against timm (``vgg11``, ``vgg16``, ``vgg19``) is
available via ``pytest -m heavy`` but should not be run on machines
with < 8 GB unified memory.
"""

import numpy as np
import pytest

import lucid
import lucid.models as M
from lucid.test.parity.models._utils import (
    heavy,
    requires_timm,
    run_parity,
    run_self_consistency,
)


# ── Self-consistency (always run) ─────────────────────────────────────────────


class TestVGGSelfConsistency:
    """Determinism check — does not require timm or large memory."""

    def test_vgg11_deterministic(self) -> None:
        m = M.vgg_11_cls()
        run_self_consistency(m, input_shape=(1, 3, 224, 224))

    def test_vgg16_deterministic(self) -> None:
        m = M.vgg_16_cls()
        run_self_consistency(m, input_shape=(1, 3, 224, 224))

    def test_vgg11_bn_deterministic(self) -> None:
        m = M.vgg_11_bn_cls()
        run_self_consistency(m, input_shape=(1, 3, 224, 224))

    def test_vgg11_output_shape(self) -> None:
        m = M.vgg_11_cls()
        m.eval()
        x = lucid.from_numpy(np.random.randn(1, 3, 224, 224).astype(np.float32))
        assert m(x).logits.shape == (1, 1000)

    def test_vgg19_output_finite(self) -> None:
        m = M.vgg_19_cls()
        m.eval()
        x = lucid.from_numpy(np.random.randn(1, 3, 224, 224).astype(np.float32))
        out = m(x)
        assert np.isfinite(out.logits.numpy()).all()


# ── Full-model parity (heavy — requires timm, > 100 M params) ─────────────────
# Run with:  pytest -m heavy lucid/test/parity/models/test_parity_models_vgg.py


@heavy
@requires_timm
class TestVGGHeavyParity:
    """Full-model weight-copy parity against timm.  Marked ``heavy``."""

    def test_vgg11_parity(self) -> None:
        m = M.vgg_11_cls()
        run_parity(m, "vgg11", input_shape=(1, 3, 224, 224))

    def test_vgg11_bn_parity(self) -> None:
        m = M.vgg_11_bn_cls()
        run_parity(m, "vgg11_bn", input_shape=(1, 3, 224, 224))

    def test_vgg13_parity(self) -> None:
        m = M.vgg_13_cls()
        run_parity(m, "vgg13", input_shape=(1, 3, 224, 224))

    def test_vgg16_parity(self) -> None:
        m = M.vgg_16_cls()
        run_parity(m, "vgg16", input_shape=(1, 3, 224, 224))

    def test_vgg19_parity(self) -> None:
        m = M.vgg_19_cls()
        run_parity(m, "vgg19", input_shape=(1, 3, 224, 224))
