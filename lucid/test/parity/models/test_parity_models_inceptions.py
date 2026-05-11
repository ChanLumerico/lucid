"""Model parity tests — Inception family.

Covers:
  Inception v3           (Szegedy et al., 2015) — 23.8 M
  Inception v4           (Szegedy et al., 2016) — 42.7 M  [slow]
  Inception-ResNet v2    (Szegedy et al., 2016) — 55.8 M  [slow]
  Xception               (Chollet, 2017)         — 22.9 M

All use 299×299 input as per the original papers.
"""

import pytest

import lucid.models as M
from lucid.test.parity.models._utils import (
    requires_timm,
    run_parity,
    run_self_consistency,
)

_INPUT = (1, 3, 299, 299)


@requires_timm
class TestInceptionParity:

    def test_inception_v3_parity(self) -> None:
        run_parity(M.inception_v3_cls(), "inception_v3", input_shape=_INPUT)

    @pytest.mark.slow
    def test_inception_v4_parity(self) -> None:
        run_parity(M.inception_v4_cls(), "inception_v4", input_shape=_INPUT)

    @pytest.mark.slow
    def test_inception_resnet_v2_parity(self) -> None:
        run_parity(
            M.inception_resnet_v2_cls(),
            "inception_resnet_v2",
            input_shape=_INPUT,
        )

    def test_xception_parity(self) -> None:
        run_parity(M.xception_cls(), "xception", input_shape=_INPUT)


# ── Self-consistency fallback (runs even without timm) ────────────────────────


class TestInceptionSelfConsistency:
    """Determinism tests that run without any external dependency."""

    def test_inception_v3_deterministic(self) -> None:
        run_self_consistency(M.inception_v3_cls(), input_shape=_INPUT)

    def test_inception_v4_deterministic(self) -> None:
        run_self_consistency(M.inception_v4_cls(), input_shape=_INPUT)

    def test_inception_resnet_v2_deterministic(self) -> None:
        run_self_consistency(M.inception_resnet_v2_cls(), input_shape=_INPUT)

    def test_xception_deterministic(self) -> None:
        run_self_consistency(M.xception_cls(), input_shape=_INPUT)
