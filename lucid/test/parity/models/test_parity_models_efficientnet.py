"""Model parity tests — EfficientNet B0–B7.

  B0: 5.3 M    → always run
  B1: 7.8 M    → always run
  B2: 9.2 M    → always run
  B3: 12.2 M   → always run
  B4: 19.3 M   → slow
  B5: 30.4 M   → slow
  B6: 43.0 M   → slow
  B7: 66.4 M   → slow

timm names: ``efficientnet_b0`` … ``efficientnet_b7``
"""

import pytest

import lucid.models as M
from lucid.test.parity.models._utils import (
    requires_timm,
    run_parity,
    run_self_consistency,
)

_INPUT = (1, 3, 224, 224)


@requires_timm
class TestEfficientNetParity:

    def test_efficientnet_b0_parity(self) -> None:
        run_parity(M.efficientnet_b0_cls(), "efficientnet_b0", input_shape=_INPUT)

    def test_efficientnet_b1_parity(self) -> None:
        run_parity(M.efficientnet_b1_cls(), "efficientnet_b1", input_shape=_INPUT)

    def test_efficientnet_b2_parity(self) -> None:
        run_parity(M.efficientnet_b2_cls(), "efficientnet_b2", input_shape=_INPUT)

    def test_efficientnet_b3_parity(self) -> None:
        run_parity(M.efficientnet_b3_cls(), "efficientnet_b3", input_shape=_INPUT)

    @pytest.mark.slow
    def test_efficientnet_b4_parity(self) -> None:
        run_parity(M.efficientnet_b4_cls(), "efficientnet_b4", input_shape=_INPUT)

    @pytest.mark.slow
    def test_efficientnet_b5_parity(self) -> None:
        run_parity(M.efficientnet_b5_cls(), "efficientnet_b5", input_shape=_INPUT)

    @pytest.mark.slow
    def test_efficientnet_b6_parity(self) -> None:
        run_parity(M.efficientnet_b6_cls(), "efficientnet_b6", input_shape=_INPUT)

    @pytest.mark.slow
    def test_efficientnet_b7_parity(self) -> None:
        run_parity(M.efficientnet_b7_cls(), "efficientnet_b7", input_shape=_INPUT)


class TestEfficientNetSelfConsistency:
    """Determinism — no external dependency."""

    def test_b0_deterministic(self) -> None:
        run_self_consistency(M.efficientnet_b0_cls(), input_shape=_INPUT)

    def test_b3_deterministic(self) -> None:
        run_self_consistency(M.efficientnet_b3_cls(), input_shape=_INPUT)
