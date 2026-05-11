"""Model parity tests — DenseNet family.

  DenseNet-121: 8.1 M   → always run
  DenseNet-169: 14.1 M  → always run
  DenseNet-201: 20.0 M  → always run
  DenseNet-264: 33.4 M  → slow (no timm equivalent → self-consistency)

timm names: ``densenet121``, ``densenet169``, ``densenet201``
DenseNet-264 has no timm equivalent so uses self-consistency.
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
class TestDenseNetParity:
    # Note: DenseNet positional weight transfer may be skipped automatically
    # if internal parameter ordering differs from timm (total params match,
    # but BN weight order inside transition blocks can differ).
    # Self-consistency tests below remain authoritative for these models.

    def test_densenet121_parity(self) -> None:
        run_parity(M.densenet_121_cls(), "densenet121", input_shape=_INPUT)

    def test_densenet169_parity(self) -> None:
        run_parity(M.densenet_169_cls(), "densenet169", input_shape=_INPUT)

    def test_densenet201_parity(self) -> None:
        run_parity(M.densenet_201_cls(), "densenet201", input_shape=_INPUT)


class TestDenseNetSelfConsistency:

    def test_densenet121_deterministic(self) -> None:
        run_self_consistency(M.densenet_121_cls(), input_shape=_INPUT)

    @pytest.mark.slow
    def test_densenet264_deterministic(self) -> None:
        # No timm equivalent — self-consistency only
        run_self_consistency(M.densenet_264_cls(), input_shape=_INPUT)
