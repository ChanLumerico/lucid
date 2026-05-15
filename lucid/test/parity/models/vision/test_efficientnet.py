"""EfficientNet parity tests (B0 – B7 vs timm efficientnet_b*).

B0 – B3 — default tier.  B4 – B7 — slow tier."""

import pytest
import lucid.models as M
from lucid.test.parity.models._registry import SPECS, ParitySpec
from lucid.test.parity.models._utils import (
    requires_timm,
    _run_parity,
    _run_self_consistency,
    _spec_param,
)

_FACTORIES = frozenset(
    {
        M.efficientnet_b0_cls,
        M.efficientnet_b1_cls,
        M.efficientnet_b2_cls,
        M.efficientnet_b3_cls,
        M.efficientnet_b4_cls,
        M.efficientnet_b5_cls,
        M.efficientnet_b6_cls,
        M.efficientnet_b7_cls,
    }
)
_SPECS = [s for s in SPECS if s.lucid_factory in _FACTORIES]
_TIMM = [s for s in _SPECS if s.timm_name is not None]
_SC = [s for s in _SPECS if s.timm_name is None]

if _TIMM:

    @requires_timm
    @pytest.mark.parametrize("spec", [_spec_param(s) for s in _TIMM])
    def test_parity(spec: ParitySpec) -> None:
        """Numeric logit parity against timm reference."""
        _run_parity(spec)


if _SC:

    @pytest.mark.parametrize("spec", [_spec_param(s) for s in _SC])
    def test_self_consistency(spec: ParitySpec) -> None:
        """Deterministic forward pass (no external reference)."""
        _run_self_consistency(spec)
