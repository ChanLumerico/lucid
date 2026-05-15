"""PVT v2 parity tests (B0–B5 + legacy pvt_tiny alias).

B0 — default tier (self-consistency).
B1 — default tier (timm parity).
B2–B5 — slow tier (self-consistency).
pvt_tiny — alias for B1 (timm parity)."""

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
        M.pvt_v2_b0_cls,
        M.pvt_v2_b1_cls,
        M.pvt_v2_b2_cls,
        M.pvt_v2_b3_cls,
        M.pvt_v2_b4_cls,
        M.pvt_v2_b5_cls,
        M.pvt_tiny_cls,
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
