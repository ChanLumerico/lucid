"""ResNeSt parity tests (14 / 26 / 50d / 101e / 200 / 269).

14 / 26 — self-consistency (no timm exact match).
50d / 101e — timm parity (slow tier).
200 / 269 — self-consistency (slow tier)."""

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
        M.resnest_14_cls,
        M.resnest_26_cls,
        M.resnest_50_cls,
        M.resnest_101_cls,
        M.resnest_200_cls,
        M.resnest_269_cls,
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
