"""PoolFormer parity tests (S12 — default; S24/S36/M36/M48 — slow; all self-consistency)."""

import pytest
import lucid.models as M
from lucid.test.parity.models._registry import SPECS, ParitySpec
from lucid.test.parity.models._utils import (
    requires_timm,
    _run_parity,
    _run_self_consistency,
    _spec_param,
)

_FACTORIES = frozenset({
    M.poolformer_s12_cls, M.poolformer_s24_cls, M.poolformer_s36_cls,
    M.poolformer_m36_cls, M.poolformer_m48_cls,
})
_SPECS = [s for s in SPECS if s.lucid_factory in _FACTORIES]
_TIMM = [s for s in _SPECS if s.timm_name is not None]
_SC = [s for s in _SPECS if s.timm_name is None]

if _TIMM:
    @requires_timm
    @pytest.mark.parametrize("spec", [_spec_param(s) for s in _TIMM])
    def test_parity(spec: ParitySpec) -> None:
        _run_parity(spec)

if _SC:
    @pytest.mark.parametrize("spec", [_spec_param(s) for s in _SC])
    def test_self_consistency(spec: ParitySpec) -> None:
        _run_self_consistency(spec)
