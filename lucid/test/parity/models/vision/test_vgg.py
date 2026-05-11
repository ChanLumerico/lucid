"""VGG parity tests (11 / 13 / 16 / 19, with and without BN).

Tier: heavy (128–144 M params).  Run with:  pytest -m heavy"""

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
        M.vgg_11_cls,
        M.vgg_11_bn_cls,
        M.vgg_13_cls,
        M.vgg_13_bn_cls,
        M.vgg_16_cls,
        M.vgg_16_bn_cls,
        M.vgg_19_cls,
        M.vgg_19_bn_cls,
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
