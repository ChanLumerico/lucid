"""Vision Transformer (ViT) parity tests.

B/16, B/32 — slow tier.  L/16, L/32 — heavy tier.
H/14 — self-consistency only (632 M params).

Attention block key naming (32%% coverage) causes timm parity to skip
until a key_transform is written for the ViT block structure."""

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
        M.vit_b_16_cls,
        M.vit_b_32_cls,
        M.vit_l_16_cls,
        M.vit_l_32_cls,
        M.vit_h_14_cls,
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
