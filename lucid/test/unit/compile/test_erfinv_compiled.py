"""``erfinv`` on the graph, by approximation rather than by primitive.

MPSGraph has no inverse error function, so this was a stub — registered
to keep the lookup total, always declining, always falling back to eager.
Giles' polynomial gives it one: two branches in
``w = -log((1 - x)(1 + x))``, both evaluated and one selected, since a
graph has no branch to take.

An approximation only earns its place if the error is stated. These
assert it rather than assume it, and the bound is the same order as the
pipeline's other float32 error — not a looser one chosen to pass.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


class Erfinv(nn.Module):
    """``tanh`` keeps the argument inside the domain, whatever the input."""

    def __init__(self, reach: float) -> None:
        super().__init__()
        self.reach = reach

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return lucid.erfinv(lucid.tanh(x) * self.reach)


def _run(reach: float) -> tuple[bool, float]:
    model = Erfinv(reach).eval()
    x = lucid.randn(1, 4, 32, 32)
    reference = model(x)
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")
    scale = float(reference.abs().max().item())
    return (
        _C_engine.compile.session_cache_size() > 0,
        float((got - reference).abs().max().item()) / scale,
    )


class TestErfinvCompiles:
    def test_the_central_region_agrees_like_any_other_op(self) -> None:
        compiled, error = _run(0.9)
        assert compiled, "erfinv fell back to eager"
        assert error < 1e-5

    def test_the_tails_agree_less_closely_and_are_still_close(self) -> None:
        """Where the polynomial is worst, and by how much.

        The tail branch takes over past ``w = 5`` and is the looser of
        the two; measured at 2e-6 relative, so a bound of 1e-4 fails on a
        real regression and passes on the approximation as it stands.
        """
        compiled, error = _run(0.999)
        assert compiled
        assert error < 1e-4

    def test_it_is_no_longer_a_stub(self) -> None:
        """Registration was always true; compiling is what changed.

        The cache growing is the difference — a stub registers and then
        declines, which reads as supported and runs eagerly.
        """
        assert _C_engine.compile.emitter_registered("erfinv")
        compiled, _error = _run(0.5)
        assert compiled
