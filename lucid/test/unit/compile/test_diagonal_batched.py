"""``diagonal`` over the trailing two axes, at any rank.

The emitter took rank 2 and declined everything else, so a batched
diagonal fell back to eager — correct, and silently not compiled, which
is the pair of properties that keeps a gap like this from being noticed.

``bandPart`` already works on the last two dimensions whatever the rank;
only the reduction axis and the output shape had to follow.
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


class Diagonal(nn.Module):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return lucid.diagonal(x)


def _run(shape: tuple[int, ...]) -> tuple[bool, float, tuple[int, ...]]:
    model = Diagonal().eval()
    x = lucid.randn(*shape)
    reference = model(x)
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")
    return (
        _C_engine.compile.session_cache_size() > 0,
        float((got - reference).abs().max().item()),
        tuple(got.shape),
    )


class TestBatchedDiagonal:
    @pytest.mark.parametrize(
        "shape",
        [(6, 6), (5, 7), (7, 5), (1, 4, 6, 6), (2, 3, 5, 5), (2, 3, 4, 6, 6)],
        ids=["square", "wide", "tall", "batched", "two-leading", "rank5"],
    )
    def test_it_compiles_and_agrees(self, shape: tuple[int, ...]) -> None:
        compiled, difference, produced = _run(shape)
        assert compiled, f"{shape} fell back to eager"
        assert difference == 0.0
        assert produced == shape[:-2] + (min(shape[-2], shape[-1]),)


class TestWhatItStillDeclines:
    """Both still fall back, and both still answer correctly."""

    def test_an_offset_diagonal_falls_back(self) -> None:
        class Offset(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.diagonal(x, offset=1)

        model = Offset().eval()
        x = lucid.randn(2, 3, 5, 5)
        reference = model(x)
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile.compile(model.to("metal"))
        got = compiled(x.to("metal")).to("cpu")
        assert _C_engine.compile.session_cache_size() == 0
        assert float((got - reference).abs().max().item()) == 0.0

    def test_a_leading_axis_pair_falls_back(self) -> None:
        """Anything but the trailing pair would need a transpose."""

        class Leading(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.diagonal(x, dim1=0, dim2=1)

        model = Leading().eval()
        x = lucid.randn(5, 5, 3)
        reference = model(x)
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile.compile(model.to("metal"))
        got = compiled(x.to("metal")).to("cpu")
        assert _C_engine.compile.session_cache_size() == 0
        assert float((got - reference).abs().max().item()) == 0.0
