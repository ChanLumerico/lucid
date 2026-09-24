"""Compiled pooling honours ``ceil_mode`` and ``count_include_pad``.

The forward records both as one-element lists (``[1]``).  The 2-D emitters
never read them and the 3-D one read them with ``int_attr``, which does not
accept a list and answered the default — so every compiled ceil-mode pool
floored, and ``count_include_pad=False`` averaged the padding in anyway.

A floored pool is the wrong size, and the trace had recorded the right one.
Returned directly it happened to come out right; consumed by the next op it
did not: a ceil-mode max pool feeding a 1×1 conv was off by 2.6, and feeding
a squeeze-and-excitation gate it aborted the process — which is how
``lucid.compile`` of an SE-ResNet died.

Every case below pools and then *uses* the result.  Ceil-mode average
pooling that counts real padding stays eager on purpose: at an overhanging
window the reference divides by the part inside the padded input, MPSGraph
by the whole kernel, and no option reconciles them.  Without padding the
two agree (ResNeSt's shortcut pool), and that case must compile.
"""

import itertools

import numpy as np
import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

_CLS = {
    ("max", 1): nn.MaxPool1d,
    ("max", 2): nn.MaxPool2d,
    ("max", 3): nn.MaxPool3d,
    ("avg", 1): nn.AvgPool1d,
    ("avg", 2): nn.AvgPool2d,
    ("avg", 3): nn.AvgPool3d,
}


class _PoolThenUse(nn.Module):
    def __init__(self, pool: nn.Module) -> None:
        super().__init__()
        self.pool = pool

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = self.pool(x)
        # Consume the pooled tensor twice, as an SE gate does.
        return y * lucid.sigmoid(y.mean(dim=tuple(range(2, y.ndim)), keepdim=True)) + 1


def _cases() -> list[tuple[str, int, bool, bool, int, int]]:
    out = []
    for kind, dim, ceil, cip, n, pad in itertools.product(
        ("max", "avg"), (1, 2, 3), (False, True), (True, False), (7, 32), (0, 1)
    ):
        if kind == "max" and not cip:
            continue
        out.append((kind, dim, ceil, cip, n, pad))
    return out


@pytest.mark.parametrize(("kind", "dim", "ceil", "cip", "n", "pad"), _cases(), ids=str)
def test_pool_then_use_matches_eager(
    kind: str, dim: int, ceil: bool, cip: bool, n: int, pad: int
) -> None:
    kw: dict[str, object] = dict(kernel_size=3, stride=2, padding=pad, ceil_mode=ceil)
    if kind == "avg":
        kw["count_include_pad"] = cip
    model = _PoolThenUse(_CLS[(kind, dim)](**kw)).to(COMPILE_DEVICE).eval()
    lucid.manual_seed(0)
    size = n if dim < 3 else min(n, 9)
    x = lucid.randn(1, 2, *(size,) * dim).to(COMPILE_DEVICE)

    want = model(x).numpy()
    compiled = lucid.compile(model)
    got = compiled(x).numpy()
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
    fell_back = bool(compiled.cache_info()["eager_only"])
    if not (kind == "avg" and ceil and cip and pad):
        assert not fell_back, f"{kind}{dim}d ceil={ceil} pad={pad} did not compile"
