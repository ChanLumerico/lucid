"""Per-op agreement between an exported package and the eager model.

An end-to-end model test proves the ops that model happens to use. It
says nothing about the rest of the table, and a wrong emitter in an
unused corner waits there until someone exports a model that reaches it.
These run one op at a time.

Every case compares values, never shapes: an emitter bound to the wrong
MIL op — ``negative_slope`` read where Lucid writes ``slope``, ``axis``
where it writes ``dim`` — produces the right shape and the wrong numbers.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class _Apply(nn.Module):
    """One traced callable, so an op can be exported on its own."""

    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


def _check(fn: object, x: lucid.Tensor, tmp_path: object, tol: float = 1e-5) -> None:
    model = _Apply(fn).eval()
    reference = model(x)
    exported = cml.export(model, x, f"{tmp_path}/op.mlpackage")
    try:
        got = exported.predict(x)
        assert got.shape == reference.shape
        scale = float(reference.abs().max().item()) or 1.0
        assert float((got - reference).abs().max().item()) / scale < tol
    finally:
        exported.close()


# ── the tables ───────────────────────────────────────────────────────────────
#
# ``positive`` marks the ops whose domain excludes the negatives that
# ``randn`` would otherwise hand them.

_UNARY = [
    ("abs", lucid.abs, False),
    ("arccos", lambda x: lucid.arccos(lucid.tanh(x)), False),
    ("arcsin", lambda x: lucid.arcsin(lucid.tanh(x)), False),
    ("arctan", lucid.arctan, False),
    ("ceil", lucid.ceil, False),
    ("cos", lucid.cos, False),
    ("cosh", lucid.cosh, False),
    ("erf", lucid.erf, False),
    ("exp", lucid.exp, False),
    ("floor", lucid.floor, False),
    ("log", lucid.log, True),
    ("log2", lucid.log2, True),
    ("neg", lambda x: -x, False),
    ("reciprocal", lucid.reciprocal, True),
    ("round", lucid.round, False),
    ("rsqrt", lucid.rsqrt, True),
    ("sign", lucid.sign, False),
    ("sin", lucid.sin, False),
    ("sinh", lucid.sinh, False),
    ("sqrt", lucid.sqrt, True),
    ("square", lucid.square, False),
    ("tan", lucid.tan, False),
]

_ACTIVATION = [
    ("elu", F.elu),
    ("mish", F.mish),
    ("selu", F.selu),
    ("softplus", F.softplus),
    ("log_softmax", lambda x: F.log_softmax(x, dim=1)),
]

_OTHER = [
    ("clip", lambda x: lucid.clip(x, -0.5, 0.5), False),
    ("flip", lambda x: lucid.flip(x, dims=1), False),
    ("masked_fill", lambda x: lucid.masked_fill(x, x > 0, 0.0), False),
    ("maximum", lambda x: lucid.maximum(x, x * 2), False),
    ("minimum", lambda x: lucid.minimum(x, x * 2), False),
    ("pad", lambda x: lucid.pad(x, (1, 1, 1, 1)), False),
    ("pow", lambda x: lucid.pow(x, 2.0), True),
    ("prod", lambda x: lucid.prod(x, dim=1), True),
    ("sum", lambda x: lucid.sum(x, dim=1), False),
    ("tile", lambda x: lucid.tile(x, (1, 1, 2, 2)), False),
]


@pytest.mark.parametrize(
    ("name", "fn", "positive"), _UNARY, ids=[c[0] for c in _UNARY]
)
def test_a_unary_op_matches(
    name: str, fn: object, positive: bool, tmp_path: object
) -> None:
    x = lucid.rand(1, 4, 6, 6) + 0.5 if positive else lucid.randn(1, 4, 6, 6)
    _check(fn, x, tmp_path)


@pytest.mark.parametrize(
    ("name", "fn"), _ACTIVATION, ids=[c[0] for c in _ACTIVATION]
)
def test_an_activation_matches(name: str, fn: object, tmp_path: object) -> None:
    _check(fn, lucid.randn(1, 4, 6, 6), tmp_path)


@pytest.mark.parametrize(
    ("name", "fn", "positive"), _OTHER, ids=[c[0] for c in _OTHER]
)
def test_a_structural_op_matches(
    name: str, fn: object, positive: bool, tmp_path: object
) -> None:
    x = lucid.rand(1, 4, 6, 6) + 0.5 if positive else lucid.randn(1, 4, 6, 6)
    _check(fn, x, tmp_path)


class TestTheOutputBufferIsReadCorrectly:
    """Core ML does not always hand back a packed array.

    On the paths that allow the Neural Engine, an output whose innermost
    dimension is not a multiple of the alignment comes back padded. Copied
    as if packed, the values interleave with the padding: right shape,
    wrong numbers, no error. ``tile`` on the last axis is the smallest
    case that produces one.
    """

    @pytest.mark.parametrize(
        "units",
        [cml.ComputeUnits.ALL, cml.ComputeUnits.CPU_ONLY, cml.ComputeUnits.CPU_AND_NE],
        ids=["all", "cpu", "cpu_and_ne"],
    )
    def test_a_padded_output_is_not_read_as_packed(
        self, units: object, tmp_path: object
    ) -> None:
        x = lucid.arange(0, 12).reshape(1, 1, 3, 4) * 1.0
        model = _Apply(lambda t: lucid.tile(t, (1, 1, 1, 2))).eval()
        reference = model(x)
        exported = cml.export(
            model, x, f"{tmp_path}/strided.mlpackage", compute_units=units
        )
        try:
            got = exported.predict(x)
            assert float((got - reference).abs().max().item()) == 0.0
        finally:
            exported.close()
