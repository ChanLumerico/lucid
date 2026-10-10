"""Export refuses a model that reads a traced value on the host (LCD-337).

The refusal used to fire only when an output was missing from the trace.
``x * 0 + lucid.full_like(x, x.sum().item())`` traces whole — the read
value enters as a constant — so it exported with the example's sum baked
in and answered every later input with it. The tracer marks any host read
of a traced value, and of an argument before an op has consumed it, and
export refuses the marked trace whatever its outputs look like.

The same reads, as the compile path meets them, are in
``lucid/test/unit/compile/test_host_reads.py``; there they run eagerly.
"""

import pytest

import lucid
import lucid.coreml as cml
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._host_reads import (
    CONSTANT_READS,
    HOST_READS,
    STATE_READS,
    Apply,
    inputs,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


@pytest.mark.parametrize("name", sorted(set(HOST_READS) - STATE_READS))
def test_a_host_read_is_refused(name: str, tmp_path: object) -> None:
    first, _second = inputs()
    with pytest.raises(ValueError, match="read on the host"):
        cml.export(
            Apply(HOST_READS[name]).eval(), first, f"{tmp_path}/{name}.mlpackage"
        )


_EXPORTABLE = {**CONSTANT_READS, **{n: HOST_READS[n] for n in STATE_READS}}


#: Exports past the host-read check, then Core ML refuses the package: the
#: ``broadcast_to`` (``tile``) of the folded int64 coordinate index is
#: declared with mismatched input and output dtypes.
_TILE_DTYPE = pytest.mark.xfail(
    strict=True,
    reason="LCD-371: Core ML tile dtype mismatch on an int64-derived broadcast_to",
)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=[_TILE_DTYPE] if n == "advanced-index-bounds" else [])
        for n in sorted(_EXPORTABLE)
    ],
)
def test_a_read_of_a_constant_exports(name: str, tmp_path: object) -> None:
    """A tensor built from Python numbers, a validation-only read, or the
    model's weights — which a package holds as constants — all export."""
    first, second = inputs()
    model = Apply(_EXPORTABLE[name]).eval()
    exported = cml.export(model, first, f"{tmp_path}/{name}.mlpackage")
    got = exported.predict(second)
    assert isinstance(got, lucid.Tensor)
    assert float((got - model(second)).abs().max().item()) <= 1e-5
