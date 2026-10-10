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
from lucid.test.unit.compile._host_reads import HOST_READS, Apply, inputs

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


@pytest.mark.parametrize("name", list(HOST_READS))
def test_a_host_read_is_refused(name: str, tmp_path: object) -> None:
    first, _second = inputs()
    with pytest.raises(ValueError, match="read on the host"):
        cml.export(
            Apply(HOST_READS[name]).eval(), first, f"{tmp_path}/{name}.mlpackage"
        )


def test_a_value_made_on_the_spot_may_be_read(tmp_path: object) -> None:
    """A tensor built from Python numbers holds no input data: it exports."""

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return x * lucid.tensor([2.0, 3.0]).tolist()[1]

    first, second = inputs()
    exported = cml.export(Apply(fn).eval(), first, f"{tmp_path}/spot.mlpackage")
    got = exported.predict(second)
    assert isinstance(got, lucid.Tensor)
    assert float((got - second * 3).abs().max().item()) <= 1e-5
