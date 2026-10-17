"""Layout probes must preserve dtype and observe both complex components."""

import numpy as np
import pytest

import lucid
from lucid.test.audit import _specs, _surface
from lucid.test.audit._axes import Context
from lucid.test.audit._axes_stability import ContiguityAxis
from lucid.test.audit._result import Status
from lucid.test._fixtures.devices import metal_available


@pytest.mark.parametrize(
    "dtype", [lucid.int64, lucid.float16, lucid.complex64, lucid.complex128]
)
def test_layout_probe_preserves_values_dtype_and_uses_an_offset_view(
    monkeypatch: pytest.MonkeyPatch,
    dtype: lucid.dtype,
) -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]])
    if dtype in (lucid.complex64, lucid.complex128):
        values = values + 1j * values
    original = lucid.tensor(values, dtype=dtype)
    call = _specs.Call([original])
    seen_views = []

    def identity(x: lucid.Tensor) -> lucid.Tensor:
        assert x.dtype == dtype
        np.testing.assert_array_equal(x.numpy(), values)
        if not x.is_contiguous():
            seen_views.append(x.storage_offset())
        return x.clone()

    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    finding = axis.run(
        _surface.Symbol("lucid.identity", "lucid", "op", identity), Context()
    )
    assert finding.status == Status.PASS, finding.detail
    assert seen_views and all(offset > 0 for offset in seen_views)


def test_layout_probe_detects_an_imaginary_only_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _specs.Call([lucid.tensor([1.0, 2.0, 3.0], dtype=lucid.complex64)])

    def corrupt_imaginary(x: lucid.Tensor) -> lucid.Tensor:
        out = x.clone()
        if not x.is_contiguous():
            out = out + lucid.tensor([1j, 1j, 1j], dtype=lucid.complex64)
        return out

    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    symbol = _surface.Symbol("lucid.corrupt", "lucid", "op", corrupt_imaginary)
    finding = axis.run(symbol, Context())
    assert finding.status == Status.FAIL, finding.detail


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_a_metal_layout_probe_says_its_layout_half_is_vacuous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Metal slice is a packed array of its own, whatever layout it reports
    # for the reference's sake (LCD-513): an op reads it as a fresh tensor, so
    # agreeing with the packed probe proves nothing about strided reads.
    original = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="metal")
    call = _specs.Call([original])
    axis = ContiguityAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    symbol = _surface.Symbol("lucid.copy", "lucid", "op", lambda x: x.clone())
    finding = axis.run(symbol, Context())
    assert finding.status == Status.PASS, finding.detail
    assert "vacuous" in finding.detail
