"""Inside an autocast scope ``sum`` / ``mean`` / ``prod`` run in their input's dtype.

CHA-35.  The three reductions were registered ``AmpPolicy::Promote``, so an
autocast scope cast their input down to the autocast dtype before reducing:
``full((64, 1024), 1.5).sum()`` under ``autocast(device_type="metal")``
answered float16 inf where the reference gives float32 98304, and an
integer or float64 sum was demoted the same way.  They are ``KeepInput``
now.  (``ForceFP32`` would have been wrong too: outside an autocast scope
the SchemaGuard reads it as "the result is a real number" and turns every
integer sum into float32.)

A half input stays half, accumulated in float32 by the kernel.  That is the
reference's CPU autocast for ``sum`` and ``mean``; its GPU autocast widens
``sum`` and ``prod`` of a float16 input to float32, which Lucid's
per-op policies cannot say yet — see
``test_autocast_float32_reduction_dtype_matches_reference``.
"""

import pytest

import lucid
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"]


def _autocast_dtype(device: str) -> lucid.dtype:
    """The autocast dtype that changes something on ``device`` — float16 is
    computed as float32 on the CPU stream, so the CPU scope uses bfloat16."""
    return lucid.float16 if device == "metal" else lucid.bfloat16


@pytest.fixture(params=_DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "metal" and not metal_available():
        pytest.skip("Metal not available on this host")
    return str(request.param)


def _name(dtype: lucid.dtype) -> str:
    return str(dtype).split(".")[-1]


@pytest.mark.parametrize("op", ["sum", "mean", "prod"])
def test_a_float32_reduction_stays_float32(op: str, device: str) -> None:
    x = lucid.full((64, 1024), 1.5 if op != "prod" else 1.0, device=device)
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        got = getattr(x, op)()
    assert got.dtype == lucid.float32
    assert got.item() == {"sum": 98304.0, "mean": 1.5, "prod": 1.0}[op]


def test_an_integer_sum_stays_integral(device: str) -> None:
    x = lucid.arange(100000, device=device)
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        got = x.sum()
    assert got.dtype == lucid.int64
    assert got.item() == 100000 * 99999 // 2


def test_a_float64_sum_stays_float64() -> None:
    x = lucid.full((1000,), 0.1, dtype=lucid.float64)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        got = x.sum()
    assert got.dtype == lucid.float64
    assert got.item() == pytest.approx(100.0, rel=1e-12)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=_name)
def test_a_half_reduction_keeps_its_dtype(device: str, dtype: lucid.dtype) -> None:
    x = lucid.full((64, 64), 1.0, dtype=dtype, device=device)
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        s, m = x.sum(), x.mean()
    assert s.dtype == dtype and m.dtype == dtype
    assert s.item() == 4096.0 and m.item() == 1.0


@pytest.mark.parametrize("op", ["sum", "mean", "prod"])
def test_autocast_float32_reduction_dtype_matches_reference(
    ref: object, op: str, device: str
) -> None:
    """A float32 input keeps its dtype under the reference's autocast on
    either stream.  A half input is where the two still part: the
    reference's GPU autocast widens ``sum`` and ``prod`` of a float16 input
    to float32 (its CPU autocast widens ``prod`` alone), and Lucid keeps the
    input dtype and accumulates in float32 — the SchemaGuard has no policy
    for "widen a half input, keep an integer" yet."""
    ref_device = "mps" if device == "metal" else "cpu"
    if ref_device == "mps" and not ref.backends.mps.is_available():  # type: ignore[attr-defined]
        pytest.skip("the reference's GPU stream is not available")
    ref_dtype = getattr(ref, _name(_autocast_dtype(device)))
    x = lucid.full((8, 8), 1.0, device=device)
    r = ref.full((8, 8), 1.0, device=ref_device)  # type: ignore[attr-defined]
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        got = getattr(x, op)()
    with ref.autocast(device_type=ref_device, dtype=ref_dtype):  # type: ignore[attr-defined]
        want = getattr(r, op)()
    assert _name(got.dtype) == str(want.dtype).split(".")[-1] == "float32"
