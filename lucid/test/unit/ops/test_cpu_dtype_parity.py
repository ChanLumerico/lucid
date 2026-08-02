"""Regression tests: the CPU backend refused dtypes Metal accepted.

Found 2026-08-02 by the audit's ``dtype`` axis, which runs every op over
every dtype on both devices and compares the two sets.

Three separate gaps, and one of them was not a refusal at all:

**Unary on int64 returned its own input.**  ``unary_op``'s I64 branch read

    op[i] = (std::int64_t)(double)(double)ip[i];

which is an identity — the function was never applied.  int64 is Lucid's
default integer dtype, so ``lucid.exp(lucid.tensor([2, 3]))`` answered
``[2, 3]`` instead of ``[7, 20]``, silently, on the CPU only, while Metal
was correct.  Of the defects this audit has found, this is the one that
produced a plausible wrong number rather than an error.

**Unary refused bool / int8 / int16 / float16.**  Accelerate has no
vector maths below 32 bits and no half accumulator, so those dtypes need
the widen-reuse-narrow shape ``binary_op`` already used.

**Reductions and convolutions refused the same set.**  ``reduce_one_axis``
handled F32 and F64 only; ``conv_nd_forward`` handled F32 and F64 only.

The tests below assert cpu/metal agreement rather than hand-written
values wherever both devices can run the op, because a hand-written
expected value encodes a convention rather than checking one.  float64 is
excluded from the comparison: it does not exist on Metal, and the engine
documents the downcast.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg
import lucid.nn.functional as F

_DEVICES = ["cpu", "metal"]

#: Every dtype that exists on both devices.
_SHARED = [
    ("bool", np.bool_),
    ("int8", np.int8),
    ("int16", np.int16),
    ("int32", np.int32),
    ("int64", np.int64),
    ("float16", np.float16),
    ("float32", np.float32),
]

_UNARY = [
    ("exp", lucid.exp),
    ("tanh", lucid.tanh),
    ("sin", lucid.sin),
    ("abs", lucid.abs),
    ("sqrt", lucid.sqrt),
]


def _tensor(values, dtype_name, np_dtype, device="cpu"):
    return lucid.tensor(
        np.ascontiguousarray(np.asarray(values).astype(np_dtype)),
        dtype=getattr(lucid, dtype_name),
        device=device,
    )


# ── the silent wrong answer ──────────────────────────────────────────────────


def test_unary_on_int64_actually_applies_the_function() -> None:
    """The defect: every unary op returned its own input on int64."""
    x = lucid.tensor(np.array([2, 3]))  # int64 by default
    assert str(x.dtype).endswith("int64")
    assert np.array_equal(
        lucid.exp(x).numpy(), [7, 20]
    ), "exp(int64) returned its input"
    assert np.array_equal(lucid.tanh(x).numpy(), [0, 0])


@pytest.mark.parametrize("name,fn", _UNARY)
def test_int64_matches_int32(name: str, fn) -> None:
    """Guard the instrument.

    The I32 path was always correct, so if I64 now disagrees with it the
    fix is wrong in a different way.  Both truncate towards zero, so the
    two must land on the same integers.
    """
    values = [1, 2, 3, 4]
    got64 = fn(_tensor(values, "int64", np.int64)).numpy()
    got32 = fn(_tensor(values, "int32", np.int32)).numpy()
    assert np.array_equal(got64, got32), name


# ── dtype parity ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype_name,np_dtype", _SHARED)
@pytest.mark.parametrize("name,fn", _UNARY)
def test_unary_accepts_every_shared_dtype_on_the_cpu(
    name: str, fn, dtype_name: str, np_dtype
) -> None:
    out = fn(_tensor(np.ones((2, 3)), dtype_name, np_dtype))
    assert str(out.dtype).endswith(dtype_name), f"{name} changed dtype"


@pytest.mark.parametrize("dtype_name,np_dtype", _SHARED)
def test_unary_agrees_across_devices(dtype_name: str, np_dtype) -> None:
    """The comparison the old sweeps could not make, because the CPU raised."""
    values = np.ones((2, 3)) * 2
    cpu_out = lucid.tanh(_tensor(values, dtype_name, np_dtype, "cpu"))
    metal_out = lucid.tanh(_tensor(values, dtype_name, np_dtype, "metal"))
    assert str(cpu_out.device) == "device('cpu')"
    assert str(metal_out.device) == "device('metal')"
    assert np.allclose(
        cpu_out.numpy().astype(np.float64),
        metal_out.numpy().astype(np.float64),
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
    ), dtype_name


@pytest.mark.parametrize("dtype_name,np_dtype", _SHARED)
def test_prod_accepts_every_shared_dtype(dtype_name: str, np_dtype) -> None:
    """``reduce_one_axis`` handled F32 and F64 only."""
    out = lucid.prod(_tensor(np.ones((2, 3)) * 2, dtype_name, np_dtype))
    value = float(np.asarray(out.numpy()).reshape(-1)[0])
    assert value == (1.0 if dtype_name == "bool" else 64.0), dtype_name


@pytest.mark.parametrize("device", _DEVICES)
def test_conv2d_accepts_float16(device: str) -> None:
    """``conv_nd_forward`` handled F32 and F64 only; Metal took F16."""
    x = _tensor(
        np.random.default_rng(0).random((1, 2, 4, 4)), "float16", np.float16, device
    )
    w = _tensor(
        np.random.default_rng(1).random((2, 2, 3, 3)), "float16", np.float16, device
    )
    out = F.conv2d(x, w, None, padding=1)
    assert out.shape == (1, 2, 4, 4)
    assert str(out.dtype).endswith("float16")


def test_conv2d_float16_matches_float32_closely() -> None:
    """Guard the instrument: accepting F16 is not enough, it has to compute.

    The half path widens to float, convolves, and rounds once — so it must
    land near the float32 answer rather than merely returning a
    correctly-shaped buffer.
    """
    rng = np.random.default_rng(3)
    x_values = rng.random((1, 2, 4, 4)) - 0.5
    w_values = rng.random((2, 2, 3, 3)) - 0.5
    half = F.conv2d(
        _tensor(x_values, "float16", np.float16),
        _tensor(w_values, "float16", np.float16),
        None,
        padding=1,
    ).numpy()
    single = F.conv2d(
        _tensor(x_values, "float32", np.float32),
        _tensor(w_values, "float32", np.float32),
        None,
        padding=1,
    ).numpy()
    assert np.allclose(half.astype(np.float64), single.astype(np.float64), atol=5e-3)
    assert np.abs(half).max() > 0.0, "the half path returned an empty buffer"


def test_half_conversion_has_one_home() -> None:
    """The helpers were duplicated to reach the reduction kernels; they are not.

    Two copies of IEEE rounding code is how two copies quietly disagree.
    """
    from pathlib import Path

    root = Path(lucid.__file__).parent / "_C"
    definitions = [
        path
        for path in root.rglob("*.h")
        if "inline float half_bits_to_float" in path.read_text()
    ]
    assert len(definitions) == 1, [str(p) for p in definitions]
    assert definitions[0].name == "Half.h"


# ── memory safety ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_scatter_add_reads_its_index_at_the_right_width(dtype_name: str) -> None:
    """The worst of the set: a bus error reachable from Python.

    The index buffer was read as ``int32`` whatever its dtype.  An int16
    index had two of its values read as one — a silently wrong result —
    and an int8 index had four read as one, producing an index far outside
    the base and an out-of-bounds write.  ``pytest -m audit`` took the
    whole process down on ``dtype-lucid.scatter_add``.
    """
    base = lucid.tensor(np.zeros((2, 4), dtype=np.float32))
    index = lucid.tensor(
        np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), dtype=getattr(lucid, dtype_name)
    )
    source = lucid.tensor(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], np.float32)
    )
    got = lucid.scatter_add(base, 1, index, source).numpy()
    assert np.allclose(got, [[1.0, 2.0, 3.0, 4.0], [8.0, 7.0, 6.0, 5.0]]), dtype_name


def test_scatter_add_refuses_an_out_of_range_index() -> None:
    """It used to write outside the buffer instead."""
    base = lucid.tensor(np.zeros((2, 4), dtype=np.float32))
    index = lucid.tensor(np.array([[99, 0, 0, 0], [0, 0, 0, 0]]), dtype=lucid.int64)
    source = lucid.tensor(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(Exception, match="out of range"):
        lucid.scatter_add(base, 1, index, source)


def test_scatter_add_still_scatters() -> None:
    """Guard the instrument: the bounds check must not have disabled the op."""
    base = lucid.tensor(np.zeros((1, 3), dtype=np.float32))
    index = lucid.tensor(np.array([[1, 1, 1]]), dtype=lucid.int64)
    source = lucid.tensor(np.array([[1.0, 2.0, 3.0]], np.float32))
    got = lucid.scatter_add(base, 1, index, source).numpy()
    assert np.allclose(got, [[0.0, 6.0, 0.0]])


# ── lu_solve pivots ──────────────────────────────────────────────────────────
# Found 2026-08-03: the full audit sweep did not finish, it died. Exit 138 is
# SIGBUS, and the last symbol it printed was the one before this op.


def _pivoting_system() -> "tuple[lucid.Tensor, lucid.Tensor]":
    """A system whose factorization genuinely permutes rows.

    A near-identity matrix pivots trivially, and every wrong pivot width
    happened to give the right answer on one — which is how this looked
    fine until the probe used a matrix with a tiny first row.
    """
    rng = np.random.default_rng(0)
    values = rng.random((5, 5))
    values[0] *= 1e-6
    return (
        lucid.tensor(values, dtype=lucid.float64),
        lucid.tensor(rng.random((5, 1)), dtype=lucid.float64),
    )


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_lu_solve_agrees_with_solve_at_every_pivot_width(dtype_name: str) -> None:
    """LAPACK reads ``ipiv`` as ``const int*``; nothing converted to it.

    int8 and bool were read past the end of their own buffer and took the
    process down with SIGBUS.  int16 and int64 were long enough to
    survive the over-read and *returned*: int16 answered ``5.9e+133`` and
    int64 answered a plausible vector that was not the solution.  Only
    int32 — the width ``lu_factor`` happens to emit — was correct.
    """
    a, b = _pivoting_system()
    reference = lucid.linalg.solve(a, b).numpy().ravel()
    lu, pivots = lucid.linalg.lu_factor(a)
    got = (
        lucid.linalg.lu_solve(lu, pivots.to(getattr(lucid, dtype_name)), b)
        .numpy()
        .ravel()
    )
    assert np.allclose(got, reference, rtol=1e-9), dtype_name


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "bool"])
def test_lu_solve_refuses_non_integer_pivots(dtype_name: str) -> None:
    """A float pivot vector had its mantissa bits read as row indices."""
    a, b = _pivoting_system()
    lu, pivots = lucid.linalg.lu_factor(a)
    with pytest.raises(TypeError, match="must be an integer tensor"):
        lucid.linalg.lu_solve(lu, pivots.to(getattr(lucid, dtype_name)), b)


def test_lu_solve_refuses_an_out_of_range_pivot() -> None:
    """A pivot of 99 on a 4x4 was accepted, and answered."""
    a = lucid.tensor(np.eye(4) + 0.1, dtype=lucid.float64)
    b = lucid.tensor(np.ones((4, 1)), dtype=lucid.float64)
    bad = lucid.tensor(np.array([99, 2, 3, 4]), dtype=lucid.int32)
    with pytest.raises(IndexError, match="out of range"):
        lucid.linalg.lu_solve(a, bad, b)


def test_lu_solve_still_solves() -> None:
    """Guard the instrument: the checks must not have disabled the op."""
    a = lucid.tensor(np.array([[3.0, 1.0], [1.0, 2.0]]), dtype=lucid.float64)
    b = lucid.tensor(np.array([[9.0], [8.0]]), dtype=lucid.float64)
    lu, pivots = lucid.linalg.lu_factor(a)
    assert np.allclose(lucid.linalg.lu_solve(lu, pivots, b).numpy(), [[2.0], [3.0]])
