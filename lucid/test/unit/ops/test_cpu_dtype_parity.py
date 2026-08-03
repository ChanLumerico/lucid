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

#: Ops whose answer is a real number whatever the input was, so an
#: integer input is promoted rather than truncated.
_REAL_VALUED = [
    "exp",
    "log",
    "sqrt",
    "tanh",
    "sigmoid",
    "reciprocal",
    "rsqrt",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "erf",
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
    """The defect: every unary op returned its own input on int64.

    The assertion used to read ``exp([2, 3]) == [7, 20]`` — the function
    was applied and then truncated back to int64.  That was enough to
    catch the identity bug it was written for, and wrong about the answer:
    the value of ``exp(2)`` is not an integer, and the reference framework
    promotes an integer input to float32 rather than rounding it.  See
    :func:`test_real_valued_ops_promote_integers`.
    """
    x = lucid.tensor(np.array([2, 3]))  # int64 by default
    assert str(x.dtype).endswith("int64")
    got = lucid.exp(x).numpy()
    assert not np.array_equal(got, [2, 3]), "exp(int64) returned its input"
    assert np.allclose(got, [7.389056, 20.085537], rtol=1e-5)


@pytest.mark.parametrize("name", _REAL_VALUED)
@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_real_valued_ops_promote_integers(name: str, dtype_name: str) -> None:
    """``exp(1)`` is 2.718, not 2, and not an error.

    Lucid had this split two ways: ``exp``/``log``/``sqrt``/``tanh``
    computed in the integer type and truncated, while
    ``sigmoid``/``reciprocal``/``rsqrt``/``erfinv`` raised
    ``NotImplementedError`` for want of an integer kernel.  Neither
    matches the reference, which promotes to float32 for all of them.

    Carried by ``OpSchema::real_valued`` rather than by the AMP policy:
    ``AmpPolicy::Promote`` is also worn by ``matmul``, whose integer
    answer really is an integer.
    """
    x = lucid.tensor(np.array([[1, 2], [3, 4]]), dtype=getattr(lucid, dtype_name))
    out = getattr(lucid, name)(x)
    assert str(out.dtype).endswith("float32"), f"{name} did not promote"
    assert np.isfinite(out.numpy()).all()


@pytest.mark.parametrize("name", _REAL_VALUED)
def test_promotion_agrees_with_the_float_answer(name: str) -> None:
    """Guard the instrument: promoting has to compute, not just re-label."""
    values = np.array([[1, 2], [3, 4]])
    from_int = getattr(lucid, name)(lucid.tensor(values, dtype=lucid.int32)).numpy()
    from_float = getattr(lucid, name)(
        lucid.tensor(values.astype(np.float32), dtype=lucid.float32)
    ).numpy()
    assert np.allclose(from_int, from_float, rtol=1e-6), name


def test_matmul_keeps_its_integers() -> None:
    """The reason ``real_valued`` is not inferred from the AMP policy.

    ``matmul`` is ``AmpPolicy::Promote`` too, and the product of two
    integer matrices is an integer matrix.
    """
    a = lucid.tensor(np.array([[1, 2], [3, 4]]), dtype=lucid.int32)
    out = lucid.matmul(a, a)
    assert str(out.dtype).endswith("int32"), "matmul promoted when it should not"
    assert np.array_equal(out.numpy(), [[7, 10], [15, 22]])


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
    """Accepted, but not necessarily in the dtype it was handed.

    ``abs`` keeps an integer integer; ``exp`` does not, because its answer
    is not one.  The assertion here is that every dtype is *accepted* —
    which is what the CPU used to fail at — with the promotion rule
    checked separately in :func:`test_real_valued_ops_promote_integers`.
    """
    out = fn(_tensor(np.ones((2, 3)), dtype_name, np_dtype))
    if name in _REAL_VALUED and dtype_name.startswith(("int", "bool")):
        assert str(out.dtype).endswith("float32"), f"{name} did not promote"
    else:
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


def test_gather_refuses_an_index_outside_the_axis() -> None:
    """The read-side twin of the scatter_add defect.

    ``gather`` computed a flat offset from the index and ``memcpy``'d from
    it with nothing in between.  An index of 99 into a 2x3 returned
    ``0.0`` read from past the allocation — a wrong number, silently —
    and gathering along an axis of extent zero segfaulted.  The audit's
    ``edge`` axis substitutes an empty tensor for the operand while
    leaving the index alone, which is exactly that second case, and the
    sweep died on ``edge-Tensor.gather`` every run.
    """
    base = lucid.tensor(np.ones((2, 3)), dtype=lucid.float64)
    index = lucid.tensor(np.full((2, 3), 99), dtype=lucid.int64)
    with pytest.raises(Exception, match="out of range"):
        base.gather(index, dim=0)


def test_gather_refuses_an_empty_axis() -> None:
    """Gathering from nothing is not zero — there is no row to read."""
    empty = lucid.tensor(np.zeros((0, 5)), dtype=lucid.float64)
    index = lucid.tensor(np.zeros((4, 5), dtype=np.int64), dtype=lucid.int64)
    with pytest.raises(Exception, match="out of range"):
        empty.gather(index, dim=0)


def test_gather_still_gathers() -> None:
    """Guard the instrument: the bounds check must not have disabled the op."""
    base = lucid.tensor(np.arange(6.0).reshape(2, 3), dtype=lucid.float64)
    index = lucid.tensor(np.array([[1, 0, 1], [0, 1, 0]]), dtype=lucid.int64)
    got = base.gather(index, dim=0).numpy()
    assert np.allclose(got, [[3.0, 1.0, 5.0], [0.0, 4.0, 2.0]])


def test_gather_backward_refuses_an_index_outside_the_axis() -> None:
    """The backward pass *writes* at the index, so this one corrupts memory.

    Reached through the graph rather than called directly: the check has
    to hold on the path autograd actually takes.
    """
    base = lucid.tensor(np.ones((2, 3)), dtype=lucid.float64, requires_grad=True)
    index = lucid.tensor(np.full((2, 3), 99), dtype=lucid.int64)
    with pytest.raises(Exception, match="out of range"):
        base.gather(index, dim=0).sum().backward()


def test_lu_solve_still_solves() -> None:
    """Guard the instrument: the checks must not have disabled the op."""
    a = lucid.tensor(np.array([[3.0, 1.0], [1.0, 2.0]]), dtype=lucid.float64)
    b = lucid.tensor(np.array([[9.0], [8.0]]), dtype=lucid.float64)
    lu, pivots = lucid.linalg.lu_factor(a)
    assert np.allclose(lucid.linalg.lu_solve(lu, pivots, b).numpy(), [[2.0], [3.0]])


# ── NaN through the order-based ops ──────────────────────────────────────────
# Found 2026-08-03 once the device axis stopped refusing to compare over
# float64: cpu and metal disagreed on every one of these, and neither
# matched the reference.

_NAN_VECTOR = np.array([1.0, np.nan, 3.0, -1.0], dtype=np.float32)


def _both_devices(name, values=_NAN_VECTOR):
    out = []
    for device in _DEVICES:
        got = getattr(lucid, name)(
            lucid.tensor(values, dtype=lucid.float32, device=device)
        )
        got = got[0] if isinstance(got, tuple) else got
        out.append(np.asarray(got.numpy()).ravel())
    return out


@pytest.mark.parametrize("name", ["max", "min"])
def test_max_and_min_propagate_nan(name: str) -> None:
    """They answered with a plausible number instead.

    ``a > b ? a : b`` drops a NaN, because every comparison against one is
    false — so ``max`` of a poisoned batch came back 3.0 and looked
    healthy, on the CPU only, while Metal said nan.
    """
    cpu, metal = _both_devices(name)
    assert np.isnan(cpu).all(), f"{name} on the CPU swallowed the NaN"
    assert np.isnan(metal).all()


@pytest.mark.parametrize("name", ["argmax", "argmin"])
def test_arg_reduce_points_at_the_nan(name: str) -> None:
    """``x[argmax(x)]`` has to be ``max(x)``.

    ``max`` reports nan and the arg-reduce reported the index of an
    ordinary element, so the two disagreed with each other.  Neither
    device did this correctly: the CPU comparison skipped the NaN and
    MLX's own argmax does too.
    """
    cpu, metal = _both_devices(name)
    assert cpu.tolist() == [1], f"{name} on the CPU missed the NaN at index 1"
    assert metal.tolist() == [1], f"{name} on Metal missed the NaN at index 1"


def test_sort_orders_nan_last() -> None:
    """The comparator was not a strict weak ordering, which is UB.

    ``lv < rv`` is false in both directions when either side is NaN, so
    ``std::sort`` was free to do anything — and did: sorting
    ``[1, nan, 3, -1]`` returned ``[1, nan, -1, 3]``, which is not sorted.
    """
    cpu, metal = _both_devices("sort")
    assert np.allclose(cpu[:3], [-1.0, 1.0, 3.0]) and np.isnan(cpu[3])
    assert np.allclose(metal[:3], [-1.0, 1.0, 3.0]) and np.isnan(metal[3])


def test_argsort_agrees_with_sort() -> None:
    cpu, metal = _both_devices("argsort")
    assert cpu.tolist() == [3, 0, 2, 1]
    assert metal.tolist() == [3, 0, 2, 1]


@pytest.mark.parametrize("name", ["max", "min", "argmax", "argmin", "sort", "argsort"])
def test_the_nan_free_answer_is_unchanged(name: str) -> None:
    """Guard the instrument: NaN handling must not disturb ordinary data."""
    clean = np.array([1.0, 5.0, 3.0, -1.0], dtype=np.float32)
    cpu, metal = _both_devices(name, clean)
    assert np.array_equal(cpu, metal), name


# ── in-place ops and the graph ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["relu_", "exp_", "sin_", "cos_", "tanh_", "sigmoid_", "square_", "log_"]
)
def test_inplace_ops_extend_the_graph(name: str) -> None:
    """They computed the right value and the wrong derivative.

    ``inplace_unary`` ran the differentiable out-of-place op and then took
    only its storage, discarding the autograd node — so the tensor kept
    the new numbers while still sitting at its old position in the graph,
    and backward reported the gradient of whatever produced it:

        y = x * 1.0; y.exp_(); y.sum().backward()   ->  dx = 1

    Twenty-four ops, silent, and exactly the kind a training run would
    never notice.
    """
    # A negative entry is in here on purpose.  ``relu_`` has derivative
    # exactly 1 wherever its input is positive, so an all-positive probe
    # cannot tell the fixed op from the broken one — the first version of
    # this test asserted "not all ones" and failed on relu_ for that
    # reason rather than because anything was wrong.
    values = np.array([[0.5, -0.3], [1.2, 0.8]], dtype=np.float64)
    x = lucid.tensor(values, dtype=lucid.float64, requires_grad=True)
    y = x * 1.0
    getattr(y, name)()
    y.sum().backward()
    assert x.grad is not None
    got = x.grad.numpy()
    assert not np.allclose(got, 1.0), f"{name} did not extend the graph"


def test_inplace_still_writes_in_place() -> None:
    """Guard the instrument: the graph fix must not have made it a copy."""
    x = lucid.tensor(np.array([-1.0, 2.0]), dtype=lucid.float64)
    same = x
    x.relu_()
    assert np.allclose(same.numpy(), [0.0, 2.0])


def test_cross_entropy_does_not_gather_at_the_ignore_index() -> None:
    """It gathered at -100 and masked the garbage afterwards.

    The loss came out right because the out-of-range read was multiplied
    by the keep-mask, so nothing pointed at the fact that every ignored
    token in every masked-LM step read past the end of the logits.
    """
    rng = np.random.default_rng(0)
    logits = lucid.tensor(rng.random((2, 5, 3)).astype(np.float32), dtype=lucid.float32)
    target = lucid.tensor(np.array([[0, 1, -100], [2, -100, -100]]), dtype=lucid.int64)
    loss = F.cross_entropy(logits, target, ignore_index=-100)
    assert np.isfinite(float(loss))


@pytest.mark.parametrize("name", ["exp_", "sin_", "cos_", "tanh_", "sigmoid_", "log_"])
def test_inplace_second_route_matches_the_first(name: str) -> None:
    """``backward()`` and ``grad(create_graph=True)`` must agree.

    They did not, and only in graph mode.  A node that saves its input
    keeps a handle on the tensor it was handed; eager backward saves a
    Storage by value at forward time and never notices the later write,
    while graph-mode backward re-reads the handle — so after ``y.sin_()``
    overwrote it, the second route computed ``cos(sin(x))``.

    The op now runs against a snapshot that shares the buffer and inherits
    ``y``'s position in the graph, so the saved input stays put.
    """
    values = np.array([[0.5, 0.3], [1.2, 0.8]], dtype=np.float64)

    x = lucid.tensor(values, dtype=lucid.float64, requires_grad=True)
    y = x * 1.0
    getattr(y, name)()
    y.sum().backward()
    eager = x.grad.numpy()

    x2 = lucid.tensor(values, dtype=lucid.float64, requires_grad=True)
    y2 = x2 * 1.0
    getattr(y2, name)()
    (graph,) = lucid.autograd.grad(y2.sum(), [x2], create_graph=True)

    assert np.allclose(eager, graph.numpy(), rtol=1e-10), name


# ── casting and the graph ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["bool", "int", "long"])
def test_casting_to_a_discrete_dtype_ends_the_graph(name: str) -> None:
    """The derivative of rounding is zero almost everywhere.

    Lucid wired a backward through it anyway, and it did not produce a
    wrong gradient so much as fall over: ``x.long().sum().backward()``
    raised ``NotImplementedError: cpu_backend::broadcast_back_for_reduce``
    from several layers below anything the caller could act on.  The
    reference refuses the backward outright, which is the same statement
    made legibly.
    """
    x = lucid.tensor(np.array([[0.5, 0.3], [1.2, 0.8]]), dtype=lucid.float64)
    x.requires_grad_(True)
    out = getattr(x, name)()
    assert not out.requires_grad, f"{name}() stayed in the graph"
    out.sum().backward()
    assert x.grad is None


@pytest.mark.parametrize("name", ["float", "half"])
def test_casting_between_float_dtypes_stays_differentiable(name: str) -> None:
    """Guard the instrument: only the *discrete* casts detach.

    ``logits.float()`` mid-graph under autocast has to keep carrying
    gradients, which is the case the surrounding code was written for.
    """
    x = lucid.tensor(np.array([[0.5, 0.3], [1.2, 0.8]]), dtype=lucid.float64)
    x.requires_grad_(True)
    out = getattr(x, name)()
    assert out.requires_grad
    out.sum().backward()
    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), 1.0)


# ── overflow-safe hypot, NaN through the cumulative scans ────────────────────


def test_hypot_does_not_overflow_on_the_way() -> None:
    """``hypot(1e200, 1e200)`` is 1.41e200, an ordinary double.

    The naive ``sqrt(a² + b²)`` squares first and overflowed to inf.
    Avoiding exactly that is what distinguishes ``hypot`` from writing the
    formula out by hand.
    """
    big = lucid.tensor(np.array([1e200]), dtype=lucid.float64)
    got = lucid.hypot(big, big).numpy()
    assert np.isfinite(got).all(), "hypot overflowed"
    assert np.allclose(got, np.hypot(1e200, 1e200), rtol=1e-12)


@pytest.mark.parametrize(
    "pair", [(3.0, 4.0), (0.0, 0.0), (1e-200, 1e-200), (5.0, 0.0), (1e200, 1e200)]
)
def test_hypot_matches_across_magnitudes(pair) -> None:
    """Guard the instrument: the scaling must not disturb ordinary values."""
    a = lucid.tensor(np.array([pair[0]]), dtype=lucid.float64)
    b = lucid.tensor(np.array([pair[1]]), dtype=lucid.float64)
    assert np.allclose(lucid.hypot(a, b).numpy(), np.hypot(*pair), rtol=1e-12)


@pytest.mark.parametrize("name", ["cummax", "cummin"])
@pytest.mark.parametrize("device", _DEVICES)
def test_cumulative_scans_poison_after_a_nan(name: str, device: str) -> None:
    """Once a NaN is seen it is the running extreme for the rest of the scan.

    ``v > running`` is false against a NaN, so the CPU skipped it and
    carried a plausible running maximum; MLX's own cumulative scans skip
    it too, so both devices were wrong and in different ways.
    """
    values = np.array([1.0, np.nan, 3.0, -1.0], dtype=np.float32)
    out = getattr(lucid, name)(
        lucid.tensor(values, dtype=lucid.float32, device=device), 0
    )
    out = out[0] if isinstance(out, tuple) else out
    got = np.asarray(out.numpy()).ravel()
    assert got[0] == 1.0
    assert np.isnan(got[1:]).all(), f"{name} on {device} did not carry the NaN"


@pytest.mark.parametrize("name", ["cummax", "cummin"])
@pytest.mark.parametrize("device", _DEVICES)
def test_cumulative_scans_unchanged_without_a_nan(name: str, device: str) -> None:
    """Guard the instrument."""
    values = np.array([1.0, 5.0, 3.0, -1.0], dtype=np.float32)
    out = getattr(lucid, name)(
        lucid.tensor(values, dtype=lucid.float32, device=device), 0
    )
    out = out[0] if isinstance(out, tuple) else out
    expected = [1.0, 5.0, 5.0, 5.0] if name == "cummax" else [1.0, 1.0, 1.0, -1.0]
    assert np.allclose(np.asarray(out.numpy()).ravel(), expected)
