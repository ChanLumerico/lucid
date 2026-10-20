"""Hypothesis boundary property tests driven by ``lucid._ops._registry._REGISTRY`` (LCD-268).

Drives every ``OpEntry`` in ``_REGISTRY`` across boundary shapes (``0-d``, ``0``
extents, length-``1`` axes, broadcast edges), dtypes (including ``float64`` and
``complex64`` on CPU/Metal), devices (``cpu`` and ``metal``), and out-of-range
indices, verifying under ``faulthandler`` that every call either:
  (a) completes without crashing,
  (b) refuses with a typed exception, or
  (c) matches the reference framework on shape and finite values when both succeed.
"""

from collections.abc import Callable
import faulthandler
import math
from types import ModuleType

from hypothesis import given, settings, strategies as st
import pytest

import lucid
from lucid._ops._registry import _REGISTRY, OpEntry
from lucid.test._fixtures.devices import metal_available
from lucid.test._fixtures.ref_framework import ref_module

faulthandler.enable()

_DEVICES: tuple[str, ...] = ("cpu", "metal") if metal_available() else ("cpu",)

_DTYPE_NAMES: tuple[str, ...] = (
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
)

_TYPED_REFUSALS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    IndexError,
    ZeroDivisionError,
    NotImplementedError,
)

# ── Hypothesis strategies ────────────────────────────────────────────────────

boundary_shapes: st.SearchStrategy[tuple[int, ...]] = st.lists(
    st.sampled_from((0, 1, 2, 3)), min_size=0, max_size=3
).map(tuple)

nonempty_1d_or_2d_shapes: st.SearchStrategy[tuple[int, ...]] = st.lists(
    st.sampled_from((0, 1, 2, 3)), min_size=1, max_size=2
).map(tuple)

dtype_names: st.SearchStrategy[str] = st.sampled_from(_DTYPE_NAMES)
devices: st.SearchStrategy[str] = st.sampled_from(_DEVICES)


@st.composite
def broadcast_shape_pairs(
    draw: st.DrawFn,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    s1 = draw(boundary_shapes)
    if draw(st.booleans()):
        # Broadcast-compatible partner (or 0-extent edge)
        rank2 = draw(st.integers(min_value=0, max_value=3))
        dims2: list[int] = []
        for i in range(rank2):
            offset = i - (rank2 - len(s1))
            if 0 <= offset < len(s1):
                dims2.append(draw(st.sampled_from((s1[offset], 1, 0))))
            else:
                dims2.append(draw(st.sampled_from((0, 1, 2))))
        return s1, tuple(dims2)
    return s1, draw(boundary_shapes)


def _make_tensor(shape: tuple[int, ...], dtype_name: str, device: str) -> lucid.Tensor:
    dt = getattr(lucid, dtype_name)
    numel = math.prod(shape)
    if numel == 0:
        return lucid.zeros(shape, dtype=dt, device=device)
    if dtype_name == "bool_":
        vals = [(i % 2) == 1 for i in range(numel)]
        return lucid.tensor(vals, dtype=dt, device=device).reshape(shape)
    if dtype_name == "complex64":
        re = lucid.arange(1, numel + 1, dtype=lucid.float32, device=device).reshape(shape)
        im = lucid.ones(shape, dtype=lucid.float32, device=device)
        return lucid.complex(re, im)
    vals_num = [(i % 5) + 1 for i in range(numel)]
    return lucid.tensor(vals_num, dtype=dt, device=device).reshape(shape)


def _make_ref_tensor(
    ref: ModuleType, shape: tuple[int, ...], dtype_name: str
) -> object:
    ref_dt = getattr(ref, "bool" if dtype_name == "bool_" else dtype_name)
    numel = math.prod(shape)
    if numel == 0:
        return ref.zeros(shape, dtype=ref_dt)
    if dtype_name == "bool_":
        vals = [(i % 2) == 1 for i in range(numel)]
        return ref.tensor(vals, dtype=ref_dt).reshape(shape)
    if dtype_name == "complex64":
        re = ref.arange(1, numel + 1, dtype=ref.float32).reshape(shape)
        im = ref.ones(shape, dtype=ref.float32)
        return ref.complex(re, im)
    vals_num = [(i % 5) + 1 for i in range(numel)]
    return ref.tensor(vals_num, dtype=ref_dt).reshape(shape)


def _extract_first_tensor(res: object) -> lucid.Tensor | None:
    if isinstance(res, lucid.Tensor):
        return res
    if isinstance(res, (tuple, list)) and res and isinstance(res[0], lucid.Tensor):
        return res[0]
    return None


def _assert_matches_ref(got_obj: object, ref_obj: object) -> None:
    got = _extract_first_tensor(got_obj)
    if got is None:
        return
    ref_t = ref_obj[0] if isinstance(ref_obj, (tuple, list)) and ref_obj else ref_obj
    if not hasattr(ref_t, "shape"):
        return
    assert tuple(got.shape) == tuple(ref_t.shape)
    if got.numel() == 0:
        return
    got_cpu = got.detach().to("cpu")
    if got_cpu.dtype in (lucid.float16, lucid.bfloat16):
        got_list = got_cpu.to(lucid.float32).reshape(-1).tolist()
        ref_list = ref_t.detach().to(ref_t.dtype).float().reshape(-1).tolist()
        assert got_list == pytest.approx(ref_list, rel=2e-2, abs=2e-2, nan_ok=True)
    elif got_cpu.dtype in (lucid.float32, lucid.float64):
        got_list = got_cpu.reshape(-1).tolist()
        ref_list = ref_t.detach().double().reshape(-1).tolist()
        assert got_list == pytest.approx(ref_list, rel=1e-4, abs=1e-4, nan_ok=True)
    elif got_cpu.dtype != lucid.complex64:
        assert got_cpu.reshape(-1).tolist() == ref_t.detach().reshape(-1).tolist()


def _call_entry(entry: OpEntry, *args: object, **kwargs: object) -> object:
    if entry.inplace:
        first = args[0]
        assert isinstance(first, lucid.Tensor)
        target = first.clone()
        assert entry.method_name is not None
        return getattr(target, entry.method_name)(*args[1:], **kwargs)
    if entry.free_fn_name is not None:
        return getattr(lucid._ops, entry.free_fn_name)(*args, **kwargs)
    assert entry.method_name is not None
    return getattr(args[0], entry.method_name)(*args[1:], **kwargs)


def _call_ref_entry(
    ref: ModuleType, entry: OpEntry, *args: object, **kwargs: object
) -> object:
    if entry.inplace:
        first = args[0]
        target = first.clone()  # type: ignore[attr-defined]
        assert entry.method_name is not None
        if not hasattr(target, entry.method_name):
            raise NotImplementedError
        return getattr(target, entry.method_name)(*args[1:], **kwargs)
    if entry.free_fn_name is not None and hasattr(ref, entry.free_fn_name):
        return getattr(ref, entry.free_fn_name)(*args, **kwargs)
    if entry.method_name is not None and hasattr(args[0], entry.method_name):
        return getattr(args[0], entry.method_name)(*args[1:], **kwargs)
    raise NotImplementedError


# ── Classification of _REGISTRY entries ──────────────────────────────────────

_SPECIAL_UNARY_KWARGS: dict[str, Callable[[lucid.Tensor], tuple[tuple[object, ...], dict[str, object]]]] = {
    "reshape": lambda x: ((-1,), {}),
    "view": lambda x: ((-1,), {}),
    "squeeze": lambda x: ((), {}),
    "squeeze_all": lambda x: ((), {}),
    "unsqueeze": lambda x: ((0,), {}),
    "flatten": lambda x: ((), {}),
    "permute": lambda x: ((tuple(reversed(range(x.ndim))),), {}),
    "transpose": lambda x: ((0, -1), {}),
    "swapaxes": lambda x: ((0, -1), {}),
    "broadcast_to": lambda x: (((2, *x.shape),), {}),
    "expand": lambda x: (((2, *x.shape),), {}),
    "as_strided": lambda x: (((0,), (1,)), {}),
    "expand_dims": lambda x: ((0,), {}),
    "repeat": lambda x: ((2,), {}),
    "repeat_method": lambda x: ((*(2 for _ in range(max(1, x.ndim))),), {}),
    "tile": lambda x: (((2,),), {}),
    "flip": lambda x: (((0,),), {}),
    "fliplr": lambda x: ((), {}),
    "flipud": lambda x: ((), {}),
    "roll": lambda x: ((1,), {}),
    "tril": lambda x: ((), {}),
    "triu": lambda x: ((), {}),
    "sort": lambda x: ((), {}),
    "argsort": lambda x: ((), {}),
    "nonzero": lambda x: ((), {}),
    "unique": lambda x: ((), {}),
    "unique_consecutive": lambda x: ((), {}),
    "topk": lambda x: ((1,), {}),
    "diagonal": lambda x: ((), {}),
    "masked_fill": lambda x: (
        (lucid.zeros(x.shape, dtype=lucid.bool_, device=x.device), 0),
        {},
    ),
    "split": lambda x: ((1,), {}),
    "split_method": lambda x: ((1,), {}),
    "chunk": lambda x: ((2,), {}),
    "unbind": lambda x: ((), {}),
    "softmax": lambda x: ((), {"dim": -1}),
    "log_softmax": lambda x: ((), {"dim": -1}),
    "cummax": lambda x: ((0,), {}),
    "cummin": lambda x: ((0,), {}),
    "cumsum": lambda x: ((0,), {}),
    "cumprod": lambda x: ((0,), {}),
    "clip": lambda x: ((), {"min": 0, "max": 2}),
    "clamp": lambda x: ((), {"min": 0, "max": 2}),
    "clamp_min": lambda x: ((0,), {}),
    "clamp_max": lambda x: ((2,), {}),
    "narrow": lambda x: ((0, 0, 0), {}),
    "kthvalue": lambda x: ((1,), {}),
    "movedim": lambda x: ((0, -1), {}),
    "unflatten": lambda x: ((0, (1, -1)), {}),
    "histc": lambda x: ((), {"bins": 4, "min": 0.0, "max": 4.0}),
    "repeat_interleave": lambda x: ((2,), {}),
}

_INDEXED_NAMES: frozenset[str] = frozenset(
    {
        "gather",
        "scatter",
        "scatter_add",
        "take",
        "index_select",
        "searchsorted",
        "bucketize",
    }
)

_VARIADIC_ZERO_NAMES: frozenset[str] = frozenset(
    {"where", "meshgrid", "cartesian_prod"}
)

_JOIN_ENTRIES: tuple[OpEntry, ...] = tuple(
    e for e in _REGISTRY if e.n_tensor_args == -1
)

_UNARY_PLAIN_ENTRIES: tuple[OpEntry, ...] = tuple(
    e
    for e in _REGISTRY
    if e.n_tensor_args == 1
    and e.name not in _SPECIAL_UNARY_KWARGS
    and e.name not in _INDEXED_NAMES
)

_UNARY_SPECIAL_ENTRIES: tuple[OpEntry, ...] = tuple(
    e for e in _REGISTRY if e.name in _SPECIAL_UNARY_KWARGS
)

_BINARY_ENTRIES: tuple[OpEntry, ...] = tuple(
    e
    for e in _REGISTRY
    if e.n_tensor_args == 2 and e.name not in _INDEXED_NAMES
)

_INDEXED_ENTRIES: tuple[OpEntry, ...] = tuple(
    e for e in _REGISTRY if e.name in _INDEXED_NAMES
)

_VARIADIC_ZERO_ENTRIES: tuple[OpEntry, ...] = tuple(
    e for e in _REGISTRY if e.name in _VARIADIC_ZERO_NAMES
)


@pytest.mark.property
def test_every_registry_entry_is_covered_by_boundary_harness() -> None:
    covered = {
        e.name
        for group in (
            _UNARY_PLAIN_ENTRIES,
            _UNARY_SPECIAL_ENTRIES,
            _BINARY_ENTRIES,
            _JOIN_ENTRIES,
            _INDEXED_ENTRIES,
            _VARIADIC_ZERO_ENTRIES,
        )
        for e in group
    }
    all_names = {e.name for e in _REGISTRY}
    assert covered == all_names


@pytest.mark.property
@settings(
    max_examples=20,
    deadline=None,
    database=None,
)
@given(shape=boundary_shapes, dtype_name=dtype_names, device=devices)
def test_unary_and_reduction_registry_ops_at_boundary_inputs(
    shape: tuple[int, ...], dtype_name: str, device: str
) -> None:
    try:
        x = _make_tensor(shape, dtype_name, device)
    except _TYPED_REFUSALS:
        return
    ref = ref_module()
    rx = _make_ref_tensor(ref, shape, dtype_name) if ref is not None else None

    for entry in _UNARY_PLAIN_ENTRIES:
        try:
            got = _call_entry(entry, x)
        except Exception as exc:
            assert isinstance(exc, _TYPED_REFUSALS), (
                f"{entry.name} raised untyped {type(exc).__name__}: {exc}"
            )
            continue
        if ref is None:
            continue
        try:
            want = _call_ref_entry(ref, entry, rx)
        except Exception:
            continue
        _assert_matches_ref(got, want)


@pytest.mark.property
@settings(
    max_examples=20,
    deadline=None,
    database=None,
)
@given(shape=boundary_shapes, dtype_name=dtype_names, device=devices)
def test_shape_and_parameterized_unary_registry_ops_at_boundary_inputs(
    shape: tuple[int, ...], dtype_name: str, device: str
) -> None:
    try:
        x = _make_tensor(shape, dtype_name, device)
    except _TYPED_REFUSALS:
        return

    for entry in _UNARY_SPECIAL_ENTRIES:
        args, kwargs = _SPECIAL_UNARY_KWARGS[entry.name](x)
        try:
            got = _call_entry(entry, x, *args, **kwargs)
        except Exception as exc:
            assert isinstance(exc, _TYPED_REFUSALS), (
                f"{entry.name} raised untyped {type(exc).__name__}: {exc}"
            )
            continue
        first = _extract_first_tensor(got)
        if first is not None:
            assert first.device.type == device


@pytest.mark.property
@settings(
    max_examples=20,
    deadline=None,
    database=None,
)
@given(
    shapes=broadcast_shape_pairs(),
    dt_a=dtype_names,
    dt_b=dtype_names,
    device=devices,
)
def test_binary_registry_ops_at_boundary_shapes_and_mixed_dtypes(
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
    dt_a: str,
    dt_b: str,
    device: str,
) -> None:
    s1, s2 = shapes
    try:
        a = _make_tensor(s1, dt_a, device)
        b = _make_tensor(s2, dt_b, device)
    except _TYPED_REFUSALS:
        return
    ref = ref_module()
    ra = _make_ref_tensor(ref, s1, dt_a) if ref is not None else None
    rb = _make_ref_tensor(ref, s2, dt_b) if ref is not None else None

    for entry in _BINARY_ENTRIES:
        try:
            got = _call_entry(entry, a, b)
        except Exception as exc:
            assert isinstance(exc, _TYPED_REFUSALS), (
                f"{entry.name} raised untyped {type(exc).__name__}: {exc}"
            )
            continue
        if ref is None:
            continue
        try:
            want = _call_ref_entry(ref, entry, ra, rb)
        except Exception:
            continue
        _assert_matches_ref(got, want)


@pytest.mark.property
@settings(max_examples=15, deadline=None, database=None)
@given(
    shape=boundary_shapes,
    n_tensors=st.sampled_from((0, 1, 2)),
    dtype_name=dtype_names,
    device=devices,
)
def test_join_and_variadic_registry_ops_at_boundary_inputs(
    shape: tuple[int, ...], n_tensors: int, dtype_name: str, device: str
) -> None:
    try:
        tensors = [_make_tensor(shape, dtype_name, device) for _ in range(n_tensors)]
    except _TYPED_REFUSALS:
        return

    for entry in _JOIN_ENTRIES:
        try:
            got = _call_entry(entry, tensors)
        except Exception as exc:
            assert isinstance(exc, _TYPED_REFUSALS), (
                f"{entry.name} raised untyped {type(exc).__name__}: {exc}"
            )
            continue
        first = _extract_first_tensor(got)
        if first is not None:
            assert first.device.type == device

    # Variadic / n_tensor_args == 0 ops: where, meshgrid, cartesian_prod
    if tensors:
        x = tensors[0]
        cond = lucid.zeros(x.shape, dtype=lucid.bool_, device=device)
        for name, call in (
            ("where", lambda: lucid.where(cond, x, x)),
            ("meshgrid", lambda: lucid.meshgrid(*tensors, indexing="ij")),
            ("cartesian_prod", lambda: lucid.cartesian_prod(*tensors)),
        ):
            try:
                _ = call()
            except Exception as exc:
                assert isinstance(exc, _TYPED_REFUSALS), (
                    f"{name} raised untyped {type(exc).__name__}: {exc}"
                )


@pytest.mark.property
@settings(max_examples=25, deadline=None, database=None)
@given(
    n_rows=st.sampled_from((0, 1, 3)),
    idx_val=st.sampled_from((-5, -1, 0, 1, 2, 3, 10)),
    device=devices,
)
def test_indexed_registry_ops_refuse_out_of_range_indices_without_crash(
    n_rows: int, idx_val: int, device: str
) -> None:
    base = lucid.ones((n_rows, 2), dtype=lucid.float32, device=device)
    idx_2d = lucid.tensor([[idx_val, idx_val]], dtype=lucid.int64, device=device)
    idx_1d = lucid.tensor([idx_val], dtype=lucid.int64, device=device)
    src = lucid.ones((1, 2), dtype=lucid.float32, device=device)
    seq = lucid.arange(n_rows, dtype=lucid.float32, device=device)
    queries = lucid.tensor([float(idx_val)], dtype=lucid.float32, device=device)

    calls: dict[str, Callable[[], object]] = {
        "gather": lambda: lucid.gather(base, 0, idx_2d),
        "scatter": lambda: lucid.scatter(base, 0, idx_2d, src),
        "scatter_add": lambda: lucid.scatter_add(base, 0, idx_2d, src),
        "take": lambda: lucid.take(base, idx_1d),
        "index_select": lambda: lucid.index_select(base, 0, idx_1d),
        "searchsorted": lambda: lucid.searchsorted(seq, queries),
        "bucketize": lambda: lucid.bucketize(queries, seq),
    }
    assert set(calls) == _INDEXED_NAMES

    for name, fn in calls.items():
        if name in ("gather", "scatter", "scatter_add", "take", "index_select"):
            limit = base.numel() if name == "take" else n_rows
            in_bounds = -limit <= idx_val < limit
            # CPU refuses every out-of-range index; Metal refuses empty axes
            # (limit == 0) and isolates out-of-range indices on the GPU without
            # a host sync when limit > 0 (LCD-228 / op-index-ranges).
            if not in_bounds and (device == "cpu" or limit == 0):
                with pytest.raises((IndexError, RuntimeError)):
                    fn()
                continue
        try:
            out = fn()
        except Exception as exc:
            assert isinstance(exc, _TYPED_REFUSALS), (
                f"{name} raised untyped {type(exc).__name__}: {exc}"
            )
            continue
        assert isinstance(out, lucid.Tensor)
        assert out.device.type == device
        _ = out.to("cpu").tolist()
