"""Every factory builds its value in the final dtype, then makes it a leaf.

``lucid.tensor([1.5, 2.0], dtype=lucid.bfloat16, requires_grad=True)``
came back a *non-leaf*: the value was made as a float32 leaf and narrowed
afterwards, so the caller held the narrowing's output and ``.grad`` never
filled — training a bfloat16 parameter did nothing, silently.  And the
fill went through a C double: ``full`` rounded ``2**60 + 1`` and refused
a complex value outright, and NumPy scalars lost their dtype
(``tensor(np.bool_(True))`` was float32).

The matrix below holds every factory to the contract on every dtype and
both devices; the value rows pin exactness.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]
ALL_DTYPES = [
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.bool,
]
_FLOATING = {
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
}
_COMPLEX = {lucid.complex64, lucid.complex128}
# MLX has neither (H3), so a Metal tensor of these is refused by design.
_METAL_ABSENT = {lucid.float64, lucid.complex128}

type Factory = Callable[[lucid.dtype, str, bool], lucid.Tensor]


def _like(fn: Callable[..., lucid.Tensor], *args: object) -> Factory:
    def make(dtype: lucid.dtype, device: str, rg: bool) -> lucid.Tensor:
        src = lucid.zeros(2, 3, device=device)
        return fn(src, *args, dtype=dtype, device=device, requires_grad=rg)

    return make


FACTORIES: dict[str, Factory] = {
    "tensor_list": lambda d, dev, rg: lucid.tensor(
        [1.5, 2.0], dtype=d, device=dev, requires_grad=rg
    ),
    "tensor_scalar": lambda d, dev, rg: lucid.tensor(
        1.5, dtype=d, device=dev, requires_grad=rg
    ),
    "tensor_ndarray": lambda d, dev, rg: lucid.tensor(
        np.array([1.5, 2.0]), dtype=d, device=dev, requires_grad=rg
    ),
    "tensor_tensor": lambda d, dev, rg: lucid.tensor(
        lucid.ones(2, device=dev), dtype=d, device=dev, requires_grad=rg
    ),
    "full": lambda d, dev, rg: lucid.full(
        (2, 3), 1, dtype=d, device=dev, requires_grad=rg
    ),
    "zeros": lambda d, dev, rg: lucid.zeros(
        2, 3, dtype=d, device=dev, requires_grad=rg
    ),
    "ones": lambda d, dev, rg: lucid.ones(2, 3, dtype=d, device=dev, requires_grad=rg),
    "empty": lambda d, dev, rg: lucid.empty(
        2, 3, dtype=d, device=dev, requires_grad=rg
    ),
    "eye": lambda d, dev, rg: lucid.eye(3, dtype=d, device=dev, requires_grad=rg),
    "arange": lambda d, dev, rg: lucid.arange(
        0, 4, dtype=d, device=dev, requires_grad=rg
    ),
    "linspace": lambda d, dev, rg: lucid.linspace(
        0, 1, 4, dtype=d, device=dev, requires_grad=rg
    ),
    "logspace": lambda d, dev, rg: lucid.logspace(
        0, 1, 4, dtype=d, device=dev, requires_grad=rg
    ),
    "rand": lambda d, dev, rg: lucid.rand(2, 3, dtype=d, device=dev, requires_grad=rg),
    "randn": lambda d, dev, rg: lucid.randn(
        2, 3, dtype=d, device=dev, requires_grad=rg
    ),
    "zeros_like": _like(lucid.zeros_like),
    "ones_like": _like(lucid.ones_like),
    "empty_like": _like(lucid.empty_like),
    "full_like": _like(lucid.full_like, 1),
    "rand_like": _like(lucid.rand_like),
    "randn_like": _like(lucid.randn_like),
}

# What the reference refuses too: integer and bool draws, a bool or
# complex ``arange``, a bool ``linspace`` / ``logspace``.
_DRAWS = {"rand", "randn", "rand_like", "randn_like"}


def _reference_refuses(name: str, dtype: lucid.dtype) -> bool:
    if name in _DRAWS:
        return dtype not in _FLOATING
    if name == "arange":
        return dtype is lucid.bool or dtype in _COMPLEX
    if name in ("linspace", "logspace"):
        return dtype is lucid.bool
    return False


# Engine gaps the reference does not have.  Strict: a fix turns the
# xfail into a failure, and the entry must come out.
_ENGINE_GAPS: set[tuple[str, lucid.dtype, str]] = {
    *{
        ("logspace", d, dev)
        for d in (
            lucid.float16,
            lucid.bfloat16,
            lucid.complex64,
            lucid.int8,
            lucid.int16,
        )
        for dev in DEVICES
    },
    ("logspace", lucid.complex128, "cpu"),
    *{
        (n, lucid.complex128, "cpu")
        for n in ("eye", "linspace", "rand", "randn", "rand_like", "randn_like")
    },
}


def _cases() -> list[object]:
    out: list[object] = []
    for name in FACTORIES:
        for dtype in ALL_DTYPES:
            if _reference_refuses(name, dtype):
                continue
            for device in DEVICES:
                if device == "metal" and dtype in _METAL_ABSENT:
                    continue
                for rg in (False, True) if dtype in _FLOATING else (False,):
                    marks = []
                    if (name, dtype, device) in _ENGINE_GAPS:
                        marks.append(
                            pytest.mark.xfail(raises=NotImplementedError, strict=True)
                        )
                    out.append(
                        pytest.param(
                            name,
                            dtype,
                            device,
                            rg,
                            marks=marks,
                            id=f"{name}-{str(dtype).removeprefix('lucid.')}-{device}-rg{int(rg)}",
                        )
                    )
    return out


def _loss(t: lucid.Tensor) -> lucid.Tensor:
    y = (t * 2).sum()
    return lucid.real(y) if t.dtype in _COMPLEX else y


@pytest.mark.parametrize(("name", "dtype", "device", "rg"), _cases())
def test_factory_returns_a_leaf_in_its_dtype(
    name: str, dtype: lucid.dtype, device: str, rg: bool
) -> None:
    t = FACTORIES[name](dtype, device, rg)
    assert t.dtype is dtype
    assert t.device.type == device
    assert t.is_leaf
    assert t.requires_grad is rg
    if rg:
        _loss(t).backward()
        assert t.grad is not None
        assert t.grad.dtype is dtype
        assert all(g == 2 for g in t.grad.reshape(-1).tolist())


@pytest.mark.parametrize("device", DEVICES)
def test_as_tensor_builds_its_dtype(device: str) -> None:
    for dtype in ALL_DTYPES:
        if device == "metal" and dtype in _METAL_ABSENT:
            continue
        t = lucid.as_tensor([1, 0], dtype=dtype, device=device)
        assert (t.dtype, t.device.type, t.is_leaf) == (dtype, device, True)


@pytest.mark.parametrize(
    "np_dtype",
    [
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    ],
)
def test_from_numpy_keeps_the_array_dtype(np_dtype: str) -> None:
    t = lucid.from_numpy(np.array([1, 0], dtype=np_dtype))
    assert t.dtype is getattr(lucid, np_dtype)
    assert t.is_leaf and not t.requires_grad


# ── exact values ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [2**60 + 1, 2**53 + 1, -(2**63), 2**63 - 1])
def test_int64_fill_is_exact(device: str, value: int) -> None:
    assert (
        lucid.full((3,), value, dtype=lucid.int64, device=device).tolist()
        == [value] * 3
    )
    src = lucid.zeros(3, device=device)
    assert lucid.full_like(src, value, dtype=lucid.int64).tolist() == [value] * 3
    assert lucid.tensor([value], dtype=lucid.int64, device=device).item() == value


@pytest.mark.parametrize("device", DEVICES)
def test_python_int_literal_is_exact(device: str) -> None:
    t = lucid.tensor([2**53 + 1, 2**60 + 1], device=device)
    assert t.dtype is lucid.int64
    assert t.tolist() == [2**53 + 1, 2**60 + 1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128], ids=str)
def test_complex_fill(device: str, dtype: lucid.dtype) -> None:
    if device == "metal" and dtype in _METAL_ABSENT:
        pytest.skip("MLX has no complex128")
    t = lucid.full((2, 2), 2 + 3j, dtype=dtype, device=device, requires_grad=True)
    assert t.dtype is dtype and t.is_leaf
    assert t.tolist() == [[2 + 3j, 2 + 3j], [2 + 3j, 2 + 3j]]
    like = lucid.full_like(lucid.zeros(3, device=device), -1j, dtype=dtype)
    assert like.tolist() == [-1j] * 3


@pytest.mark.parametrize("device", DEVICES)
def test_full_like_fills_in_the_requested_dtype(device: str) -> None:
    src = lucid.zeros(2, dtype=lucid.int32, device=device)
    t = lucid.full_like(src, 2.5, dtype=lucid.float32)
    assert t.dtype is lucid.float32 and t.tolist() == [2.5, 2.5]
    assert lucid.full_like(src, 7).dtype is lucid.int32


@pytest.mark.parametrize(
    ("value", "dtype", "error"),
    [
        (2 + 3j, lucid.float32, TypeError),
        (2 + 3j, lucid.int64, TypeError),
        (300, lucid.int8, OverflowError),
        (2**63, lucid.int64, OverflowError),
        (float("nan"), lucid.int32, ValueError),
        (float("inf"), lucid.int64, ValueError),
    ],
    ids=["complex-f32", "complex-i64", "300-i8", "2^63-i64", "nan-i32", "inf-i64"],
)
def test_fill_refuses_a_value_the_dtype_cannot_hold(
    value: complex, dtype: lucid.dtype, error: type[Exception]
) -> None:
    with pytest.raises(error):
        lucid.full((2,), value, dtype=dtype)
    with pytest.raises(error):
        lucid.tensor([value], dtype=dtype)


def test_fill_truncates_a_finite_float_into_an_integer() -> None:
    assert lucid.full((2,), -2.7, dtype=lucid.int32).tolist() == [-2, -2]
    assert lucid.tensor([2.7, -2.7], dtype=lucid.int64).tolist() == [2, -2]


def test_float_overflow_rounds_to_infinity() -> None:
    t = lucid.tensor([1e6, -1e6], dtype=lucid.float16)
    assert t.float().tolist() == [float("inf"), float("-inf")]
    assert lucid.tensor([1e40], dtype=lucid.float32).item() == float("inf")


# ── dtype inference ──────────────────────────────────────────────────────

_NUMPY_SCALARS: list[tuple[object, lucid.dtype]] = [
    (np.bool_(True), lucid.bool),
    (np.int8(3), lucid.int8),
    (np.int32(3), lucid.int32),
    (np.int64(3), lucid.int64),
    (np.uint8(3), lucid.int16),
    (np.float16(1.5), lucid.float16),
    (np.float32(1.5), lucid.float32),
    (np.float64(1.5), lucid.float64),
    (np.complex64(1 + 2j), lucid.complex64),
    (np.complex128(1 + 2j), lucid.complex128),
]


@pytest.mark.parametrize(("value", "dtype"), _NUMPY_SCALARS, ids=lambda v: repr(v))
def test_numpy_scalar_keeps_its_dtype(value: object, dtype: lucid.dtype) -> None:
    t = lucid.tensor(value)
    assert t.dtype is dtype
    assert t.item() == value
    assert lucid.tensor([value, value]).dtype is dtype


def test_python_floats_follow_the_default_dtype() -> None:
    previous = lucid.get_default_dtype()
    lucid.set_default_dtype(lucid.float64)
    try:
        assert lucid.tensor([1.5]).dtype is lucid.float64
        assert lucid.tensor(1.5).dtype is lucid.float64
        assert lucid.tensor([1j]).dtype is lucid.complex128
        assert lucid.tensor([1, 2]).dtype is lucid.int64
        assert lucid.tensor([]).dtype is lucid.float64
    finally:
        lucid.set_default_dtype(previous)
    assert lucid.tensor([1.5]).dtype is lucid.float32
    assert lucid.tensor([1j]).dtype is lucid.complex64


_INFERENCE_ROWS: list[object] = [
    [np.float16(1), 2.0],
    [np.float16(1), 3],
    [np.int32(1), 2],
    [np.float64(1), 1j],
    [np.bool_(True), np.bool_(False)],
    [np.int8(1), np.int16(2)],
    [True, 2],
    [1, 2.5],
    [1, 2j],
]


@pytest.mark.parametrize("row", _INFERENCE_ROWS, ids=repr)
def test_literal_inference_matches_reference(
    ref: ModuleType, row: list[object]
) -> None:
    want = str(ref.tensor(row).dtype).rsplit(".", 1)[-1]
    assert str(lucid.tensor(row).dtype) == f"lucid.{want}"


@pytest.mark.parametrize(
    "case",
    ["int64-2^60+1", "int64-2^53+1", "complex64", "complex128", "bf16-leaf"],
)
def test_exact_rows_match_reference(ref: ModuleType, case: str) -> None:
    r = ref
    if case == "int64-2^60+1":
        want = r.full((2,), 2**60 + 1, dtype=r.int64).tolist()
        assert lucid.full((2,), 2**60 + 1, dtype=lucid.int64).tolist() == want
    elif case == "int64-2^53+1":
        want = r.tensor([2**53 + 1]).tolist()
        assert lucid.tensor([2**53 + 1]).tolist() == want
    elif case in ("complex64", "complex128"):
        want = r.full((2,), 2 + 3j, dtype=getattr(r, case)).tolist()
        assert lucid.full((2,), 2 + 3j, dtype=getattr(lucid, case)).tolist() == want
    else:
        rt = r.tensor([1.5, 2.0], dtype=r.bfloat16, requires_grad=True)
        lt = lucid.tensor([1.5, 2.0], dtype=lucid.bfloat16, requires_grad=True)
        assert lt.is_leaf == rt.is_leaf
