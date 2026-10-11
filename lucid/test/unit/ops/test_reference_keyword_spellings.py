"""The reference's keyword spellings reach the op, not the engine binding.

``tril(x, diagonal=1)``, ``flatten(x, start_dim=1)``, ``tile(x, dims=...)``
and ``all(x, dim=1)`` went straight through to a pybind11 binding that
takes ``k`` / ``start_axis`` / ``reps`` and no ``dim`` at all, and failed
with "incompatible function arguments" (LCD-386).  Lucid's own names keep
their place; the reference's are accepted beside them, and each answer is
checked against the reference over every spelling — the free function,
the Tensor method, positional and keyword — on both devices.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

_MATRIX = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]


@pytest.mark.parametrize("op", ["tril", "triu"])
@pytest.mark.parametrize("offset", [-1, 0, 2])
def test_tril_triu_take_diagonal(
    ref: ModuleType, device: str, op: str, offset: int
) -> None:
    x = lucid.tensor(_MATRIX, device=device)
    want = getattr(ref, op)(ref.tensor(_MATRIX), diagonal=offset).tolist()
    for got in (
        getattr(lucid, op)(x, diagonal=offset),
        getattr(lucid, op)(x, offset),
        getattr(lucid, op)(x, k=offset),
        getattr(lucid, op)(input=x, diagonal=offset),
        getattr(x, op)(diagonal=offset),
        getattr(x, op)(offset),
    ):
        assert got.tolist() == want


def test_flatten_takes_start_and_end_dim(ref: ModuleType, device: str) -> None:
    x = lucid.zeros(2, 3, 4, 5, device=device)
    r = ref.zeros(2, 3, 4, 5)
    for start, end in [(0, -1), (1, -1), (1, 2), (0, 1)]:
        want = tuple(ref.flatten(r, start_dim=start, end_dim=end).shape)
        assert lucid.flatten(x, start_dim=start, end_dim=end).shape == want
        assert x.flatten(start_dim=start, end_dim=end).shape == want
        assert x.flatten(start, end_dim=end).shape == want
        assert lucid.flatten(x, start, end).shape == want


def test_tile_takes_dims(ref: ModuleType, device: str) -> None:
    x = lucid.tensor([[1, 2], [3, 4]], device=device)
    for dims in [(2,), (2, 3), (2, 1, 2)]:
        want = ref.tile(ref.tensor([[1, 2], [3, 4]]), dims=dims).tolist()
        assert lucid.tile(x, dims=dims).tolist() == want
        assert x.tile(dims=dims).tolist() == want
        assert lucid.tile(x, dims).tolist() == want


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.tril(m.ones(2, 2), 1, diagonal=1),
        lambda m: m.flatten(m.ones(2, 2), 0, start_dim=0),
        lambda m: m.tile(m.ones(2), (2,), dims=(2,)),
        lambda m: m.tile(m.ones(2)),
    ],
    ids=["tril-both", "flatten-both", "tile-both", "tile-neither"],
)
def test_two_spellings_of_one_argument_are_refused(
    call: Callable[[ModuleType], object],
) -> None:
    with pytest.raises(TypeError):
        call(lucid)


_LOGICAL_DATA = {
    "bool": [[True, False, True], [True, True, True]],
    "int64": [[1, 0, 3], [2, 5, -1]],
    "float32": [[0.5, 0.0, float("nan")], [1.0, -2.0, 3.0]],
}


def _dtypes(m: ModuleType, name: str) -> object:
    return getattr(m, name)


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("dtype", sorted(_LOGICAL_DATA))
@pytest.mark.parametrize("dim", [None, 0, 1, -1, (0, 1)], ids=lambda d: f"dim={d}")
@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
def test_any_all_reduce_over_dim(
    ref: ModuleType, device: str, op: str, dtype: str, dim: object, keepdim: bool
) -> None:
    data = _LOGICAL_DATA[dtype]
    x = lucid.tensor(data, dtype=_dtypes(lucid, dtype), device=device)
    r = ref.tensor(data, dtype=_dtypes(ref, dtype))
    if dim is None:
        want = (
            getattr(ref, op)(r, dim=None, keepdim=keepdim)
            if keepdim
            else getattr(ref, op)(r)
        )
    else:
        want = getattr(ref, op)(r, dim=dim, keepdim=keepdim)
    calls = [
        getattr(lucid, op)(x, dim=dim, keepdim=keepdim),
        getattr(x, op)(dim=dim, keepdim=keepdim),
        getattr(x, op)(dim, keepdim),
    ]
    for got in calls:
        assert got.dtype == lucid.bool
        assert got.shape == tuple(want.shape)
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize("op", ["any", "all"])
def test_any_all_over_an_empty_axis(ref: ModuleType, device: str, op: str) -> None:
    x = lucid.zeros(0, 3, device=device)
    for dim in (0, 1):
        want = getattr(ref, op)(ref.zeros(0, 3), dim=dim)
        got = getattr(lucid, op)(x, dim=dim)
        assert got.shape == tuple(want.shape)
        assert got.tolist() == want.tolist()


def test_any_all_read_both_parts_of_a_complex_entry(device: str) -> None:
    # 2j is non-zero although its real part is 0: Metal's complex-to-bool
    # cast reads the real part only.
    z = lucid.complex(
        lucid.tensor([[0.0, 0.0], [1.0, 0.0]], device=device),
        lucid.tensor([[2.0, 0.0], [0.0, 0.0]], device=device),
    )
    assert lucid.any(z, dim=1).tolist() == [True, True]
    assert lucid.all(z, dim=0).tolist() == [True, False]
