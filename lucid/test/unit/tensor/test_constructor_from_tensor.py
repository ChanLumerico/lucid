"""``Tensor(t, dtype=..., device=...)`` honours both (LCD-331).

The constructor's signature and docstring offer ``dtype`` and ``device``
for every kind of ``data``, and a tensor source dropped them: ``Tensor(
zeros(2), dtype=float64, device="metal")`` was a float32 CPU tensor.  The
reference's legacy constructor refuses those keywords for a tensor, as it
does for a list, which Lucid accepts; Lucid honours them for a tensor too,
by the same rule as for a list.  A conversion is a new tensor that stays in
the source's graph; no conversion is a view of the source, as before.
"""

import pytest

import lucid

_SOURCES = ["float32", "int64", "bool"]
_TARGETS = [None, "float32", "float16", "int32"]


@pytest.mark.parametrize("source", _SOURCES)
@pytest.mark.parametrize("target", _TARGETS, ids=str)
@pytest.mark.parametrize("to_device", [None, "cpu", "metal"], ids=str)
def test_dtype_and_device_are_honoured(
    device: str, source: str, target: str | None, to_device: str | None
) -> None:
    t = lucid.tensor([[1, 0, 2]], dtype=getattr(lucid, source), device=device)
    dtype = None if target is None else getattr(lucid, target)
    got = lucid.Tensor(t, dtype=dtype, device=to_device)
    want = lucid.tensor(t, dtype=dtype, device=to_device)
    assert got.dtype == want.dtype == (t.dtype if dtype is None else dtype)
    assert got.device == want.device == lucid.device(to_device or device)
    assert got.tolist() == want.tolist()
    assert got.is_leaf and not got.requires_grad


@pytest.mark.parametrize("to_device", [None, "metal"], ids=str)
def test_an_engine_impl_source_is_converted_too(to_device: str | None) -> None:
    impl = lucid.zeros(2)._impl
    got = lucid.Tensor(impl, dtype=lucid.float16, device=to_device)
    assert got.dtype == lucid.float16
    assert got.device == lucid.device(to_device or "cpu")


def test_no_conversion_is_a_view_of_the_source(device: str) -> None:
    t = lucid.zeros(3, device=device)
    for alias in (
        lucid.Tensor(t),
        lucid.Tensor(t, dtype=lucid.float32),
        lucid.Tensor(t, device=device),
    ):
        alias[0] = 7.0
        assert t[0].item() == 7.0
        t[0] = 0.0


def test_a_conversion_stays_in_the_source_graph(device: str) -> None:
    w = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    v = lucid.Tensor(w, dtype=lucid.float16, requires_grad=True)
    assert v.dtype == lucid.float16 and v.requires_grad and not v.is_leaf
    (v.float() * v.float()).sum().backward()
    assert w.grad is not None and w.grad.tolist() == [2.0, 4.0]


def test_requires_grad_false_detaches_a_conversion(device: str) -> None:
    w = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    v = lucid.Tensor(w, dtype=lucid.float16)
    assert v.is_leaf and not v.requires_grad


def test_unconverted_tensor_preserves_requires_grad_and_graph(device: str) -> None:
    w = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    alias_w = lucid.Tensor(w)
    assert alias_w.requires_grad is True
    assert not alias_w.is_leaf

    with pytest.raises(RuntimeError, match="view of a leaf"):
        alias_w.add_(1.0)

    y = w * 2.0
    alias_y = lucid.Tensor(y)
    assert alias_y.requires_grad is True
    alias_y.add_(1.0)
    assert y.tolist() == [3.0, 5.0]
    assert y.requires_grad is True
    y.sum().backward()
    assert w.grad is not None and w.grad.tolist() == [2.0, 2.0]

