"""Public ``broadcast_to`` and ``expand_as`` are CPU views, as ``expand`` is.

Both copied, so their results could be written and never shared their
input's memory.  Now a later write to the input shows through them, and a
write through them is refused: a broadcast axis repeats its elements.  The
engine's own internal broadcasting still copies.  On Metal the result is a
copy, and a write through it is refused the same way.
"""

import pytest

import lucid


def test_broadcast_to_is_a_view_of_its_input() -> None:
    x = lucid.tensor([1.0, 2.0, 3.0])
    b = lucid.broadcast_to(x, (2, 3))
    x.mul_(10.0)
    assert b.tolist() == [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]


def test_the_method_form_is_a_view_too() -> None:
    x = lucid.tensor([[1.0], [2.0]])
    b = x.broadcast_to((2, 3))
    x.add_(1.0)
    assert b.tolist() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]


def test_a_write_through_a_broadcast_is_refused_while_its_input_lives() -> None:
    x = lucid.tensor([1.0, 2.0, 3.0])
    b = lucid.broadcast_to(x, (2, 3))
    with pytest.raises(Exception, match="overlap"):
        b.add_(1.0)
    assert x.tolist() == [1.0, 2.0, 3.0]


def test_a_write_through_a_broadcast_is_refused_once_its_input_is_gone() -> None:
    # The broadcast axis still repeats its elements, whoever else holds the
    # buffer, and the reference refuses the write either way.  This used to
    # hand the broadcast a dense buffer of its own and go through.
    b = lucid.broadcast_to(lucid.tensor([1.0, 2.0, 3.0]), (2, 3))
    with pytest.raises(RuntimeError, match="overlap"):
        b.add_(1.0)
    assert b.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


def test_contiguous_gives_a_writable_copy() -> None:
    x = lucid.tensor([1.0, 2.0, 3.0])
    c = lucid.broadcast_to(x, (2, 3)).contiguous()
    c.add_(1.0)
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert c.tolist() == [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]


def test_expand_as_is_a_view() -> None:
    x = lucid.tensor([[1.0, 2.0, 3.0]])
    e = x.expand_as(lucid.zeros(4, 3))
    x.fill_(7.0)
    assert e.tolist() == [[7.0, 7.0, 7.0]] * 4


def test_the_gradient_sums_over_the_broadcast_axes() -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (lucid.broadcast_to(w, (4, 3)) * 2.0).sum().backward()
    assert w.grad is not None
    assert w.grad.tolist() == [8.0, 8.0, 8.0]


def test_matrix_power_zero_stays_writable() -> None:
    eye = lucid.linalg.matrix_power(lucid.randn(2, 3, 3), 0)
    eye.add_(1.0)
    assert eye.tolist()[1][0] == [2.0, 1.0, 1.0]


@pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
def test_metal_broadcast_to_write_refuses_like_cpu() -> None:
    """A write through ``broadcast_to`` is refused on Metal, as on the CPU.

    This test used to assert the Metal write was silently dropped — it pinned
    the defect of debug-metal-view-writes-silent (CHA-73) as if it were the
    intent.
    """
    for dev in ("cpu", "metal"):
        x = lucid.tensor([1.0, 2.0, 3.0]).to(dev)
        b = lucid.broadcast_to(x, (2, 3))
        with pytest.raises(RuntimeError, match="overlap"):
            b.add_(1.0)
        assert x.to("cpu").tolist() == [1.0, 2.0, 3.0]
