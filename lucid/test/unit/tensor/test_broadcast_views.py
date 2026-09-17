"""Public ``broadcast_to`` and ``expand_as`` are CPU views, as ``expand`` is.

Both copied, so their results could be written and never shared their
input's memory.  Now a later write to the input shows through them, and a
write through them is refused: a broadcast axis repeats its elements.  The
engine's own internal broadcasting still copies, and Metal keeps copy
semantics.
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


def test_a_broadcast_nothing_else_shares_can_be_written() -> None:
    # Once its input is gone the buffer is the broadcast's alone, so an
    # in-place op gives it a buffer of its own rather than refusing.
    b = lucid.broadcast_to(lucid.tensor([1.0, 2.0, 3.0]), (2, 3))
    b.add_(1.0)
    assert b.tolist() == [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]


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
def test_metal_broadcast_to_stays_a_copy() -> None:
    x = lucid.tensor([1.0, 2.0, 3.0]).to("metal")
    b = lucid.broadcast_to(x, (2, 3))
    b.add_(1.0)
    assert x.to("cpu").tolist() == [1.0, 2.0, 3.0]
