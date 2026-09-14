"""matmul takes a vector on either side, as the reference and NumPy do.

A vector is a matrix of one row on the left or one column on the right, and
the axis it gained is dropped from the product again.
"""

import pytest

import lucid


def _matrix() -> lucid.Tensor:
    return lucid.tensor([[1.0, 2.0], [3.0, 4.0]])


def test_a_matrix_times_a_vector_is_a_vector() -> None:
    assert (_matrix() @ lucid.tensor([1.0, 1.0])).tolist() == [3.0, 7.0]


def test_a_vector_times_a_matrix_is_a_vector() -> None:
    assert (lucid.tensor([1.0, 1.0]) @ _matrix()).tolist() == [4.0, 6.0]


def test_two_vectors_give_a_scalar() -> None:
    out = lucid.matmul(lucid.tensor([1.0, 1.0]), lucid.tensor([2.0, 3.0]))
    assert tuple(out.shape) == ()
    assert out.item() == 5.0


def test_a_vector_against_a_batch() -> None:
    v = lucid.ones(2)
    assert tuple((v @ lucid.ones(4, 2, 3)).shape) == (4, 3)
    assert tuple((lucid.ones(4, 3, 2) @ v).shape) == (4, 3)


def test_the_gradient_keeps_the_vector_s_shape() -> None:
    a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    v = lucid.tensor([1.0, 1.0], requires_grad=True)
    (a @ v).sum().backward()
    assert v.grad is not None and a.grad is not None
    assert v.grad.tolist() == [4.0, 6.0]
    assert a.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_a_scalar_operand_is_refused() -> None:
    with pytest.raises(Exception, match="at least 1-D"):
        lucid.matmul(lucid.tensor(2.0), lucid.ones(2))
