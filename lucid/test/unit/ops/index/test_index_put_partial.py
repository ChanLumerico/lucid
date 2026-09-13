"""``index_put`` with fewer index tensors than dimensions.

It used to raise ``NotImplementedError``.  The reference indexes the
leading dimensions and takes the rest whole; the parity tier checks the
numbers against it.
"""

import pytest

import lucid


def test_one_index_writes_whole_rows() -> None:
    rows = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = lucid.index_put(lucid.zeros(4, 3), (lucid.tensor([0, 2]),), rows)
    assert out.numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [4.0, 5.0, 6.0],
        [0.0, 0.0, 0.0],
    ]


def test_a_row_value_broadcasts_over_the_indexed_rows() -> None:
    row = lucid.tensor([7.0, 8.0])
    out = lucid.index_put(lucid.zeros(3, 2), (lucid.tensor([0, 2]),), row)
    assert out.numpy().tolist() == [[7.0, 8.0], [0.0, 0.0], [7.0, 8.0]]


def test_repeated_rows_accumulate() -> None:
    out = lucid.index_put(
        lucid.zeros(2, 2), (lucid.tensor([1, 1]),), lucid.ones(2, 2), accumulate=True
    )
    assert out.numpy().tolist() == [[0.0, 0.0], [2.0, 2.0]]


def test_too_many_indices_is_an_index_error() -> None:
    index = lucid.tensor([0])
    with pytest.raises(IndexError):
        lucid.index_put(lucid.zeros(2), (index, index), lucid.tensor(1.0))
