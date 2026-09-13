"""Parity: ``index_put`` with fewer index tensors than dimensions.

The reference indexes the leading dimensions and takes the rest whole;
Lucid used to raise.  Exact agreement, including repeated indices under
``accumulate=True`` and values that broadcast over the whole block.
"""

from typing import Any

import numpy as np
import pytest

import lucid

_CASES: dict[str, tuple[tuple[int, ...], list[list[int]], tuple[int, ...], bool]] = {
    "rows_of_4x3": ((4, 3), [[0, 2]], (2, 3), False),
    "repeated_rows_accumulate": ((4, 3), [[1, 1, 3]], (3, 3), True),
    "3d_by_two_indices": ((2, 3, 4), [[0, 1], [2, 0]], (2, 4), False),
    "3d_scalar_value": ((2, 3, 4), [[1]], (), False),
    "3d_row_broadcast": ((2, 3, 4), [[0, 1]], (4,), False),
    "full_indexing": ((4, 3), [[0, 1], [2, 0]], (2,), False),
}


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_partial_index_put_matches_the_reference(ref: Any, case: str) -> None:
    shape, index_lists, value_shape, accumulate = _CASES[case]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    values = rng.standard_normal(value_shape).astype(np.float32)
    indices = [np.array(i, dtype=np.int64) for i in index_lists]
    got = lucid.index_put(
        lucid.from_numpy(x.copy()),
        tuple(lucid.from_numpy(i) for i in indices),
        lucid.from_numpy(values.copy()),
        accumulate=accumulate,
    ).numpy()
    want = ref.from_numpy(x.copy()).index_put(
        tuple(ref.from_numpy(i) for i in indices),
        ref.from_numpy(values.copy()),
        accumulate=accumulate,
    )
    np.testing.assert_array_equal(got, want.numpy())
