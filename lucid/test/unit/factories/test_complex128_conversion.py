"""An explicit complex128 dtype must not route through the float32 fallback."""

import numpy as np
import pytest
import warnings

import lucid


@pytest.mark.parametrize("as_list", [False, True])
def test_explicit_complex128_preserves_both_components(as_list: bool) -> None:
    values = np.array([1.0 + 2.0j, 1.0000000001 - 3.0000000001j], dtype=np.complex128)
    x = lucid.tensor(values.tolist() if as_list else values, dtype=lucid.complex128)
    assert x.dtype == lucid.complex128
    np.testing.assert_array_equal(x.numpy(), values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("strided", [False, True])
def test_array_import_never_performs_an_unused_byte_cast(dtype, strided: bool) -> None:
    values = np.array([1e30, np.inf, np.nan, -1e30], dtype=dtype).reshape(2, 2)
    if np.issubdtype(dtype, np.complexfloating):
        values.imag = [[1, 2], [3, 4]]
    if strided:
        values = values.T[:, ::-1]
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = lucid.tensor(values)
    assert not emitted, [str(warning.message) for warning in emitted]
    np.testing.assert_array_equal(result.numpy(), values)
