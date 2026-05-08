"""Reference parity for reduction ops."""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestReductionParity:
    @pytest.fixture
    def x_pair(self, ref: Any) -> tuple[lucid.Tensor, Any]:
        np.random.seed(0)
        x = np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float32)
        return lucid.tensor(x.copy()), ref.tensor(x.copy())

    def test_sum(self, x_pair: tuple[lucid.Tensor, Any], ref: Any) -> None:
        l, r = x_pair
        assert_close(lucid.sum(l), ref.sum(r), atol=1e-5)

    def test_mean(self, x_pair: tuple[lucid.Tensor, Any], ref: Any) -> None:
        l, r = x_pair
        assert_close(lucid.mean(l), ref.mean(r), atol=1e-6)

    def test_max(self, x_pair: tuple[lucid.Tensor, Any], ref: Any) -> None:
        l, r = x_pair
        # Both call ``max`` differently — Lucid returns a value, ref
        # returns a (values, indices) tuple when an axis is given.
        # Full-tensor: both produce a scalar.
        assert_close(lucid.max(l), ref.max(r), atol=0.0)

    def test_argmax_axis(self, x_pair: tuple[lucid.Tensor, Any], ref: Any) -> None:
        # Default ``argmax`` semantics differ — the reference returns a
        # global flat argmax, Lucid returns per-row.  Pin to ``dim=1``
        # for a value-equivalent comparison.
        l, r = x_pair
        l_out = lucid.argmax(l, dim=1).numpy()
        r_out = ref.argmax(r, dim=1).detach().cpu().numpy()
        np.testing.assert_array_equal(l_out, r_out)

    def test_cumsum_dim0(self, x_pair: tuple[lucid.Tensor, Any], ref: Any) -> None:
        l, r = x_pair
        assert_close(lucid.cumsum(l, dim=0), ref.cumsum(r, dim=0), atol=1e-5)

    def test_var_unbiased(self, x_pair: tuple[lucid.Tensor, Any], ref: Any) -> None:
        l, r = x_pair
        # Reference framework's default var uses Bessel correction (=1).
        assert_close(lucid.var(l, correction=1), ref.var(r), atol=1e-5)
