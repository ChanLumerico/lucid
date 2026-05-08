"""nn.init — initializer functions."""

import numpy as np
import pytest

import lucid
import lucid.nn.init as init


class TestZerosOnes:
    def test_zeros_(self) -> None:
        t = lucid.zeros(3, 4)
        init.zeros_(t)
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_ones_(self) -> None:
        t = lucid.zeros(3, 4)
        init.ones_(t)
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 4)))


class TestConstant:
    def test_constant_(self) -> None:
        t = lucid.zeros(2, 3)
        init.constant_(t, 7.5)
        np.testing.assert_array_equal(t.numpy(), np.full((2, 3), 7.5))


class TestUniformNormal:
    def test_uniform_(self) -> None:
        t = lucid.zeros(1000)
        init.uniform_(t, a=-1.0, b=1.0)
        arr = t.numpy()
        assert (arr >= -1.0).all() and (arr <= 1.0).all()

    def test_normal_(self) -> None:
        t = lucid.zeros(10_000)
        init.normal_(t, mean=0.0, std=1.0)
        arr = t.numpy()
        assert abs(arr.mean()) < 0.05
        assert abs(arr.std() - 1.0) < 0.05


class TestXavier:
    def test_xavier_uniform_(self) -> None:
        t = lucid.zeros(64, 64)
        init.xavier_uniform_(t)
        # Should be non-zero.
        assert t.numpy().std() > 0.0

    def test_xavier_normal_(self) -> None:
        t = lucid.zeros(64, 64)
        init.xavier_normal_(t)
        assert t.numpy().std() > 0.0


class TestKaiming:
    def test_kaiming_uniform_(self) -> None:
        t = lucid.zeros(64, 64)
        init.kaiming_uniform_(t, a=0.0, nonlinearity="relu")
        assert t.numpy().std() > 0.0

    def test_kaiming_normal_(self) -> None:
        t = lucid.zeros(64, 64)
        init.kaiming_normal_(t, a=0.0, nonlinearity="relu")
        assert t.numpy().std() > 0.0


class TestOrthogonal:
    def test_orthogonal_qr(self) -> None:
        t = lucid.zeros(8, 4)
        init.orthogonal_(t)
        arr = t.numpy()
        # Q^T Q ≈ I for tall matrices.
        np.testing.assert_allclose(arr.T @ arr, np.eye(4), atol=1e-3)


class TestSparse:
    def test_zeros_per_column(self) -> None:
        t = lucid.zeros(10, 4)
        init.sparse_(t, sparsity=0.5)
        arr = t.numpy()
        zeros_per_col = (arr == 0).sum(axis=0)
        # Each column should have exactly 5 zeros (50% of 10).
        assert (zeros_per_col == 5).all()


class TestDirac:
    def test_4d_identity(self) -> None:
        t = lucid.zeros(4, 4, 3, 3)
        init.dirac_(t)
        arr = t.numpy()
        # Centre is (1, 1).  arr[i, i, 1, 1] == 1, all else 0.
        np.testing.assert_array_equal(np.diag(arr[..., 1, 1]), [1.0] * 4)
        assert arr.sum() == 4.0


class TestCalculateGain:
    def test_relu(self) -> None:
        import math
        assert abs(init.calculate_gain("relu") - math.sqrt(2.0)) < 1e-6

    def test_linear(self) -> None:
        assert init.calculate_gain("linear") == 1.0
