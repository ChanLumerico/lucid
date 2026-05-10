"""nn.* activation modules — they wrap functional equivalents."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestReLU:
    def test_basic(self) -> None:
        m = nn.ReLU()
        out = m(lucid.tensor([-1.0, 0.0, 1.0])).numpy()
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])


class TestLeakyReLU:
    def test_negative_slope(self) -> None:
        m = nn.LeakyReLU(negative_slope=0.1)
        out = m(lucid.tensor([-1.0, 0.0, 1.0])).numpy()
        np.testing.assert_allclose(out, [-0.1, 0.0, 1.0], atol=1e-6)


class TestSigmoidTanh:
    def test_sigmoid_at_zero(self) -> None:
        out = nn.Sigmoid()(lucid.tensor([0.0])).item()
        assert abs(out - 0.5) < 1e-6

    def test_tanh_at_zero(self) -> None:
        assert abs(nn.Tanh()(lucid.tensor([0.0])).item()) < 1e-6


class TestSoftmax:
    def test_default(self) -> None:
        m = nn.Softmax(dim=1)
        x = lucid.tensor([[1.0, 1.0, 1.0]])
        out = m(x).numpy()
        np.testing.assert_allclose(out, [[1 / 3, 1 / 3, 1 / 3]], atol=1e-6)


class TestSoftmax2d:
    def test_4d_only(self) -> None:
        m = nn.Softmax2d()
        x = lucid.tensor([[[[1.0, 1.0]], [[1.0, 1.0]]]])  # (1, 2, 1, 2)
        out = m(x).numpy()
        # Softmax along channel dim.
        np.testing.assert_allclose(out.sum(axis=1), np.ones((1, 1, 2)), atol=1e-6)

    def test_rejects_non_4d(self) -> None:
        m = nn.Softmax2d()
        with pytest.raises(ValueError):
            m(lucid.zeros(2, 3))


class TestRReLU:
    def test_eval_midpoint_slope(self) -> None:
        m = nn.RReLU()
        m.eval()
        out = m(lucid.tensor([-2.0, 1.0])).numpy()
        mid = (1.0 / 8.0 + 1.0 / 3.0) / 2.0
        np.testing.assert_allclose(out, [-2.0 * mid, 1.0], atol=1e-5)


class TestThreshold:
    def test_replace(self) -> None:
        m = nn.Threshold(threshold=0.5, value=-9.0)
        out = m(lucid.tensor([-1.0, 0.0, 1.0])).numpy()
        np.testing.assert_array_equal(out, [-9.0, -9.0, 1.0])


class TestHardtanh:
    def test_default(self) -> None:
        m = nn.Hardtanh()
        out = m(lucid.tensor([-2.0, -0.5, 0.5, 2.0])).numpy()
        np.testing.assert_array_equal(out, [-1.0, -0.5, 0.5, 1.0])


class TestGLU:
    def test_basic(self) -> None:
        m = nn.GLU(dim=-1)
        x = lucid.tensor([[1.0, 1.0, 0.0, 0.0]])
        # glu splits last dim in half: a * sigmoid(b).
        out = m(x).numpy()
        # b = [0, 0] → sigmoid = [0.5, 0.5]; a = [1, 1] → out = [0.5, 0.5].
        np.testing.assert_allclose(out, [[0.5, 0.5]], atol=1e-6)


class TestPReLU:
    def test_basic(self) -> None:
        m = nn.PReLU(num_parameters=1)
        out = m(lucid.tensor([-1.0, 0.0, 1.0]))
        assert out.shape == (3,)


class TestCosineSimilarityModule:
    def test_orthogonal(self) -> None:
        m = nn.CosineSimilarity(dim=1)
        a = lucid.tensor([[1.0, 0.0]])
        b = lucid.tensor([[0.0, 1.0]])
        assert abs(m(a, b).item()) < 1e-5


class TestPairwiseDistanceModule:
    def test_l2(self) -> None:
        m = nn.PairwiseDistance(p=2.0)
        a = lucid.tensor([[0.0, 0.0]])
        b = lucid.tensor([[3.0, 4.0]])
        assert abs(m(a, b).item() - 5.0) < 1e-4
