"""``lucid.einops`` — rearrange / reduce / repeat."""

import numpy as np

import lucid


class TestRearrange:
    def test_basic(self, device: str) -> None:
        x = lucid.arange(0.0, 24.0, 1.0, device=device).reshape(2, 3, 4)
        out = lucid.einops.rearrange(x, "a b c -> a c b")
        assert out.shape == (2, 4, 3)

    def test_merge_dims(self, device: str) -> None:
        x = lucid.zeros(2, 3, 4, device=device)
        out = lucid.einops.rearrange(x, "a b c -> a (b c)")
        assert out.shape == (2, 12)


class TestReduce:
    def test_sum(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        out = lucid.einops.reduce(x, "h w -> h", "sum")
        np.testing.assert_array_equal(out.numpy(), [3.0, 7.0])

    def test_mean(self, device: str) -> None:
        x = lucid.tensor([[1.0, 3.0], [5.0, 7.0]], device=device)
        out = lucid.einops.reduce(x, "h w -> h", "mean")
        np.testing.assert_array_equal(out.numpy(), [2.0, 6.0])


class TestRepeat:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = lucid.einops.repeat(x, "n -> n k", k=2)
        assert out.shape == (3, 2)
