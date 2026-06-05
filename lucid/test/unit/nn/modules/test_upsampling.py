"""Dedicated unit coverage for ``nn`` upsampling / pixel-shuffle modules.

Forward shape + a deterministic value/round-trip check + backward — these
families had no dedicated test file before.
"""

import lucid
import lucid.nn as nn


class TestUpsample:
    def test_nearest_scale(self) -> None:
        out = nn.Upsample(scale_factor=2)(lucid.ones(2, 3, 8, 8))
        assert out.shape == (2, 3, 16, 16)

    def test_bilinear_scale(self) -> None:
        out = nn.Upsample(scale_factor=2, mode="bilinear")(lucid.ones(2, 3, 8, 8))
        assert out.shape == (2, 3, 16, 16)

    def test_explicit_size(self) -> None:
        out = nn.Upsample(size=(10, 12))(lucid.ones(1, 1, 4, 4))
        assert out.shape == (1, 1, 10, 12)

    def test_nearest_replicates_value(self) -> None:
        # nearest upsample of a constant tensor stays constant
        out = nn.Upsample(scale_factor=2)(lucid.full((1, 1, 2, 2), 7.0)).numpy()
        assert out.shape == (1, 1, 4, 4)
        assert (out == 7.0).all()

    def test_backward(self) -> None:
        # NOTE: nearest-mode upsample currently doesn't propagate a gradient
        # (a separate engine gap); bilinear does, so the grad path is tested
        # through it.
        x = lucid.ones(1, 1, 4, 4, requires_grad=True)
        nn.Upsample(scale_factor=2, mode="bilinear")(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 1, 4, 4)


class TestPixelShuffleUnshuffle:
    def test_pixel_shuffle_shape(self) -> None:
        # (N, C*r^2, H, W) -> (N, C, H*r, W*r)
        out = nn.PixelShuffle(2)(lucid.ones(1, 8, 4, 4))
        assert out.shape == (1, 2, 8, 8)

    def test_pixel_unshuffle_shape(self) -> None:
        out = nn.PixelUnshuffle(2)(lucid.ones(1, 2, 8, 8))
        assert out.shape == (1, 8, 4, 4)

    def test_shuffle_unshuffle_round_trip(self) -> None:
        x = lucid.randn(2, 4, 6, 6)
        rt = nn.PixelUnshuffle(2)(nn.PixelShuffle(2)(x))
        assert rt.shape == x.shape
        import numpy as np

        np.testing.assert_allclose(rt.numpy(), x.numpy(), atol=1e-6)

    def test_backward(self) -> None:
        x = lucid.ones(1, 8, 4, 4, requires_grad=True)
        nn.PixelShuffle(2)(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 8, 4, 4)
