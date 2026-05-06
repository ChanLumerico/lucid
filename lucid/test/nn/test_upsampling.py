"""Tests for the upsampling modules added in this pass."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


class TestUpsamplingNearest2d:
    def test_doubles_spatial_dims(self) -> None:
        layer: nn.UpsamplingNearest2d = nn.UpsamplingNearest2d(scale_factor=2)
        x: lucid.Tensor = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        out: np.ndarray = layer(x).numpy()
        assert out.shape == (1, 1, 4, 4)
        # Each input cell is replicated into a 2×2 block.
        np.testing.assert_allclose(
            out[0, 0],
            np.array(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]
            ),
        )

    def test_size_target(self) -> None:
        layer: nn.UpsamplingNearest2d = nn.UpsamplingNearest2d(size=(6, 6))
        x: lucid.Tensor = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        assert layer(x).shape == (1, 1, 6, 6)


class TestUpsamplingBilinear2d:
    def test_align_corners_default_true(self) -> None:
        # ``UpsamplingBilinear2d`` pre-sets ``align_corners=True`` — the
        # corners of the input land exactly at the corners of the output.
        layer: nn.UpsamplingBilinear2d = nn.UpsamplingBilinear2d(scale_factor=2)
        x: lucid.Tensor = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        out: np.ndarray = layer(x).numpy()
        assert out.shape == (1, 1, 4, 4)
        # Corners preserved exactly.
        assert out[0, 0, 0, 0] == pytest.approx(1.0)
        assert out[0, 0, 0, -1] == pytest.approx(2.0)
        assert out[0, 0, -1, 0] == pytest.approx(3.0)
        assert out[0, 0, -1, -1] == pytest.approx(4.0)


class TestPixelShuffle:
    def test_doubles_spatial_dims(self) -> None:
        # ``pixel_shuffle`` rearranges (N, C·r², H, W) → (N, C, H·r, W·r).
        x: lucid.Tensor = lucid.tensor(
            np.arange(48, dtype=np.float32).reshape(1, 12, 2, 2)
        )
        out: lucid.Tensor = F.pixel_shuffle(x, 2)
        assert out.shape == (1, 3, 4, 4)

    def test_round_trip(self) -> None:
        x: lucid.Tensor = lucid.tensor(
            np.arange(48, dtype=np.float32).reshape(1, 12, 2, 2)
        )
        rt: lucid.Tensor = F.pixel_unshuffle(F.pixel_shuffle(x, 2), 2)
        np.testing.assert_allclose(rt.numpy(), x.numpy())

    def test_pixel_shuffle_rejects_bad_channel_count(self) -> None:
        x: lucid.Tensor = lucid.randn(1, 5, 2, 2)  # 5 not divisible by 4
        with pytest.raises(ValueError, match="divisible"):
            F.pixel_shuffle(x, 2)

    def test_module_matches_functional(self) -> None:
        x: lucid.Tensor = lucid.tensor(
            np.arange(48, dtype=np.float32).reshape(1, 12, 2, 2)
        )
        np.testing.assert_allclose(
            nn.PixelShuffle(2)(x).numpy(),
            F.pixel_shuffle(x, 2).numpy(),
        )


class TestMultiHeadAttentionForward:
    def test_matches_module(self) -> None:
        # ``F.multi_head_attention_forward`` should produce the same output as
        # the module path for a basic self-attention call.
        mha: nn.MultiheadAttention = nn.MultiheadAttention(embed_dim=8, num_heads=2)
        mha.eval()
        q: lucid.Tensor = lucid.randn(3, 1, 8)  # (T, B, E)
        module_out, _ = mha(q, q, q, need_weights=False)
        fn_out, _ = F.multi_head_attention_forward(
            q, q, q,
            embed_dim_to_check=8,
            num_heads=2,
            in_proj_weight=mha.in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            bias_k=mha.bias_k,
            bias_v=mha.bias_v,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=mha.out_proj_weight,
            out_proj_bias=mha.out_proj_bias,
            training=False,
            need_weights=False,
        )
        np.testing.assert_allclose(fn_out.numpy(), module_out.numpy(), atol=1e-5)
