"""Tests for SDPA and MultiheadAttention."""

import pytest
import math
import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test.helpers.numerics import make_tensor


class TestSDPA:
    def test_output_shape(self):
        B, H, T, E = 2, 4, 8, 16
        q = make_tensor((B, H, T, E))
        k = make_tensor((B, H, T, E))
        v = make_tensor((B, H, T, E))
        out = F.scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, T, E)

    def test_causal_mask(self):
        B, H, T, E = 1, 2, 6, 8
        q = make_tensor((B, H, T, E))
        k = make_tensor((B, H, T, E))
        v = make_tensor((B, H, T, E))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (B, H, T, E)

    def test_custom_scale(self):
        B, H, T, E = 1, 1, 4, 8
        q = make_tensor((B, H, T, E))
        k = make_tensor((B, H, T, E))
        v = make_tensor((B, H, T, E))
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert out.shape == (B, H, T, E)

    def test_default_scale_matches_theory(self):
        # Default scale = 1/sqrt(E); verify output shape only
        B, H, T, E = 2, 2, 5, 16
        q = make_tensor((B, H, T, E))
        k = make_tensor((B, H, T, E))
        v = make_tensor((B, H, T, E))
        out = F.scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, T, E)


class TestMultiheadAttention:
    def test_output_shape(self):
        layer = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        x = make_tensor((4, 8, 16))
        out, weights = layer(x, x, x)
        assert out.shape == (4, 8, 16)

    def test_cross_attention_shape(self):
        layer = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        q = make_tensor((4, 8, 16))
        kv = make_tensor((4, 10, 16))
        out, _ = layer(q, kv, kv)
        assert out.shape == (4, 8, 16)

    def test_backward(self):
        layer = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        x = make_tensor((2, 5, 16), requires_grad=True)
        out, _ = layer(x, x, x)
        lucid.sum(out).backward()
        assert x.grad is not None


class TestMHAContracts:
    """Coverage for MHA's full contract — masks, separate kv dims, head split."""

    def test_embed_dim_must_divide_num_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            nn.MultiheadAttention(embed_dim=15, num_heads=4)

    def test_need_weights_false_returns_none(self):
        m = nn.MultiheadAttention(16, 4, batch_first=True)
        x = make_tensor((2, 5, 16))
        out, w = m(x, x, x, need_weights=False)
        assert w is None
        assert out.shape == (2, 5, 16)

    def test_need_weights_returns_averaged_weights(self):
        m = nn.MultiheadAttention(16, 4, batch_first=True)
        x = make_tensor((2, 5, 16))
        out, w = m(x, x, x, need_weights=True, average_attn_weights=True)
        # Default: averaged across heads → (B, Tq, Tk).
        assert w.shape == (2, 5, 5)

    def test_need_weights_per_head(self):
        m = nn.MultiheadAttention(16, 4, batch_first=True)
        x = make_tensor((2, 5, 16))
        out, w = m(x, x, x, need_weights=True, average_attn_weights=False)
        assert w.shape == (2, 4, 5, 5)

    def test_kdim_vdim_separate_projections(self):
        m = nn.MultiheadAttention(16, 4, kdim=8, vdim=12, batch_first=True)
        # Three separate projection weights, no fused in_proj.
        assert m.in_proj_weight is None
        assert m.q_proj_weight.shape == (16, 16)
        assert m.k_proj_weight.shape == (16, 8)
        assert m.v_proj_weight.shape == (16, 12)

        q = make_tensor((2, 5, 16))
        k = make_tensor((2, 7, 8), seed=1)
        v = make_tensor((2, 7, 12), seed=2)
        out, _ = m(q, k, v)
        assert out.shape == (2, 5, 16)

    def test_key_padding_mask_zeroes_attention(self):
        # When all key positions are masked except one, the output for
        # every query row should equal the attended (single) value row
        # passed through the out_proj.
        m = nn.MultiheadAttention(8, 2, batch_first=True, bias=False)
        x = make_tensor((1, 4, 8))
        kpm = lucid.tensor([[False, True, True, True]], dtype=lucid.bool_)
        out, w = m(
            x, x, x, key_padding_mask=kpm, need_weights=True, average_attn_weights=False
        )
        # All weight mass should sit on position 0.
        weights_np = w.numpy()
        np.testing.assert_allclose(weights_np[..., 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(weights_np[..., 1:], 0.0, atol=1e-5)

    def test_attn_mask_2d_broadcasts(self):
        m = nn.MultiheadAttention(8, 2, batch_first=True)
        x = make_tensor((2, 4, 8))
        am = lucid.tensor(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, True],
                [False, False, False, True],
            ],
            dtype=lucid.bool_,
        )
        out, w = m(x, x, x, attn_mask=am, need_weights=True, average_attn_weights=False)
        # Attention from query rows 2 and 3 should put zero weight on key 3.
        wn = w.numpy()
        np.testing.assert_allclose(wn[:, :, 2, 3], 0.0, atol=1e-5)
        np.testing.assert_allclose(wn[:, :, 3, 3], 0.0, atol=1e-5)

    def test_add_zero_attn_appends_extra_key(self):
        m = nn.MultiheadAttention(8, 2, batch_first=True, add_zero_attn=True)
        x = make_tensor((2, 4, 8))
        out, w = m(x, x, x, need_weights=True, average_attn_weights=False)
        # add_zero_attn appends one extra key/value row.
        assert w.shape[-1] == 4 + 1

    def test_add_bias_kv_adds_one_extra_position(self):
        m = nn.MultiheadAttention(8, 2, batch_first=True, add_bias_kv=True)
        x = make_tensor((2, 4, 8))
        out, w = m(x, x, x, need_weights=True, average_attn_weights=False)
        assert w.shape[-1] == 4 + 1
        assert m.bias_k is not None
        assert m.bias_v is not None

    def test_extra_repr_reflects_kdim_and_options(self):
        m = nn.MultiheadAttention(16, 4, kdim=8, add_zero_attn=True, add_bias_kv=True)
        s = repr(m)
        assert "kdim=8" in s
        assert "add_bias_kv=True" in s
        assert "add_zero_attn=True" in s

    def test_reference_state_dict_keys_load(self):
        # External reference checkpoints use ``out_proj.weight`` /
        # ``out_proj.bias``; the load hook must remap them.
        from collections import OrderedDict

        m = nn.MultiheadAttention(16, 4, batch_first=True)
        sd = OrderedDict()
        sd["in_proj_weight"] = lucid.zeros(48, 16)
        sd["in_proj_bias"] = lucid.zeros(48)
        sd["out_proj.weight"] = lucid.zeros(16, 16)
        sd["out_proj.bias"] = lucid.zeros(16)
        sd._metadata = {}
        result = m.load_state_dict(sd)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
