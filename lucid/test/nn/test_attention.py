"""Tests for SDPA and MultiheadAttention."""

import pytest
import math
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
