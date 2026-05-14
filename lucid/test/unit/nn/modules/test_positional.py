"""Unit tests for ``lucid.nn`` positional-encoding modules + functional pair.

Covers the three Lucid-specific extensions to PyTorch's nn surface:

    * :class:`lucid.nn.SinusoidalEmbedding`
    * :class:`lucid.nn.SinusoidalEmbedding2D`
    * :class:`lucid.nn.RotaryEmbedding` + ``apply_rotary_emb``
"""

import math

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional embedding — 1D (Vaswani et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────


class TestSinusoidalEmbedding:
    def test_shape(self) -> None:
        pe = nn.SinusoidalEmbedding(num_positions=32, embedding_dim=16)
        assert tuple(pe().shape) == (32, 16)

    def test_odd_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="even embedding_dim"):
            nn.SinusoidalEmbedding(num_positions=8, embedding_dim=7)

    def test_pos_zero_is_alternating_unit(self) -> None:
        # At position 0, sin(·) = 0 and cos(·) = 1 → table row alternates
        # 0, 1, 0, 1, ...
        table = nn.SinusoidalEmbedding(num_positions=4, embedding_dim=8)()
        row0 = [float(table[0, i].item()) for i in range(8)]
        for i in range(0, 8, 2):
            assert math.isclose(row0[i], 0.0, abs_tol=1e-6)
            assert math.isclose(row0[i + 1], 1.0, abs_tol=1e-6)

    def test_distinct_rows(self) -> None:
        # Every position should map to a distinct vector.
        table = nn.SinusoidalEmbedding(num_positions=10, embedding_dim=8)()
        for p in range(1, 10):
            diff = float(((table[p] - table[0]) ** 2).sum().item())
            assert diff > 1e-6, f"row {p} equals row 0"

    def test_functional_matches_module(self) -> None:
        module_out = nn.SinusoidalEmbedding(num_positions=8, embedding_dim=6)()
        func_out = F.sinusoidal_embedding(num_positions=8, embedding_dim=6)
        diff = float(((module_out - func_out) ** 2).sum().item())
        assert diff < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional embedding — 2D (DETR §A.4)
# ─────────────────────────────────────────────────────────────────────────────


class TestSinusoidalEmbedding2D:
    def test_shape_row_major(self) -> None:
        pe = nn.SinusoidalEmbedding2D(height=4, width=5, embedding_dim=8)
        out = pe()
        assert tuple(out.shape) == (20, 8)

    def test_non_div_by_4_rejected(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            nn.SinusoidalEmbedding2D(height=4, width=4, embedding_dim=6)

    def test_split_layout(self) -> None:
        """First half of each vector should depend only on column, second
        half only on row — by design."""
        pe = nn.SinusoidalEmbedding2D(height=3, width=4, embedding_dim=8)()
        half = 4
        # (r=0, c=0) and (r=0, c=1) should differ in first half (cols differ)
        # but match in second half (same row).
        v00 = pe[0]  # r=0, c=0
        v01 = pe[1]  # r=0, c=1
        v10 = pe[4]  # r=1, c=0
        diff_first_half_cols = float(((v00[:half] - v01[:half]) ** 2).sum().item())
        diff_second_half_cols = float(((v00[half:] - v01[half:]) ** 2).sum().item())
        diff_first_half_rows = float(((v00[:half] - v10[:half]) ** 2).sum().item())
        diff_second_half_rows = float(((v00[half:] - v10[half:]) ** 2).sum().item())
        assert diff_first_half_cols > 1e-6  # cols differ → first half differs
        assert diff_second_half_cols < 1e-10  # same row → second half identical
        assert diff_first_half_rows < 1e-10  # same col → first half identical
        assert diff_second_half_rows > 1e-6  # rows differ → second half differs

    def test_functional_matches_module(self) -> None:
        m = nn.SinusoidalEmbedding2D(height=4, width=4, embedding_dim=8)()
        f = F.sinusoidal_embedding_2d(height=4, width=4, embedding_dim=8)
        diff = float(((m - f) ** 2).sum().item())
        assert diff < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Rotary position embedding (Su et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────


class TestRotaryEmbedding:
    def test_buffer_shapes(self) -> None:
        rope = nn.RotaryEmbedding(head_dim=16, max_position_embeddings=32)
        cos, sin = rope()
        assert tuple(cos.shape) == (32, 16)
        assert tuple(sin.shape) == (32, 16)

    def test_odd_head_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="even head_dim"):
            nn.RotaryEmbedding(head_dim=7, max_position_embeddings=8)

    def test_pos_zero_is_identity(self) -> None:
        rope = nn.RotaryEmbedding(head_dim=8, max_position_embeddings=4)
        cos, sin = rope()
        assert math.isclose(float(cos[0, 0].item()), 1.0, abs_tol=1e-6)
        assert math.isclose(float(sin[0, 0].item()), 0.0, abs_tol=1e-6)

    def test_apply_preserves_shape_and_norm(self) -> None:
        B, H, T, D = 2, 4, 6, 8
        rope = nn.RotaryEmbedding(head_dim=D, max_position_embeddings=16)
        cos, sin = rope()
        q = lucid.rand((B, H, T, D))
        k = lucid.rand((B, H, T, D))
        q_rot, k_rot = F.apply_rotary_emb(q, k, cos, sin)
        assert tuple(q_rot.shape) == (B, H, T, D)
        assert tuple(k_rot.shape) == (B, H, T, D)

        # Rotation is orthogonal — per-vector L2 norm must be preserved.
        q_norm = float((q * q).sum().item())
        q_rot_norm = float((q_rot * q_rot).sum().item())
        assert math.isclose(q_norm, q_rot_norm, rel_tol=1e-4)
