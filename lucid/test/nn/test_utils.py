"""Tests for ``lucid.nn.utils`` helpers."""

import math

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestClipGradNorm:
    def test_clip_grad_norm_returns_total(self) -> None:
        m: nn.Linear = nn.Linear(4, 2)
        x: lucid.Tensor = lucid.randn(3, 4)
        m(x).sum().backward()
        total: lucid.Tensor = nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        assert float(total.item()) >= 0.0

    def test_clip_grad_norm_shrinks_grads(self) -> None:
        m: nn.Linear = nn.Linear(4, 2)
        x: lucid.Tensor = lucid.randn(3, 4) * 100.0  # large grads
        m(x).sum().backward()
        # Save grad norms before clipping.
        nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.1)
        post: float = (
            sum(float((p.grad * p.grad).sum().item()) for p in m.parameters()) ** 0.5
        )
        # Should be at most max_norm + a small epsilon.
        assert post <= 0.1 + 1e-3


class TestParametersToVector:
    def test_round_trip_preserves_data(self) -> None:
        m: nn.Linear = nn.Linear(4, 2)
        flat: lucid.Tensor = nn.utils.parameters_to_vector(m.parameters())
        # Modify outside of autograd, then write back.
        modified: lucid.Tensor = flat * 2.0
        nn.utils.vector_to_parameters(modified, m.parameters())
        flat_after: lucid.Tensor = nn.utils.parameters_to_vector(m.parameters())
        np.testing.assert_allclose(flat_after.numpy(), flat.numpy() * 2.0)

    def test_total_size_matches(self) -> None:
        m: nn.Module = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        flat: lucid.Tensor = nn.utils.parameters_to_vector(m.parameters())
        expected: int = sum(int(p._impl.numel()) for p in m.parameters())
        assert int(flat._impl.numel()) == expected

    def test_vector_to_parameters_size_mismatch_raises(self) -> None:
        m: nn.Linear = nn.Linear(4, 2)
        with pytest.raises(ValueError, match="does not match"):
            nn.utils.vector_to_parameters(lucid.zeros(3), m.parameters())

    def test_vector_to_parameters_shape_check(self) -> None:
        m: nn.Linear = nn.Linear(4, 2)
        with pytest.raises(ValueError, match="1-D"):
            nn.utils.vector_to_parameters(lucid.zeros(2, 5), m.parameters())

    def test_empty_parameter_list(self) -> None:
        flat: lucid.Tensor = nn.utils.parameters_to_vector([])
        assert int(flat._impl.numel()) == 0


class TestWeightNorm:
    def test_initial_weight_unchanged(self) -> None:
        m: nn.Linear = nn.Linear(4, 3)
        orig: np.ndarray = m.weight.numpy().copy()
        nn.utils.weight_norm(m, "weight", dim=0)
        np.testing.assert_allclose(m.weight.numpy(), orig, atol=1e-5)

    def test_g_and_v_replace_weight_in_parameters(self) -> None:
        m: nn.Linear = nn.Linear(4, 3)
        nn.utils.weight_norm(m, "weight", dim=0)
        names: list[str] = list(dict(m.named_parameters()).keys())
        assert "weight_g" in names
        assert "weight_v" in names
        # The reparametrised ``weight`` is no longer a leaf Parameter.
        assert "weight" not in names

    def test_forward_and_backward(self) -> None:
        m: nn.Linear = nn.Linear(4, 3)
        nn.utils.weight_norm(m, "weight", dim=0)
        x: lucid.Tensor = lucid.randn(2, 4)
        out: lucid.Tensor = m(x)
        assert out.shape == (2, 3)
        out.sum().backward()
        assert m.weight_g.grad is not None
        assert m.weight_v.grad is not None

    def test_remove_weight_norm_restores_leaf(self) -> None:
        m: nn.Linear = nn.Linear(4, 3)
        orig: np.ndarray = m.weight.numpy().copy()
        nn.utils.weight_norm(m, "weight", dim=0)
        nn.utils.remove_weight_norm(m, "weight")
        names: list[str] = list(dict(m.named_parameters()).keys())
        assert "weight" in names
        assert "weight_g" not in names
        np.testing.assert_allclose(m.weight.numpy(), orig, atol=1e-5)
