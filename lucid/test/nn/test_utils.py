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


class TestSpectralNorm:
    def test_registers_orig_and_buffers(self) -> None:
        m: nn.Linear = nn.Linear(8, 4)
        nn.utils.spectral_norm(m)
        assert hasattr(m, "weight_orig")
        assert hasattr(m, "weight_u")
        assert hasattr(m, "weight_v")

    def test_constrains_max_singular_value(self) -> None:
        m: nn.Linear = nn.Linear(8, 4)
        nn.utils.spectral_norm(m)
        m.train()
        x: lucid.Tensor = lucid.randn(2, 8)
        for _ in range(20):
            _ = m(x)  # let power iteration converge
        sigma: float = float(np.linalg.svd(m.weight.numpy(), compute_uv=False)[0])
        # σ_max should be ~1 after spectral_norm + enough iterations.
        assert abs(sigma - 1.0) < 0.05

    def test_remove_restores_leaf(self) -> None:
        m: nn.Linear = nn.Linear(8, 4)
        nn.utils.spectral_norm(m)
        nn.utils.remove_spectral_norm(m)
        names: list[str] = list(dict(m.named_parameters()).keys())
        assert "weight" in names
        assert "weight_orig" not in names


class TestPackSequence:
    def test_packed_data_and_lengths(self) -> None:
        seqs: list[lucid.Tensor] = [
            lucid.tensor([1.0, 2.0, 3.0]),
            lucid.tensor([4.0, 5.0]),
            lucid.tensor([6.0]),
        ]
        ps = nn.utils.pack_sequence(seqs)
        # ``data`` collects per-timestep elements in the time-major order
        # that ``pack_padded_sequence`` produces from a padded batch.
        # batch_sizes encodes how many sequences are still alive at each step.
        np.testing.assert_array_equal(ps.batch_sizes.numpy().astype(int), [3, 2, 1])
        # Total number of valid elements should match the sum of lengths.
        assert int(ps.data._impl.numel()) == 6

    def test_rejects_unsorted_lengths(self) -> None:
        seqs: list[lucid.Tensor] = [
            lucid.tensor([1.0]),
            lucid.tensor([2.0, 3.0, 4.0]),  # longer than first
        ]
        with pytest.raises(ValueError, match="decreasing order"):
            nn.utils.pack_sequence(seqs, enforce_sorted=True)


class TestParametrize:
    def test_register_and_remove(self) -> None:
        from lucid.nn.utils import parametrize

        m: nn.Linear = nn.Linear(4, 2)

        class Identity(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x

        parametrize.register_parametrization(m, "weight", Identity())
        assert parametrize.is_parametrized(m, "weight")
        # Forward still works while parametrised.
        out: lucid.Tensor = m(lucid.randn(1, 4))
        assert out.shape == (1, 2)
        # Removing restores the leaf parameter.
        parametrize.remove_parametrizations(m, "weight")
        assert not parametrize.is_parametrized(m)
        assert "weight" in dict(m.named_parameters())

    def test_parametrizations_weight_norm_alias(self) -> None:
        # ``parametrizations.weight_norm`` should produce the same buffers
        # as the legacy ``weight_norm`` entry.
        m: nn.Linear = nn.Linear(4, 2)
        nn.utils.parametrizations.weight_norm(m)
        assert hasattr(m, "weight_g")
        assert hasattr(m, "weight_v")


class TestPrune:
    def test_l1_unstructured_keeps_correct_count(self) -> None:
        m: nn.Linear = nn.Linear(8, 4)  # 32 weights
        nn.utils.prune.l1_unstructured(m, "weight", amount=0.5)
        mask: np.ndarray = m.weight_mask.numpy()
        # Half the elements should be zeroed.
        assert int((mask == 0).sum()) == 16

    def test_random_unstructured_runs(self) -> None:
        m: nn.Linear = nn.Linear(8, 4)
        nn.utils.prune.random_unstructured(m, "weight", amount=0.5)
        mask: np.ndarray = m.weight_mask.numpy()
        # Random pruning has variance — just sanity-check the mask shape
        # and that some elements are zero.
        assert mask.shape == (4, 8)
        assert int((mask == 0).sum()) > 0

    def test_identity_keeps_all(self) -> None:
        m: nn.Linear = nn.Linear(4, 2)
        nn.utils.prune.identity(m, "weight")
        assert (m.weight_mask.numpy() == 1.0).all()

    def test_remove_restores_leaf(self) -> None:
        m: nn.Linear = nn.Linear(8, 4)
        nn.utils.prune.l1_unstructured(m, "weight", amount=0.5)
        nn.utils.prune.remove(m, "weight")
        assert not nn.utils.prune.is_pruned(m)
        assert "weight" in dict(m.named_parameters())
        # ``weight_orig`` and ``weight_mask`` are gone.
        assert not hasattr(m, "weight_orig")

    def test_invalid_amount_rejected(self) -> None:
        m: nn.Linear = nn.Linear(4, 2)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            nn.utils.prune.l1_unstructured(m, "weight", amount=1.5)
