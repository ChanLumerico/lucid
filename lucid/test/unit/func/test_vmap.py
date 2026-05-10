"""Unit tests for lucid.func.vmap — C++ vmapped dispatch."""

import pytest
import lucid
import lucid.func as func


# ── basic vectorisation ───────────────────────────────────────────────────────


class TestVmapBasic:
    def test_elementwise_fn(self) -> None:
        """Batched elementwise op produces correct output shape and values."""
        f = lambda x: x * 2.0
        X = lucid.arange(6.0).reshape(3, 2)
        Y = func.vmap(f)(X)
        assert list(Y.shape) == [3, 2]
        expected = X * 2.0
        assert lucid.allclose(Y, expected)

    def test_scalar_output(self) -> None:
        """Each batch element produces a scalar; output is 1-D."""
        f = lambda x: (x ** 2).sum(dim=-1)
        X = lucid.ones(4, 3)
        Y = func.vmap(f)(X)
        assert list(Y.shape) == [4]
        assert lucid.allclose(Y, lucid.full((4,), 3.0))

    def test_matmul_batched(self) -> None:
        """Batched matrix-vector multiply via C++ dispatch."""
        W = lucid.eye(4)
        f = lambda x: x @ W
        X = lucid.randn(8, 4)
        Y = func.vmap(f, in_dims=0)(X)
        assert list(Y.shape) == [8, 4]
        assert lucid.allclose(Y, X)

    def test_linear_broadcast(self) -> None:
        """Weight has no batch dim (in_dims=None); only x is batched."""
        W = lucid.randn(5, 3)
        b = lucid.zeros(5)
        f = lambda x, w, bias: x @ w.T + bias
        X = lucid.randn(7, 3)
        Y = func.vmap(f, in_dims=(0, None, None))(X, W, b)
        assert list(Y.shape) == [7, 5]

    def test_nested_ops(self) -> None:
        """Multi-op function correctly vectorised (use swapaxes for batch-safe transpose)."""
        # Inside vmap, x is (B, n, n); use swapaxes to transpose last two dims
        f = lambda x: lucid.relu(x @ lucid.swapaxes(x, -2, -1))
        X = lucid.randn(4, 3, 3)
        Y = func.vmap(f)(X)
        assert list(Y.shape) == [4, 3, 3]

    def test_tuple_output(self) -> None:
        """Function returning a tuple of tensors."""
        f = lambda x: (x, x * 2)
        X = lucid.randn(5, 4)
        a, b = func.vmap(f)(X)
        assert list(a.shape) == [5, 4]
        assert list(b.shape) == [5, 4]
        assert lucid.allclose(b, X * 2)


# ── in_dims / out_dims variants ───────────────────────────────────────────────


class TestVmapDims:
    def test_in_dim_1(self) -> None:
        """Batch along dim 1: each column becomes a batch element."""
        f = lambda x: x * 2  # elementwise, preserves shape
        X = lucid.randn(3, 5)
        # vmap moves batch-dim 1 to front → (5, 3), calls f → (5, 3)
        Y = func.vmap(f, in_dims=1)(X)
        assert list(Y.shape) == [5, 3]

    def test_negative_in_dim(self) -> None:
        """Negative in_dim resolves correctly."""
        f = lambda x: x * 2
        X = lucid.randn(4, 6)
        Y = func.vmap(f, in_dims=-2)(X)
        assert list(Y.shape) == [4, 6]

    def test_out_dims(self) -> None:
        """Batch dim placed at non-zero output position."""
        f = lambda x: x
        X = lucid.randn(3, 4)
        Y = func.vmap(f, in_dims=0, out_dims=1)(X)
        assert list(Y.shape) == [4, 3]

    def test_inconsistent_batch_sizes_raises(self) -> None:
        f = lambda x, y: x + y
        X = lucid.randn(3, 4)
        Y = lucid.randn(5, 4)
        with pytest.raises(ValueError, match="inconsistent batch sizes"):
            func.vmap(f)(X, Y)

    def test_in_dim_out_of_range_raises(self) -> None:
        f = lambda x: x
        X = lucid.randn(3)
        with pytest.raises(ValueError, match="out of range"):
            func.vmap(f, in_dims=2)(X)


# ── chunk_size ────────────────────────────────────────────────────────────────


class TestVmapChunk:
    def test_chunk_same_as_full(self) -> None:
        """chunk_size produces the same result as full-batch dispatch."""
        f = lambda x: x * 3
        X = lucid.randn(12, 4)
        full = func.vmap(f)(X)
        chunked = func.vmap(f, chunk_size=4)(X)
        assert lucid.allclose(full, chunked)

    def test_chunk_single_element(self) -> None:
        """chunk_size=1 processes each element separately."""
        f = lambda x: (x ** 2).sum(dim=-1, keepdim=True)
        X = lucid.randn(6, 3)
        Y = func.vmap(f, chunk_size=1)(X)
        assert list(Y.shape) == [6, 1]


# ── composability with autograd ───────────────────────────────────────────────


class TestVmapGradCompose:
    def test_grad_inside_vmap(self) -> None:
        """vmap(grad(fn)) gives per-sample gradients."""
        f = lambda x: (x ** 2).sum()
        df = func.grad(f)
        X = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        grads = func.vmap(df)(X)
        assert list(grads.shape) == [2, 2]
        expected = X * 2.0
        assert lucid.allclose(grads, expected)

    def test_vmap_inside_grad(self) -> None:
        """grad(vmap(fn)) differentiates through batched computation."""
        W = lucid.eye(3)
        f = lambda X: (func.vmap(lambda x: x @ W)(X) ** 2).sum()
        X = lucid.randn(4, 3).requires_grad_(True)
        loss = f(X)
        loss.backward()
        assert X.grad is not None
        assert list(X.grad.shape) == [4, 3]

    def test_per_sample_grad_matches_manual(self) -> None:
        """Per-sample gradients match manually computed gradients."""
        X = lucid.tensor([[1.0, 0.0], [0.0, 2.0]])
        f = lambda x: (x ** 2).sum()
        df = func.grad(f)

        per_sample = func.vmap(df)(X)

        manual = []
        for i in range(X.shape[0]):
            xi = X[i].detach().requires_grad_(True)
            loss = (xi ** 2).sum()
            loss.backward()
            manual.append(xi.grad.detach())
        manual_t = lucid.stack(manual, dim=0)

        assert lucid.allclose(per_sample, manual_t)
