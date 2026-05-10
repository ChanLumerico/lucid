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
        f = lambda x: (x**2).sum(dim=-1)
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
        f = lambda x: (x**2).sum(dim=-1, keepdim=True)
        X = lucid.randn(6, 3)
        Y = func.vmap(f, chunk_size=1)(X)
        assert list(Y.shape) == [6, 1]


# ── composability with autograd ───────────────────────────────────────────────


class TestVmapGradCompose:
    def test_grad_inside_vmap(self) -> None:
        """vmap(grad(fn)) gives per-sample gradients."""
        f = lambda x: (x**2).sum()
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
        f = lambda x: (x**2).sum()
        df = func.grad(f)

        per_sample = func.vmap(df)(X)

        manual = []
        for i in range(X.shape[0]):
            xi = X[i].detach().requires_grad_(True)
            loss = (xi**2).sum()
            loss.backward()
            manual.append(xi.grad.detach())
        manual_t = lucid.stack(manual, dim=0)

        assert lucid.allclose(per_sample, manual_t)


# ── Stage 2: element-level isolation ─────────────────────────────────────────


class TestVmapIsolation:
    """vmap over transforms that require per-element isolation (jacrev/jacfwd/hessian).

    Stage 1 (move-batch-dim + call-once) gives wrong shapes for these:
      vmap(jacrev(fn)) → (B, out, B, in)  ← wrong
    Stage 2 isolation gives:
      vmap(jacrev(fn)) → (B, out, in)     ← correct
    """

    def test_vmap_jacrev_vector_output_shape(self) -> None:
        """vmap(jacrev(fn)) shape is (B, out, in) for fn: R^n → R^m."""
        f = lambda x: lucid.stack([x.sum(), (x**2).sum()])  # R^3 → R^2
        B, n = 4, 3
        X = lucid.randn(B, n)
        J = func.vmap(func.jacrev(f))(X)
        assert list(J.shape) == [B, 2, n]

    def test_vmap_jacrev_matches_manual(self) -> None:
        """Each vmap(jacrev) element matches individually computed Jacobian."""
        f = lambda x: lucid.stack([x[0] * x[1], x[1] ** 2])  # R^3 → R^2
        B, n = 3, 3
        X = lucid.randn(B, n)
        jf = func.jacrev(f)

        batched = func.vmap(jf)(X)
        for b in range(B):
            ref = jf(X[b])
            assert lucid.allclose(batched[b], ref, atol=1e-5)

    def test_vmap_jacrev_scalar_fn(self) -> None:
        """vmap(jacrev(fn)) for scalar fn gives (B, n) — same as vmap(grad)."""
        f = lambda x: (x**2).sum()
        B, n = 4, 3
        X = lucid.randn(B, n)
        J = func.vmap(func.jacrev(f))(X)
        g = func.vmap(func.grad(f))(X)
        assert list(J.shape) == [B, n]
        assert lucid.allclose(J, g, atol=1e-5)

    def test_vmap_jacfwd_vector_output_shape(self) -> None:
        """vmap(jacfwd(fn)) shape is (B, out, in) for fn: R^n → R^m."""
        f = lambda x: lucid.stack([(x * 2).sum(), (x**2).sum()])
        B, n = 3, 4
        X = lucid.randn(B, n)
        J = func.vmap(func.jacfwd(f))(X)
        assert list(J.shape) == [B, 2, n]

    def test_vmap_jacfwd_matches_jacrev(self) -> None:
        """vmap(jacfwd) and vmap(jacrev) agree on same function."""
        f = lambda x: lucid.stack([x.sum(), (x**3).sum()])
        B, n = 3, 3
        X = lucid.randn(B, n)
        Jrev = func.vmap(func.jacrev(f))(X)
        Jfwd = func.vmap(func.jacfwd(f))(X)
        assert lucid.allclose(Jrev, Jfwd, atol=1e-4)

    def test_vmap_hessian_shape(self) -> None:
        """vmap(hessian(fn)) shape is (B, n, n) for scalar fn."""
        f = lambda x: (x**2).sum()
        B, n = 3, 4
        X = lucid.randn(B, n)
        H = func.vmap(func.hessian(f))(X)
        assert list(H.shape) == [B, n, n]

    def test_vmap_hessian_matches_manual(self) -> None:
        """Each vmap(hessian) element matches individual hessian call."""
        f = lambda x: (x**3).sum()
        B, n = 3, 3
        X = lucid.randn(B, n)
        hf = func.hessian(f)

        batched = func.vmap(hf)(X)
        for b in range(B):
            ref = hf(X[b])
            assert lucid.allclose(batched[b], ref, atol=1e-4)

    def test_vmap_jacrev_non_zero_in_dim(self) -> None:
        """Isolation works when batch dim is not 0."""
        f = lambda x: lucid.stack([x.sum(), (x**2).sum()])
        B, n = 3, 4
        X = lucid.randn(n, B)  # batch on dim 1
        J = func.vmap(func.jacrev(f), in_dims=1)(X)
        assert list(J.shape) == [B, 2, n]


# ── randomness enforcement ────────────────────────────────────────────────────


class TestVmapRandomness:
    """randomness='error' raises; 'different' allows random ops."""

    def test_error_mode_raises_on_randn(self) -> None:
        f = lambda x: x + lucid.randn(*x.shape)
        with pytest.raises(RuntimeError, match="random op"):
            func.vmap(f, randomness="error")(lucid.ones(4, 3))

    def test_error_mode_raises_on_rand(self) -> None:
        f = lambda x: x * lucid.rand(*x.shape)
        with pytest.raises(RuntimeError, match="random op"):
            func.vmap(f, randomness="error")(lucid.ones(3, 2))

    def test_different_mode_allows_randn(self) -> None:
        f = lambda x: x + lucid.randn(*x.shape)
        out = func.vmap(f, randomness="different")(lucid.zeros(4, 3))
        assert list(out.shape) == [4, 3]

    def test_error_is_default(self) -> None:
        f = lambda x: x + lucid.randn(*x.shape)
        with pytest.raises(RuntimeError):
            func.vmap(f)(lucid.ones(2, 2))

    def test_no_random_op_passes_error_mode(self) -> None:
        f = lambda x: x * 2.0
        out = func.vmap(f, randomness="error")(lucid.ones(3, 4))
        assert list(out.shape) == [3, 4]

    def test_nested_vmap_inner_overrides_outer(self) -> None:
        # inner vmap with randomness='different' pushes 'different' onto the
        # thread-local stack, overriding the outer 'error' guard. Innermost
        # randomness setting wins — correct RAII semantics.
        f = lambda x: x + lucid.randn(*x.shape)
        inner = func.vmap(f, randomness="different")
        out = func.vmap(inner, randomness="error")(lucid.ones(2, 3, 4))
        assert list(out.shape) == [2, 3, 4]


# ── strategy parameter ────────────────────────────────────────────────────────


class TestVmapStrategy:
    """strategy='auto'|'isolated'|'vectorized' dispatch control."""

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            func.vmap(lambda x: x, strategy="bad")

    def test_isolated_gives_correct_jacrev_shape(self) -> None:
        f = lambda x: lucid.stack([x.sum(), (x**2).sum()])
        J = func.vmap(func.jacrev(f), strategy="isolated")(lucid.randn(4, 3))
        assert list(J.shape) == [4, 2, 3]

    def test_vectorized_strategy_for_plain_fn(self) -> None:
        f = lambda x: x * 2.0
        X = lucid.arange(12.0).reshape(4, 3)
        out_vec = func.vmap(f, strategy="vectorized")(X)
        out_auto = func.vmap(f, strategy="auto")(X)
        assert lucid.allclose(out_vec, out_auto)

    def test_isolated_strategy_explicit_for_user_fn(self) -> None:
        """User can force isolation for any function, e.g. custom vjp wrappers."""
        results = []

        def f(x: lucid.Tensor) -> lucid.Tensor:
            out, vjp_fn = func.vjp(lambda z: (z**2).sum(), x)
            g = vjp_fn(lucid.ones_like(out))[0]
            assert g is not None
            return g

        X = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        per_sample = func.vmap(f, strategy="isolated")(X)
        assert list(per_sample.shape) == [2, 2]
        expected = X * 2.0
        assert lucid.allclose(per_sample, expected, atol=1e-5)


# ── chunk_size in isolation mode ──────────────────────────────────────────────


class TestVmapChunkIsolation:
    """chunk_size bounds intermediate memory in isolation mode."""

    def test_chunk_matches_full_isolation(self) -> None:
        f = lambda x: lucid.stack([x.sum(), (x**2).sum()])
        jf = func.jacrev(f)
        X = lucid.randn(8, 3)
        full = func.vmap(jf)(X)
        chunked = func.vmap(jf, chunk_size=3)(X)
        assert lucid.allclose(full, chunked, atol=1e-5)

    def test_chunk_size_1_matches_full(self) -> None:
        f = lambda x: (x**3).sum()
        jf = func.jacrev(f)
        X = lucid.randn(5, 4)
        full = func.vmap(jf)(X)
        chunk1 = func.vmap(jf, chunk_size=1)(X)
        assert lucid.allclose(full, chunk1, atol=1e-5)


# ── linearize linear_fn isolation ─────────────────────────────────────────────


class TestVmapLinearize:
    """vmap(linear_fn) from linearize uses element isolation automatically."""

    def test_linear_fn_vmap_shape(self) -> None:
        f = lambda x: lucid.stack([x.sum(), (x**2).sum()])
        x0 = lucid.tensor([1.0, 2.0, 3.0])
        _, lin = func.linearize(f, x0)
        B, n = 4, 3
        tangents = lucid.randn(B, n)
        jvp_batch = func.vmap(lin)(tangents)
        assert list(jvp_batch.shape) == [B, 2]

    def test_linear_fn_matches_manual_jvp(self) -> None:
        f = lambda x: lucid.stack([x.sum(), (x**2).sum()])
        x0 = lucid.tensor([1.0, 0.0, -1.0])
        _, lin = func.linearize(f, x0)
        t = lucid.tensor([1.0, 0.0, 0.0])
        _, jvp_manual = func.jvp(f, (x0,), (t,))
        lin_out = lin(t)
        assert lucid.allclose(lin_out, jvp_manual, atol=1e-5)
