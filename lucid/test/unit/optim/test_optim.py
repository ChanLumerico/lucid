"""``lucid.optim`` — optimizers + LR schedulers."""

import numpy as np
import pytest

import lucid
import lucid.optim as optim


def _quadratic_problem() -> tuple[lucid.Tensor, callable]:
    """Return ``(x, fn)`` where ``fn(x) = (x − target)²`` minimised at
    ``target = 1.0`` with ``x`` initialised at the origin."""
    x = lucid.tensor([0.0, 0.0], requires_grad=True)
    target = lucid.tensor([1.0, 1.0])

    def loss() -> lucid.Tensor:
        return ((x - target) ** 2).sum()

    return x, loss


class TestSGD:
    def test_converges(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.SGD([x], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        np.testing.assert_allclose(x.numpy(), [1.0, 1.0], atol=1e-3)


class TestAdam:
    def test_converges(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.Adam([x], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        np.testing.assert_allclose(x.numpy(), [1.0, 1.0], atol=1e-3)


class TestAdamW:
    def test_step_runs(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.AdamW([x], lr=0.1, weight_decay=0.01)
        for _ in range(50):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        # After 50 steps the value should have moved meaningfully.
        assert (x.numpy() > 0.5).all()


class TestRMSprop:
    def test_converges_loose(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.RMSprop([x], lr=0.05)
        for _ in range(500):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        np.testing.assert_allclose(x.numpy(), [1.0, 1.0], atol=5e-3)


class TestZeroGrad:
    def test_zero_grad_resets(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.SGD([x], lr=0.1)
        loss_fn().backward()
        assert x.grad is not None
        opt.zero_grad()
        # ``zero_grad`` may leave ``grad`` as zeros or None — both
        # satisfy "no leftover gradient".
        if x.grad is not None:
            np.testing.assert_array_equal(x.grad.numpy(), [0.0, 0.0])


class TestLRScheduler:
    def test_step_lr_decays(self) -> None:
        x, _ = _quadratic_problem()
        opt = optim.SGD([x], lr=1.0)
        if not hasattr(optim, "lr_scheduler"):
            pytest.skip("lr_scheduler module not exposed")
        sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        sched.step()
        # After one step the LR should be halved.
        for group in opt.param_groups:
            assert abs(group["lr"] - 0.5) < 1e-6


class TestOptimizerGpuMemory:
    """Regression: the GPU optimizer step must not leak active MLX memory.

    ``Optimizer::step`` writes each param back as an UNEVALUATED MLX array;
    without a per-step eval the lazy graph pins every prior step's compute →
    unbounded active-memory / RSS growth on the GPU path (only loops that call
    ``.item()`` every step happen to mask it). A leak grows linearly with steps;
    the batched eval flush keeps it flat.
    """

    def test_no_active_memory_leak(self) -> None:
        mx = pytest.importorskip("mlx.core")
        import lucid.nn as nn

        m = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)
        ).to("metal")
        opt = optim.Adam(m.parameters(), lr=1e-3)
        x = lucid.randn(16, 64).to("metal")
        t = lucid.randn(16, 64).to("metal")

        def step() -> None:
            opt.zero_grad()
            loss = ((m(x) - t) ** 2).mean()
            loss.backward()
            opt.step()

        for _ in range(20):  # warm up — lazy alloc / first-step state settles
            step()
        mx.synchronize()
        before = mx.get_active_memory()
        for _ in range(300):  # NO per-step .item() — the leak-exposing pattern
            step()
        mx.synchronize()
        delta = mx.get_active_memory() - before
        # Pre-fix this grew ~60-90 B/step (≈ 20+ KB over 300, unbounded). Flat now.
        assert delta < 8192, (
            f"optimizer leaked {delta} B over 300 steps "
            f"({delta / 300:.1f} B/step) — Optimizer::step eval flush regressed"
        )
