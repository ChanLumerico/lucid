"""
Comprehensive tests for all learning rate schedulers in lucid.optim.lr_scheduler.

Each scheduler is tested for:
  - Initial LR value
  - LR after expected decay steps
  - Monotonicity / periodicity properties
  - get_last_lr() consistency
  - Multi-param-group correctness
"""

import math
import pytest
import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.optim.lr_scheduler import (
    StepLR, ExponentialLR, MultiStepLR, CosineAnnealingLR,
    LambdaLR, CyclicLR, ReduceLROnPlateau, NoamScheduler,
    MultiplicativeLR, LinearLR, ConstantLR, PolynomialLR,
    CosineAnnealingWarmRestarts, OneCycleLR,
    SequentialLR, ChainedScheduler,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _sgd(lr: float = 0.1) -> optim.SGD:
    model = nn.Linear(4, 2)
    return optim.SGD(model.parameters(), lr=lr)


def _step_n(sched: object, n: int) -> None:
    """Advance scheduler n steps."""
    for _ in range(n):
        sched.step()


def _lrs(opt: optim.SGD) -> list[float]:
    return [g["lr"] for g in opt.param_groups]


# ── StepLR ────────────────────────────────────────────────────────────────────

class TestStepLR:
    def test_initial_lr_unchanged(self):
        opt = _sgd(0.1)
        sched = StepLR(opt, step_size=3, gamma=0.1)
        assert _lrs(opt) == pytest.approx([0.1])

    def test_lr_unchanged_before_step_size(self):
        opt = _sgd(0.1)
        sched = StepLR(opt, step_size=3, gamma=0.1)
        _step_n(sched, 2)
        assert _lrs(opt) == pytest.approx([0.1])

    def test_lr_decays_at_step_size(self):
        opt = _sgd(0.1)
        sched = StepLR(opt, step_size=3, gamma=0.1)
        _step_n(sched, 3)
        assert _lrs(opt) == pytest.approx([0.01])

    def test_lr_decays_twice(self):
        opt = _sgd(0.1)
        sched = StepLR(opt, step_size=2, gamma=0.5)
        _step_n(sched, 4)
        assert _lrs(opt) == pytest.approx([0.1 * 0.5 * 0.5])

    def test_get_last_lr(self):
        opt = _sgd(0.1)
        sched = StepLR(opt, step_size=5, gamma=0.1)
        _step_n(sched, 5)
        assert sched.get_last_lr() == pytest.approx([0.01])

    def test_gamma_1_no_change(self):
        opt = _sgd(0.05)
        sched = StepLR(opt, step_size=1, gamma=1.0)
        _step_n(sched, 10)
        assert _lrs(opt) == pytest.approx([0.05])


# ── ExponentialLR ─────────────────────────────────────────────────────────────

class TestExponentialLR:
    def test_decays_every_epoch(self):
        opt = _sgd(1.0)
        sched = ExponentialLR(opt, gamma=0.9)
        _step_n(sched, 1)
        assert _lrs(opt) == pytest.approx([0.9])
        _step_n(sched, 1)
        assert _lrs(opt) == pytest.approx([0.81])

    def test_n_epochs(self):
        opt = _sgd(1.0)
        sched = ExponentialLR(opt, gamma=0.5)
        _step_n(sched, 5)
        assert _lrs(opt) == pytest.approx([0.5 ** 5])

    def test_monotonically_decreasing(self):
        opt = _sgd(1.0)
        sched = ExponentialLR(opt, gamma=0.8)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(_lrs(opt)[0])
        assert all(lrs[i] > lrs[i + 1] for i in range(len(lrs) - 1))

    def test_gamma_1_no_change(self):
        opt = _sgd(0.1)
        sched = ExponentialLR(opt, gamma=1.0)
        _step_n(sched, 10)
        assert _lrs(opt) == pytest.approx([0.1])


# ── MultiStepLR ───────────────────────────────────────────────────────────────

class TestMultiStepLR:
    def test_no_change_before_milestone(self):
        opt = _sgd(0.1)
        sched = MultiStepLR(opt, milestones=[5, 10], gamma=0.1)
        _step_n(sched, 4)
        assert _lrs(opt) == pytest.approx([0.1])

    def test_decays_at_first_milestone(self):
        opt = _sgd(0.1)
        sched = MultiStepLR(opt, milestones=[5, 10], gamma=0.1)
        _step_n(sched, 5)
        assert _lrs(opt) == pytest.approx([0.01])

    def test_decays_at_both_milestones(self):
        opt = _sgd(1.0)
        sched = MultiStepLR(opt, milestones=[3, 7], gamma=0.5)
        _step_n(sched, 7)
        assert _lrs(opt) == pytest.approx([0.25])

    def test_milestones_unsorted(self):
        opt = _sgd(1.0)
        sched = MultiStepLR(opt, milestones=[7, 3], gamma=0.5)  # unsorted
        _step_n(sched, 3)
        assert _lrs(opt) == pytest.approx([0.5])


# ── CosineAnnealingLR ─────────────────────────────────────────────────────────

class TestCosineAnnealingLR:
    def test_starts_at_base_lr(self):
        opt = _sgd(0.1)
        sched = CosineAnnealingLR(opt, T_max=10)
        sched.step()  # epoch 1
        # At epoch 1, should be slightly below 0.1
        assert _lrs(opt)[0] < 0.1

    def test_reaches_eta_min_at_T_max(self):
        opt = _sgd(0.1)
        sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        _step_n(sched, 10)
        assert _lrs(opt)[0] == pytest.approx(0.0, abs=1e-6)

    def test_eta_min_floor(self):
        opt = _sgd(0.1)
        sched = CosineAnnealingLR(opt, T_max=5, eta_min=0.01)
        _step_n(sched, 5)
        assert _lrs(opt)[0] == pytest.approx(0.01, rel=1e-4)

    def test_decreasing_first_half(self):
        opt = _sgd(1.0)
        sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(_lrs(opt)[0])
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


# ── LambdaLR ──────────────────────────────────────────────────────────────────

class TestLambdaLR:
    def test_constant_lambda(self):
        opt = _sgd(0.1)
        sched = LambdaLR(opt, lr_lambda=lambda e: 1.0)
        _step_n(sched, 5)
        assert _lrs(opt) == pytest.approx([0.1])

    def test_halving_lambda(self):
        opt = _sgd(1.0)
        sched = LambdaLR(opt, lr_lambda=lambda e: 0.5 ** e)
        _step_n(sched, 3)
        assert _lrs(opt) == pytest.approx([0.5 ** 3])

    def test_linear_warmup(self):
        opt = _sgd(1.0)
        sched = LambdaLR(opt, lr_lambda=lambda e: min(1.0, e / 5))
        _step_n(sched, 5)
        assert _lrs(opt) == pytest.approx([1.0])
        _step_n(sched, 5)  # stays at 1.0
        assert _lrs(opt) == pytest.approx([1.0])

    def test_per_group_lambdas(self):
        model1 = nn.Linear(4, 2)
        model2 = nn.Linear(4, 2)
        opt = optim.SGD(
            [{"params": model1.parameters(), "lr": 0.1},
             {"params": model2.parameters(), "lr": 0.01}],
            lr=0.1,
        )
        sched = LambdaLR(opt, lr_lambda=[lambda e: 0.5, lambda e: 2.0])
        _step_n(sched, 1)
        lrs = _lrs(opt)
        assert lrs[0] == pytest.approx(0.05)
        assert lrs[1] == pytest.approx(0.02)


# ── CyclicLR ──────────────────────────────────────────────────────────────────

class TestCyclicLR:
    def test_starts_at_base(self):
        opt = _sgd(0.01)
        sched = CyclicLR(opt, base_lr=0.01, max_lr=0.1, step_size_up=4)
        sched.step()
        lr = _lrs(opt)[0]
        assert lr >= 0.01 and lr <= 0.1

    def test_lr_bounded_triangular(self):
        opt = _sgd(0.01)
        sched = CyclicLR(opt, base_lr=0.01, max_lr=0.1, step_size_up=4, mode="triangular")
        for _ in range(20):
            sched.step()
            lr = _lrs(opt)[0]
            assert 0.01 - 1e-6 <= lr <= 0.1 + 1e-6, f"LR {lr} out of bounds"

    def test_triangular2_mode(self):
        opt = _sgd(0.01)
        sched = CyclicLR(opt, base_lr=0.01, max_lr=0.1, step_size_up=4, mode="triangular2")
        for _ in range(20):
            sched.step()
            lr = _lrs(opt)[0]
            assert lr >= 0.01 - 1e-6


# ── ReduceLROnPlateau ─────────────────────────────────────────────────────────

class TestReduceLROnPlateau:
    def test_no_reduction_while_improving(self):
        opt = _sgd(0.1)
        sched = ReduceLROnPlateau(opt, patience=3, factor=0.5)
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            sched.step(loss)
        assert _lrs(opt) == pytest.approx([0.1])

    def test_reduces_after_patience(self):
        opt = _sgd(0.1)
        sched = ReduceLROnPlateau(opt, patience=3, factor=0.5, mode="min")
        sched.step(1.0)  # initial best
        for _ in range(3):  # patience consecutive no-improvement
            sched.step(1.0)
        assert _lrs(opt) == pytest.approx([0.05])

    def test_mode_max(self):
        opt = _sgd(0.1)
        sched = ReduceLROnPlateau(opt, patience=2, factor=0.5, mode="max")
        sched.step(0.5)  # initial best
        sched.step(0.5)
        sched.step(0.5)  # 2 bad epochs → reduce
        assert _lrs(opt) == pytest.approx([0.05])

    def test_min_lr_floor(self):
        opt = _sgd(0.1)
        sched = ReduceLROnPlateau(opt, patience=1, factor=0.01, min_lr=0.001)
        sched.step(1.0)
        for _ in range(10):  # force many reductions
            sched.step(1.0)
        assert _lrs(opt)[0] >= 0.001 - 1e-9

    def test_resets_bad_count_on_improvement(self):
        opt = _sgd(0.1)
        sched = ReduceLROnPlateau(opt, patience=3, factor=0.5)
        sched.step(1.0)
        sched.step(1.0)  # 1 bad
        sched.step(0.8)  # improvement → reset
        sched.step(0.8)  # 1 bad again
        sched.step(0.8)  # 2 bad
        assert _lrs(opt) == pytest.approx([0.1])  # patience=3, not hit yet


# ── NoamScheduler ─────────────────────────────────────────────────────────────

class TestNoamScheduler:
    def test_warmup_phase_increasing(self):
        opt = _sgd(1.0)
        sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(_lrs(opt)[0])
        # During warmup, LR should increase
        assert lrs[0] < lrs[5]

    def test_decay_after_warmup(self):
        opt = _sgd(1.0)
        sched = NoamScheduler(opt, d_model=512, warmup_steps=5)
        _step_n(sched, 10)  # past warmup
        lr_after_10 = _lrs(opt)[0]
        _step_n(sched, 100)  # much later
        lr_after_110 = _lrs(opt)[0]
        assert lr_after_110 < lr_after_10

    def test_lr_positive(self):
        opt = _sgd(1.0)
        sched = NoamScheduler(opt, d_model=256, warmup_steps=100)
        for _ in range(200):
            sched.step()
            assert _lrs(opt)[0] > 0


# ── MultiplicativeLR ──────────────────────────────────────────────────────────

class TestMultiplicativeLR:
    def test_multiplies_each_epoch(self):
        opt = _sgd(1.0)
        sched = MultiplicativeLR(opt, lr_lambda=lambda e: 0.9)
        _step_n(sched, 3)
        assert _lrs(opt) == pytest.approx([0.9 ** 3])

    def test_lambda_1_no_change(self):
        opt = _sgd(0.1)
        sched = MultiplicativeLR(opt, lr_lambda=lambda e: 1.0)
        _step_n(sched, 10)
        assert _lrs(opt) == pytest.approx([0.1])

    def test_epoch_dependent_factor(self):
        opt = _sgd(1.0)
        # Factor that doubles LR each epoch: 2.0
        sched = MultiplicativeLR(opt, lr_lambda=lambda e: 2.0)
        _step_n(sched, 3)
        assert _lrs(opt) == pytest.approx([8.0])


# ── LinearLR ──────────────────────────────────────────────────────────────────

class TestLinearLR:
    def test_starts_at_start_factor(self):
        opt = _sgd(1.0)
        sched = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=10)
        sched.step()
        assert _lrs(opt)[0] > 0.1  # progressed slightly from start

    def test_reaches_end_factor(self):
        opt = _sgd(1.0)
        sched = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5)
        _step_n(sched, 5)
        assert _lrs(opt) == pytest.approx([1.0])

    def test_monotonically_increasing(self):
        opt = _sgd(1.0)
        sched = LinearLR(opt, start_factor=0.0, end_factor=1.0, total_iters=10)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(_lrs(opt)[0])
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_plateau_after_total_iters(self):
        opt = _sgd(1.0)
        sched = LinearLR(opt, start_factor=0.5, end_factor=1.0, total_iters=4)
        _step_n(sched, 4)
        lr_at_4 = _lrs(opt)[0]
        _step_n(sched, 4)
        lr_at_8 = _lrs(opt)[0]
        assert lr_at_4 == pytest.approx(lr_at_8)


# ── ConstantLR ────────────────────────────────────────────────────────────────

class TestConstantLR:
    def test_scaled_during_total_iters(self):
        opt = _sgd(1.0)
        sched = ConstantLR(opt, factor=0.5, total_iters=5)
        _step_n(sched, 3)
        assert _lrs(opt) == pytest.approx([0.5])

    def test_restores_after_total_iters(self):
        opt = _sgd(1.0)
        sched = ConstantLR(opt, factor=0.5, total_iters=5)
        _step_n(sched, 6)
        assert _lrs(opt) == pytest.approx([1.0])

    def test_factor_1_no_change(self):
        opt = _sgd(0.1)
        sched = ConstantLR(opt, factor=1.0, total_iters=3)
        _step_n(sched, 5)
        assert _lrs(opt) == pytest.approx([0.1])


# ── PolynomialLR ──────────────────────────────────────────────────────────────

class TestPolynomialLR:
    def test_reaches_eta_min(self):
        opt = _sgd(1.0)
        sched = PolynomialLR(opt, total_iters=10, power=1.0, eta_min=0.0)
        _step_n(sched, 10)
        assert _lrs(opt) == pytest.approx([0.0])

    def test_eta_min_floor(self):
        opt = _sgd(1.0)
        sched = PolynomialLR(opt, total_iters=5, power=2.0, eta_min=0.1)
        _step_n(sched, 5)
        assert _lrs(opt) == pytest.approx([0.1])

    def test_monotonically_decreasing(self):
        opt = _sgd(1.0)
        sched = PolynomialLR(opt, total_iters=8, power=1.0, eta_min=0.0)
        lrs = []
        for _ in range(8):
            sched.step()
            lrs.append(_lrs(opt)[0])
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_power_2_decays_faster(self):
        opt1 = _sgd(1.0)
        opt2 = _sgd(1.0)
        s1 = PolynomialLR(opt1, total_iters=10, power=1.0, eta_min=0.0)
        s2 = PolynomialLR(opt2, total_iters=10, power=2.0, eta_min=0.0)
        _step_n(s1, 5)
        _step_n(s2, 5)
        assert _lrs(opt2)[0] < _lrs(opt1)[0]


# ── CosineAnnealingWarmRestarts ───────────────────────────────────────────────

class TestCosineAnnealingWarmRestarts:
    def test_restarts_at_T0(self):
        opt = _sgd(1.0)
        sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=1, eta_min=0.0)
        _step_n(sched, 5)  # first cycle complete
        lr_after_restart = _lrs(opt)[0]
        _step_n(sched, 5)  # second cycle complete
        lr_after_second = _lrs(opt)[0]
        # Both should be at/near eta_min at end of cycle
        assert lr_after_restart == pytest.approx(0.0, abs=1e-5)
        assert lr_after_second == pytest.approx(0.0, abs=1e-5)

    def test_lr_bounded(self):
        opt = _sgd(1.0)
        sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=0.01)
        for _ in range(30):
            sched.step()
            lr = _lrs(opt)[0]
            assert 0.01 - 1e-6 <= lr <= 1.0 + 1e-6

    def test_T_mult_extends_cycles(self):
        opt = _sgd(1.0)
        sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=0.0)
        # After T_0=5 steps, T_i becomes 5*2=10 for the next cycle
        _step_n(sched, 5)
        assert sched._T_i == 10


# ── OneCycleLR ────────────────────────────────────────────────────────────────

class TestOneCycleLR:
    def test_reaches_max_lr_at_warmup(self):
        opt = _sgd(0.01)
        total = 10
        sched = OneCycleLR(opt, max_lr=0.1, total_steps=total,
                           pct_start=0.3, anneal_strategy="cos")
        warmup = int(total * 0.3)
        _step_n(sched, warmup)
        # At end of warmup should be near max_lr
        assert _lrs(opt)[0] == pytest.approx(0.1, rel=0.1)

    def test_lr_bounded(self):
        opt = _sgd(0.01)
        sched = OneCycleLR(opt, max_lr=0.1, total_steps=20,
                           pct_start=0.3, anneal_strategy="cos")
        for _ in range(20):
            sched.step()
            lr = _lrs(opt)[0]
            assert lr > 0, f"LR became non-positive: {lr}"

    def test_linear_strategy(self):
        opt = _sgd(0.01)
        sched = OneCycleLR(opt, max_lr=0.1, total_steps=20,
                           pct_start=0.3, anneal_strategy="linear")
        _step_n(sched, 20)
        # Should have gone through full cycle without error
        assert _lrs(opt)[0] > 0


# ── SequentialLR ──────────────────────────────────────────────────────────────

class TestSequentialLR:
    def test_switches_at_milestone(self):
        opt = _sgd(1.0)
        s1 = ConstantLR(opt, factor=0.5, total_iters=100)
        s2 = ExponentialLR(opt, gamma=0.9)
        sched = SequentialLR(opt, schedulers=[s1, s2], milestones=[5])

        # First 4 steps: ConstantLR (factor=0.5 → lr=0.5)
        _step_n(sched, 4)
        assert _lrs(opt)[0] == pytest.approx(0.5)

        # After milestone 5: ExponentialLR takes over
        _step_n(sched, 3)
        assert _lrs(opt)[0] < 0.5  # ExponentialLR decays

    def test_get_last_lr(self):
        opt = _sgd(1.0)
        s1 = StepLR(opt, step_size=5, gamma=0.5)
        s2 = ExponentialLR(opt, gamma=0.8)
        sched = SequentialLR(opt, schedulers=[s1, s2], milestones=[5])
        _step_n(sched, 3)
        assert sched.get_last_lr() is not None

    def test_multiple_milestones(self):
        opt = _sgd(1.0)
        s1 = ConstantLR(opt, factor=1.0, total_iters=100)
        s2 = ConstantLR(opt, factor=0.5, total_iters=100)
        s3 = ConstantLR(opt, factor=0.1, total_iters=100)
        sched = SequentialLR(opt, schedulers=[s1, s2, s3], milestones=[3, 6])
        _step_n(sched, 7)
        assert _lrs(opt)[0] == pytest.approx(0.1)


# ── ChainedScheduler ──────────────────────────────────────────────────────────

class TestChainedScheduler:
    def test_both_schedulers_applied(self):
        opt = _sgd(1.0)
        s1 = ExponentialLR(opt, gamma=0.9)
        s2 = ExponentialLR(opt, gamma=0.8)
        sched = ChainedScheduler([s1, s2])
        sched.step()
        # Both applied: 1.0 * 0.9 * 0.8 = 0.72
        assert _lrs(opt)[0] == pytest.approx(0.9 * 0.8, rel=1e-3)

    def test_get_last_lr(self):
        opt = _sgd(0.1)
        s1 = StepLR(opt, step_size=3, gamma=0.5)
        s2 = ExponentialLR(opt, gamma=0.9)
        sched = ChainedScheduler([s1, s2])
        _step_n(sched, 5)
        assert sched.get_last_lr() is not None

    def test_step_count(self):
        opt = _sgd(1.0)
        s1 = ExponentialLR(opt, gamma=0.9)
        s2 = ExponentialLR(opt, gamma=0.9)
        sched = ChainedScheduler([s1, s2])
        _step_n(sched, 5)
        assert s1._step_count == 5
        assert s2._step_count == 5


# ── Multi-param-group correctness ─────────────────────────────────────────────

class TestMultiParamGroup:
    def _two_group_sgd(self) -> optim.SGD:
        m1, m2 = nn.Linear(4, 2), nn.Linear(4, 2)
        return optim.SGD(
            [{"params": m1.parameters(), "lr": 0.1},
             {"params": m2.parameters(), "lr": 0.01}],
            lr=0.1,
        )

    def test_step_lr_both_groups(self):
        opt = self._two_group_sgd()
        sched = StepLR(opt, step_size=1, gamma=0.5)
        sched.step()
        lrs = _lrs(opt)
        assert lrs[0] == pytest.approx(0.05)
        assert lrs[1] == pytest.approx(0.005)

    def test_exponential_lr_both_groups(self):
        opt = self._two_group_sgd()
        sched = ExponentialLR(opt, gamma=0.9)
        _step_n(sched, 2)
        lrs = _lrs(opt)
        assert lrs[0] == pytest.approx(0.1 * 0.9 ** 2, rel=1e-4)
        assert lrs[1] == pytest.approx(0.01 * 0.9 ** 2, rel=1e-4)

    def test_cosine_annealing_both_groups(self):
        opt = self._two_group_sgd()
        sched = CosineAnnealingLR(opt, T_max=5, eta_min=0.0)
        _step_n(sched, 5)
        lrs = _lrs(opt)
        assert lrs[0] == pytest.approx(0.0, abs=1e-6)
        assert lrs[1] == pytest.approx(0.0, abs=1e-6)
