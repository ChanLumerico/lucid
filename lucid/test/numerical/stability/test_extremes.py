"""Extreme-value handling — inf / nan / subnormal / overflow.

A correct framework propagates inf/nan deterministically and never
silently turns a poison value into a finite one.  These tests pin that
contract on every device.
"""

import numpy as np
import pytest

import lucid


@pytest.mark.stability
class TestNaNPropagation:
    def test_add_nan(self, device: str) -> None:
        a = lucid.tensor([1.0, float("nan"), 3.0], device=device)
        b = lucid.tensor([1.0, 1.0, 1.0], device=device)
        out = (a + b).numpy()
        assert out[0] == 2.0
        assert np.isnan(out[1])
        assert out[2] == 4.0

    def test_mul_nan(self, device: str) -> None:
        a = lucid.tensor([float("nan")], device=device)
        b = lucid.tensor([0.0], device=device)
        # 0 * nan = nan (IEEE-754).
        out = (a * b).numpy()
        assert np.isnan(out[0])

    def test_reduction_nan(self, device: str) -> None:
        a = lucid.tensor([1.0, float("nan"), 3.0], device=device)
        # sum with a nan inside is nan.
        assert np.isnan(lucid.sum(a).item())

    def test_max_with_nan_safe_via_nansum(self, device: str) -> None:
        # ``nansum`` skips nans by contract.
        a = lucid.tensor([1.0, float("nan"), 3.0], device=device)
        assert lucid.nansum(a).item() == 4.0


@pytest.mark.stability
class TestInfPropagation:
    def test_inf_minus_inf(self, device: str) -> None:
        a = lucid.tensor([float("inf")], device=device)
        b = lucid.tensor([float("inf")], device=device)
        out = (a - b).numpy()
        assert np.isnan(out[0])

    def test_inf_div_inf(self, device: str) -> None:
        a = lucid.tensor([float("inf")], device=device)
        b = lucid.tensor([float("inf")], device=device)
        out = (a / b).numpy()
        assert np.isnan(out[0])

    def test_finite_div_inf(self, device: str) -> None:
        a = lucid.tensor([1.0], device=device)
        b = lucid.tensor([float("inf")], device=device)
        assert (a / b).item() == 0.0

    def test_inf_times_zero(self, device: str) -> None:
        a = lucid.tensor([float("inf")], device=device)
        b = lucid.tensor([0.0], device=device)
        out = (a * b).numpy()
        assert np.isnan(out[0])


@pytest.mark.stability
class TestExpOverflow:
    def test_exp_large_positive(self, device: str) -> None:
        # exp(large) → +inf in f32; must not become a finite junk value.
        a = lucid.tensor([100.0], device=device)
        out = a.exp().numpy()
        assert np.isinf(out[0]) and out[0] > 0

    def test_exp_large_negative(self, device: str) -> None:
        a = lucid.tensor([-100.0], device=device)
        out = a.exp().numpy()
        # exp(-100) ≈ 3.7e-44 — either flush-to-zero or a subnormal is
        # acceptable; what's *not* acceptable is producing a finite
        # value of any meaningful magnitude.
        assert 0.0 <= out[0] < 1e-40


@pytest.mark.stability
class TestLogDomain:
    def test_log_zero(self, device: str) -> None:
        a = lucid.tensor([0.0], device=device)
        out = a.log().numpy()
        # log(0) = -inf.
        assert np.isinf(out[0]) and out[0] < 0

    def test_log_negative(self, device: str) -> None:
        a = lucid.tensor([-1.0], device=device)
        out = a.log().numpy()
        # log(neg) = nan.
        assert np.isnan(out[0])


@pytest.mark.stability
class TestSqrtDomain:
    def test_sqrt_zero(self, device: str) -> None:
        a = lucid.tensor([0.0], device=device)
        assert a.sqrt().item() == 0.0

    def test_sqrt_negative(self, device: str) -> None:
        a = lucid.tensor([-1.0], device=device)
        assert np.isnan(a.sqrt().item())


@pytest.mark.stability
class TestDivByZero:
    def test_finite_div_zero(self, device: str) -> None:
        a = lucid.tensor([1.0], device=device)
        b = lucid.tensor([0.0], device=device)
        out = (a / b).numpy()
        # 1/0 = +inf in IEEE.
        assert np.isinf(out[0])

    def test_zero_div_zero(self, device: str) -> None:
        a = lucid.tensor([0.0], device=device)
        b = lucid.tensor([0.0], device=device)
        out = (a / b).numpy()
        # 0/0 = nan.
        assert np.isnan(out[0])


@pytest.mark.stability
class TestSoftmaxStability:
    def test_softmax_large_logits(self, device: str) -> None:
        # softmax must subtract max for stability — feed huge values.
        from lucid.nn.functional import softmax
        x = lucid.tensor([1000.0, 1001.0, 1002.0], device=device)
        out = softmax(x, dim=0).numpy()
        # Output must sum to 1 and contain no nans.
        assert not np.any(np.isnan(out))
        assert abs(out.sum() - 1.0) < 1e-5

    def test_log_softmax_large_logits(self, device: str) -> None:
        from lucid.nn.functional import log_softmax
        x = lucid.tensor([1000.0, 1001.0, 1002.0], device=device)
        out = log_softmax(x, dim=0).numpy()
        # No nans, no infs.
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


@pytest.mark.stability
class TestNanToNum:
    def test_replace_nan(self, device: str) -> None:
        a = lucid.tensor([1.0, float("nan"), 2.0], device=device)
        out = lucid.nan_to_num(a, nan=0.0).numpy()
        np.testing.assert_array_equal(out, [1.0, 0.0, 2.0])

    def test_replace_posinf(self, device: str) -> None:
        a = lucid.tensor([1.0, float("inf"), 2.0], device=device)
        out = lucid.nan_to_num(a, posinf=999.0).numpy()
        assert out[1] == 999.0

    def test_replace_neginf(self, device: str) -> None:
        a = lucid.tensor([1.0, float("-inf"), 2.0], device=device)
        out = lucid.nan_to_num(a, neginf=-999.0).numpy()
        assert out[1] == -999.0
