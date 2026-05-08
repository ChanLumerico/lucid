"""Unit tests for the P8-C distribution additions:
``Poisson`` / ``Binomial`` / ``NegativeBinomial`` / ``Pareto`` /
``Weibull`` / ``HalfNormal`` / ``HalfCauchy`` / ``FisherSnedecor`` /
``RelaxedBernoulli`` / ``RelaxedOneHotCategorical`` /
``MixtureSameFamily``, the four new transforms, and the four new KL
pairs.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.distributions as D


# ── discrete additions ───────────────────────────────────────────────────


class TestPoisson:
    def test_mean_eq_rate(self) -> None:
        p = D.Poisson(3.0)
        assert abs(p.mean.item() - 3.0) < 1e-6

    def test_log_prob_known(self) -> None:
        # log p(2 | 3) = 2·log(3) − 3 − lgamma(3) = 2·log(3) − 3 − log(2).
        expected = 2.0 * math.log(3.0) - 3.0 - math.log(2.0)
        assert abs(
            D.Poisson(3.0).log_prob(lucid.tensor(2.0)).item() - expected
        ) < 1e-4

    def test_sample_nonnegative_int(self) -> None:
        s = D.Poisson(5.0).sample((50,)).numpy()
        assert (s >= 0).all()
        assert np.equal(np.floor(s), s).all()


class TestBinomial:
    def test_mean(self) -> None:
        assert abs(D.Binomial(10, probs=0.3).mean.item() - 3.0) < 1e-6

    def test_sample_in_range(self) -> None:
        s = D.Binomial(10, probs=0.5).sample((50,)).numpy()
        assert (s >= 0).all() and (s <= 10).all()

    def test_log_prob_known(self) -> None:
        # log p(0 | 10, 0.3) = 10·log(0.7).
        v = D.Binomial(10, probs=0.3).log_prob(lucid.tensor(0.0)).item()
        assert abs(v - 10.0 * math.log(0.7)) < 1e-4

    def test_normal_approx_path(self) -> None:
        # n above the cutoff → Normal-approx; mean should still match.
        s = D.Binomial(1000, probs=0.5).sample((50,)).numpy()
        assert abs(s.mean() - 500.0) < 30.0  # ~3σ.


class TestNegativeBinomial:
    def test_mean_known(self) -> None:
        # E[X] = r·p / (1−p).  r=5, p=0.5 ⇒ 5.
        assert abs(D.NegativeBinomial(5.0, probs=0.5).mean.item() - 5.0) < 1e-6

    def test_sample_nonnegative(self) -> None:
        s = D.NegativeBinomial(5.0, probs=0.5).sample((20,)).numpy()
        assert (s >= 0).all()


# ── continuous additions ────────────────────────────────────────────────


class TestPareto:
    def test_sample_above_scale(self) -> None:
        s = D.Pareto(1.5, 3.0).sample((20,)).numpy()
        assert (s >= 1.5).all()

    def test_mean_known(self) -> None:
        # μ = α·scale/(α−1) for α>1.  α=3, scale=1 ⇒ 1.5.
        assert abs(D.Pareto(1.0, 3.0).mean.item() - 1.5) < 1e-5


class TestWeibull:
    def test_sample_nonneg(self) -> None:
        assert (D.Weibull(2.0, 1.5).sample((20,)).numpy() >= 0).all()


class TestHalfNormal:
    def test_sample_nonneg(self) -> None:
        assert (D.HalfNormal(1.0).sample((50,)).numpy() >= 0).all()

    def test_mean_known(self) -> None:
        # E[|X|] = σ·√(2/π).
        assert abs(
            D.HalfNormal(1.0).mean.item() - math.sqrt(2.0 / math.pi)
        ) < 1e-5


class TestHalfCauchy:
    def test_sample_nonneg(self) -> None:
        assert (D.HalfCauchy(1.0).sample((20,)).numpy() >= 0).all()


class TestFisherSnedecor:
    def test_sample_positive(self) -> None:
        s = D.FisherSnedecor(5.0, 10.0).sample((20,)).numpy()
        assert (s > 0).all()

    def test_mean_known(self) -> None:
        # μ = d2/(d2−2) for d2>2.  d2=10 ⇒ 1.25.
        assert abs(D.FisherSnedecor(5.0, 10.0).mean.item() - 1.25) < 1e-5


# ── relaxed (Concrete) ────────────────────────────────────────────────────


class TestRelaxedBernoulli:
    def test_sample_in_open_unit(self) -> None:
        s = D.RelaxedBernoulli(temperature=0.5, probs=0.5).rsample((20,)).numpy()
        assert (s > 0).all() and (s < 1).all()

    def test_low_temperature_sharpens(self) -> None:
        # Very low τ → samples close to 0/1.
        s = D.RelaxedBernoulli(temperature=0.05, probs=0.7).rsample((100,)).numpy()
        assert ((s < 0.1) | (s > 0.9)).mean() > 0.6  # most samples are extreme


class TestRelaxedOneHotCategorical:
    def test_sample_simplex(self) -> None:
        roc = D.RelaxedOneHotCategorical(0.5, probs=lucid.tensor([0.2, 0.5, 0.3]))
        s = roc.rsample().numpy()
        assert abs(s.sum() - 1.0) < 1e-5
        assert (s > 0).all()


# ── MixtureSameFamily ────────────────────────────────────────────────────


class TestMixtureSameFamily:
    def _mix(self) -> D.MixtureSameFamily:
        return D.MixtureSameFamily(
            D.Categorical(probs=lucid.tensor([0.5, 0.5])),
            D.Normal(lucid.tensor([-2.0, 2.0]), lucid.tensor([0.5, 0.5])),
        )

    def test_mean_zero_for_symmetric_mixture(self) -> None:
        assert abs(self._mix().mean.item()) < 1e-6

    def test_log_prob_reasonable(self) -> None:
        # Log-prob at x=0 should be lower than at the modes ±2.
        m = self._mix()
        lp_0 = m.log_prob(lucid.tensor(0.0)).item()
        lp_2 = m.log_prob(lucid.tensor(2.0)).item()
        assert lp_2 > lp_0


# ── new transforms ───────────────────────────────────────────────────────


class TestNewTransforms:
    def test_power_inverse_roundtrip(self) -> None:
        p = D.PowerTransform(2.0)
        x = lucid.tensor(3.0)
        np.testing.assert_allclose(p.inv(p(x)).numpy(), x.numpy(), atol=1e-5)

    def test_softmax_sums_to_one(self) -> None:
        out = D.SoftmaxTransform()(lucid.tensor([1.0, 2.0, 3.0])).numpy()
        assert abs(out.sum() - 1.0) < 1e-5

    def test_lower_cholesky_diag_positive(self) -> None:
        x = lucid.tensor([[0.5, 0.0], [1.5, 0.3]])
        y = D.LowerCholeskyTransform()(x).numpy()
        # Diagonal entries are softplus-positive.
        assert y[0, 0] > 0 and y[1, 1] > 0
        # Upper triangle is zero.
        assert y[0, 1] == 0


# ── new KL pairs ─────────────────────────────────────────────────────────


class TestExtendedKL:
    def test_kl_poisson_self_zero(self) -> None:
        v = D.kl_divergence(D.Poisson(3.0), D.Poisson(3.0)).item()
        assert abs(v) < 1e-5

    def test_kl_poisson_known(self) -> None:
        # KL(Pois(3) || Pois(4)) = 3·log(3/4) + 1.
        expected = 3.0 * math.log(3.0 / 4.0) + 1.0
        v = D.kl_divergence(D.Poisson(3.0), D.Poisson(4.0)).item()
        assert abs(v - expected) < 1e-4

    def test_kl_dirichlet_self_zero(self) -> None:
        d = D.Dirichlet(lucid.tensor([1.0, 2.0, 3.0]))
        v = D.kl_divergence(d, d).item()
        assert abs(v) < 1e-3

    def test_kl_mvn_translation(self) -> None:
        # KL(N(0, I) || N(μ, I)) = 0.5 · ‖μ‖².
        kl = D.kl_divergence(
            D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2)),
            D.MultivariateNormal(lucid.tensor([1.0, 0.0]), scale_tril=lucid.eye(2)),
        ).item()
        assert abs(kl - 0.5) < 1e-4

    def test_kl_mvn_self_zero(self) -> None:
        kl = D.kl_divergence(
            D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2)),
            D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2)),
        ).item()
        assert abs(kl) < 1e-5


# ── public surface ───────────────────────────────────────────────────────


class TestSurface:
    def test_all_p8c_visible(self) -> None:
        for name in (
            # discrete
            "Poisson", "Binomial", "NegativeBinomial",
            # continuous
            "Pareto", "Weibull", "HalfNormal", "HalfCauchy", "FisherSnedecor",
            # relaxed
            "RelaxedBernoulli", "RelaxedOneHotCategorical",
            # mixture
            "MixtureSameFamily",
            # transforms
            "PowerTransform", "SoftmaxTransform",
            "StickBreakingTransform", "LowerCholeskyTransform",
        ):
            assert hasattr(D, name), f"D.{name} missing"
