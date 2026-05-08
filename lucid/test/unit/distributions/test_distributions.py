"""``lucid.distributions`` — full surface coverage.

Tests are CPU-only (the gamma sampler iteratively retries with engine
ops in a tight loop and is significantly heavier on Metal — pinning
to CPU keeps the test tier fast and deterministic).  Distribution
correctness is the same on either device.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.distributions as D

# ── continuous univariate ───────────────────────────────────────────────


class TestNormal:
    def test_log_prob_at_zero(self) -> None:
        v = D.Normal(0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v - (-0.5 * math.log(2.0 * math.pi))) < 1e-5

    def test_entropy(self) -> None:
        v = D.Normal(0.0, 1.0).entropy().item()
        assert abs(v - (0.5 + 0.5 * math.log(2.0 * math.pi))) < 1e-5

    def test_rsample_grad_flows(self) -> None:
        loc = lucid.tensor([0.0], requires_grad=True)
        scale = lucid.tensor([1.0], requires_grad=True)
        s = D.Normal(loc, scale).rsample((4,))
        assert s.requires_grad

    def test_cdf_icdf_roundtrip(self) -> None:
        n = D.Normal(0.0, 1.0)
        u = lucid.tensor([0.25, 0.5, 0.75])
        np.testing.assert_allclose(n.cdf(n.icdf(u)).numpy(), u.numpy(), atol=1e-4)


class TestLogNormalUniform:
    def test_lognormal_positive(self) -> None:
        s = D.LogNormal(0.0, 1.0).sample((20,)).numpy()
        assert (s > 0).all()

    def test_uniform_in_range(self) -> None:
        s = D.Uniform(2.0, 5.0).sample((50,)).numpy()
        assert (s >= 2.0).all() and (s < 5.0).all()


class TestExponentialLaplaceCauchy:
    def test_exp_log_prob_at_one(self) -> None:
        v = D.Exponential(1.0).log_prob(lucid.tensor(1.0)).item()
        assert abs(v + 1.0) < 1e-5

    def test_laplace_at_loc(self) -> None:
        v = D.Laplace(0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v + math.log(2.0)) < 1e-5

    def test_cauchy_at_loc(self) -> None:
        v = D.Cauchy(0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v + math.log(math.pi)) < 1e-5


class TestStudentT:
    def test_log_prob(self) -> None:
        # ψ(0 | df=5, loc=0, scale=1) = lgamma(3) − lgamma(2.5) − 0.5·log(5π).
        expected = math.lgamma(3.0) - math.lgamma(2.5) - 0.5 * math.log(5.0 * math.pi)
        v = D.StudentT(5.0, 0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v - expected) < 1e-4


class TestParetoWeibullHalfFisher:
    def test_pareto_min(self) -> None:
        s = D.Pareto(1.5, 3.0).sample((20,)).numpy()
        assert (s >= 1.5).all()

    def test_weibull_nonneg(self) -> None:
        s = D.Weibull(2.0, 1.5).sample((20,)).numpy()
        assert (s >= 0).all()

    def test_halfnormal_nonneg(self) -> None:
        s = D.HalfNormal(1.0).sample((50,)).numpy()
        assert (s >= 0).all()

    def test_halfnormal_mean(self) -> None:
        # E[|X|] = σ·√(2/π).
        v = D.HalfNormal(1.0).mean.item()
        assert abs(v - math.sqrt(2.0 / math.pi)) < 1e-5

    def test_halfcauchy_nonneg(self) -> None:
        s = D.HalfCauchy(1.0).sample((20,)).numpy()
        assert (s >= 0).all()

    def test_fisher_mean(self) -> None:
        # μ = d2/(d2−2) for d2>2.
        v = D.FisherSnedecor(5.0, 10.0).mean.item()
        assert abs(v - 1.25) < 1e-5


# ── discrete ────────────────────────────────────────────────────────────


class TestBernoulliGeometric:
    def test_bern_binary(self) -> None:
        s = D.Bernoulli(probs=0.5).sample((100,)).numpy()
        assert set(s.tolist()) <= {0.0, 1.0}

    def test_bern_log_prob(self) -> None:
        v = D.Bernoulli(probs=0.3).log_prob(lucid.tensor(1.0)).item()
        assert abs(v - math.log(0.3)) < 1e-5


class TestCategoricalOneHot:
    def test_log_prob(self) -> None:
        c = D.Categorical(probs=lucid.tensor([0.2, 0.5, 0.3]))
        assert abs(c.log_prob(lucid.tensor(1)).item() - math.log(0.5)) < 1e-5

    def test_one_hot(self) -> None:
        s = D.OneHotCategorical(probs=lucid.tensor([0.2, 0.5, 0.3])).sample().numpy()
        assert s.sum() == 1.0


class TestPoissonBinomial:
    def test_poisson_log_prob(self) -> None:
        # log p(2 | 3) = 2·log(3) − 3 − log(2).
        expected = 2.0 * math.log(3.0) - 3.0 - math.log(2.0)
        v = D.Poisson(3.0).log_prob(lucid.tensor(2.0)).item()
        assert abs(v - expected) < 1e-4

    def test_binomial_mean(self) -> None:
        v = D.Binomial(10, probs=0.3).mean.item()
        assert abs(v - 3.0) < 1e-6

    def test_negbin_mean(self) -> None:
        v = D.NegativeBinomial(5.0, probs=0.5).mean.item()
        assert abs(v - 5.0) < 1e-6


# ── multivariate / wrappers ─────────────────────────────────────────────


class TestMultivariateNormal:
    def test_log_prob_at_origin(self) -> None:
        mvn = D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2))
        v = mvn.log_prob(lucid.zeros(2)).item()
        assert abs(v - (-math.log(2.0 * math.pi))) < 1e-4

    def test_entropy_unit(self) -> None:
        mvn = D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2))
        assert abs(mvn.entropy().item() - (1.0 + math.log(2.0 * math.pi))) < 1e-4


class TestIndependent:
    def test_log_prob_summed(self) -> None:
        base = D.Normal(lucid.tensor([0.0, 0.0]), lucid.tensor([1.0, 1.0]))
        ind = D.Independent(base, 1)
        v = lucid.tensor([0.0, 0.0])
        expected = 2.0 * base.log_prob(v).numpy()[0]
        assert abs(ind.log_prob(v).item() - expected) < 1e-5


class TestMixtureSameFamily:
    def test_symmetric_mean_zero(self) -> None:
        m = D.MixtureSameFamily(
            D.Categorical(probs=lucid.tensor([0.5, 0.5])),
            D.Normal(lucid.tensor([-2.0, 2.0]), lucid.tensor([0.5, 0.5])),
        )
        assert abs(m.mean.item()) < 1e-6


class TestRelaxed:
    def test_relaxed_bern_in_unit(self) -> None:
        s = D.RelaxedBernoulli(temperature=0.5, probs=0.5).rsample((20,)).numpy()
        assert (s > 0).all() and (s < 1).all()

    def test_relaxed_ohc_simplex(self) -> None:
        s = (
            D.RelaxedOneHotCategorical(0.5, probs=lucid.tensor([0.2, 0.5, 0.3]))
            .rsample()
            .numpy()
        )
        assert abs(s.sum() - 1.0) < 1e-5


# ── transforms ──────────────────────────────────────────────────────────


class TestTransforms:
    def test_exp_round_trip(self) -> None:
        x = lucid.tensor(1.5)
        e = D.ExpTransform()
        np.testing.assert_allclose(e.inv(e(x)).numpy(), x.numpy(), atol=1e-5)

    def test_normal_through_exp_eq_lognormal(self) -> None:
        n = D.Normal(0.0, 1.0)
        td = D.TransformedDistribution(n, D.ExpTransform())
        ref = D.LogNormal(0.0, 1.0)
        for v in (0.5, 1.0, 2.0, 3.0):
            assert (
                abs(
                    td.log_prob(lucid.tensor(v)).item()
                    - ref.log_prob(lucid.tensor(v)).item()
                )
                < 1e-4
            )


# ── KL pairs ────────────────────────────────────────────────────────────


class TestKL:
    def test_kl_normal_known(self) -> None:
        # KL(N(0,1) || N(1,2)) = log(2) + (1+1)/(2·4) − 0.5.
        kl = D.kl_divergence(D.Normal(0.0, 1.0), D.Normal(1.0, 2.0)).item()
        expected = math.log(2.0) + 0.25 - 0.5
        assert abs(kl - expected) < 1e-5

    def test_kl_self_zero_normal(self) -> None:
        kl = D.kl_divergence(D.Normal(0.0, 1.0), D.Normal(0.0, 1.0)).item()
        assert abs(kl) < 1e-5

    def test_kl_poisson_known(self) -> None:
        expected = 3.0 * math.log(3.0 / 4.0) + 1.0
        v = D.kl_divergence(D.Poisson(3.0), D.Poisson(4.0)).item()
        assert abs(v - expected) < 1e-4

    def test_kl_mvn_translation(self) -> None:
        kl = D.kl_divergence(
            D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2)),
            D.MultivariateNormal(lucid.tensor([1.0, 0.0]), scale_tril=lucid.eye(2)),
        ).item()
        assert abs(kl - 0.5) < 1e-4
