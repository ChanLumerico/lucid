"""Unit tests for ``lucid.distributions`` (P8 — first wave)."""

import math

import numpy as np
import pytest

import lucid
import lucid.distributions as D


# ── Normal ────────────────────────────────────────────────────────────────


class TestNormal:
    def test_log_prob_at_zero(self) -> None:
        n = D.Normal(0.0, 1.0)
        assert abs(n.log_prob(lucid.tensor(0.0)).item() + 0.5 * math.log(2.0 * math.pi)) < 1e-5

    def test_entropy(self) -> None:
        n = D.Normal(0.0, 1.0)
        expected = 0.5 + 0.5 * math.log(2.0 * math.pi)
        assert abs(n.entropy().item() - expected) < 1e-5

    def test_rsample_shape(self) -> None:
        n = D.Normal(lucid.tensor([0.0, 1.0]), lucid.tensor([1.0, 2.0]))
        assert n.rsample((5,)).shape == (5, 2)

    def test_rsample_grad_flows(self) -> None:
        loc = lucid.tensor([0.0], requires_grad=True)
        scale = lucid.tensor([1.0], requires_grad=True)
        sample = D.Normal(loc, scale).rsample((4,))
        assert sample.requires_grad

    def test_cdf_icdf_roundtrip(self) -> None:
        n = D.Normal(0.0, 1.0)
        u = lucid.tensor([0.25, 0.5, 0.75])
        np.testing.assert_allclose(n.cdf(n.icdf(u)).numpy(), u.numpy(), atol=1e-4)


class TestLogNormal:
    def test_sample_positive(self) -> None:
        ln = D.LogNormal(0.0, 1.0)
        s = ln.sample((20,)).numpy()
        assert (s > 0).all()


# ── Uniform ───────────────────────────────────────────────────────────────


class TestUniform:
    def test_sample_in_range(self) -> None:
        u = D.Uniform(2.0, 5.0)
        s = u.sample((50,)).numpy()
        assert (s >= 2.0).all() and (s < 5.0).all()

    def test_log_prob(self) -> None:
        u = D.Uniform(0.0, 2.0)
        assert abs(u.log_prob(lucid.tensor(1.0)).item() - math.log(0.5)) < 1e-5

    def test_entropy(self) -> None:
        u = D.Uniform(0.0, math.e)  # log(width) = 1.
        assert abs(u.entropy().item() - 1.0) < 1e-5


# ── Exponential / Laplace / Cauchy ────────────────────────────────────────


class TestExponential:
    def test_mean(self) -> None:
        assert abs(D.Exponential(2.0).mean.item() - 0.5) < 1e-6

    def test_log_prob(self) -> None:
        e = D.Exponential(1.0)
        # log p(1) = log(1) - 1 = -1.
        assert abs(e.log_prob(lucid.tensor(1.0)).item() + 1.0) < 1e-5

    def test_cdf_icdf_roundtrip(self) -> None:
        e = D.Exponential(1.5)
        u = lucid.tensor([0.1, 0.5, 0.9])
        np.testing.assert_allclose(e.cdf(e.icdf(u)).numpy(), u.numpy(), atol=1e-5)


class TestLaplace:
    def test_log_prob_at_loc(self) -> None:
        # log p(loc | loc, scale) = -log(2 · scale).
        l = D.Laplace(0.0, 1.0)
        assert abs(l.log_prob(lucid.tensor(0.0)).item() + math.log(2.0)) < 1e-5


class TestCauchy:
    def test_log_prob_at_loc(self) -> None:
        # log p(loc | loc, scale) = -log(π · scale).
        c = D.Cauchy(0.0, 1.0)
        assert abs(c.log_prob(lucid.tensor(0.0)).item() + math.log(math.pi)) < 1e-5


# ── Bernoulli / Geometric ─────────────────────────────────────────────────


class TestBernoulli:
    def test_sample_binary(self) -> None:
        b = D.Bernoulli(probs=0.7)
        s = b.sample((100,)).numpy()
        assert set(s.flatten().tolist()) <= {0.0, 1.0}

    def test_log_prob(self) -> None:
        b = D.Bernoulli(probs=0.3)
        # log p(1) = log 0.3, log p(0) = log 0.7.
        assert abs(b.log_prob(lucid.tensor(1.0)).item() - math.log(0.3)) < 1e-5
        assert abs(b.log_prob(lucid.tensor(0.0)).item() - math.log(0.7)) < 1e-5


class TestGeometric:
    def test_sample_nonnegative_integer(self) -> None:
        g = D.Geometric(probs=0.5)
        s = g.sample((20,)).numpy()
        assert (s >= 0).all()
        assert np.all(np.equal(np.floor(s), s))


# ── Categorical / OneHotCategorical ───────────────────────────────────────


class TestCategorical:
    def test_log_prob(self) -> None:
        c = D.Categorical(probs=lucid.tensor([0.2, 0.5, 0.3]))
        assert abs(c.log_prob(lucid.tensor(1)).item() - math.log(0.5)) < 1e-5

    def test_entropy_uniform(self) -> None:
        c = D.Categorical(probs=lucid.tensor([0.25, 0.25, 0.25, 0.25]))
        assert abs(c.entropy().item() - math.log(4.0)) < 1e-5

    def test_sample_in_range(self) -> None:
        c = D.Categorical(probs=lucid.tensor([0.5, 0.5]))
        for _ in range(10):
            v = c.sample().item()
            assert v in (0, 1)


class TestOneHotCategorical:
    def test_sample_one_hot(self) -> None:
        c = D.OneHotCategorical(probs=lucid.tensor([0.5, 0.3, 0.2]))
        s = c.sample().numpy()
        assert s.sum() == 1.0


# ── Gamma family ──────────────────────────────────────────────────────────


class TestGamma:
    def test_mean_variance(self) -> None:
        g = D.Gamma(2.0, 1.0)
        assert abs(g.mean.item() - 2.0) < 1e-6
        assert abs(g.variance.item() - 2.0) < 1e-6

    def test_log_prob(self) -> None:
        # log p(1 | α=2, β=1) = 0 + 0 - 1 - lgamma(2) = -1.
        g = D.Gamma(2.0, 1.0)
        assert abs(g.log_prob(lucid.tensor(1.0)).item() + 1.0) < 1e-4

    def test_sample_positive(self) -> None:
        g = D.Gamma(2.0, 1.0)
        s = g.sample((20,)).numpy()
        assert (s > 0).all()


class TestChi2:
    def test_mean_eq_df(self) -> None:
        # Chi2(df).mean = df.
        c = D.Chi2(4.0)
        assert abs(c.mean.item() - 4.0) < 1e-5


class TestBeta:
    def test_mean(self) -> None:
        b = D.Beta(2.0, 3.0)
        assert abs(b.mean.item() - 0.4) < 1e-5

    def test_sample_in_unit_interval(self) -> None:
        b = D.Beta(2.0, 3.0)
        s = b.sample((20,)).numpy()
        assert (s >= 0).all() and (s <= 1).all()


class TestDirichlet:
    def test_mean_normalized(self) -> None:
        d = D.Dirichlet(lucid.tensor([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            d.mean.numpy(), [1 / 6, 2 / 6, 3 / 6], atol=1e-5
        )

    def test_sample_simplex(self) -> None:
        d = D.Dirichlet(lucid.tensor([1.0, 2.0, 3.0]))
        s = d.sample().numpy()
        assert abs(s.sum() - 1.0) < 1e-5
        assert (s >= 0).all()


# ── MultivariateNormal ────────────────────────────────────────────────────


class TestMultivariateNormal:
    def _mvn(self) -> D.MultivariateNormal:
        return D.MultivariateNormal(
            lucid.zeros(2),
            scale_tril=lucid.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

    def test_log_prob_at_origin(self) -> None:
        # log p(0 | 0, I) = -D/2 · log 2π = -log 2π.
        assert abs(
            self._mvn().log_prob(lucid.zeros(2)).item() + math.log(2.0 * math.pi)
        ) < 1e-4

    def test_entropy(self) -> None:
        # H = D/2 · (1 + log 2π) for unit covariance.
        expected = 1.0 + math.log(2.0 * math.pi)
        assert abs(self._mvn().entropy().item() - expected) < 1e-4

    def test_sample_shape(self) -> None:
        assert self._mvn().sample((5,)).shape == (5, 2)


# ── kl_divergence ─────────────────────────────────────────────────────────


class TestKL:
    def test_normal_normal_closed_form(self) -> None:
        kl = D.kl_divergence(D.Normal(0.0, 1.0), D.Normal(1.0, 2.0))
        # log(2) + (1 + 1) / 8 - 0.5 = log 2 + 0.25 - 0.5
        expected = math.log(2.0) + (1 + 1) / (2 * 4) - 0.5
        assert abs(kl.item() - expected) < 1e-5

    def test_self_kl_is_zero(self) -> None:
        for d in (
            D.Normal(0.0, 1.0),
            D.Bernoulli(probs=lucid.tensor(0.5)),
            D.Categorical(probs=lucid.tensor([0.3, 0.7])),
            D.Exponential(1.0),
            D.Gamma(2.0, 1.0),
            D.Beta(2.0, 3.0),
        ):
            kl_val = D.kl_divergence(d, d).item()
            assert abs(kl_val) < 1e-3, f"{type(d).__name__}: KL(d,d)={kl_val}"

    def test_unregistered_raises(self) -> None:
        # Cauchy/Cauchy has no closed form (∞ in general).
        with pytest.raises(NotImplementedError):
            D.kl_divergence(D.Cauchy(0.0, 1.0), D.Cauchy(0.0, 1.0))


# ── constraints ───────────────────────────────────────────────────────────


class TestConstraints:
    def test_real_accepts_finite(self) -> None:
        c = D.constraints.real
        assert bool(c.check(lucid.tensor([0.0, -1.0, 100.0])).all().item())

    def test_positive_rejects_zero(self) -> None:
        c = D.constraints.positive
        chk = c.check(lucid.tensor([0.0, 1.0, -1.0])).numpy().tolist()
        assert chk == [False, True, False]

    def test_simplex_accepts_unit(self) -> None:
        s = D.constraints.simplex
        assert bool(s.check(lucid.tensor([0.3, 0.5, 0.2])).all().item())

    def test_simplex_rejects_non_unit(self) -> None:
        s = D.constraints.simplex
        assert not bool(s.check(lucid.tensor([0.3, 0.3, 0.3])).all().item())


# ── public surface ────────────────────────────────────────────────────────


class TestSurface:
    def test_distributions_visible(self) -> None:
        for name in (
            "Distribution", "ExponentialFamily",
            "Normal", "LogNormal", "Uniform",
            "Exponential", "Laplace", "Cauchy",
            "Gamma", "Chi2", "Beta", "Dirichlet",
            "Bernoulli", "Geometric",
            "Categorical", "OneHotCategorical",
            "MultivariateNormal",
            "kl_divergence", "register_kl",
            "constraints",
        ):
            assert hasattr(D, name), f"lucid.distributions.{name} missing"

    def test_h8_no_top_level_leak(self) -> None:
        # H8: distributions live only under lucid.distributions.
        for name in (
            "Normal", "Bernoulli", "Categorical",
            "Gamma", "Beta", "MultivariateNormal",
            "kl_divergence", "Distribution",
        ):
            assert not hasattr(lucid, name), (
                f"lucid.{name} should not exist at top level — H8 forbids "
                f"distributions shortcuts"
            )
