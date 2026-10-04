"""Distribution formulas checked against oracles that do not share their code.

Each class is one issue.  A test either compares with a value from an
independent oracle (a closed form evaluated in NumPy, SciPy constants, a
numerical integral of the density) and runs everywhere, or compares with
the reference framework through the ``ref`` fixture and is marked
``parity``.  Every check runs on CPU and on Metal.

* ``TestFormulaErrors`` (CHA-143) — four closed forms that were silently
  wrong on both devices.
* ``TestSupportBoundaries`` (CHA-144) — ``-inf`` from overflowing
  ``log(1 + exp(l))``, NaN from ``0 · log 0`` at the edge of a support, and
  ``-inf`` logits (masked categories) refused by ``constraints.real``.
* ``TestScalarConstantsFollowTheDevice`` (CHA-148) — Python numbers held as
  0-dim host tensors that raised ``DeviceMismatch`` against Metal tensors.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D

DEVICES = ["cpu", "metal"]


def _t(values: Any, device: str) -> lucid.Tensor:
    return lucid.tensor(np.asarray(values, dtype=np.float32), device=device)


def _np(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.numpy(), dtype=np.float64)


def _ref_np(t: Any) -> np.ndarray:
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


# ── CHA-143: formula errors ──────────────────────────────────────────────────

_PRECISION = np.array([[2.0, 0.6], [0.6, 1.0]])
_PRECISION_3 = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])


def _mvn_log_prob_from_precision(
    x: np.ndarray, loc: np.ndarray, precision: np.ndarray
) -> float:
    """``log N(x; loc, P⁻¹)`` written with ``P`` directly — no factorisation."""
    diff = x - loc
    _, logdet = np.linalg.slogdet(precision)
    d = len(loc)
    return float(
        -0.5 * diff @ precision @ diff + 0.5 * logdet - 0.5 * d * np.log(2 * np.pi)
    )


def _continuous_bernoulli_moments(lam: float) -> tuple[float, float]:
    """Mean and variance of ``ContinuousBernoulli(λ)`` by integrating its density."""
    x = np.linspace(0.0, 1.0, 200_001)
    density = lam**x * (1.0 - lam) ** (1.0 - x)
    z = _trapezoid(density, x)
    m1 = _trapezoid(x * density, x) / z
    m2 = _trapezoid(x * x * density, x) / z
    return m1, m2 - m1 * m1


@pytest.mark.parametrize("device", DEVICES)
class TestFormulaErrors:
    """CHA-143 — MVN(precision), Dirichlet entropy, RelaxedBernoulli density,
    ContinuousBernoulli variance."""

    @pytest.mark.parametrize(
        "precision,loc,x",
        [
            (_PRECISION, [0.5, -1.0], [1.0, 0.3]),
            (_PRECISION_3, [0.0, 0.0, 0.0], [0.2, -0.4, 1.1]),
        ],
        ids=["2d", "3d"],
    )
    def test_mvn_precision_log_prob(
        self, device: str, precision: np.ndarray, loc: list[float], x: list[float]
    ) -> None:
        """``inv(chol(P)).mT`` is upper-triangular; ``log_prob`` read it as lower.

        Before: -2.53343 for the 2-d case, against -3.07553.
        """
        d = D.MultivariateNormal(
            _t(loc, device), precision_matrix=_t(precision, device)
        )
        got = float(_np(d.log_prob(_t(x, device))))
        want = _mvn_log_prob_from_precision(np.array(x), np.array(loc), precision)
        assert got == pytest.approx(want, abs=1e-5)

    def test_mvn_precision_agrees_with_covariance(self, device: str) -> None:
        cov = np.linalg.inv(_PRECISION_3)
        x = _t([[0.2, -0.4, 1.1], [1.0, 0.0, -2.0]], device)
        by_precision = D.MultivariateNormal(
            _t([0.1, 0.2, 0.3], device), precision_matrix=_t(_PRECISION_3, device)
        )
        by_covariance = D.MultivariateNormal(
            _t([0.1, 0.2, 0.3], device), covariance_matrix=_t(cov, device)
        )
        np.testing.assert_allclose(
            _np(by_precision.log_prob(x)), _np(by_covariance.log_prob(x)), atol=1e-5
        )
        np.testing.assert_allclose(
            _np(by_precision.entropy()), _np(by_covariance.entropy()), atol=1e-5
        )
        tril = _np(by_precision.scale_tril)
        np.testing.assert_allclose(np.triu(tril, 1), 0.0, atol=0.0)
        np.testing.assert_allclose(tril, np.linalg.cholesky(cov), atol=1e-5)

    def test_mvn_precision_batched(self, device: str) -> None:
        precisions = np.stack([_PRECISION, np.array([[1.5, -0.3], [-0.3, 0.8]])])
        d = D.MultivariateNormal(
            _t([0.0, 0.0], device), precision_matrix=_t(precisions, device)
        )
        got = _np(d.log_prob(_t([1.0, 0.3], device)))
        want = [
            _mvn_log_prob_from_precision(np.array([1.0, 0.3]), np.zeros(2), p)
            for p in precisions
        ]
        np.testing.assert_allclose(got, want, atol=1e-5)

    def test_dirichlet_entropy(self, device: str) -> None:
        """ψ(α₀) was counted twice.  Before: ``[3.874, -1.217]``.

        Expected values: ``scipy.stats.dirichlet(α).entropy()``.
        """
        d = D.Dirichlet(_t([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], device))
        np.testing.assert_allclose(
            _np(d.entropy()), [-1.2443445622221003, -1.1621229335906549], atol=1e-5
        )

    def test_dirichlet_entropy_of_the_flat_simplex(self, device: str) -> None:
        """``Dirichlet(1, …, 1)`` is uniform on the simplex: ``H = −log((K−1)!)``."""
        for k in (2, 3, 5):
            got = float(_np(D.Dirichlet(_t(np.ones(k), device)).entropy()))
            assert got == pytest.approx(
                -np.log(float(np.prod(np.arange(1, k)))), abs=1e-5
            )

    def test_relaxed_bernoulli_log_prob(self, device: str) -> None:
        """Before: ``[-0.357, -3.251, -6.947]`` — an exponent of ``τ+1`` and no
        sigmoid Jacobian."""
        d = D.RelaxedBernoulli(_t(0.7, device), probs=_t(0.3, device))
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.1, 0.5, 0.9], device))),
            [0.54798782, -0.53102827, -0.51020932],
            atol=1e-5,
        )

    @pytest.mark.parametrize("temperature,logit", [(2.0, -1.5), (1.0, 0.7), (3.5, 2.0)])
    def test_relaxed_bernoulli_density_integrates_to_one(
        self, device: str, temperature: float, logit: float
    ) -> None:
        """Bounded at both ends for ``τ ≥ 1``, so integrated in ``y`` directly."""
        y = np.linspace(1e-6, 1.0 - 1e-6, 400_001)
        d = D.RelaxedBernoulli(_t(temperature, device), logits=_t(logit, device))
        density = np.exp(_np(d.log_prob(_t(y, device))))
        assert _trapezoid(density, y) == pytest.approx(1.0, abs=2e-4)

    @pytest.mark.parametrize("temperature,prob", [(0.6, 0.4), (0.7, 0.5), (0.9, 0.8)])
    def test_relaxed_bernoulli_density_integrates_to_one_below_unit_temperature(
        self, device: str, temperature: float, prob: float
    ) -> None:
        """For ``τ < 1`` the density blows up at 0 and 1, so integrate over
        ``z = logit(y)``: ``∫ p(y) dy = ∫ p(σ(z)) σ(z)(1 − σ(z)) dz``.

        Float32 cannot hold a ``y`` closer to 1 than ``σ(16.6)``; the
        parameters keep the mass beyond ``|z| = 16`` under 2e-4.
        """
        z = np.linspace(-16.0, 16.0, 400_001)
        y = (1.0 / (1.0 + np.exp(-z))).astype(np.float32).astype(np.float64)
        y = y[(y > 0.0) & (y < 1.0)]
        z = np.log(y) - np.log1p(-y)  # the z each representable y stands for
        d = D.RelaxedBernoulli(_t(temperature, device), probs=_t(prob, device))
        log_p = _np(d.log_prob(_t(y, device)))
        integrand = np.exp(log_p + np.log(y) + np.log1p(-y))
        assert _trapezoid(integrand, z) == pytest.approx(1.0, abs=1e-3)

    def test_continuous_bernoulli_variance(self, device: str) -> None:
        """Before: ``[0.0695, 0.0833, 0.0]`` — a wrong ``E[X²]`` with the
        negative results clamped to 0."""
        probs = [0.01, 0.2, 0.3, 0.45, 0.4999, 0.5, 0.5003, 0.55, 0.65, 0.7, 0.99]
        d = D.ContinuousBernoulli(probs=_t(probs, device))
        want = [_continuous_bernoulli_moments(p)[1] for p in probs]
        np.testing.assert_allclose(_np(d.variance), want, atol=2e-6)

    def test_continuous_bernoulli_mean_near_one_half(self, device: str) -> None:
        """The closed-form mean cancels catastrophically near ½ — in float32 it
        was off by 0.5 at λ = 0.4999 and by 0.25 at λ = 0.5001."""
        probs = [0.38, 0.45, 0.4999, 0.5001, 0.5002, 0.501, 0.505, 0.62]
        d = D.ContinuousBernoulli(probs=_t(probs, device))
        want = [_continuous_bernoulli_moments(p)[0] for p in probs]
        np.testing.assert_allclose(_np(d.mean), want, atol=2e-6)

    def test_continuous_bernoulli_moments_from_large_logits(self, device: str) -> None:
        """``sigmoid(±100)`` rounds to 0 or 1; the moments come from the logit."""
        d = D.ContinuousBernoulli(logits=_t([100.0, -100.0], device))
        np.testing.assert_allclose(_np(d.mean), [0.99, 0.01], atol=1e-6)
        np.testing.assert_allclose(_np(d.variance), [1e-4, 1e-4], atol=1e-7)

    def test_continuous_bernoulli_moment_gradients_are_finite_at_one_half(
        self, device: str
    ) -> None:
        """The closed form is evaluated everywhere and masked; its pole at ½
        must not reach the gradient."""
        p = _t([0.2, 0.5, 0.45], device)
        p.requires_grad_(True)
        d = D.ContinuousBernoulli(probs=p)
        (d.mean.sum() + d.variance.sum()).backward()
        grad = _np(p.grad)
        assert np.isfinite(grad).all(), grad
        assert grad[1] == pytest.approx(1.0 / 3.0, abs=1e-5)

    @pytest.mark.parity
    def test_parity(self, device: str, ref: Any) -> None:
        RD = ref.distributions
        rt = lambda v: ref.tensor(np.asarray(v, dtype=np.float32))  # noqa: E731

        lucid_mvn = D.MultivariateNormal(
            _t([0.5, -1.0], device), precision_matrix=_t(_PRECISION, device)
        )
        ref_mvn = RD.MultivariateNormal(
            rt([0.5, -1.0]), precision_matrix=rt(_PRECISION)
        )
        np.testing.assert_allclose(
            _np(lucid_mvn.log_prob(_t([1.0, 0.3], device))),
            _ref_np(ref_mvn.log_prob(rt([1.0, 0.3]))),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            _np(lucid_mvn.scale_tril), _ref_np(ref_mvn.scale_tril), atol=1e-5
        )

        conc = [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]
        np.testing.assert_allclose(
            _np(D.Dirichlet(_t(conc, device)).entropy()),
            _ref_np(RD.Dirichlet(rt(conc)).entropy()),
            atol=1e-5,
        )

        y = [0.05, 0.3, 0.5, 0.9]
        np.testing.assert_allclose(
            _np(
                D.RelaxedBernoulli(_t(0.7, device), probs=_t(0.3, device)).log_prob(
                    _t(y, device)
                )
            ),
            _ref_np(RD.RelaxedBernoulli(rt(0.7), probs=rt(0.3)).log_prob(rt(y))),
            atol=1e-5,
        )

        # Away from the reference's own float32 cancellation band (0.4–0.499).
        probs = [0.05, 0.2, 0.3, 0.5, 0.65, 0.7, 0.95]
        lucid_cb = D.ContinuousBernoulli(probs=_t(probs, device))
        ref_cb = RD.ContinuousBernoulli(probs=rt(probs))
        np.testing.assert_allclose(_np(lucid_cb.mean), _ref_np(ref_cb.mean), atol=1e-5)
        np.testing.assert_allclose(
            _np(lucid_cb.variance), _ref_np(ref_cb.variance), atol=1e-5
        )


# ── CHA-144: support boundaries ──────────────────────────────────────────────

_INF = float("inf")


@pytest.mark.parametrize("device", DEVICES)
class TestSupportBoundaries:
    """CHA-144 — overflow at large logits, ``0 · log 0`` NaN, and masked logits.

    Expected values are the reference framework's (it holds an exact 0 or 1
    probability one epsilon inside ``[0, 1]``, hence the ``1e-7``-sized
    residues), or SciPy's where noted.
    """

    def test_bernoulli_large_logits(self, device: str) -> None:
        """``log(1 + exp(l))`` overflowed at ``l ≈ 89``: before, ``[-inf]``
        and an entropy of ``inf``."""
        d = D.Bernoulli(logits=_t([100.0, -100.0], device))
        np.testing.assert_allclose(
            _np(d.log_prob(_t([1.0, 1.0], device))), [0.0, -100.0], atol=1e-5
        )
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.0, 0.0], device))), [-100.0, 0.0], atol=1e-5
        )
        np.testing.assert_allclose(_np(d.entropy()), [0.0, 0.0], atol=1e-5)

    def test_bernoulli_degenerate_probs(self, device: str) -> None:
        """``probs=0`` gave an infinite logit and ``0 · (-inf) = NaN``."""
        zero = D.Bernoulli(probs=_t([0.0], device))
        np.testing.assert_allclose(
            _np(zero.log_prob(_t([0.0, 1.0], device))), [0.0, -15.94238472], atol=1e-5
        )
        np.testing.assert_allclose(_np(zero.entropy()), [0.0], atol=1e-6)
        one = D.Bernoulli(probs=_t([1.0], device))
        np.testing.assert_allclose(
            _np(one.log_prob(_t([1.0, 0.0], device))), [0.0, -15.94238472], atol=1e-5
        )

    def test_bernoulli_gradient_at_large_logits(self, device: str) -> None:
        logits = _t([100.0, -100.0, 0.0], device)
        logits.requires_grad_(True)
        d = D.Bernoulli(logits=logits)
        d.log_prob(_t([1.0, 1.0, 1.0], device)).sum().backward()
        grad = _np(logits.grad)
        # d/dl [l − softplus(l)] = 1 − σ(l).
        np.testing.assert_allclose(grad, [0.0, 1.0, 0.5], atol=1e-5)

    def test_gamma_family_at_zero(self, device: str) -> None:
        """``(α − 1) log x`` at ``α = 1, x = 0`` was NaN."""
        gamma = D.Gamma(_t(1.0, device), _t(1.0, device))
        np.testing.assert_allclose(_np(gamma.log_prob(_t(0.0, device))), 0.0, atol=1e-6)
        gamma = D.Gamma(_t(1.0, device), _t(2.5, device))
        np.testing.assert_allclose(
            _np(gamma.log_prob(_t([0.0, 1.0], device))),
            [0.91629076, -1.58370924],
            atol=1e-5,
        )
        chi2 = D.Chi2(_t(2.0, device))
        np.testing.assert_allclose(
            _np(chi2.log_prob(_t(0.0, device))), -0.69314718, atol=1e-5
        )

    def test_beta_and_dirichlet_at_the_boundary(self, device: str) -> None:
        flat = D.Beta(_t(1.0, device), _t(1.0, device))
        np.testing.assert_allclose(
            _np(flat.log_prob(_t([0.0, 1.0], device))), [0.0, 0.0], atol=1e-6
        )
        beta = D.Beta(_t(1.0, device), _t(3.0, device))
        np.testing.assert_allclose(
            _np(beta.log_prob(_t([0.0, 0.5], device))),
            [1.09861231, -0.28768206],
            atol=1e-5,
        )
        # Dirichlet(1, 1, 2) on a face of the simplex:
        # log Γ(4) − log Γ(2) + (2 − 1) log 0.5 = log 6 − log 2.
        d = D.Dirichlet(_t([1.0, 1.0, 2.0], device))
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.0, 0.5, 0.5], device))), np.log(3.0), atol=1e-5
        )

    def test_gradients_stay_finite_at_the_boundary(self, device: str) -> None:
        """The guard swaps the operand, so no ``0 · inf`` reaches backward."""
        a = _t(1.0, device)
        b = _t(1.0, device)
        a.requires_grad_(True)
        b.requires_grad_(True)
        D.Beta(a, b).log_prob(_t([0.0, 1.0], device)).sum().backward()
        assert np.isfinite([_np(a.grad), _np(b.grad)]).all()

        alpha = _t(1.0, device)
        alpha.requires_grad_(True)
        D.Gamma(alpha, _t(1.0, device)).log_prob(
            _t([0.0, 2.0], device)
        ).sum().backward()
        # Σ over both values of d/dα = log β + log x − ψ(α), at α = β = 1:
        # x = 0 contributes only −ψ(1) (its log x term is held at 0), x = 2
        # contributes log 2 − ψ(1).
        euler = 0.5772156649015329
        assert float(_np(alpha.grad).sum()) == pytest.approx(
            np.log(2.0) + 2 * euler, abs=1e-4
        )

    def test_poisson_geometric_negative_binomial_at_zero(self, device: str) -> None:
        """Validation off: their stricter parameter constraints are CHA-145."""
        poisson = D.Poisson(_t(0.0, device), validate_args=False)
        np.testing.assert_allclose(
            _np(poisson.log_prob(_t([0.0, 1.0], device))), [0.0, -_INF], atol=2e-6
        )
        geom = D.Geometric(_t(1.0, device), validate_args=False)
        np.testing.assert_allclose(_np(geom.log_prob(_t(0.0, device))), 0.0, atol=1e-6)
        np.testing.assert_allclose(_np(geom.entropy()), 0.0, atol=1e-6)
        nb = D.NegativeBinomial(
            _t(3.0, device), probs=_t(0.0, device), validate_args=False
        )
        np.testing.assert_allclose(_np(nb.log_prob(_t(0.0, device))), 0.0, atol=2e-6)

    def test_negative_binomial_large_logits(self, device: str) -> None:
        """Through ``probs`` the answer was ``-inf`` for both."""
        nb = D.NegativeBinomial(_t(3.0, device), logits=_t([-100.0, 100.0], device))
        np.testing.assert_allclose(
            _np(nb.log_prob(_t(1.0, device))), [-98.90139008, -298.90139771], atol=1e-4
        )
        nb = D.NegativeBinomial(_t(3.0, device), probs=_t(0.4, device))
        np.testing.assert_allclose(
            _np(nb.log_prob(_t([0.0, 2.0, 7.0], device))),
            [-1.5324769, -1.57329893, -4.36299324],
            atol=1e-5,
        )

    def test_fisher_snedecor_at_zero(self, device: str) -> None:
        """``scipy.stats.f.logpdf([0, 1], 2, 5)``.  The reference itself gives
        NaN at 0; the density there is exactly 1."""
        d = D.FisherSnedecor(_t(2.0, device), _t(5.0, device))
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.0, 1.0], device))), [0.0, -1.17765283], atol=1e-5
        )

    def test_categorical_entropy_with_zero_probability(self, device: str) -> None:
        log2 = np.log(2.0)
        cat = D.Categorical(probs=_t([0.0, 0.5, 0.5], device))
        np.testing.assert_allclose(_np(cat.entropy()), log2, atol=1e-6)
        ohc = D.OneHotCategorical(probs=_t([0.0, 0.5, 0.5], device))
        np.testing.assert_allclose(_np(ohc.entropy()), log2, atol=1e-6)
        np.testing.assert_allclose(
            _np(ohc.log_prob(_t([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device))),
            [-log2, -_INF],
            atol=1e-6,
        )

    def test_masked_logits(self, device: str) -> None:
        """The action-mask idiom: ``real`` refused ``-inf``, so this raised."""
        logits = _t([-_INF, 0.0, 1.0], device)
        logits.requires_grad_(True)
        d = D.Categorical(logits=logits)
        index = lucid.tensor([1, 0], dtype=lucid.int64, device=device)
        np.testing.assert_allclose(
            _np(d.log_prob(index)), [-np.log1p(np.e), -_INF], atol=1e-5
        )
        entropy = d.entropy()
        p = np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum()
        np.testing.assert_allclose(_np(entropy), -(p * np.log(p)).sum(), atol=1e-6)
        entropy.backward()
        grad = _np(logits.grad)
        assert np.isfinite(grad).all(), grad
        assert grad[0] == 0.0

    def test_real_constraint_rejects_only_nan(self, device: str) -> None:
        check = D.constraints.real.check(_t([-_INF, _INF, 0.0, np.nan], device))
        assert [bool(v) for v in check.numpy()] == [True, True, True, False]
        with pytest.raises(ValueError):
            D.Categorical(logits=_t([np.nan, 0.0], device))

    def test_categorical_kl_with_zero_probability(self, device: str) -> None:
        p = D.Categorical(probs=_t([0.0, 0.5, 0.5], device))
        q = D.Categorical(probs=_t([0.2, 0.3, 0.5], device))
        np.testing.assert_allclose(_np(D.kl_divergence(p, q)), 0.25541282, atol=1e-6)
        assert np.isposinf(_np(D.kl_divergence(q, p)))
        masked = _t([-_INF, 0.0, 1.0], device)
        masked.requires_grad_(True)
        D.kl_divergence(D.Categorical(logits=masked), q).backward()
        assert np.isfinite(_np(masked.grad)).all(), _np(masked.grad)

    def test_multinomial_with_zero_probability(self, device: str) -> None:
        by_probs = D.Multinomial(5, probs=_t([0.0, 0.5, 0.5], device))
        by_logits = D.Multinomial(5, logits=_t([-_INF, 0.0, 0.0], device))
        counts = _t([0.0, 2.0, 3.0], device)
        np.testing.assert_allclose(
            _np(by_probs.log_prob(counts)), -1.16315079, atol=1e-5
        )
        np.testing.assert_allclose(
            _np(by_logits.log_prob(counts)), -1.16315079, atol=1e-5
        )

    def test_continuous_bernoulli_large_logits(self, device: str) -> None:
        """``log(1 + exp(l))`` and a normaliser computed through ``λ`` gave
        ``inf``.  The exact value is ``0.3 l − softplus(l) + log|l|``; the
        reference holds ``λ`` one epsilon inside ``[0, 1]`` and answers
        ``[-67.23, -27.23]`` instead."""
        d = D.ContinuousBernoulli(logits=_t([100.0, -100.0], device))
        want = [0.3 * 100 - 100 + np.log(100.0), -0.3 * 100 + np.log(100.0)]
        np.testing.assert_allclose(_np(d.log_prob(_t(0.3, device))), want, atol=1e-4)

    @pytest.mark.parity
    def test_parity(self, device: str, ref: Any) -> None:
        RD = ref.distributions
        rt = lambda v: ref.tensor(np.asarray(v, dtype=np.float32))  # noqa: E731

        def same(lucid_value: lucid.Tensor, ref_value: Any) -> None:
            np.testing.assert_allclose(
                _np(lucid_value), _ref_np(ref_value), atol=1e-5, rtol=1e-5
            )

        for logits in ([100.0, -100.0, 3.0], [0.0, 20.0, -40.0]):
            for v in ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]):
                same(
                    D.Bernoulli(logits=_t(logits, device)).log_prob(_t(v, device)),
                    RD.Bernoulli(logits=rt(logits)).log_prob(rt(v)),
                )
            same(
                D.Bernoulli(logits=_t(logits, device)).entropy(),
                RD.Bernoulli(logits=rt(logits)).entropy(),
            )
        probs, v = [0.0, 1.0, 0.3], [0.0, 1.0, 1.0]
        same(
            D.Bernoulli(probs=_t(probs, device)).log_prob(_t(v, device)),
            RD.Bernoulli(probs=rt(probs)).log_prob(rt(v)),
        )
        same(
            D.Bernoulli(probs=_t(probs, device)).entropy(),
            RD.Bernoulli(probs=rt(probs)).entropy(),
        )
        same(
            D.Gamma(_t([1.0, 2.0], device), _t([2.5, 1.0], device)).log_prob(
                _t([0.0, 0.0], device)
            ),
            RD.Gamma(rt([1.0, 2.0]), rt([2.5, 1.0])).log_prob(rt([0.0, 0.0])),
        )
        same(
            D.Beta(_t([1.0, 1.0], device), _t([1.0, 3.0], device)).log_prob(
                _t([0.0, 1.0], device)
            ),
            RD.Beta(rt([1.0, 1.0]), rt([1.0, 3.0])).log_prob(rt([0.0, 1.0])),
        )
        same(
            D.Poisson(_t([0.0, 2.0], device), validate_args=False).log_prob(
                _t([0.0, 3.0], device)
            ),
            RD.Poisson(rt([0.0, 2.0])).log_prob(rt([0.0, 3.0])),
        )
        same(
            D.NegativeBinomial(
                _t(3.0, device), logits=_t([-100.0, 100.0, 0.5], device)
            ).log_prob(_t(2.0, device)),
            RD.NegativeBinomial(rt(3.0), logits=rt([-100.0, 100.0, 0.5])).log_prob(
                rt(2.0)
            ),
        )
        same(
            D.FisherSnedecor(_t(7.0, device), _t(3.0, device)).log_prob(
                _t([0.2, 1.0, 3.5], device)
            ),
            RD.FisherSnedecor(rt(7.0), rt(3.0)).log_prob(rt([0.2, 1.0, 3.5])),
        )
        q_probs = [0.2, 0.3, 0.5]
        for kwargs in ({"probs": [0.0, 0.5, 0.5]}, {"logits": [-_INF, 0.0, 1.0]}):
            lucid_kw = {k: _t(v, device) for k, v in kwargs.items()}
            ref_kw = {k: rt(v) for k, v in kwargs.items()}
            same(
                D.Categorical(**lucid_kw).entropy(),
                RD.Categorical(**ref_kw).entropy(),
            )
            same(
                D.OneHotCategorical(**lucid_kw).entropy(),
                RD.OneHotCategorical(**ref_kw).entropy(),
            )
            same(
                D.kl_divergence(
                    D.Categorical(**lucid_kw),
                    D.Categorical(probs=_t(q_probs, device)),
                ),
                RD.kl_divergence(
                    RD.Categorical(**ref_kw), RD.Categorical(probs=rt(q_probs))
                ),
            )
        same(
            D.ContinuousBernoulli(logits=_t([0.5, -3.0, 8.0], device)).log_prob(
                _t(0.3, device)
            ),
            RD.ContinuousBernoulli(logits=rt([0.5, -3.0, 8.0])).log_prob(rt(0.3)),
        )


# ── CHA-148: constants on the wrong device ───────────────────────────────────


def _on(t: lucid.Tensor, device: str) -> lucid.Tensor:
    assert t.device.type == device, f"result on {t.device}, expected {device}"
    return t


@pytest.mark.parametrize("device", DEVICES)
class TestScalarConstantsFollowTheDevice:
    """CHA-148 — a Python number held as a 0-dim host tensor met a Metal
    tensor and raised ``DeviceMismatch``.  Values are the reference's."""

    def test_student_t_entropy(self, device: str) -> None:
        d = D.StudentT(_t([3.0, 5.0], device))
        np.testing.assert_allclose(
            _np(_on(d.entropy(), device)), [1.77347744, 1.62750196], atol=1e-5
        )

    def test_multinomial_with_an_int_count(self, device: str) -> None:
        d = D.Multinomial(5, probs=_t([0.2, 0.3, 0.5], device))
        np.testing.assert_allclose(_np(_on(d.mean, device)), [1.0, 1.5, 2.5])
        np.testing.assert_allclose(
            _np(_on(d.variance, device)), [0.8, 1.05, 1.25], atol=1e-6
        )
        np.testing.assert_allclose(
            _np(_on(d.log_prob(_t([1.0, 2.0, 2.0], device)), device)),
            -2.00248051,
            atol=1e-5,
        )
        lucid.manual_seed(0)
        sample = _on(d.sample((4,)), device)
        assert (_np(sample).sum(axis=-1) == 5).all()

    def test_affine_and_power_transforms_with_number_parameters(
        self, device: str
    ) -> None:
        x = _t([1.0, 2.0], device)
        affine = D.transforms.AffineTransform(2.0, -3.0)
        np.testing.assert_allclose(_np(_on(affine(x), device)), [-1.0, -4.0])
        np.testing.assert_allclose(_np(_on(affine.inv(x), device)), [1 / 3, 0.0])
        np.testing.assert_allclose(
            _np(_on(affine.log_abs_det_jacobian(x, affine(x)), device)),
            [np.log(3.0)] * 2,
            atol=1e-6,
        )
        power = D.transforms.PowerTransform(2.5)
        np.testing.assert_allclose(
            _np(_on(power(x), device)), [1.0, 2.0**2.5], rtol=1e-6
        )
        np.testing.assert_allclose(
            _np(_on(power.inv(x), device)), [1.0, 2.0**0.4], rtol=1e-6
        )
        np.testing.assert_allclose(
            _np(_on(power.log_abs_det_jacobian(x, power(x)), device)),
            np.log(2.5) + 1.5 * np.log([1.0, 2.0]),
            atol=1e-6,
        )

    def test_transformed_distributions_built_on_them(self, device: str) -> None:
        base = D.Normal(_t([0.0, 1.0], device), _t([1.0, 0.5], device))
        affine = D.TransformedDistribution(
            base, [D.transforms.AffineTransform(2.0, 3.0)]
        )
        lucid.manual_seed(0)
        _on(affine.rsample((3,)), device)
        y = _t([2.5, 4.0], device)
        # N(2 + 3·loc, 3·scale) scored directly.
        want = _np(D.Normal(_t([2.0, 5.0], device), _t([3.0, 1.5], device)).log_prob(y))
        np.testing.assert_allclose(
            _np(_on(affine.log_prob(y), device)), want, atol=1e-5
        )
        power = D.TransformedDistribution(
            D.Exponential(_t([1.0, 2.0], device)), [D.transforms.PowerTransform(2.5)]
        )
        lucid.manual_seed(0)
        _on(power.sample((3,)), device)
        assert np.isfinite(
            _np(_on(power.log_prob(_t([0.5, 2.0], device)), device))
        ).all()

    def test_relaxed_distributions_with_a_number_temperature(self, device: str) -> None:
        rb = D.RelaxedBernoulli(0.7, probs=_t([0.3], device))
        np.testing.assert_allclose(
            _np(_on(rb.log_prob(_t([0.1, 0.5, 0.9], device)), device)),
            [0.54798782, -0.53102827, -0.51020932],
            atol=1e-5,
        )
        lucid.manual_seed(0)
        _on(rb.rsample((3,)), device)
        roc = D.RelaxedOneHotCategorical(0.5, probs=_t([0.2, 0.3, 0.5], device))
        lucid.manual_seed(0)
        sample = _on(roc.rsample((3,)), device)
        assert np.isfinite(_np(_on(roc.log_prob(sample), device))).all()

    @pytest.mark.parity
    def test_parity(self, device: str, ref: Any) -> None:
        RD = ref.distributions
        rt = lambda v: ref.tensor(np.asarray(v, dtype=np.float32))  # noqa: E731
        np.testing.assert_allclose(
            _np(D.StudentT(_t([3.0, 5.0], device), 1.0, 2.0).entropy()),
            _ref_np(RD.StudentT(rt([3.0, 5.0]), 1.0, 2.0).entropy()),
            atol=1e-5,
        )
        counts = [1.0, 2.0, 2.0]
        np.testing.assert_allclose(
            _np(
                D.Multinomial(5, probs=_t([0.2, 0.3, 0.5], device)).log_prob(
                    _t(counts, device)
                )
            ),
            _ref_np(RD.Multinomial(5, probs=rt([0.2, 0.3, 0.5])).log_prob(rt(counts))),
            atol=1e-5,
        )
        y = [2.5, 4.0]
        lucid_td = D.TransformedDistribution(
            D.Normal(_t([0.0, 1.0], device), _t([1.0, 0.5], device)),
            [D.transforms.AffineTransform(2.0, 3.0)],
        )
        ref_td = RD.TransformedDistribution(
            RD.Normal(rt([0.0, 1.0]), rt([1.0, 0.5])),
            [RD.transforms.AffineTransform(2.0, 3.0)],
        )
        np.testing.assert_allclose(
            _np(lucid_td.log_prob(_t(y, device))),
            _ref_np(ref_td.log_prob(rt(y))),
            atol=1e-5,
        )
        y = [0.5, 2.0]
        lucid_td = D.TransformedDistribution(
            D.Exponential(_t([1.0, 2.0], device)), [D.transforms.PowerTransform(2.5)]
        )
        ref_td = RD.TransformedDistribution(
            RD.Exponential(rt([1.0, 2.0])), [RD.transforms.PowerTransform(2.5)]
        )
        np.testing.assert_allclose(
            _np(lucid_td.log_prob(_t(y, device))),
            _ref_np(ref_td.log_prob(rt(y))),
            atol=1e-5,
        )
