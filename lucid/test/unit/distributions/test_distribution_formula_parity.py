"""Distribution formulas checked against oracles that do not share their code.

Each class is one issue.  A test either compares with a value from an
independent oracle (a closed form evaluated in NumPy, SciPy constants, a
numerical integral of the density) and runs everywhere, or compares with
the reference framework through the ``ref`` fixture and is marked
``parity``.  Every check runs on CPU and on Metal.

* ``TestFormulaErrors`` (CHA-143) — four closed forms that were silently
  wrong on both devices.
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
