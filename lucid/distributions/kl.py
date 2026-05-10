"""``kl_divergence`` registry — pairwise dispatch keyed on distribution
type, mirroring the reference framework's ``register_kl`` mechanism.
"""

from collections.abc import Callable

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.bernoulli import Bernoulli
from lucid.distributions.categorical import Categorical
from lucid.distributions.distribution import Distribution
from lucid.distributions.exponential import Cauchy, Exponential, Laplace
from lucid.distributions.gamma import Beta, Gamma
from lucid.distributions.normal import Normal
from lucid.distributions.student import StudentT

_KLFn = Callable[[Distribution, Distribution], Tensor]
_KL_REGISTRY: dict[tuple[type, type], _KLFn] = {}


def register_kl(p_cls: type, q_cls: type) -> Callable[[_KLFn], _KLFn]:
    """Decorator that registers a closed-form ``KL(p || q)`` for the
    pair ``(p_cls, q_cls)``.  Mirrors the reference framework's API."""

    def _decorator(fn: _KLFn) -> _KLFn:
        _KL_REGISTRY[(p_cls, q_cls)] = fn
        return fn

    return _decorator


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    """Compute ``KL(p || q)`` if a pairwise registration exists.

    Falls through to the most-derived registered ancestor — exact-class
    matching is checked first, then walks the MRO of ``type(p)`` × ``type(q)``.

    If no analytical pair is registered and ``p`` supports ``rsample``,
    a Monte Carlo estimate (1 sample) is returned as a fallback.  This
    matches the reference framework's behaviour for rare distribution pairs.
    """
    key = (type(p), type(q))
    if key in _KL_REGISTRY:
        return _KL_REGISTRY[key](p, q)
    for p_base in type(p).__mro__:
        for q_base in type(q).__mro__:
            if (p_base, q_base) in _KL_REGISTRY:
                return _KL_REGISTRY[(p_base, q_base)](p, q)
    # MC fallback for distributions with rsample.
    if p.has_rsample:
        s = p.rsample()
        return p.log_prob(s) - q.log_prob(s)
    raise NotImplementedError(
        f"kl_divergence({type(p).__name__}, {type(q).__name__}) "
        f"is not registered and p does not support rsample for MC estimation."
    )


# ── concrete pairs ────────────────────────────────────────────────────────


@register_kl(Normal, Normal)
def _kl_normal_normal(p: Normal, q: Normal) -> Tensor:
    var_p = p.variance
    var_q = q.variance
    return (
        (q.scale.log() - p.scale.log())
        + (var_p + (p.loc - q.loc) ** 2) / (2.0 * var_q)
        - 0.5
    )


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p: Bernoulli, q: Bernoulli) -> Tensor:
    p_p = p._probs.clip(1e-7, 1.0 - 1e-7)
    p_q = q._probs.clip(1e-7, 1.0 - 1e-7)
    return p_p * (p_p.log() - p_q.log()) + (1.0 - p_p) * (
        (1.0 - p_p).log() - (1.0 - p_q).log()
    )


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p: Categorical, q: Categorical) -> Tensor:
    p_log = p._log_probs
    q_log = q._log_probs
    return (p._probs * (p_log - q_log)).sum(dim=-1)


@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p: Exponential, q: Exponential) -> Tensor:
    rate_ratio = q.rate / p.rate
    return rate_ratio - 1.0 - rate_ratio.log()


@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p: Gamma, q: Gamma) -> Tensor:
    a_p, b_p = p.concentration, p.rate
    a_q, b_q = q.concentration, q.rate
    t1 = a_q * (b_p / b_q).log()
    t2 = lucid.lgamma(a_q) - lucid.lgamma(a_p)
    t3 = (a_p - a_q) * lucid.digamma(a_p)
    t4 = (b_q - b_p) * (a_p / b_p)
    return t1 + t2 + t3 + t4


@register_kl(Beta, Beta)
def _kl_beta_beta(p: Beta, q: Beta) -> Tensor:
    sum_p = p.concentration1 + p.concentration0
    sum_q = q.concentration1 + q.concentration0
    t1 = lucid.lgamma(sum_p) - lucid.lgamma(sum_q)
    t2 = (
        lucid.lgamma(q.concentration1)
        - lucid.lgamma(p.concentration1)
        + lucid.lgamma(q.concentration0)
        - lucid.lgamma(p.concentration0)
    )
    t3 = (p.concentration1 - q.concentration1) * (
        lucid.digamma(p.concentration1) - lucid.digamma(sum_p)
    )
    t4 = (p.concentration0 - q.concentration0) * (
        lucid.digamma(p.concentration0) - lucid.digamma(sum_p)
    )
    return t1 + t2 + t3 + t4


# ── P8-C: extended pairwise registrations ────────────────────────────────


from lucid.distributions.discrete import Poisson


@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p: Poisson, q: Poisson) -> Tensor:
    # KL(Pois(λ_p) || Pois(λ_q)) = λ_p · log(λ_p / λ_q) − λ_p + λ_q.
    return p.rate * (p.rate.log() - q.rate.log()) - p.rate + q.rate


from lucid.distributions.gamma import Dirichlet


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p: Dirichlet, q: Dirichlet) -> Tensor:
    # KL(Dir(α) || Dir(β)) closed form using digamma + lgamma.
    a = p.concentration
    b = q.concentration
    sum_a = a.sum(dim=-1, keepdim=True)
    sum_b = b.sum(dim=-1, keepdim=True)
    t1 = (
        lucid.lgamma(sum_a).squeeze(-1)
        - lucid.lgamma(a).sum(dim=-1)
        - lucid.lgamma(sum_b).squeeze(-1)
        + lucid.lgamma(b).sum(dim=-1)
    )
    t2 = ((a - b) * (lucid.digamma(a) - lucid.digamma(sum_a))).sum(dim=-1)
    return t1 + t2


from lucid.distributions.multivariate import MultivariateNormal
from lucid.distributions.exponential import Laplace
from lucid.distributions.continuous_extra import HalfNormal
from lucid.distributions.normal import LogNormal
from lucid.distributions.uniform import Uniform
from lucid.distributions.bernoulli import Geometric
from lucid.distributions.independent import Independent
from lucid.distributions.extra import Gumbel

_EULER_GAMMA: float = 0.5772156649015329


@register_kl(MultivariateNormal, MultivariateNormal)
def _kl_mvn_mvn(p: MultivariateNormal, q: MultivariateNormal) -> Tensor:
    # KL(N(μ_p, Σ_p) || N(μ_q, Σ_q))
    #   = 0.5·[tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p)
    #          − D + log(|Σ_q| / |Σ_p|)]
    # All work on the lower-triangular factors to avoid forming Σ⁻¹.
    Lp, Lq = p.scale_tril, q.scale_tril
    D = int(p._D)

    diag_p = Lp.diagonal(dim1=-2, dim2=-1)
    diag_q = Lq.diagonal(dim1=-2, dim2=-1)
    log_det_p = 2.0 * diag_p.log().sum(dim=-1)
    log_det_q = 2.0 * diag_q.log().sum(dim=-1)

    M = lucid.linalg.solve_triangular(Lq, Lp, upper=False)
    trace_term = (M * M).sum(dim=(-2, -1))

    diff = (q.loc - p.loc).unsqueeze(-1)
    z = lucid.linalg.solve_triangular(Lq, diff, upper=False)
    maha = (z * z).sum(dim=(-2, -1))

    return 0.5 * (trace_term + maha - float(D) + log_det_q - log_det_p)


# ── Additional closed-form KL pairs ──────────────────────────────────────────


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p: Laplace, q: Laplace) -> Tensor:
    # From Gil et al. 2011 §3.2
    # KL = -log(b1/b2) + |μ1-μ2|/b2 + (b1/b2)·exp(-|μ1-μ2|/b1) - 1
    scale_ratio: Tensor = p.scale / q.scale
    loc_diff: Tensor = (p.loc - q.loc).abs()
    return (
        -scale_ratio.log()
        + loc_diff / q.scale
        + scale_ratio * (-(loc_diff / p.scale)).exp()
        - 1.0
    )


@register_kl(HalfNormal, HalfNormal)
def _kl_halfnormal_halfnormal(p: HalfNormal, q: HalfNormal) -> Tensor:
    # HalfNormal(σ) = |Normal(0, σ)|; KL reduces to the underlying Normals.
    return _kl_normal_normal(p._base, q._base)  # type: ignore[attr-defined]


@register_kl(LogNormal, LogNormal)
def _kl_lognormal_lognormal(p: LogNormal, q: LogNormal) -> Tensor:
    # LogNormal(μ, σ²) = exp(Normal(μ, σ²)); KL equals KL of the underlying.
    return _kl_normal_normal(p._base, q._base)  # type: ignore[attr-defined]


@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p: Gumbel, q: Gumbel) -> Tensor:
    # KL(Gumbel(μ1,β1) || Gumbel(μ2,β2))
    # ct1 = β1/β2,  ct2 = μ2/β2,  ct3 = μ1/β2
    # KL = -log(ct1) - ct2 + ct3
    #     + ct1 · euler_gamma
    #     + exp(ct2 + lgamma(1 + ct1) - ct3)
    #     - (1 + euler_gamma)
    ct1: Tensor = p.scale / q.scale
    ct2: Tensor = q.loc / q.scale
    ct3: Tensor = p.loc / q.scale
    t1 = -ct1.log() - ct2 + ct3
    t2 = ct1 * _EULER_GAMMA
    t3 = ((ct2 + lucid.lgamma(1.0 + ct1) - ct3)).exp()
    return t1 + t2 + t3 - (1.0 + _EULER_GAMMA)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p: Uniform, q: Uniform) -> Tensor:
    # KL = log((b2-a2)/(b1-a1)) if [a1,b1] ⊆ [a2,b2], else +∞
    result: Tensor = ((q.high - q.low) / (p.high - p.low)).log()
    # Mask entries where p is not supported inside q
    p_outside_q = (q.low > p.low) | (q.high < p.high)
    inf_val = lucid.full_like(result, float("inf"))
    return lucid.where(p_outside_q, inf_val, result)


@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p: Geometric, q: Geometric) -> Tensor:
    # KL(Geom(p1) || Geom(p2)) = -H(p) - log(1-p2)/p1 - log(p2)
    p_logits: Tensor = p.probs.log() - (1.0 - p.probs).log()  # log(p/(1-p))
    q_logits: Tensor = q.probs.log() - (1.0 - q.probs).log()
    return -p.entropy() - lucid.log1p(-q.probs) / p.probs - q_logits


@register_kl(Independent, Independent)
def _kl_independent_independent(p: Independent, q: Independent) -> Tensor:
    # KL sums over the reinterpreted batch dims.
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError(
            "kl_divergence(Independent, Independent) requires equal "
            "reinterpreted_batch_ndims."
        )
    kl: Tensor = kl_divergence(p.base_dist, q.base_dist)
    # Sum over the last `reinterpreted_batch_ndims` dims.
    n = p.reinterpreted_batch_ndims
    for _ in range(n):
        kl = kl.sum(dim=-1)
    return kl


# ── MC-based KL pairs (no closed form) ───────────────────────────────────────


def _kl_monte_carlo(p: Distribution, q: Distribution, n_samples: int = 1) -> Tensor:
    """Monte Carlo estimate of KL(p||q) using a single ``rsample``.

    Requires ``p`` to support ``rsample``.
    """
    if not p.has_rsample:
        raise NotImplementedError(
            f"kl_divergence({type(p).__name__}, {type(q).__name__}): "
            "no closed-form KL and p does not support rsample for MC estimation."
        )
    samples = p.rsample((n_samples,))
    return (p.log_prob(samples) - q.log_prob(samples)).mean(0)


@register_kl(StudentT, StudentT)
def _kl_studentt_studentt(p: StudentT, q: StudentT) -> Tensor:
    """Monte-Carlo KL for StudentT — no closed form exists."""
    return _kl_monte_carlo(p, q)


@register_kl(Cauchy, Cauchy)
def _kl_cauchy_cauchy(p: Cauchy, q: Cauchy) -> Tensor:
    """Monte-Carlo KL for Cauchy (df=1 StudentT) — no closed form exists."""
    return _kl_monte_carlo(p, q)


@register_kl(Normal, Laplace)
def _kl_normal_laplace(p: Normal, q: Laplace) -> Tensor:
    """KL(Normal(μ,σ²) || Laplace(m,b)).

    Computed analytically via the moment-generating function of a folded Normal.
    """
    # E_p[log p(X)] = -0.5*(1 + log(2π) + log(σ²))  (normal entropy, negated)
    # E_p[log q(X)] = -log(2b) - E_p[|X - m|] / b
    # E_p[|X - m|] = σ * sqrt(2/π) * exp(-z²/2) + (μ - m) * erf(z/sqrt(2))
    #   where z = (μ - m) / σ
    import math

    mu, sigma = p.loc, p.scale
    m, b = q.loc, q.scale
    z = (mu - m) / sigma
    # E[|X - m|] where X ~ N(mu, sigma^2)
    # = sigma * sqrt(2/pi) * exp(-z^2/2) + (mu - m) * erf(z / sqrt(2))
    _sqrt2 = math.sqrt(2.0)
    _sqrt2pi = math.sqrt(2.0 * math.pi)
    exp_term = sigma * (2.0 / math.pi) ** 0.5 * (-0.5 * z * z).exp()
    erf_term = (mu - m) * lucid.erf(z / _sqrt2)
    e_abs = exp_term + erf_term

    log_p = p.entropy().neg() - 0.5 * (2.0 * math.pi * math.e)  # -H(p)
    # Actually compute directly:
    # KL = H_cross(p, q) - H(p)  where H(p) = 0.5*(1 + log(2πσ²))
    # H_cross = log(2b) + e_abs / b
    h_cross = (2.0 * b).log() + e_abs / b
    h_p = 0.5 * (1.0 + math.log(2.0 * math.pi) + (sigma * sigma).log())
    return h_cross - h_p
