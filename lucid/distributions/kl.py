"""``kl_divergence`` registry — pairwise dispatch keyed on distribution
type, mirroring the reference framework's ``register_kl`` mechanism.
"""

from collections.abc import Callable

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.bernoulli import Bernoulli
from lucid.distributions.categorical import Categorical
from lucid.distributions.distribution import Distribution
from lucid.distributions.exponential import Exponential
from lucid.distributions.gamma import Beta, Gamma
from lucid.distributions.normal import Normal

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
    """
    key = (type(p), type(q))
    if key in _KL_REGISTRY:
        return _KL_REGISTRY[key](p, q)
    for p_base in type(p).__mro__:
        for q_base in type(q).__mro__:
            if (p_base, q_base) in _KL_REGISTRY:
                return _KL_REGISTRY[(p_base, q_base)](p, q)
    raise NotImplementedError(
        f"kl_divergence({type(p).__name__}, {type(q).__name__}) "
        f"is not registered."
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
        lucid.lgamma(q.concentration1) - lucid.lgamma(p.concentration1)
        + lucid.lgamma(q.concentration0) - lucid.lgamma(p.concentration0)
    )
    t3 = (p.concentration1 - q.concentration1) * (
        lucid.digamma(p.concentration1) - lucid.digamma(sum_p)
    )
    t4 = (p.concentration0 - q.concentration0) * (
        lucid.digamma(p.concentration0) - lucid.digamma(sum_p)
    )
    return t1 + t2 + t3 + t4
