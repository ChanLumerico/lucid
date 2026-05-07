"""``lucid.distributions`` — probability distributions with the
reference-framework's interface (sample / rsample / log_prob / entropy / …).

Mirrors ``torch.distributions`` for the most-used univariate and
discrete families plus ``MultivariateNormal``, ``Beta``, ``Gamma``, and
``Dirichlet`` from the gamma family.  See [[api-python-distributions]]
for the full surface.
"""

from lucid.distributions import constraints
from lucid.distributions.bernoulli import Bernoulli, Geometric
from lucid.distributions.categorical import Categorical, OneHotCategorical
from lucid.distributions.distribution import Distribution, ExponentialFamily
from lucid.distributions.exponential import Cauchy, Exponential, Laplace
from lucid.distributions.gamma import Beta, Chi2, Dirichlet, Gamma
from lucid.distributions.kl import kl_divergence, register_kl
from lucid.distributions.multivariate import MultivariateNormal
from lucid.distributions.normal import LogNormal, Normal
from lucid.distributions.uniform import Uniform

__all__ = [
    # base
    "Distribution",
    "ExponentialFamily",
    # constraints submodule
    "constraints",
    # univariate continuous
    "Normal",
    "LogNormal",
    "Uniform",
    "Exponential",
    "Laplace",
    "Cauchy",
    # gamma family
    "Gamma",
    "Chi2",
    "Beta",
    "Dirichlet",
    # discrete
    "Bernoulli",
    "Geometric",
    "Categorical",
    "OneHotCategorical",
    # multivariate
    "MultivariateNormal",
    # kl
    "kl_divergence",
    "register_kl",
]
