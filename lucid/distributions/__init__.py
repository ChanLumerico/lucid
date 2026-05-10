"""``lucid.distributions`` — probability distributions with the
reference-framework's interface (sample / rsample / log_prob / entropy / …).

Mirrors the reference framework's distributions surface for the most-used univariate and
discrete families plus ``MultivariateNormal``, ``Beta``, ``Gamma``, and
``Dirichlet`` from the gamma family.  See [[api-python-distributions]]
for the full surface.
"""

from lucid.distributions import constraints
from lucid.distributions.bernoulli import Bernoulli, Geometric
from lucid.distributions.categorical import Categorical, OneHotCategorical
from lucid.distributions.continuous_extra import (
    FisherSnedecor,
    HalfCauchy,
    HalfNormal,
    Pareto,
    Weibull,
)
from lucid.distributions.discrete import Binomial, NegativeBinomial, Poisson
from lucid.distributions.distribution import Distribution, ExponentialFamily
from lucid.distributions.exponential import Cauchy, Exponential, Laplace
from lucid.distributions.gamma import Beta, Chi2, Dirichlet, Gamma
from lucid.distributions.independent import Independent
from lucid.distributions.kl import kl_divergence, register_kl
from lucid.distributions.mixture import MixtureSameFamily
from lucid.distributions.multivariate import MultivariateNormal
from lucid.distributions.normal import LogNormal, Normal
from lucid.distributions.relaxed import (
    RelaxedBernoulli,
    RelaxedOneHotCategorical,
)
from lucid.distributions.student import StudentT
from lucid.distributions.transforms import (
    AbsTransform,
    AffineTransform,
    CatTransform,
    ComposeTransform,
    CorrCholeskyTransform,
    CumulativeDistributionTransform,
    ExpTransform,
    IndependentTransform,
    LowerCholeskyTransform,
    PowerTransform,
    ReshapeTransform,
    SigmoidTransform,
    SoftmaxTransform,
    StackTransform,
    StickBreakingTransform,
    TanhTransform,
    Transform,
    TransformedDistribution,
)
from lucid.distributions.extra import (
    ContinuousBernoulli,
    Gumbel,
    InverseGamma,
    Kumaraswamy,
    Multinomial,
)
from lucid.distributions.matrix import LKJCholesky, Wishart
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
    "StudentT",
    "Pareto",
    "Weibull",
    "HalfNormal",
    "HalfCauchy",
    "FisherSnedecor",
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
    "Poisson",
    "Binomial",
    "NegativeBinomial",
    # extra univariate continuous
    "Gumbel",
    "InverseGamma",
    "Kumaraswamy",
    # discrete (extra)
    "Multinomial",
    # continuous [0,1]
    "ContinuousBernoulli",
    # relaxed (Concrete) — differentiable approximations
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    # multivariate
    "MultivariateNormal",
    # matrix-valued
    "Wishart",
    "LKJCholesky",
    # wrappers
    "Independent",
    "TransformedDistribution",
    "MixtureSameFamily",
    # transforms
    "Transform",
    "AbsTransform",
    "AffineTransform",
    "CatTransform",
    "ComposeTransform",
    "CorrCholeskyTransform",
    "CumulativeDistributionTransform",
    "ExpTransform",
    "IndependentTransform",
    "LowerCholeskyTransform",
    "PowerTransform",
    "ReshapeTransform",
    "SigmoidTransform",
    "SoftmaxTransform",
    "StackTransform",
    "StickBreakingTransform",
    "TanhTransform",
    # kl
    "kl_divergence",
    "register_kl",
]
