"""Matrix-valued distributions: ``Wishart``, ``LKJCholesky``.

Both distributions operate over symmetric positive-definite matrices or
their Cholesky factors, making them useful for learning covariance and
correlation structure in Bayesian models.

All computation is done in pure Lucid — no external libraries.
"""

import math

import lucid
import lucid.linalg as _la
import lucid.special as _sp
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import as_tensor as _as_tensor
from lucid.distributions.constraints import (
    Constraint,
    positive,
    positive_definite,
)
from lucid.distributions.distribution import Distribution
from lucid.distributions.gamma import Beta, _sample_standard_gamma

# ── Wishart ───────────────────────────────────────────────────────────────────


class Wishart(Distribution):
    """Wishart distribution over symmetric positive-definite matrices.

    ``Wishart(df, covariance_matrix=Σ)`` models sample covariance matrices
    arising from ``df`` independent draws from ``Normal(0, Σ)``.

    Parameters
    ----------
    df : Tensor | float
        Degrees of freedom.  Must satisfy ``df > dim - 1``.
    covariance_matrix : Tensor | None
        Positive-definite scale matrix ``Σ``.  Mutually exclusive with
        ``precision_matrix`` and ``scale_tril``.
    precision_matrix : Tensor | None
        Positive-definite precision matrix ``Σ⁻¹``.
    scale_tril : Tensor | None
        Lower Cholesky factor ``L`` of ``Σ`` (``Σ = L Lᵀ``).
    """

    arg_constraints = {"df": positive}
    has_rsample: bool = False

    def __init__(
        self,
        df: Tensor | float,
        covariance_matrix: Tensor | None = None,
        *,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        n_spec = sum(
            x is not None for x in (covariance_matrix, precision_matrix, scale_tril)
        )
        if n_spec != 1:
            raise ValueError(
                "Wishart: pass exactly one of covariance_matrix, "
                "precision_matrix, or scale_tril."
            )

        if scale_tril is not None:
            self.scale_tril = _as_tensor(scale_tril)
        elif covariance_matrix is not None:
            cov = _as_tensor(covariance_matrix)
            self.scale_tril = _la.cholesky(cov)
        else:  # precision_matrix
            prec = _as_tensor(precision_matrix)  # type: ignore[arg-type]
            self.scale_tril = _la.cholesky(_la.inv(prec))  # type: ignore[arg-type]

        self.df = _as_tensor(df)
        dim: int = int(self.scale_tril.shape[-1])
        self._dim = dim

        # Pre-compute covariance for mean/variance.
        self._cov = self.scale_tril @ self.scale_tril.mT

        batch_shape: tuple[int, ...] = tuple(self.scale_tril.shape[:-2])
        event_shape: tuple[int, ...] = (dim, dim)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        return positive_definite

    @property
    def mean(self) -> Tensor:
        """``df · Σ``."""
        return self.df * self._cov

    @property
    def variance(self) -> Tensor:
        """``Var[W_{ij}] = df · (Σ_{ij}² + Σ_{ii} Σ_{jj})``."""
        S = self._cov
        diag = S.diagonal(dim1=-2, dim2=-1)  # (..., d)
        diag_outer = diag.unsqueeze(-1) * diag.unsqueeze(-2)  # (..., d, d)
        return self.df * (S * S + diag_outer)

    # -- sampling (Bartlett decomposition) ------------------------------------

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a sample via the Bartlett decomposition.

        Each sample is ``L A Aᵀ Lᵀ`` where ``L = scale_tril`` and ``A``
        is a random lower-triangular matrix with:

        * diagonal ``A_{ii} ~ sqrt(χ²_{df − i})``  for ``i = 0, …, d−1``,
        * off-diagonal ``A_{ij} ~ N(0, 1)``  for ``i > j``.
        """
        d: int = self._dim
        df_val: float = float(self.df.item())
        L: Tensor = self.scale_tril  # (d, d)
        dt = L.dtype
        dev = L.device

        # ── diagonal: chi-squared samples ──────────────────────────────────
        # chi²(k) = 2 · Gamma(k/2, rate=1)
        # concentrations for each diagonal position: (df−0)/2, (df−1)/2, …
        chi2_conc = lucid.tensor(
            [(df_val - i) / 2.0 for i in range(d)], dtype=dt, device=dev
        )
        # _sample_standard_gamma(conc, sample_shape) → (*sample_shape, d)
        std_g = _sample_standard_gamma(chi2_conc, sample_shape)
        sqrt_chi = (2.0 * std_g).sqrt()  # (*sample_shape, d)

        # ── off-diagonal: standard normals ──────────────────────────────────
        Z = lucid.randn(*sample_shape, d, d, dtype=dt, device=dev)
        lower_mask = lucid.tril(lucid.ones(d, d, dtype=dt, device=dev), k=-1)  # type: ignore[arg-type]
        Z_lower = Z * lower_mask  # strictly lower triangular

        # ── assemble A ──────────────────────────────────────────────────────
        A = Z_lower + lucid.diag_embed(sqrt_chi)  # (*sample_shape, d, d)

        # ── W = L A Aᵀ Lᵀ ──────────────────────────────────────────────────
        return (L @ A @ A.mT @ L.mT).detach()

    # -- log probability ──────────────────────────────────────────────────────

    def log_prob(self, value: Tensor) -> Tensor:
        """Log-density of the Wishart distribution.

        .. code::

            log p(X) = (df − d − 1)/2 · log|X|
                     − tr(Σ⁻¹ X) / 2
                     − df · d / 2 · log 2
                     − df/2 · log|Σ|
                     − log Γ_d(df/2)

        where ``Γ_d`` is the multivariate gamma function.
        """
        d: int = self._dim
        df: Tensor = self.df
        L: Tensor = self.scale_tril

        # log |X| via Cholesky (numerically stable)
        log_det_X = lucid.logdet(value)  # scalar or (...)
        t1 = (df - d - 1.0) / 2.0 * log_det_X

        # tr(Σ⁻¹ X) using Cholesky solve: Σ⁻¹ X = L⁻ᵀ (L⁻¹ X)
        # tr(AB) = (A ∘ Bᵀ).sum()  — but we compute via solve_triangular.
        # cholesky_solve(B, L) returns Σ⁻¹ B.
        # We need tr(Σ⁻¹ X) = sum of diagonal of Σ⁻¹ X.
        # Implemented as: Σ⁻¹ X = inv(Σ) @ X then trace.
        sigma_inv = _la.inv(self._cov)
        t2 = -0.5 * (sigma_inv @ value).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # log |Σ| = 2 · sum(log(diag(L)))
        log_det_sigma = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        t3 = -df / 2.0 * log_det_sigma

        # constant terms
        t4 = -df * d / 2.0 * math.log(2.0)
        t5 = -_sp.multigammaln(df / 2.0, d)

        return t1 + t2 + t3 + t4 + t5


# ── LKJCholesky ───────────────────────────────────────────────────────────────


class LKJCholesky(Distribution):
    """LKJ distribution over Cholesky factors of correlation matrices.

    ``LKJCholesky(dim, concentration=η)`` places probability proportional to
    ``det(R)^(η−1)`` over correlation matrices ``R``.  This distribution
    operates on the **lower Cholesky factor** ``L`` of ``R`` (``R = L Lᵀ``).

    When ``concentration == 1`` the distribution is uniform over Cholesky
    factors of correlation matrices.

    Parameters
    ----------
    dim : int
        Size of the square matrix (must be ≥ 2).
    concentration : Tensor | float
        Shape parameter ``η > 0``.  Larger values concentrate mass near the
        identity matrix (low correlation).

    Sampling
    --------
    Uses the vectorised Onion method from Lewandowski et al. (2009),
    §3.  Each off-diagonal element of row ``i`` is constructed so that the
    partial row has the correct norm.
    """

    arg_constraints = {"concentration": positive}
    has_rsample: bool = False

    def __init__(
        self,
        dim: int,
        concentration: Tensor | float = 1.0,
        validate_args: bool | None = None,
    ) -> None:
        if dim < 2:
            raise ValueError(f"LKJCholesky: dim must be ≥ 2, got {dim}.")
        self.dim = dim
        self.concentration = _as_tensor(concentration)

        # Pre-compute Beta parameters for the Onion sampler (vectorised).
        # marginal_conc = η + (d − 2) / 2
        d: int = dim
        eta: Tensor = self.concentration
        marginal_conc = eta + 0.5 * (d - 2)

        offset = lucid.arange(
            0, d, 1, dtype=eta.dtype, device=eta.device
        )  # [0, 1, ..., d-1]
        beta_conc1 = offset + 0.5  # (d,)
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset  # (*batch, d)

        self._beta = Beta(beta_conc1, beta_conc0)

        batch_shape: tuple[int, ...] = tuple(self.concentration.shape)
        event_shape: tuple[int, ...] = (dim, dim)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        return positive_definite  # approximate — actual support is corr-cholesky

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a sample via the vectorised Onion method (Lewandowski 2009 §3).

        Returns the lower Cholesky factor ``L`` of a random correlation matrix.
        """
        d: int = self.dim
        dtype = self.concentration.dtype
        dev = self.concentration.device

        y = self._beta.sample(sample_shape).unsqueeze(-1)  # (*s, *b, d, 1)

        # u_normal: (*s, *b, d, d) strictly lower triangular standard normal
        full_shape = self._extended_shape(sample_shape)  # (*s, *b, d, d)
        u_normal = lucid.randn(*full_shape, dtype=dtype, device=dev)
        # Zero strictly upper triangle (keep only lower triangle with k=-1)
        lower_mask = lucid.tril(lucid.ones(d, d, dtype=dtype, device=dev), k=-1)  # type: ignore[arg-type]
        u_normal = u_normal * lower_mask

        # Normalise rows to lie on the unit hypersphere (ignoring row 0 which
        # is all zeros in the lower triangle).
        row_norms = (u_normal * u_normal).sum(dim=-1, keepdim=True).sqrt()
        # Avoid division by zero on rows that are entirely zero.
        # Avoid division by zero: replace zero norms with 1.0 (those rows
        # are all zeros anyway, so the division result is still zero).
        row_norms = lucid.where(
            row_norms < 1e-10, lucid.ones_like(row_norms), row_norms
        )
        u_hypersphere = u_normal / row_norms

        # Scale: w = sqrt(y) * u_hypersphere
        w = y.sqrt() * u_hypersphere

        # Diagonal: d_i = sqrt(1 − ‖w_i‖²), clamped for numerical stability.
        w_sq_sum = (w * w).sum(dim=-1)  # (*s, *b, d)
        eps = 1e-6
        diag_elems = lucid.where(
            1.0 - w_sq_sum < eps,
            lucid.full_like(w_sq_sum, eps),
            1.0 - w_sq_sum,
        ).sqrt()

        # Assemble L = strictly-lower part + diagonal
        L = w + lucid.diag_embed(diag_elems)
        return L.detach()

    def log_prob(self, value: Tensor) -> Tensor:
        """Log-density of the LKJ distribution on a Cholesky factor.

        The density is proportional to
        ``∏_{i=2}^{d} L_{ii}^{order_i}``
        where ``order_i = 2(η − 1) + (d − i)``.

        The normaliser follows Eq. (15) of Lewandowski et al. (2009).
        """
        d: int = self.dim
        eta: Tensor = self.concentration

        # Extract diagonal elements (exclude L_{11} = 1).
        diag = value.diagonal(dim1=-2, dim2=-1)[..., 1:]  # (..., d-1)

        # order_i for i = 2, ..., d (1-indexed)
        order = lucid.arange(2, d + 1, 1, dtype=eta.dtype, device=eta.device)
        order_coeff = 2.0 * (eta - 1.0).unsqueeze(-1) + d - order  # (*batch, d-1)
        unnorm_lp = (order_coeff * diag.log()).sum(dim=-1)

        return unnorm_lp - self._log_normalizer()

    def _log_normalizer(self) -> Tensor:
        """Log of the LKJ normalisation constant.

        From the Stan functions reference (based on Lewandowski 2009, page 1999):

        .. code::

            log Z(η, d) = (d-1)*(d-2)/4 * log(π)
                        + Σ_{j=1}^{d-1} [lmgamma(η + (d-j-1)/2, 1)
                                         − lgamma(η + (d-j-1)/2)]
        """
        d: int = self.dim
        dm1 = d - 1
        eta: Tensor = self.concentration

        # alpha = η + (d - 1) / 2 (scalar part of the beta parameterisation)
        alpha = eta + 0.5 * dm1

        # numerator = mvlgamma(alpha - 0.5, dm1)
        # denominator = lgamma(alpha) * dm1
        numerator = _sp.multigammaln(alpha - 0.5, dm1)
        denominator = lucid.lgamma(alpha) * dm1

        # pi-constant adjustment
        pi_constant = 0.5 * dm1 * math.log(math.pi)

        return pi_constant + numerator - denominator
