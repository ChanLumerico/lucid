"""``MultivariateNormal`` parameterised by a Cholesky factor."""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    real,
)
from lucid.distributions.distribution import Distribution


class MultivariateNormal(Distribution):
    r"""Multivariate Normal (Gaussian) distribution in :math:`\mathbb{R}^D`.

    ``MultivariateNormal(loc=μ, ...)`` defines the :math:`D`-dimensional
    Gaussian with mean vector :math:`\mu` and a covariance structure
    expressed in one of three equivalent forms.  Internally all forms are
    converted to the lower-triangular Cholesky factor :math:`L` of
    :math:`\Sigma` (i.e. :math:`\Sigma = L L^\top`), which is the numerically
    preferred representation for both sampling and log-probability evaluation.

    Specify **exactly one** of:

    - ``covariance_matrix`` :math:`\Sigma` (positive-definite, :math:`D \times D`),
    - ``precision_matrix`` :math:`\Sigma^{-1}` (positive-definite, inverted via
      Cholesky internally), or
    - ``scale_tril`` :math:`L` (lower-triangular with positive diagonal).

    Parameters
    ----------
    loc : Tensor
        Mean vector :math:`\mu \in \mathbb{R}^D` of shape ``(..., D)``.
    covariance_matrix : Tensor | None, optional
        Full covariance matrix :math:`\Sigma` of shape ``(..., D, D)``.
    precision_matrix : Tensor | None, optional
        Precision matrix :math:`\Sigma^{-1}` of shape ``(..., D, D)``.
    scale_tril : Tensor | None, optional
        Lower-triangular Cholesky factor :math:`L` of shape ``(..., D, D)``
        with positive diagonal entries.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    loc : Tensor
        Mean vector :math:`\mu`.
    scale_tril : Tensor
        Cholesky factor :math:`L` (always populated, regardless of which
        parameterisation was used for construction).

    Notes
    -----
    **PDF**:

    .. math::

        p(x; \mu, \Sigma) = (2\pi)^{-D/2} |\Sigma|^{-1/2}
        \exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)

    **Log-PDF** (numerically stable via Cholesky, avoiding explicit inversion):

    .. math::

        \log p(x) = -\tfrac{1}{2}\|L^{-1}(x-\mu)\|^2
                   - \sum_i \log L_{ii}
                   - \tfrac{D}{2}\log(2\pi)

    where :math:`\|L^{-1}(x-\mu)\|^2` is the squared **Mahalanobis distance**
    computed via triangular solve (no explicit matrix inversion).

    **Moments**:

    - Mean: :math:`E[X] = \mu`
    - Mode: :math:`\mu` (Gaussian is unimodal)
    - Marginal variances: diagonal of :math:`\Sigma = L L^\top`

    **Entropy**:

    .. math::

        H[X] = \tfrac{D}{2}(1 + \log(2\pi)) + \sum_i \log L_{ii}

    **Reparameterised sampling** uses the Cholesky factorisation:

    .. math::

        X = \mu + L \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_D)

    Gradients propagate through :math:`\mu` and :math:`L` unobstructed.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import MultivariateNormal
    >>> # 2-d isotropic Gaussian
    >>> dist = MultivariateNormal(
    ...     loc=lucid.zeros(2),
    ...     covariance_matrix=lucid.eye(2),
    ... )
    >>> samples = dist.rsample((50,))
    >>> samples.shape  # (50, 2)
    (50, 2)
    >>> # Log-prob at the mean (maximum)
    >>> dist.log_prob(lucid.zeros(2))
    """

    arg_constraints = {"loc": real}
    support: Constraint | None = real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor,
        covariance_matrix: Tensor | None = None,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a MultivariateNormal distribution.

        Exactly one of ``covariance_matrix``, ``precision_matrix``, or
        ``scale_tril`` must be provided.  All parameterisations are
        internally converted to the lower-triangular Cholesky factor ``L``
        via :func:`lucid.linalg.cholesky` or triangular inversion.

        Parameters
        ----------
        loc : Tensor
            Mean vector :math:`\mu \in \mathbb{R}^D` of shape ``(..., D)``.
        covariance_matrix : Tensor | None, optional
            Full covariance matrix :math:`\Sigma` of shape ``(..., D, D)``.
        precision_matrix : Tensor | None, optional
            Precision matrix :math:`\Sigma^{-1}` of shape ``(..., D, D)``.
        scale_tril : Tensor | None, optional
            Lower-triangular Cholesky factor :math:`L` of shape ``(..., D, D)``
            with positive diagonal.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If not exactly one of the covariance parameterisations is given.
        """
        n_set = sum(
            x is not None for x in (covariance_matrix, precision_matrix, scale_tril)
        )
        if n_set != 1:
            raise ValueError(
                "MultivariateNormal: exactly one of covariance_matrix / "
                "precision_matrix / scale_tril must be provided."
            )
        self.loc = loc
        if scale_tril is not None:
            self.scale_tril = scale_tril
        elif covariance_matrix is not None:
            self.scale_tril = lucid.linalg.cholesky(covariance_matrix)
        else:
            assert precision_matrix is not None
            # P = Lᵀ⁻¹ · L⁻¹  ⇒  L = (chol(P)⁻ᵀ).
            l_p = lucid.linalg.cholesky(precision_matrix)
            self.scale_tril = lucid.linalg.inv(l_p).mT  # type: ignore[attr-defined]

        D = int(self.loc.shape[-1])
        self._D = D
        super().__init__(
            batch_shape=tuple(self.loc.shape[:-1]),
            event_shape=(D,),
            validate_args=validate_args,
        )

    @property
    def covariance_matrix(self) -> Tensor:
        r"""Full covariance matrix :math:`\Sigma = L L^\top`.

        Returns
        -------
        Tensor
            Positive-definite covariance matrix of shape ``(*batch_shape, D, D)``.
        """
        return self.scale_tril @ self.scale_tril.mT

    @property
    def mean(self) -> Tensor:
        r"""Mean of the MultivariateNormal: :math:`E[X] = \mu`.

        Returns
        -------
        Tensor
            Mean vector of shape ``(*batch_shape, D)``.
        """
        return self.loc

    @property
    def mode(self) -> Tensor:
        r"""Mode of the MultivariateNormal: equal to the mean :math:`\mu`.

        The Gaussian is unimodal; its unique maximum is at the mean.

        Returns
        -------
        Tensor
            Mode vector of shape ``(*batch_shape, D)``.
        """
        return self.loc

    @property
    def variance(self) -> Tensor:
        r"""Marginal variances — diagonal entries of :math:`\Sigma = LL^\top`.

        Returns
        -------
        Tensor
            Variance vector of shape ``(*batch_shape, D)``.
        """
        return (self.scale_tril * self.scale_tril).sum(dim=-1)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples via the Cholesky factorisation.

        .. math::

            X = \mu + L \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_D)

        Gradients propagate through both :math:`\mu` and :math:`L`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples of shape ``(*sample_shape, *batch_shape, D)``.
        """
        shape = self._extended_shape(sample_shape)
        eps = lucid.randn(*shape, dtype=self.loc.dtype, device=self.loc.device)
        # x = loc + L · eps  — matmul with the Cholesky factor.
        return self.loc + (self.scale_tril @ eps.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the MultivariateNormal distribution.

        Computed via the Cholesky factor to avoid explicit matrix inversion:

        .. math::

            \log p(x) = -\tfrac{1}{2}\|L^{-1}(x-\mu)\|^2
                        - \sum_i \log L_{ii}
                        - \tfrac{D}{2}\log(2\pi)

        Parameters
        ----------
        value : Tensor
            Observation vectors of shape ``(*batch_shape, D)``.

        Returns
        -------
        Tensor
            Log-density values of shape ``batch_shape``.
        """
        # diff = (value − loc),  M = L⁻¹ diff,  log p = − 0.5 ‖M‖² − sum(log diag(L)) − 0.5 D log(2π)
        diff = (value - self.loc).unsqueeze(-1)
        M = lucid.linalg.solve_triangular(self.scale_tril, diff, upper=False)
        sq = (M * M).sum(dim=(-2, -1))
        diag = self.scale_tril.diagonal(dim1=-2, dim2=-1)
        log_det = diag.log().sum(dim=-1)
        return -0.5 * sq - log_det - 0.5 * self._D * math.log(2.0 * math.pi)

    def entropy(self) -> Tensor:
        r"""Entropy of the MultivariateNormal distribution.

        .. math::

            H[X] = \tfrac{D}{2}(1 + \log(2\pi)) + \sum_i \log L_{ii}

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        diag = self.scale_tril.diagonal(dim1=-2, dim2=-1)
        log_det = diag.log().sum(dim=-1)
        return 0.5 * self._D * (1.0 + math.log(2.0 * math.pi)) + log_det
