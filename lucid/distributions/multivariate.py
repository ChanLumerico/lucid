"""``MultivariateNormal`` parameterised by a Cholesky factor."""

import math
from typing import override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import _lazy_param
from lucid.distributions.constraints import (
    Constraint,
    lower_cholesky,
    positive_definite,
    real_vector,
)
from lucid.distributions.distribution import Distribution


def _precision_to_scale_tril(precision: Tensor) -> Tensor:
    r"""The *lower*-triangular :math:`L` with :math:`L L^\top = P^{-1}`.

    The direct route — ``inv(chol(P)).mT`` — gives an *upper*-triangular
    factor: from :math:`P = C C^\top`,
    :math:`\Sigma = P^{-1} = C^{-\top} C^{-1}`, and :math:`C^{-\top}` is
    upper.  It squares back to :math:`\Sigma` but every consumer of
    ``scale_tril`` (the lower-triangular solve in :meth:`log_prob`, the KL
    pair) reads it as lower, so the density came out wrong.

    Reversing both axes turns the lower Cholesky factor of the flipped
    matrix into an upper factor :math:`U` with :math:`P = U U^\top`; then
    :math:`\Sigma = U^{-\top} U^{-1}` and :math:`L = U^{-\top}` is lower,
    obtained by one triangular solve of :math:`U^\top L = I`.
    """
    flipped = lucid.linalg.cholesky(lucid.flip(precision, (-2, -1)))
    l_inv = lucid.flip(flipped, (-2, -1)).mT
    eye = lucid.eye(
        int(precision.shape[-1]), dtype=precision.dtype, device=precision.device
    )
    eye = eye.broadcast_to(tuple(l_inv.shape))
    return lucid.linalg.solve_triangular(l_inv, eye, upper=False)


def _scale_tril_of(
    covariance_matrix: Tensor | None, precision_matrix: Tensor | None
) -> Tensor:
    """The lower Cholesky factor of the covariance given by one of the two.

    Shared by every family parameterised by a covariance, a precision or a
    ``scale_tril``.  Called only after the given matrix has been validated,
    so a matrix that is not positive-definite is refused by validation with
    a ``ValueError`` and reaches the factorisation only when validation is
    off — where :class:`lucid.linalg.LinAlgError` reports it.
    """
    if covariance_matrix is not None:
        return lucid.linalg.cholesky(covariance_matrix)
    if precision_matrix is None:
        raise ValueError("pass a covariance_matrix or a precision_matrix")
    return _precision_to_scale_tril(precision_matrix)


def _precision_of(scale_tril: Tensor) -> Tensor:
    r"""The precision :math:`(L L^\top)^{-1} = L^{-\top} L^{-1}`."""
    eye = lucid.eye(
        int(scale_tril.shape[-1]), dtype=scale_tril.dtype, device=scale_tril.device
    ).broadcast_to(tuple(scale_tril.shape))
    l_inv = lucid.linalg.solve_triangular(scale_tril, eye, upper=False)
    return l_inv.mT @ l_inv


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
    >>> # Log-prob at the mean (maximum): -log(2π)
    >>> dist.log_prob(lucid.zeros(2))
    tensor(-1.838)
    """

    arg_constraints = {
        "loc": real_vector,
        "covariance_matrix": positive_definite,
        "precision_matrix": positive_definite,
        "scale_tril": lower_cholesky,
    }
    support: Constraint | None = real_vector
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
            If not exactly one of the covariance parameterisations is given,
            or — under validation — the one given is not positive-definite
            (a ``scale_tril``: not lower-triangular with a positive
            diagonal).
        lucid.linalg.LinAlgError
            With validation off, if a covariance or precision matrix has no
            Cholesky factorisation.
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
        # The matrix given is stored under its own name, so validation checks
        # it before anything factorises it.
        if scale_tril is not None:
            self.scale_tril = scale_tril
        elif covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        elif precision_matrix is not None:
            self.precision_matrix = precision_matrix

        D = int(self.loc.shape[-1])
        self._D = D
        super().__init__(
            batch_shape=tuple(self.loc.shape[:-1]),
            event_shape=(D,),
            validate_args=validate_args,
        )
        if scale_tril is None:
            self.scale_tril = _scale_tril_of(covariance_matrix, precision_matrix)

    @_lazy_param
    def covariance_matrix(self) -> Tensor:
        r"""Full covariance matrix :math:`\Sigma = L L^\top` — as given, or
        derived from ``scale_tril`` on access.

        Returns
        -------
        Tensor
            Positive-definite covariance matrix of shape ``(*batch_shape, D, D)``.
        """
        return self.scale_tril @ self.scale_tril.mT

    @_lazy_param
    def precision_matrix(self) -> Tensor:
        r"""Precision matrix :math:`\Sigma^{-1}` — as given, or derived from
        ``scale_tril`` on access.

        Returns
        -------
        Tensor
            Positive-definite precision matrix of shape
            ``(*batch_shape, D, D)``.
        """
        return _precision_of(self.scale_tril)

    @override
    @property
    def mean(self) -> Tensor:
        r"""Mean of the MultivariateNormal: :math:`E[X] = \mu`.

        Returns
        -------
        Tensor
            Mean vector of shape ``(*batch_shape, D)``.
        """
        return self.loc

    @override
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

    @override
    @property
    def variance(self) -> Tensor:
        r"""Marginal variances — diagonal entries of :math:`\Sigma = LL^\top`.

        Returns
        -------
        Tensor
            Variance vector of shape ``(*batch_shape, D)``.
        """
        return (self.scale_tril * self.scale_tril).sum(dim=-1)

    @override
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

    @override
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
        # diff = (value − loc),  M = L⁻¹ diff
        # log p = − 0.5 ‖M‖² − sum(log diag(L)) − 0.5 D log(2π)
        diff = (value - self.loc).unsqueeze(-1)
        M = lucid.linalg.solve_triangular(self.scale_tril, diff, upper=False)
        sq = (M * M).sum(dim=(-2, -1))
        diag = self.scale_tril.diagonal(dim1=-2, dim2=-1)
        log_det = diag.log().sum(dim=-1)
        return -0.5 * sq - log_det - 0.5 * self._D * math.log(2.0 * math.pi)

    @override
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
