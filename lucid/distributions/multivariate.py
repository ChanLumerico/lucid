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
    """Multivariate Gaussian with mean ``loc`` (..., D) and covariance
    expressed as the lower-triangular Cholesky factor ``scale_tril``
    (..., D, D).

    Specify exactly one of ``covariance_matrix``, ``precision_matrix``,
    or ``scale_tril`` — Lucid converts to ``scale_tril`` internally
    (the only form the rsample / log_prob math actually uses).
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
        n_set = sum(
            x is not None
            for x in (covariance_matrix, precision_matrix, scale_tril)
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
            self.scale_tril = lucid.linalg.inv(l_p).mT

        D = int(self.loc.shape[-1])
        self._D = D
        super().__init__(
            batch_shape=tuple(self.loc.shape[:-1]),
            event_shape=(D,),
            validate_args=validate_args,
        )

    @property
    def covariance_matrix(self) -> Tensor:
        return self.scale_tril @ self.scale_tril.mT

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return (self.scale_tril * self.scale_tril).sum(dim=-1)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = lucid.randn(*shape, dtype=self.loc.dtype, device=self.loc.device)
        # x = loc + L · eps  — matmul with the Cholesky factor.
        return self.loc + (self.scale_tril @ eps.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        # diff = (value − loc),  M = L⁻¹ diff,  log p = − 0.5 ‖M‖² − sum(log diag(L)) − 0.5 D log(2π)
        diff = (value - self.loc).unsqueeze(-1)
        M = lucid.linalg.solve_triangular(self.scale_tril, diff, upper=False)
        sq = (M * M).sum(dim=(-2, -1))
        diag = self.scale_tril.diagonal(dim1=-2, dim2=-1)
        log_det = diag.log().sum(dim=-1)
        return -0.5 * sq - log_det - 0.5 * self._D * math.log(2.0 * math.pi)

    def entropy(self) -> Tensor:
        diag = self.scale_tril.diagonal(dim1=-2, dim2=-1)
        log_det = diag.log().sum(dim=-1)
        return 0.5 * self._D * (1.0 + math.log(2.0 * math.pi)) + log_det
