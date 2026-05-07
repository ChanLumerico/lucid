"""``Independent`` — re-interpret rightmost batch dims of a base
distribution as event dims.

Used to turn a batch of independent univariates into a single
multivariate distribution with diagonal covariance, e.g.

    base = Normal(loc.shape == (B, D), scale.shape == (B, D))
    Independent(base, 1)  # event_shape == (D,), batch_shape == (B,)
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.distribution import Distribution


class Independent(Distribution):
    """Wrap ``base`` and treat ``reinterpreted_batch_ndims`` rightmost
    batch dimensions as event dimensions.  Sums log_probs over the
    re-interpreted axes."""

    def __init__(
        self,
        base_distribution: Distribution,
        reinterpreted_batch_ndims: int,
        validate_args: bool | None = None,
    ) -> None:
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                f"Independent: reinterpreted_batch_ndims "
                f"{reinterpreted_batch_ndims} exceeds base batch ndim "
                f"{len(base_distribution.batch_shape)}"
            )
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = int(reinterpreted_batch_ndims)
        b = tuple(base_distribution.batch_shape)
        e = tuple(base_distribution.event_shape)
        new_batch = b[: len(b) - self.reinterpreted_batch_ndims]
        new_event = b[len(b) - self.reinterpreted_batch_ndims :] + e
        super().__init__(
            batch_shape=new_batch,
            event_shape=new_event,
            validate_args=validate_args,
        )

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self.base_dist.has_rsample

    @property
    def support(self):  # type: ignore[override]
        return self.base_dist.support

    @property
    def mean(self) -> Tensor:
        return self.base_dist.mean

    @property
    def mode(self) -> Tensor:
        return self.base_dist.mode

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self.base_dist.rsample(sample_shape)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self.base_dist.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        log_p = self.base_dist.log_prob(value)
        # Sum over the rightmost ``reinterpreted_batch_ndims`` axes.
        if self.reinterpreted_batch_ndims == 0:
            return log_p
        # Lucid's sum accepts a list of dims.
        dims = list(
            range(
                log_p.ndim - self.reinterpreted_batch_ndims, log_p.ndim
            )
        )
        return log_p.sum(dim=dims)

    def entropy(self) -> Tensor:
        h = self.base_dist.entropy()
        if self.reinterpreted_batch_ndims == 0:
            return h
        dims = list(
            range(
                h.ndim - self.reinterpreted_batch_ndims, h.ndim
            )
        )
        return h.sum(dim=dims)
