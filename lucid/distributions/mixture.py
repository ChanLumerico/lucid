"""``MixtureSameFamily`` — finite mixture of identically-typed components."""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.categorical import Categorical
from lucid.distributions.distribution import Distribution


class MixtureSameFamily(Distribution):
    """Mixture of identically-typed component distributions.

    ``mixture_distribution`` (a :class:`Categorical` over ``K`` components)
    selects which component generates each sample;
    ``component_distribution`` is a single distribution whose rightmost
    batch dim has size ``K``.

    Sampling is non-reparameterised — drawing a categorical index breaks
    differentiability through the mixture weights.
    """

    def __init__(
        self,
        mixture_distribution: Categorical,
        component_distribution: Distribution,
        validate_args: bool | None = None,
    ) -> None:
        if not isinstance(mixture_distribution, Categorical):
            raise ValueError(
                "MixtureSameFamily: mixture_distribution must be Categorical"
            )
        # Component dist's rightmost batch dim is the number of components.
        comp_batch = tuple(component_distribution.batch_shape)
        if len(comp_batch) == 0 or comp_batch[-1] != mixture_distribution._num_events:
            raise ValueError(
                f"MixtureSameFamily: component_distribution.batch_shape "
                f"{comp_batch} must end with K={mixture_distribution._num_events}"
            )
        self.mixture_distribution = mixture_distribution
        self.component_distribution = component_distribution
        # Drop the K dim from the batch shape.
        new_batch = comp_batch[:-1]
        super().__init__(
            batch_shape=new_batch,
            event_shape=tuple(component_distribution.event_shape),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        # E[X] = Σ_k π_k · μ_k along the K dim.
        probs: Tensor = self.mixture_distribution._probs  # (..., K)
        comp_mean: Tensor = self.component_distribution.mean  # (..., K, *event)
        # Multiply along the K axis.  ``comp_mean`` may have trailing
        # event dims; broadcast ``probs`` accordingly.
        weight = probs.reshape(list(probs.shape) + [1] * len(self._event_shape))
        return (weight * comp_mean).sum(dim=-1 - len(self._event_shape))

    @property
    def variance(self) -> Tensor:
        # Var = E[Var(X|k)] + Var(E[X|k]) — law of total variance.
        probs: Tensor = self.mixture_distribution._probs
        comp_mean: Tensor = self.component_distribution.mean
        comp_var: Tensor = self.component_distribution.variance
        weight = probs.reshape(list(probs.shape) + [1] * len(self._event_shape))
        ax = -1 - len(self._event_shape)
        mean_of_mean = (weight * comp_mean).sum(dim=ax, keepdim=True)
        within: Tensor = (weight * comp_var).sum(dim=ax)
        between: Tensor = (weight * (comp_mean - mean_of_mean) ** 2).sum(dim=ax)
        return within + between

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # 1. Draw component indices from the mixture categorical.
        comp_idx: Tensor = self.mixture_distribution.sample(sample_shape)
        # 2. Draw one sample per component for the same shape, then gather
        #    along the K axis.  Cheap when K is modest; exact otherwise.
        comp_samples: Tensor = self.component_distribution.sample(sample_shape)
        # comp_samples shape: (*sample, *batch, K, *event); pick along K.
        ax: int = comp_samples.ndim - 1 - len(self._event_shape)
        idx_unsq: Tensor = comp_idx.to(lucid.int64)
        # Expand idx to match comp_samples shape (insert event dims as 1
        # then broadcast).
        idx_shape: list[int] = list(idx_unsq.shape) + [1] + [1] * len(self._event_shape)
        idx_b: Tensor = (
            idx_unsq.reshape(idx_shape)
            .broadcast_to(
                list(comp_samples.shape[:ax]) + [1] + list(comp_samples.shape[ax + 1 :])
            )
            .contiguous()
        )
        gathered: Tensor = lucid.gather(comp_samples, idx_b, ax)
        return gathered.squeeze(ax)

    def log_prob(self, value: Tensor) -> Tensor:
        # log p(x) = logsumexp_k [ log π_k + log p_k(x) ].
        log_pi: Tensor = self.mixture_distribution._log_probs  # (..., K)
        # Insert K-axis into value so component_distribution sees it as a
        # broadcast input over K.  Value shape: (..., *event); we want
        # (..., 1, *event) so the component log_prob produces (..., K, *event_collapsed).
        v_unsq: Tensor = value.reshape(
            list(value.shape[: value.ndim - len(self._event_shape)])
            + [1]
            + list(value.shape[value.ndim - len(self._event_shape) :])
        )
        log_pk: Tensor = self.component_distribution.log_prob(v_unsq)
        return lucid.logsumexp(log_pi + log_pk, dim=-1)
