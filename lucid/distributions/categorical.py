"""``Categorical`` (discrete with K outcomes) and ``OneHotCategorical``."""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    integer_interval,
    real,
    simplex,
)
from lucid.distributions.distribution import Distribution


from lucid.distributions._util import as_tensor as _as_tensor


def _normalize_probs(probs: Tensor) -> Tensor:
    """Project a non-negative tensor onto the K-simplex along the last
    dim (matches the reference framework's preprocessing)."""
    return probs / probs.sum(dim=-1, keepdim=True)


class Categorical(Distribution):
    """Distribution over ``{0, 1, ..., K−1}`` parameterised by ``probs``
    (each row a probability vector) or by ``logits`` (un-normalised log
    probs).  Specify exactly one."""

    arg_constraints = {"probs": simplex, "logits": real}
    has_enumerate_support = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Categorical: pass exactly one of `probs` or `logits`.")
        if probs is not None:
            self.probs = _normalize_probs(_as_tensor(probs))
            self._is_logits = False
            shape = tuple(self.probs.shape)
        else:
            self.logits = _as_tensor(logits)  # type: ignore[arg-type]
            self._is_logits = True
            shape = tuple(self.logits.shape)
        self._num_events = shape[-1]
        super().__init__(
            batch_shape=shape[:-1],
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        return integer_interval(0, self._num_events - 1)

    @property
    def _log_probs(self) -> Tensor:
        if self._is_logits:
            from lucid.nn.functional.activations import log_softmax

            return log_softmax(self.logits, dim=-1)
        return self.probs.log()

    @property
    def _probs(self) -> Tensor:
        if self._is_logits:
            from lucid.nn.functional.activations import softmax

            return softmax(self.logits, dim=-1)
        return self.probs

    @property
    def mean(self) -> Tensor:
        # mean of Categorical isn't well-defined (no metric on labels)
        # but reference framework returns NaN with the right shape.
        return lucid.full(
            self._batch_shape,
            float("nan"),
            device=self._probs.device,
            dtype=self._probs.dtype,
        )

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # Gumbel-max trick, no replacement: draw G ~ −log(−log U), pick argmax.
        shape = tuple(sample_shape) + tuple(self._batch_shape) + (self._num_events,)
        u = lucid.rand(*shape, dtype=self._probs.dtype, device=self._probs.device)
        u = u.clip(1e-7, 1.0 - 1e-7)
        gumbel = -(-(u.log())).log()
        scores = self._log_probs + gumbel
        return scores.argmax(dim=-1).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        log_p = self._log_probs
        # Reshape value so it matches log_p[:-1] exactly, then add a
        # trailing length-1 axis for the ``gather`` index.
        target_shape = tuple(log_p.shape[:-1])
        v = value
        if tuple(v.shape) != target_shape:
            if v.ndim == 0 or v.shape == (1,):
                v = lucid.full(
                    target_shape,
                    float(v.item()),
                    dtype=v.dtype,
                    device=v.device,
                )
            else:
                v = v + lucid.zeros(target_shape, dtype=v.dtype, device=v.device)
        v_long = v.to(lucid.int64).unsqueeze(-1)
        gathered = lucid.gather(log_p, v_long, dim=-1)
        return gathered.squeeze(-1)

    def entropy(self) -> Tensor:
        log_p = self._log_probs
        return -(self._probs * log_p).sum(dim=-1)


class OneHotCategorical(Distribution):
    """``Categorical`` with one-hot samples.  Useful for relaxations and
    REINFORCE-style training."""

    arg_constraints = {"probs": simplex, "logits": real}

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        self._cat = Categorical(probs=probs, logits=logits, validate_args=False)
        if probs is not None:
            self.probs = self._cat.probs
        else:
            self.logits = self._cat.logits
        super().__init__(
            batch_shape=tuple(self._cat._batch_shape),
            event_shape=(self._cat._num_events,),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        return simplex

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        idx = self._cat.sample(sample_shape)
        from lucid.nn.functional.sparse import one_hot

        return one_hot(idx, num_classes=self._cat._num_events).to(
            self._cat._probs.dtype
        )

    def log_prob(self, value: Tensor) -> Tensor:
        # value is one-hot — log_prob = sum(value * log_probs).
        return (value * self._cat._log_probs).sum(dim=-1)

    def entropy(self) -> Tensor:
        return self._cat.entropy()
