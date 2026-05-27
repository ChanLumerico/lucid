"""Composition transforms (Albumentations-compatible).

Containers that orchestrate child transforms: ``OneOf``, ``SomeOf``,
``Sequential``, ``OneOrOther``.  A selected child is always applied (its
own ``p`` acts as a selection *weight* for ``OneOf``), so children run
with their probability gate bypassed.  Each child handles its own
multi-target dispatch, so composition works on tensors or full samples.
"""

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms._base import (
    Empty,
    Transform,
    TransformLike,
    _NoParams,
)


def _force(tf: TransformLike, inputs: object) -> object:
    """Apply ``tf`` ignoring its probability gate (restore ``p`` after)."""
    old = getattr(tf, "p", 1.0)
    if isinstance(tf, Transform):
        tf.p = 1.0
        try:
            return tf(inputs)
        finally:
            tf.p = old
    return tf(inputs)


class _Container(_NoParams, Transform[Empty]):
    """Base for composition transforms (children do the real work)."""

    def __init__(self, transforms: list[TransformLike], p: float = 1.0) -> None:
        super().__init__(p=p)
        self.transforms = list(transforms)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:  # unused
        return img

    def _gate_or_pass(self, inputs: object) -> bool:
        """Return True if the container should run this call."""
        return not (self.p < 1.0 and _random.rand() >= self.p)


class OneOf(_Container):
    r"""Apply exactly one child, chosen by child-``p`` weight (Albumentations ``OneOf``)."""

    def __init__(self, transforms: list[TransformLike], p: float = 0.5) -> None:
        super().__init__(transforms, p=p)
        weights = [float(getattr(t, "p", 1.0)) for t in self.transforms]
        total = sum(weights)
        self._weights = [w / total for w in weights] if total > 0 else None

    def __call__(self, inputs: object) -> object:
        if not self.transforms or not self._gate_or_pass(inputs):
            return inputs
        r = _random.rand()
        acc = 0.0
        chosen = self.transforms[-1]
        weights = self._weights or [1.0 / len(self.transforms)] * len(self.transforms)
        for tf, w in zip(self.transforms, weights):
            acc += w
            if r <= acc:
                chosen = tf
                break
        return _force(chosen, inputs)

    def __repr__(self) -> str:
        return f"OneOf({self.transforms}, p={self.p})"


class SomeOf(_Container):
    r"""Apply ``n`` randomly-chosen children (Albumentations ``SomeOf``).

    Parameters
    ----------
    transforms : list of Transform
    n : int
        Number of children to apply (without replacement).
    p : float, optional, default=1.0
    """

    def __init__(
        self, transforms: list[TransformLike], n: int = 1, p: float = 1.0
    ) -> None:
        super().__init__(transforms, p=p)
        self.n = n

    def __call__(self, inputs: object) -> object:
        if not self.transforms or not self._gate_or_pass(inputs):
            return inputs
        pool = list(range(len(self.transforms)))
        picks: list[int] = []
        for _ in range(min(self.n, len(pool))):
            k = _random.randint(0, len(pool))
            picks.append(pool.pop(k))
        picks.sort()
        for i in picks:
            inputs = _force(self.transforms[i], inputs)
        return inputs

    def __repr__(self) -> str:
        return f"SomeOf({self.transforms}, n={self.n}, p={self.p})"


class Sequential(_Container):
    r"""Apply all children in order, gated by ``p`` (Albumentations ``Sequential``)."""

    def __init__(self, transforms: list[TransformLike], p: float = 1.0) -> None:
        super().__init__(transforms, p=p)

    def __call__(self, inputs: object) -> object:
        if not self._gate_or_pass(inputs):
            return inputs
        for tf in self.transforms:
            inputs = tf(inputs)
        return inputs

    def __repr__(self) -> str:
        return f"Sequential({self.transforms}, p={self.p})"


class OneOrOther(_Container):
    r"""Apply ``first`` with probability ``p`` else ``second`` (Albumentations ``OneOrOther``)."""

    def __init__(
        self, first: TransformLike, second: TransformLike, p: float = 0.5
    ) -> None:
        super().__init__([first, second], p=1.0)
        self.first = first
        self.second = second
        self.switch = p

    def __call__(self, inputs: object) -> object:
        chosen = self.first if _random.rand() < self.switch else self.second
        return _force(chosen, inputs)

    def __repr__(self) -> str:
        return f"OneOrOther(p={self.switch})"
