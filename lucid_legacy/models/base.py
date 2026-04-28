from typing import Self

import lucid.nn as nn

from lucid.weights import WeightEntry, apply


class PreTrainedModelMixin:
    def from_pretrained(self, weights: WeightEntry, strict: bool = True) -> Self:
        if not isinstance(self, nn.Module):
            raise RuntimeError(
                "from_pretrained must be called from 'nn.Module' instances."
            )
        apply(self, weights, strict=strict)
        return self
