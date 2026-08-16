"""DreamerV3 family — Hafner et al., 2023.

Under construction: the configuration and the distributional heads are in
place; the model and its factories are not.  What is exported here is
what exists, so the family contract has something to check rather than an
empty package that reads as a broken family.
"""

from lucid.models.generative.dreamer_v3._config import (
    DREAMER_V3_SIZES,
    DreamerV3Config,
)
from lucid.models.generative.dreamer_v3._heads import TwoHotHead
from lucid.models.generative.dreamer_v3._objectives import (
    ReturnNormaliser,
    free_bits_kl,
)

__all__ = [
    "DreamerV3Config",
    "DREAMER_V3_SIZES",
    "TwoHotHead",
    "free_bits_kl",
    "ReturnNormaliser",
]
