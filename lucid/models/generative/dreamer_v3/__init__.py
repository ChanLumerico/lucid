"""DreamerV3 family — Hafner et al., Nature 640 (2025)."""

from lucid.models.generative.dreamer_v3._config import (
    DREAMER_V3_SIZES,
    DreamerV3Config,
)
from lucid.models.generative.dreamer_v3._heads import TwoHotHead
from lucid.models.generative.dreamer_v3._model import (
    DreamerV3BehaviorOutput,
    DreamerV3ForWorldModeling,
    DreamerV3Model,
    DreamerV3Output,
)
from lucid.models.generative.dreamer_v3._objectives import (
    ReturnNormaliser,
    free_bits_kl,
)
from lucid.models.generative.dreamer_v3._pretrained import (
    dreamer_v3_12m,
    dreamer_v3_12m_world_model,
    dreamer_v3_25m,
    dreamer_v3_25m_world_model,
    dreamer_v3_50m,
    dreamer_v3_50m_world_model,
    dreamer_v3_100m,
    dreamer_v3_100m_world_model,
    dreamer_v3_200m,
    dreamer_v3_200m_world_model,
    dreamer_v3_400m,
    dreamer_v3_400m_world_model,
)

__all__ = [
    "DreamerV3Config",
    "DREAMER_V3_SIZES",
    "DreamerV3Model",
    "DreamerV3ForWorldModeling",
    "DreamerV3Output",
    "DreamerV3BehaviorOutput",
    "TwoHotHead",
    "free_bits_kl",
    "ReturnNormaliser",
    "dreamer_v3_12m",
    "dreamer_v3_12m_world_model",
    "dreamer_v3_25m",
    "dreamer_v3_25m_world_model",
    "dreamer_v3_50m",
    "dreamer_v3_50m_world_model",
    "dreamer_v3_100m",
    "dreamer_v3_100m_world_model",
    "dreamer_v3_200m",
    "dreamer_v3_200m_world_model",
    "dreamer_v3_400m",
    "dreamer_v3_400m_world_model",
]
