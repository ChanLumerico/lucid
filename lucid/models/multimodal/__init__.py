"""Models whose input spans more than one modality.

The domain exists because the alternative is a lie about where a model
belongs.  A CLIP filed under ``vision`` would hide a text Transformer
that is half its parameters and all of its transfer story; filed under
``text`` it would hide the image tower.  The zoo classifies by
structure, and two encoders trained to land in one representation are a
structure neither single-modality domain describes.

Concrete families live in sub-packages (``clip``).  What is exported
here is the infrastructure a new family builds on — the base config and
the tower blocks — so a second family needs only its own ``_config.py``,
``_model.py`` and ``_pretrained.py``, which is the arrangement
:mod:`lucid.models.generative` and :mod:`lucid.models.text` already use.

The blocks here are deliberately not in :mod:`lucid.nn`.  They encode a
particular pre-norm-with-``QuickGELU`` shape because that is what the
released contrastive checkpoints were trained in;
:class:`lucid.nn.TransformerEncoderLayer` is the general form and will
not load those weights.
"""

from lucid.models.multimodal._config import (
    MultimodalActivation,
    MultimodalModelConfig,
)
from lucid.models.multimodal._towers import (
    QuickGELU,
    ResidualAttentionBlock,
    TransformerTower,
)

__all__ = [
    "MultimodalActivation",
    "MultimodalModelConfig",
    "QuickGELU",
    "ResidualAttentionBlock",
    "TransformerTower",
]
