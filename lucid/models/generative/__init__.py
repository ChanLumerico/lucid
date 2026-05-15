"""Image-generative model families — Phase 5 of the model zoo.

Concrete families (VAE / DDPM / NCSN) will populate this package in follow-up
commits.  The infrastructure exported here — base configs, output dataclasses,
noise schedulers — is in place so each family only needs to add its own
``_config.py``, ``_model.py``, and ``_pretrained.py``.

Positional / timestep encoding primitives (``SinusoidalEmbedding``,
``TimestepEmbedding``) live in :mod:`lucid.nn`; family code imports them
from there rather than redefining locally.
"""

from lucid.models.generative._config import (
    BetaSchedule,
    DiffusionModelConfig,
    GenerativeActivation,
    GenerativeModelConfig,
)
from lucid.models.generative._schedulers import DDPMScheduler, DiffusionScheduler

__all__ = [
    "BetaSchedule",
    "DDPMScheduler",
    "DiffusionModelConfig",
    "GenerativeActivation",
    "GenerativeModelConfig",
    "DiffusionScheduler",
]
