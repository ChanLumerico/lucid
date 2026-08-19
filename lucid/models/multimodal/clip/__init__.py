"""CLIP family — Radford et al., ICML 2021."""

from lucid.models.multimodal.clip._config import CLIPConfig
from lucid.models.multimodal.clip._model import (
    CLIP,
    CLIPForZeroShotImageClassification,
    CLIPOutput,
    CLIPZeroShotOutput,
)
from lucid.models.multimodal.clip._pretrained import (
    clip_vit_base_16,
    clip_vit_base_16_zero_shot,
    clip_vit_base_32,
    clip_vit_base_32_zero_shot,
    clip_vit_large_14,
    clip_vit_large_14_336,
    clip_vit_large_14_336_zero_shot,
    clip_vit_large_14_zero_shot,
)

__all__ = [
    "CLIPConfig",
    "CLIP",
    "CLIPForZeroShotImageClassification",
    "CLIPOutput",
    "CLIPZeroShotOutput",
    "clip_vit_base_32",
    "clip_vit_base_16",
    "clip_vit_large_14",
    "clip_vit_large_14_336",
    "clip_vit_base_32_zero_shot",
    "clip_vit_base_16_zero_shot",
    "clip_vit_large_14_zero_shot",
    "clip_vit_large_14_336_zero_shot",
]
