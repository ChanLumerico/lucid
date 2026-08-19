"""CLIP family — Radford et al., ICML 2021."""

from lucid.models.multimodal.clip._config import CLIPConfig
from lucid.models.multimodal.clip._model import (
    CLIP,
    CLIPForZeroShotImageClassification,
    CLIPOutput,
    CLIPZeroShotOutput,
)
from lucid.models.multimodal.clip._weights import (
    CLIPViTBase16Weights,
    CLIPViTBase32Weights,
    CLIPViTLarge14_336Weights,
    CLIPViTLarge14Weights,
)
from lucid.models.multimodal.clip._tokenizer import (
    CLIP_EOS,
    CLIP_SOS,
    CLIPTokenizer,
    CLIPTokenizerFast,
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
    "CLIPTokenizer",
    "CLIPTokenizerFast",
    "CLIP_SOS",
    "CLIP_EOS",
    "CLIPViTBase32Weights",
    "CLIPViTBase16Weights",
    "CLIPViTLarge14Weights",
    "CLIPViTLarge14_336Weights",
    "clip_vit_base_32",
    "clip_vit_base_16",
    "clip_vit_large_14",
    "clip_vit_large_14_336",
    "clip_vit_base_32_zero_shot",
    "clip_vit_base_16_zero_shot",
    "clip_vit_large_14_zero_shot",
    "clip_vit_large_14_336_zero_shot",
]
