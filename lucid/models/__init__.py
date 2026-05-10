"""Lucid 3.0 model zoo — public surface.

See ``obsidian/roadmap/roadmap-models-zoo-v3.md`` for the design.
"""

from lucid.models._auto import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation,
)
from lucid.models._base import ModelConfig, PretrainedModel
from lucid.models._hub import (
    PretrainedEntry,
    download,
    load_from_pretrained_entry,
)
from lucid.models._mixins import (
    BackboneMixin,
    ClassificationHeadMixin,
    FeatureInfo,
)
from lucid.models._output import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
    ImageClassificationOutput,
    MaskedLMOutput,
    ModelOutput,
    ObjectDetectionOutput,
    SemanticSegmentationOutput,
    Seq2SeqLMOutput,
)
from lucid.models._registry import (
    create_model,
    is_model,
    list_models,
    model_entrypoint,
    register_model,
)

__all__ = [
    # Auto
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForImageClassification",
    "AutoModelForMaskedLM",
    "AutoModelForObjectDetection",
    "AutoModelForSemanticSegmentation",
    # Base
    "ModelConfig",
    "PretrainedModel",
    # Hub
    "PretrainedEntry",
    "download",
    "load_from_pretrained_entry",
    # Mixins
    "BackboneMixin",
    "ClassificationHeadMixin",
    "FeatureInfo",
    # Output
    "BaseModelOutput",
    "BaseModelOutputWithPooling",
    "CausalLMOutput",
    "ImageClassificationOutput",
    "MaskedLMOutput",
    "ModelOutput",
    "ObjectDetectionOutput",
    "SemanticSegmentationOutput",
    "Seq2SeqLMOutput",
    # Registry
    "create_model",
    "is_model",
    "list_models",
    "model_entrypoint",
    "register_model",
]
