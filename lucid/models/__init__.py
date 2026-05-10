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

# Vision families — importing triggers @register_model decorators
from lucid.models.vision.resnet import (  # noqa: E402
    ResNet,
    ResNetConfig,
    ResNetForImageClassification,
    resnet_18,
    resnet_18_cls,
    resnet_34,
    resnet_34_cls,
    resnet_50,
    resnet_50_cls,
    resnet_101,
    resnet_101_cls,
    resnet_152,
    resnet_152_cls,
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
    # Vision — ResNet
    "ResNet",
    "ResNetConfig",
    "ResNetForImageClassification",
    "resnet_18",
    "resnet_18_cls",
    "resnet_34",
    "resnet_34_cls",
    "resnet_50",
    "resnet_50_cls",
    "resnet_101",
    "resnet_101_cls",
    "resnet_152",
    "resnet_152_cls",
]
