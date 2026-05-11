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

# Vision families — in chronological order of publication
# 1998 — LeNet-5 (LeCun et al.)
from lucid.models.vision.lenet import (  # noqa: E402
    LeNet,
    LeNetConfig,
    LeNetForImageClassification,
    lenet_5,
    lenet_5_cls,
    lenet_5_relu,
    lenet_5_relu_cls,
)

# 2012 — AlexNet (Krizhevsky, Sutskever & Hinton)
from lucid.models.vision.alexnet import (  # noqa: E402
    AlexNet,
    AlexNetConfig,
    AlexNetForImageClassification,
    alexnet,
    alexnet_cls,
)

# 2014 — VGG (Simonyan & Zisserman)
from lucid.models.vision.vgg import (  # noqa: E402
    VGG,
    VGGConfig,
    VGGForImageClassification,
    vgg_11,
    vgg_11_bn,
    vgg_11_cls,
    vgg_11_bn_cls,
    vgg_13,
    vgg_13_bn,
    vgg_13_cls,
    vgg_13_bn_cls,
    vgg_16,
    vgg_16_bn,
    vgg_16_cls,
    vgg_16_bn_cls,
    vgg_19,
    vgg_19_bn,
    vgg_19_cls,
    vgg_19_bn_cls,
)

# 2014 — GoogLeNet / Inception v1 (Szegedy et al.)
from lucid.models.vision.googlenet import (  # noqa: E402
    GoogLeNet,
    GoogLeNetConfig,
    GoogLeNetForImageClassification,
    GoogLeNetOutput,
    googlenet,
    googlenet_cls,
)
# 2017 — MobileNet v1 (Howard et al.)
from lucid.models.vision.mobilenet import (  # noqa: E402
    MobileNetV1,
    MobileNetV1Config,
    MobileNetV1ForImageClassification,
    mobilenet_v1, mobilenet_v1_cls,
    mobilenet_v1_075, mobilenet_v1_075_cls,
    mobilenet_v1_050, mobilenet_v1_050_cls,
    mobilenet_v1_025, mobilenet_v1_025_cls,
)
# 2016 — DenseNet (Huang et al.)
from lucid.models.vision.densenet import (  # noqa: E402
    DenseNet,
    DenseNetConfig,
    DenseNetForImageClassification,
    densenet_121, densenet_121_cls,
    densenet_169, densenet_169_cls,
    densenet_201, densenet_201_cls,
    densenet_264, densenet_264_cls,
)
# 2015 — ResNet (He et al.)
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
    # Vision — MobileNet v1 (2017)
    "MobileNetV1",
    "MobileNetV1Config",
    "MobileNetV1ForImageClassification",
    "mobilenet_v1", "mobilenet_v1_cls",
    "mobilenet_v1_075", "mobilenet_v1_075_cls",
    "mobilenet_v1_050", "mobilenet_v1_050_cls",
    "mobilenet_v1_025", "mobilenet_v1_025_cls",
    # Vision — DenseNet (2016)
    "DenseNet",
    "DenseNetConfig",
    "DenseNetForImageClassification",
    "densenet_121", "densenet_121_cls",
    "densenet_169", "densenet_169_cls",
    "densenet_201", "densenet_201_cls",
    "densenet_264", "densenet_264_cls",
    # Vision — GoogLeNet (2014)
    "GoogLeNet",
    "GoogLeNetConfig",
    "GoogLeNetForImageClassification",
    "GoogLeNetOutput",
    "googlenet",
    "googlenet_cls",
    # Vision — VGG (2014)
    "VGG",
    "VGGConfig",
    "VGGForImageClassification",
    "vgg_11",
    "vgg_11_bn",
    "vgg_11_cls",
    "vgg_11_bn_cls",
    "vgg_13",
    "vgg_13_bn",
    "vgg_13_cls",
    "vgg_13_bn_cls",
    "vgg_16",
    "vgg_16_bn",
    "vgg_16_cls",
    "vgg_16_bn_cls",
    "vgg_19",
    "vgg_19_bn",
    "vgg_19_cls",
    "vgg_19_bn_cls",
    # Vision — AlexNet (2012)
    "AlexNet",
    "AlexNetConfig",
    "AlexNetForImageClassification",
    "alexnet",
    "alexnet_cls",
    # Vision — LeNet (1998)
    "LeNet",
    "LeNetConfig",
    "LeNetForImageClassification",
    "lenet_5",
    "lenet_5_cls",
    "lenet_5_relu",
    "lenet_5_relu_cls",
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
