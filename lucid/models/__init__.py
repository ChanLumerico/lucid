"""Lucid 3.0 model zoo — public surface."""

# fmt: off

from lucid.models._auto import (
    AutoConfig, AutoModel,
    AutoModelForCausalLM, AutoModelForImageClassification,
    AutoModelForMaskedLM, AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation,
)
from lucid.models._base   import ModelConfig, PretrainedModel
from lucid.models._hub    import PretrainedEntry, download, load_from_pretrained_entry
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import (
    ModelOutput, BaseModelOutput, BaseModelOutputWithPooling,
    ImageClassificationOutput, ObjectDetectionOutput, SemanticSegmentationOutput,
    CausalLMOutput, MaskedLMOutput, Seq2SeqLMOutput,
)
from lucid.models._registry import create_model, is_model, list_models, model_entrypoint, register_model

# Vision families — chronological order of publication
# 1998 — LeNet-5 (LeCun et al.)
from lucid.models.vision.lenet import (
    LeNetConfig, LeNet, LeNetForImageClassification,
    lenet_5, lenet_5_cls, lenet_5_relu, lenet_5_relu_cls,
)
# 2012 — AlexNet (Krizhevsky, Sutskever & Hinton)
from lucid.models.vision.alexnet import (
    AlexNetConfig, AlexNet, AlexNetForImageClassification,
    alexnet, alexnet_cls,
)
# 2014 — VGG (Simonyan & Zisserman)
from lucid.models.vision.vgg import (
    VGGConfig, VGG, VGGForImageClassification,
    vgg_11, vgg_11_bn, vgg_11_cls, vgg_11_bn_cls,
    vgg_13, vgg_13_bn, vgg_13_cls, vgg_13_bn_cls,
    vgg_16, vgg_16_bn, vgg_16_cls, vgg_16_bn_cls,
    vgg_19, vgg_19_bn, vgg_19_cls, vgg_19_bn_cls,
)
# 2014 — GoogLeNet / Inception v1 (Szegedy et al.)
from lucid.models.vision.googlenet import (
    GoogLeNetConfig, GoogLeNet, GoogLeNetForImageClassification, GoogLeNetOutput,
    googlenet, googlenet_cls,
)
# 2015 — ResNet (He et al.)
from lucid.models.vision.resnet import (
    ResNetConfig, ResNet, ResNetForImageClassification,
    resnet_18, resnet_18_cls,
    resnet_34, resnet_34_cls,
    resnet_50, resnet_50_cls,
    resnet_101, resnet_101_cls,
    resnet_152, resnet_152_cls,
)
# 2016 — DenseNet (Huang et al.)
from lucid.models.vision.densenet import (
    DenseNetConfig, DenseNet, DenseNetForImageClassification,
    densenet_121, densenet_121_cls,
    densenet_169, densenet_169_cls,
    densenet_201, densenet_201_cls,
    densenet_264, densenet_264_cls,
)
# 2017 — MobileNet v1 (Howard et al.)
from lucid.models.vision.mobilenet import (
    MobileNetV1Config, MobileNetV1, MobileNetV1ForImageClassification,
    mobilenet_v1,     mobilenet_v1_cls,
    mobilenet_v1_075, mobilenet_v1_075_cls,
    mobilenet_v1_050, mobilenet_v1_050_cls,
    mobilenet_v1_025, mobilenet_v1_025_cls,
)
# 2020 — ViT (Dosovitskiy et al.)
from lucid.models.vision.vit import (
    ViTConfig, ViT, ViTForImageClassification,
    vit_b_16, vit_b_16_cls,
    vit_b_32, vit_b_32_cls,
    vit_l_16, vit_l_16_cls,
    vit_l_32, vit_l_32_cls,
    vit_h_14, vit_h_14_cls,
)
# 2019 — EfficientNet (Tan & Le)
from lucid.models.vision.efficientnet import (
    EfficientNetConfig, EfficientNet, EfficientNetForImageClassification,
    efficientnet_b0, efficientnet_b0_cls,
    efficientnet_b1, efficientnet_b1_cls,
    efficientnet_b2, efficientnet_b2_cls,
    efficientnet_b3, efficientnet_b3_cls,
    efficientnet_b4, efficientnet_b4_cls,
    efficientnet_b5, efficientnet_b5_cls,
    efficientnet_b6, efficientnet_b6_cls,
    efficientnet_b7, efficientnet_b7_cls,
)

__all__ = [
    # ── Infrastructure ────────────────────────────────────────────────────────
    "ModelConfig", "PretrainedModel",
    "PretrainedEntry", "download", "load_from_pretrained_entry",
    "BackboneMixin", "ClassificationHeadMixin", "FeatureInfo",
    "ModelOutput", "BaseModelOutput", "BaseModelOutputWithPooling",
    "ImageClassificationOutput", "ObjectDetectionOutput", "SemanticSegmentationOutput",
    "CausalLMOutput", "MaskedLMOutput", "Seq2SeqLMOutput",
    "AutoConfig", "AutoModel",
    "AutoModelForCausalLM", "AutoModelForImageClassification",
    "AutoModelForMaskedLM", "AutoModelForObjectDetection", "AutoModelForSemanticSegmentation",
    "create_model", "is_model", "list_models", "model_entrypoint", "register_model",
    # ── Vision (1998) LeNet ───────────────────────────────────────────────────
    "LeNetConfig", "LeNet", "LeNetForImageClassification",
    "lenet_5", "lenet_5_cls", "lenet_5_relu", "lenet_5_relu_cls",
    # ── Vision (2012) AlexNet ─────────────────────────────────────────────────
    "AlexNetConfig", "AlexNet", "AlexNetForImageClassification",
    "alexnet", "alexnet_cls",
    # ── Vision (2014) VGG ────────────────────────────────────────────────────
    "VGGConfig", "VGG", "VGGForImageClassification",
    "vgg_11", "vgg_11_bn", "vgg_11_cls", "vgg_11_bn_cls",
    "vgg_13", "vgg_13_bn", "vgg_13_cls", "vgg_13_bn_cls",
    "vgg_16", "vgg_16_bn", "vgg_16_cls", "vgg_16_bn_cls",
    "vgg_19", "vgg_19_bn", "vgg_19_cls", "vgg_19_bn_cls",
    # ── Vision (2014) GoogLeNet ───────────────────────────────────────────────
    "GoogLeNetConfig", "GoogLeNet", "GoogLeNetForImageClassification", "GoogLeNetOutput",
    "googlenet", "googlenet_cls",
    # ── Vision (2015) ResNet ──────────────────────────────────────────────────
    "ResNetConfig", "ResNet", "ResNetForImageClassification",
    "resnet_18", "resnet_18_cls", "resnet_34", "resnet_34_cls",
    "resnet_50", "resnet_50_cls", "resnet_101", "resnet_101_cls", "resnet_152", "resnet_152_cls",
    # ── Vision (2016) DenseNet ────────────────────────────────────────────────
    "DenseNetConfig", "DenseNet", "DenseNetForImageClassification",
    "densenet_121", "densenet_121_cls", "densenet_169", "densenet_169_cls",
    "densenet_201", "densenet_201_cls", "densenet_264", "densenet_264_cls",
    # ── Vision (2017) MobileNet v1 ────────────────────────────────────────────
    "MobileNetV1Config", "MobileNetV1", "MobileNetV1ForImageClassification",
    "mobilenet_v1", "mobilenet_v1_cls",
    "mobilenet_v1_075", "mobilenet_v1_075_cls",
    "mobilenet_v1_050", "mobilenet_v1_050_cls",
    "mobilenet_v1_025", "mobilenet_v1_025_cls",
    # ── Vision (2020) ViT ────────────────────────────────────────────────────
    "ViTConfig", "ViT", "ViTForImageClassification",
    "vit_b_16", "vit_b_16_cls", "vit_b_32", "vit_b_32_cls",
    "vit_l_16", "vit_l_16_cls", "vit_l_32", "vit_l_32_cls",
    "vit_h_14", "vit_h_14_cls",
    # ── Vision (2019) EfficientNet ────────────────────────────────────────────
    "EfficientNetConfig", "EfficientNet", "EfficientNetForImageClassification",
    "efficientnet_b0", "efficientnet_b0_cls", "efficientnet_b1", "efficientnet_b1_cls",
    "efficientnet_b2", "efficientnet_b2_cls", "efficientnet_b3", "efficientnet_b3_cls",
    "efficientnet_b4", "efficientnet_b4_cls", "efficientnet_b5", "efficientnet_b5_cls",
    "efficientnet_b6", "efficientnet_b6_cls", "efficientnet_b7", "efficientnet_b7_cls",
]
