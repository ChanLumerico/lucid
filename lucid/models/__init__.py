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
# 2022 — ConvNeXt (Liu et al.)
from lucid.models.vision.convnext import (
    ConvNeXtConfig, ConvNeXt, ConvNeXtForImageClassification,
    convnext_t, convnext_t_cls, convnext_s, convnext_s_cls,
    convnext_b, convnext_b_cls, convnext_l, convnext_l_cls,
    convnext_xl, convnext_xl_cls,
)
# 2021 — Swin Transformer (Liu et al.)
from lucid.models.vision.swin import (
    SwinConfig, SwinTransformer, SwinTransformerForImageClassification,
    swin_t, swin_t_cls, swin_s, swin_s_cls,
    swin_b, swin_b_cls, swin_l, swin_l_cls,
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
# 2013 — ZFNet (Zeiler & Fergus)
from lucid.models.vision.zfnet import (
    ZFNetConfig, ZFNet, ZFNetForImageClassification,
    zfnet, zfnet_cls,
)
# 2015 — Inception v3 (Szegedy et al.)
from lucid.models.vision.inception import (
    InceptionConfig, InceptionV3, InceptionV3ForImageClassification, InceptionV3Output,
    inception_v3, inception_v3_cls,
)
# 2016 — Inception v4 (Szegedy et al.)
from lucid.models.vision.inception_v4 import (
    InceptionV4Config, InceptionV4, InceptionV4ForImageClassification, InceptionV4Output,
    inception_v4, inception_v4_cls,
)
# 2016 — Inception-ResNet v2 (Szegedy et al.)
from lucid.models.vision.inception_resnet import (
    InceptionResNetConfig, InceptionResNetV2, InceptionResNetV2ForImageClassification,
    InceptionResNetOutput,
    inception_resnet_v2, inception_resnet_v2_cls,
)
# 2017 — Xception (Chollet)
from lucid.models.vision.xception import (
    XceptionConfig, Xception, XceptionForImageClassification, XceptionOutput,
    xception, xception_cls,
)
# 2017 — ResNeXt (Xie et al.)
from lucid.models.vision.resnext import (
    ResNeXtConfig, ResNeXt, ResNeXtForImageClassification,
    resnext_50_32x4d, resnext_50_32x4d_cls,
    resnext_101_32x4d, resnext_101_32x4d_cls,
    resnext_101_32x8d, resnext_101_32x8d_cls,
)
# 2018 — SENet (Hu et al.)
from lucid.models.vision.senet import (
    SENetConfig, SENet, SENetForImageClassification,
    se_resnet_18, se_resnet_18_cls,
    se_resnet_34, se_resnet_34_cls,
    se_resnet_50, se_resnet_50_cls,
    se_resnet_101, se_resnet_101_cls,
    se_resnet_152, se_resnet_152_cls,
)
# 2019 — SKNet (Li et al.)
from lucid.models.vision.sknet import (
    SKNetConfig, SKNet, SKNetForImageClassification,
    sk_resnet_50, sk_resnet_50_cls,
    sk_resnet_101, sk_resnet_101_cls,
    sk_resnext_50_32x4d, sk_resnext_50_32x4d_cls,
)
# 2018 — MobileNet v2 (Sandler et al.)
from lucid.models.vision.mobilenet_v2 import (
    MobileNetV2Config, MobileNetV2, MobileNetV2ForImageClassification,
    mobilenet_v2, mobilenet_v2_cls,
    mobilenet_v2_075, mobilenet_v2_075_cls,
)
# 2019 — MobileNet v3 (Howard et al.)
from lucid.models.vision.mobilenet_v3 import (
    MobileNetV3Config, MobileNetV3, MobileNetV3ForImageClassification,
    mobilenet_v3_large, mobilenet_v3_large_cls,
    mobilenet_v3_small, mobilenet_v3_small_cls,
)
# 2024 — MobileNet v4 (Qin et al.)
from lucid.models.vision.mobilenet_v4 import (
    MobileNetV4Config, MobileNetV4, MobileNetV4ForImageClassification,
    mobilenet_v4_conv_small, mobilenet_v4_conv_small_cls,
)
# 2020 — ResNeSt (Zhang et al.)
from lucid.models.vision.resnest import (
    ResNeStConfig, ResNeSt, ResNeStForImageClassification,
    resnest_50, resnest_50_cls,
    resnest_101, resnest_101_cls,
)
# 2019 — CSPNet (Wang et al.)
from lucid.models.vision.cspnet import (
    CSPNetConfig, CSPNet, CSPNetForImageClassification,
    cspresnet_50, cspresnet_50_cls,
)
# 2021 — CoAtNet (Dai et al.)
from lucid.models.vision.coatnet import (
    CoAtNetConfig, CoAtNet, CoAtNetForImageClassification,
    coatnet_0, coatnet_0_cls,
)
# 2021 — CvT (Wu et al.)
from lucid.models.vision.cvt import (
    CvTConfig, CvT, CvTForImageClassification,
    cvt_13, cvt_13_cls,
)
# 2021 — CrossViT (Chen et al.)
from lucid.models.vision.crossvit import (
    CrossViTConfig, CrossViT, CrossViTForImageClassification,
    crossvit_9, crossvit_9_cls,
)
# 2021 — PVT (Wang et al.)
from lucid.models.vision.pvt import (
    PVTConfig, PVT, PVTForImageClassification,
    pvt_tiny, pvt_tiny_cls,
)
# 2022 — EfficientFormer (Li et al.)
from lucid.models.vision.efficientformer import (
    EfficientFormerConfig, EfficientFormer, EfficientFormerForImageClassification,
    efficientformer_l1, efficientformer_l1_cls,
)
# 2022 — MaxViT (Tu et al.)
from lucid.models.vision.maxvit import (
    MaxViTConfig, MaxViT, MaxViTForImageClassification,
    maxvit_t, maxvit_t_cls,
)
# 2023 — InceptionNeXt (Yu et al.)
from lucid.models.vision.inception_next import (
    InceptionNeXtConfig, InceptionNeXt, InceptionNeXtForImageClassification,
    inception_next_t, inception_next_t_cls,
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
    # ── Vision (2022) ConvNeXt ────────────────────────────────────────────────
    "ConvNeXtConfig", "ConvNeXt", "ConvNeXtForImageClassification",
    "convnext_t", "convnext_t_cls", "convnext_s", "convnext_s_cls",
    "convnext_b", "convnext_b_cls", "convnext_l", "convnext_l_cls",
    "convnext_xl", "convnext_xl_cls",
    # ── Vision (2021) Swin Transformer ───────────────────────────────────────
    "SwinConfig", "SwinTransformer", "SwinTransformerForImageClassification",
    "swin_t", "swin_t_cls", "swin_s", "swin_s_cls",
    "swin_b", "swin_b_cls", "swin_l", "swin_l_cls",
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
    # ── Vision (2013) ZFNet ───────────────────────────────────────────────────
    "ZFNetConfig", "ZFNet", "ZFNetForImageClassification",
    "zfnet", "zfnet_cls",
    # ── Vision (2015) Inception v3 ────────────────────────────────────────────
    "InceptionConfig", "InceptionV3", "InceptionV3ForImageClassification", "InceptionV3Output",
    "inception_v3", "inception_v3_cls",
    # ── Vision (2016) Inception v4 ────────────────────────────────────────────
    "InceptionV4Config", "InceptionV4", "InceptionV4ForImageClassification", "InceptionV4Output",
    "inception_v4", "inception_v4_cls",
    # ── Vision (2016) Inception-ResNet v2 ────────────────────────────────────
    "InceptionResNetConfig", "InceptionResNetV2", "InceptionResNetV2ForImageClassification",
    "InceptionResNetOutput",
    "inception_resnet_v2", "inception_resnet_v2_cls",
    # ── Vision (2017) Xception ────────────────────────────────────────────────
    "XceptionConfig", "Xception", "XceptionForImageClassification", "XceptionOutput",
    "xception", "xception_cls",
    # ── Vision (2017) ResNeXt ─────────────────────────────────────────────────
    "ResNeXtConfig", "ResNeXt", "ResNeXtForImageClassification",
    "resnext_50_32x4d", "resnext_50_32x4d_cls",
    "resnext_101_32x4d", "resnext_101_32x4d_cls",
    "resnext_101_32x8d", "resnext_101_32x8d_cls",
    # ── Vision (2018) SENet ───────────────────────────────────────────────────
    "SENetConfig", "SENet", "SENetForImageClassification",
    "se_resnet_18", "se_resnet_18_cls", "se_resnet_34", "se_resnet_34_cls",
    "se_resnet_50", "se_resnet_50_cls", "se_resnet_101", "se_resnet_101_cls",
    "se_resnet_152", "se_resnet_152_cls",
    # ── Vision (2019) SKNet ───────────────────────────────────────────────────
    "SKNetConfig", "SKNet", "SKNetForImageClassification",
    "sk_resnet_50", "sk_resnet_50_cls", "sk_resnet_101", "sk_resnet_101_cls",
    "sk_resnext_50_32x4d", "sk_resnext_50_32x4d_cls",
    # ── Vision (2018) MobileNet v2 ────────────────────────────────────────────
    "MobileNetV2Config", "MobileNetV2", "MobileNetV2ForImageClassification",
    "mobilenet_v2", "mobilenet_v2_cls", "mobilenet_v2_075", "mobilenet_v2_075_cls",
    # ── Vision (2019) MobileNet v3 ────────────────────────────────────────────
    "MobileNetV3Config", "MobileNetV3", "MobileNetV3ForImageClassification",
    "mobilenet_v3_large", "mobilenet_v3_large_cls",
    "mobilenet_v3_small", "mobilenet_v3_small_cls",
    # ── Vision (2024) MobileNet v4 ────────────────────────────────────────────
    "MobileNetV4Config", "MobileNetV4", "MobileNetV4ForImageClassification",
    "mobilenet_v4_conv_small", "mobilenet_v4_conv_small_cls",
    # ── Vision (2020) ResNeSt ────────────────────────────────────────────────
    "ResNeStConfig", "ResNeSt", "ResNeStForImageClassification",
    "resnest_50", "resnest_50_cls", "resnest_101", "resnest_101_cls",
    # ── Vision (2019) CSPNet ─────────────────────────────────────────────────
    "CSPNetConfig", "CSPNet", "CSPNetForImageClassification",
    "cspresnet_50", "cspresnet_50_cls",
    # ── Vision (2021) CoAtNet ────────────────────────────────────────────────
    "CoAtNetConfig", "CoAtNet", "CoAtNetForImageClassification",
    "coatnet_0", "coatnet_0_cls",
    # ── Vision (2021) CvT ────────────────────────────────────────────────────
    "CvTConfig", "CvT", "CvTForImageClassification",
    "cvt_13", "cvt_13_cls",
    # ── Vision (2021) CrossViT ───────────────────────────────────────────────
    "CrossViTConfig", "CrossViT", "CrossViTForImageClassification",
    "crossvit_9", "crossvit_9_cls",
    # ── Vision (2021) PVT ────────────────────────────────────────────────────
    "PVTConfig", "PVT", "PVTForImageClassification",
    "pvt_tiny", "pvt_tiny_cls",
    # ── Vision (2022) EfficientFormer ────────────────────────────────────────
    "EfficientFormerConfig", "EfficientFormer", "EfficientFormerForImageClassification",
    "efficientformer_l1", "efficientformer_l1_cls",
    # ── Vision (2022) MaxViT ─────────────────────────────────────────────────
    "MaxViTConfig", "MaxViT", "MaxViTForImageClassification",
    "maxvit_t", "maxvit_t_cls",
    # ── Vision (2023) InceptionNeXt ──────────────────────────────────────────
    "InceptionNeXtConfig", "InceptionNeXt", "InceptionNeXtForImageClassification",
    "inception_next_t", "inception_next_t_cls",
]
