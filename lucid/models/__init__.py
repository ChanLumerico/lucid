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
    ImageClassificationOutput, ObjectDetectionOutput, InstanceSegmentationOutput,
    SemanticSegmentationOutput, CausalLMOutput, MaskedLMOutput, Seq2SeqLMOutput,
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
# 2013 — ZFNet (Zeiler & Fergus)
from lucid.models.vision.zfnet import (
    ZFNetConfig, ZFNet, ZFNetForImageClassification,
    zfnet, zfnet_cls,
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
    resnet_200, resnet_200_cls,
    resnet_269, resnet_269_cls,
    wide_resnet_50, wide_resnet_50_cls,
    wide_resnet_101, wide_resnet_101_cls,
)
# 2015 — Inception v3 (Szegedy et al.)
from lucid.models.vision.inception import (
    InceptionConfig, InceptionV3, InceptionV3ForImageClassification, InceptionV3Output,
    inception_v3, inception_v3_cls,
)
# 2016 — DenseNet (Huang et al.)
from lucid.models.vision.densenet import (
    DenseNetConfig, DenseNet, DenseNetForImageClassification,
    densenet_121, densenet_121_cls,
    densenet_169, densenet_169_cls,
    densenet_201, densenet_201_cls,
    densenet_264, densenet_264_cls,
)
# 2016 — Inception-ResNet v2 (Szegedy et al.)
from lucid.models.vision.inception_resnet import (
    InceptionResNetConfig, InceptionResNetV2, InceptionResNetV2ForImageClassification,
    InceptionResNetOutput,
    inception_resnet_v2, inception_resnet_v2_cls,
)
# 2017 — ResNeXt (Xie et al.)
from lucid.models.vision.resnext import (
    ResNeXtConfig, ResNeXt, ResNeXtForImageClassification,
    resnext_50_32x4d, resnext_50_32x4d_cls,
    resnext_101_32x4d, resnext_101_32x4d_cls,
    resnext_101_32x8d, resnext_101_32x8d_cls,
)
# 2017 — Xception (Chollet)
from lucid.models.vision.xception import (
    XceptionConfig, Xception, XceptionForImageClassification, XceptionOutput,
    xception, xception_cls,
)
# 2017 — MobileNet v1 (Howard et al.)
from lucid.models.vision.mobilenet import (
    MobileNetV1Config, MobileNetV1, MobileNetV1ForImageClassification,
    mobilenet_v1,     mobilenet_v1_cls,
    mobilenet_v1_075, mobilenet_v1_075_cls,
    mobilenet_v1_050, mobilenet_v1_050_cls,
    mobilenet_v1_025, mobilenet_v1_025_cls,
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
# 2018 — MobileNet v2 (Sandler et al.)
from lucid.models.vision.mobilenet_v2 import (
    MobileNetV2Config, MobileNetV2, MobileNetV2ForImageClassification,
    mobilenet_v2, mobilenet_v2_cls,
    mobilenet_v2_075, mobilenet_v2_075_cls,
)
# 2019 — SKNet (Li et al.)
from lucid.models.vision.sknet import (
    SKNetConfig, SKNet, SKNetForImageClassification,
    sk_resnet_18, sk_resnet_18_cls,
    sk_resnet_34, sk_resnet_34_cls,
    sk_resnet_50, sk_resnet_50_cls,
    sk_resnet_101, sk_resnet_101_cls,
    sk_resnext_50_32x4d, sk_resnext_50_32x4d_cls,
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
# 2019 — MobileNet v3 (Howard et al.)
from lucid.models.vision.mobilenet_v3 import (
    MobileNetV3Config, MobileNetV3, MobileNetV3ForImageClassification,
    mobilenet_v3_large, mobilenet_v3_large_cls,
    mobilenet_v3_small, mobilenet_v3_small_cls,
)
# 2019 — CSPNet (Wang et al.)
from lucid.models.vision.cspnet import (
    CSPNetConfig, CSPNet, CSPNetForImageClassification,
    cspresnet_50, cspresnet_50_cls,
)
# 2020 — ResNeSt (Zhang et al.)
from lucid.models.vision.resnest import (
    ResNeStConfig, ResNeSt, ResNeStForImageClassification,
    resnest_14, resnest_14_cls,
    resnest_26, resnest_26_cls,
    resnest_50, resnest_50_cls,
    resnest_101, resnest_101_cls,
    resnest_200, resnest_200_cls,
    resnest_269, resnest_269_cls,
)
# 2020 — ViT (Dosovitskiy et al.)
from lucid.models.vision.vit import (
    ViTConfig, ViT, ViTForImageClassification,
    vit_base_16, vit_base_16_cls,
    vit_base_32, vit_base_32_cls,
    vit_large_16, vit_large_16_cls,
    vit_large_32, vit_large_32_cls,
    vit_huge_14, vit_huge_14_cls,
)
# 2021 — Swin Transformer (Liu et al.)
from lucid.models.vision.swin import (
    SwinConfig, SwinTransformer, SwinTransformerForImageClassification,
    swin_tiny, swin_tiny_cls, swin_small, swin_small_cls,
    swin_base, swin_base_cls, swin_large, swin_large_cls,
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
    cvt_21, cvt_21_cls,
    cvt_w24, cvt_w24_cls,
)
# 2021 — CrossViT (Chen et al.)
from lucid.models.vision.crossvit import (
    CrossViTConfig, CrossViT, CrossViTForImageClassification,
    crossvit_9, crossvit_9_cls,
    crossvit_tiny, crossvit_tiny_cls,
    crossvit_small, crossvit_small_cls,
    crossvit_base, crossvit_base_cls,
    crossvit_15, crossvit_15_cls,
    crossvit_18, crossvit_18_cls,
)
# 2021 — PVT (Wang et al.)
from lucid.models.vision.pvt import (
    PVTConfig, PVT, PVTForImageClassification,
    pvt_v2_b0, pvt_v2_b0_cls,
    pvt_v2_b1, pvt_v2_b1_cls,
    pvt_v2_b2, pvt_v2_b2_cls,
    pvt_v2_b3, pvt_v2_b3_cls,
    pvt_v2_b4, pvt_v2_b4_cls,
    pvt_v2_b5, pvt_v2_b5_cls,
    pvt_tiny, pvt_tiny_cls,
)
# 2022 — ConvNeXt (Liu et al.)
from lucid.models.vision.convnext import (
    ConvNeXtConfig, ConvNeXt, ConvNeXtForImageClassification,
    convnext_tiny, convnext_tiny_cls, convnext_small, convnext_small_cls,
    convnext_base, convnext_base_cls, convnext_large, convnext_large_cls,
    convnext_xlarge, convnext_xlarge_cls,
)
# 2022 — EfficientFormer (Li et al.)
from lucid.models.vision.efficientformer import (
    EfficientFormerConfig, EfficientFormer, EfficientFormerForImageClassification,
    efficientformer_l1, efficientformer_l1_cls,
    efficientformer_l3, efficientformer_l3_cls,
    efficientformer_l7, efficientformer_l7_cls,
)
# 2022 — MaxViT (Tu et al.)
from lucid.models.vision.maxvit import (
    MaxViTConfig, MaxViT, MaxViTForImageClassification,
    maxvit_tiny, maxvit_tiny_cls,
    maxvit_small, maxvit_small_cls,
    maxvit_base, maxvit_base_cls,
    maxvit_large, maxvit_large_cls,
    maxvit_xlarge, maxvit_xlarge_cls,
)
# 2023 — InceptionNeXt (Yu et al.)
from lucid.models.vision.inception_next import (
    InceptionNeXtConfig, InceptionNeXt, InceptionNeXtForImageClassification,
    inception_next_tiny, inception_next_tiny_cls,
)
# 2014 — R-CNN (Girshick et al.)
from lucid.models.vision.rcnn import (
    RCNNConfig, RCNNForObjectDetection,
    rcnn,
)
# 2015 — Fast R-CNN (Girshick)
from lucid.models.vision.fast_rcnn import (
    FastRCNNConfig, FastRCNNForObjectDetection,
    fast_rcnn,
)
# 2015 — Faster R-CNN (Ren et al.)
from lucid.models.vision.faster_rcnn import (
    FasterRCNNConfig, FasterRCNNForObjectDetection,
    faster_rcnn,
)
# 2017 — Mask R-CNN (He et al.)
from lucid.models.vision.mask_rcnn import (
    MaskRCNNConfig, MaskRCNNForObjectDetection,
    mask_rcnn,
)
# 2020 — DETR (Carion et al.)
from lucid.models.vision.detr import (
    DETRConfig, DETRForObjectDetection,
    detr_resnet50,
)
# 2020 — EfficientDet (Tan et al.)
from lucid.models.vision.efficientdet import (
    EfficientDetConfig, EfficientDetForObjectDetection, efficientdet_config,
    efficientdet_d0, efficientdet_d1, efficientdet_d2, efficientdet_d3,
    efficientdet_d4, efficientdet_d5, efficientdet_d6, efficientdet_d7,
)
# 2016–2020 — YOLO family (Redmon et al., Bochkovskiy et al.)
from lucid.models.vision.yolo import (
    YOLOV1Config, YOLOV1ForObjectDetection,
    yolo_v1, yolo_v1_tiny,
    YOLOV2Config, YOLOV2ForObjectDetection,
    yolo_v2, yolo_v2_tiny,
    YOLOV3Config, YOLOV3ForObjectDetection,
    yolo_v3, yolo_v3_tiny,
    YOLOV4Config, YOLOV4ForObjectDetection,
    yolo_v4,
)

__all__ = [
    # ── Infrastructure ────────────────────────────────────────────────────────
    "ModelConfig", "PretrainedModel",
    "PretrainedEntry", "download", "load_from_pretrained_entry",
    "BackboneMixin", "ClassificationHeadMixin", "FeatureInfo",
    "ModelOutput", "BaseModelOutput", "BaseModelOutputWithPooling",
    "ImageClassificationOutput", "ObjectDetectionOutput", "InstanceSegmentationOutput",
    "SemanticSegmentationOutput", "CausalLMOutput", "MaskedLMOutput", "Seq2SeqLMOutput",
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
    # ── Vision (2013) ZFNet ───────────────────────────────────────────────────
    "ZFNetConfig", "ZFNet", "ZFNetForImageClassification",
    "zfnet", "zfnet_cls",
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
    "resnet_200", "resnet_200_cls", "resnet_269", "resnet_269_cls",
    "wide_resnet_50", "wide_resnet_50_cls", "wide_resnet_101", "wide_resnet_101_cls",
    # ── Vision (2015) Inception v3 ────────────────────────────────────────────
    "InceptionConfig", "InceptionV3", "InceptionV3ForImageClassification", "InceptionV3Output",
    "inception_v3", "inception_v3_cls",
    # ── Vision (2016) DenseNet ────────────────────────────────────────────────
    "DenseNetConfig", "DenseNet", "DenseNetForImageClassification",
    "densenet_121", "densenet_121_cls", "densenet_169", "densenet_169_cls",
    "densenet_201", "densenet_201_cls", "densenet_264", "densenet_264_cls",
    # ── Vision (2016) Inception-ResNet v2 ────────────────────────────────────
    "InceptionResNetConfig", "InceptionResNetV2", "InceptionResNetV2ForImageClassification",
    "InceptionResNetOutput",
    "inception_resnet_v2", "inception_resnet_v2_cls",
    # ── Vision (2017) ResNeXt ─────────────────────────────────────────────────
    "ResNeXtConfig", "ResNeXt", "ResNeXtForImageClassification",
    "resnext_50_32x4d", "resnext_50_32x4d_cls",
    "resnext_101_32x4d", "resnext_101_32x4d_cls",
    "resnext_101_32x8d", "resnext_101_32x8d_cls",
    # ── Vision (2017) Xception ────────────────────────────────────────────────
    "XceptionConfig", "Xception", "XceptionForImageClassification", "XceptionOutput",
    "xception", "xception_cls",
    # ── Vision (2017) MobileNet v1 ────────────────────────────────────────────
    "MobileNetV1Config", "MobileNetV1", "MobileNetV1ForImageClassification",
    "mobilenet_v1", "mobilenet_v1_cls",
    "mobilenet_v1_075", "mobilenet_v1_075_cls",
    "mobilenet_v1_050", "mobilenet_v1_050_cls",
    "mobilenet_v1_025", "mobilenet_v1_025_cls",
    # ── Vision (2018) SENet ───────────────────────────────────────────────────
    "SENetConfig", "SENet", "SENetForImageClassification",
    "se_resnet_18", "se_resnet_18_cls", "se_resnet_34", "se_resnet_34_cls",
    "se_resnet_50", "se_resnet_50_cls", "se_resnet_101", "se_resnet_101_cls",
    "se_resnet_152", "se_resnet_152_cls",
    # ── Vision (2018) MobileNet v2 ────────────────────────────────────────────
    "MobileNetV2Config", "MobileNetV2", "MobileNetV2ForImageClassification",
    "mobilenet_v2", "mobilenet_v2_cls", "mobilenet_v2_075", "mobilenet_v2_075_cls",
    # ── Vision (2019) SKNet ───────────────────────────────────────────────────
    "SKNetConfig", "SKNet", "SKNetForImageClassification",
    "sk_resnet_18", "sk_resnet_18_cls", "sk_resnet_34", "sk_resnet_34_cls",
    "sk_resnet_50", "sk_resnet_50_cls", "sk_resnet_101", "sk_resnet_101_cls",
    "sk_resnext_50_32x4d", "sk_resnext_50_32x4d_cls",
    # ── Vision (2019) EfficientNet ────────────────────────────────────────────
    "EfficientNetConfig", "EfficientNet", "EfficientNetForImageClassification",
    "efficientnet_b0", "efficientnet_b0_cls", "efficientnet_b1", "efficientnet_b1_cls",
    "efficientnet_b2", "efficientnet_b2_cls", "efficientnet_b3", "efficientnet_b3_cls",
    "efficientnet_b4", "efficientnet_b4_cls", "efficientnet_b5", "efficientnet_b5_cls",
    "efficientnet_b6", "efficientnet_b6_cls", "efficientnet_b7", "efficientnet_b7_cls",
    # ── Vision (2019) MobileNet v3 ────────────────────────────────────────────
    "MobileNetV3Config", "MobileNetV3", "MobileNetV3ForImageClassification",
    "mobilenet_v3_large", "mobilenet_v3_large_cls",
    "mobilenet_v3_small", "mobilenet_v3_small_cls",
    # ── Vision (2019) CSPNet ─────────────────────────────────────────────────
    "CSPNetConfig", "CSPNet", "CSPNetForImageClassification",
    "cspresnet_50", "cspresnet_50_cls",
    # ── Vision (2020) ResNeSt ────────────────────────────────────────────────
    "ResNeStConfig", "ResNeSt", "ResNeStForImageClassification",
    "resnest_14", "resnest_14_cls", "resnest_26", "resnest_26_cls",
    "resnest_50", "resnest_50_cls", "resnest_101", "resnest_101_cls",
    "resnest_200", "resnest_200_cls", "resnest_269", "resnest_269_cls",
    # ── Vision (2020) ViT ────────────────────────────────────────────────────
    "ViTConfig", "ViT", "ViTForImageClassification",
    "vit_base_16", "vit_base_16_cls", "vit_base_32", "vit_base_32_cls",
    "vit_large_16", "vit_large_16_cls", "vit_large_32", "vit_large_32_cls",
    "vit_huge_14", "vit_huge_14_cls",
    # ── Vision (2021) Swin Transformer ───────────────────────────────────────
    "SwinConfig", "SwinTransformer", "SwinTransformerForImageClassification",
    "swin_tiny", "swin_tiny_cls", "swin_small", "swin_small_cls",
    "swin_base", "swin_base_cls", "swin_large", "swin_large_cls",
    # ── Vision (2021) CoAtNet ────────────────────────────────────────────────
    "CoAtNetConfig", "CoAtNet", "CoAtNetForImageClassification",
    "coatnet_0", "coatnet_0_cls",
    # ── Vision (2021) CvT ────────────────────────────────────────────────────
    "CvTConfig", "CvT", "CvTForImageClassification",
    "cvt_13", "cvt_13_cls", "cvt_21", "cvt_21_cls", "cvt_w24", "cvt_w24_cls",
    # ── Vision (2021) CrossViT ───────────────────────────────────────────────
    "CrossViTConfig", "CrossViT", "CrossViTForImageClassification",
    "crossvit_9", "crossvit_9_cls",
    "crossvit_tiny", "crossvit_tiny_cls", "crossvit_small", "crossvit_small_cls",
    "crossvit_base", "crossvit_base_cls", "crossvit_15", "crossvit_15_cls",
    "crossvit_18", "crossvit_18_cls",
    # ── Vision (2021) PVT ────────────────────────────────────────────────────
    "PVTConfig", "PVT", "PVTForImageClassification",
    "pvt_v2_b0", "pvt_v2_b0_cls", "pvt_v2_b1", "pvt_v2_b1_cls",
    "pvt_v2_b2", "pvt_v2_b2_cls", "pvt_v2_b3", "pvt_v2_b3_cls",
    "pvt_v2_b4", "pvt_v2_b4_cls", "pvt_v2_b5", "pvt_v2_b5_cls",
    "pvt_tiny", "pvt_tiny_cls",
    # ── Vision (2022) ConvNeXt ────────────────────────────────────────────────
    "ConvNeXtConfig", "ConvNeXt", "ConvNeXtForImageClassification",
    "convnext_tiny", "convnext_tiny_cls", "convnext_small", "convnext_small_cls",
    "convnext_base", "convnext_base_cls", "convnext_large", "convnext_large_cls",
    "convnext_xlarge", "convnext_xlarge_cls",
    # ── Vision (2022) EfficientFormer ────────────────────────────────────────
    "EfficientFormerConfig", "EfficientFormer", "EfficientFormerForImageClassification",
    "efficientformer_l1", "efficientformer_l1_cls",
    "efficientformer_l3", "efficientformer_l3_cls",
    "efficientformer_l7", "efficientformer_l7_cls",
    # ── Vision (2022) MaxViT ─────────────────────────────────────────────────
    "MaxViTConfig", "MaxViT", "MaxViTForImageClassification",
    "maxvit_tiny", "maxvit_tiny_cls",
    "maxvit_small", "maxvit_small_cls", "maxvit_base", "maxvit_base_cls",
    "maxvit_large", "maxvit_large_cls", "maxvit_xlarge", "maxvit_xlarge_cls",
    # ── Vision (2023) InceptionNeXt ──────────────────────────────────────────
    "InceptionNeXtConfig", "InceptionNeXt", "InceptionNeXtForImageClassification",
    "inception_next_tiny", "inception_next_tiny_cls",
    # ── Vision (2014) R-CNN ───────────────────────────────────────────────────
    "RCNNConfig", "RCNNForObjectDetection",
    "rcnn",
    # ── Vision (2015) Fast R-CNN ──────────────────────────────────────────────
    "FastRCNNConfig", "FastRCNNForObjectDetection",
    "fast_rcnn",
    # ── Vision (2015) Faster R-CNN ────────────────────────────────────────────
    "FasterRCNNConfig", "FasterRCNNForObjectDetection",
    "faster_rcnn",
    # ── Vision (2017) Mask R-CNN ──────────────────────────────────────────────
    "MaskRCNNConfig", "MaskRCNNForObjectDetection",
    "mask_rcnn",
    # ── Vision (2020) DETR ────────────────────────────────────────────────────
    "DETRConfig", "DETRForObjectDetection",
    "detr_resnet50",
    # ── Vision (2020) EfficientDet ────────────────────────────────────────────
    "EfficientDetConfig", "EfficientDetForObjectDetection", "efficientdet_config",
    "efficientdet_d0", "efficientdet_d1", "efficientdet_d2", "efficientdet_d3",
    "efficientdet_d4", "efficientdet_d5", "efficientdet_d6", "efficientdet_d7",
    # ── Vision (2016) YOLOv1 ──────────────────────────────────────────────────
    "YOLOV1Config", "YOLOV1ForObjectDetection",
    "yolo_v1", "yolo_v1_tiny",
    # ── Vision (2017) YOLOv2 ──────────────────────────────────────────────────
    "YOLOV2Config", "YOLOV2ForObjectDetection",
    "yolo_v2", "yolo_v2_tiny",
    # ── Vision (2018) YOLOv3 ──────────────────────────────────────────────────
    "YOLOV3Config", "YOLOV3ForObjectDetection",
    "yolo_v3", "yolo_v3_tiny",
    # ── Vision (2020) YOLOv4 ──────────────────────────────────────────────────
    "YOLOV4Config", "YOLOV4ForObjectDetection",
    "yolo_v4",
]
