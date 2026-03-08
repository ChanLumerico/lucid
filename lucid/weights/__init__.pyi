from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class WeightEntry:
    url: str
    sha256: str
    tag: str
    dataset: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    @property
    def config(self) -> Optional[Dict[str, Any]]: ...

class LeNet_1_Weights(Enum):
    MNIST: WeightEntry
    DEFAULT: WeightEntry

class LeNet_4_Weights(Enum):
    MNIST: WeightEntry
    DEFAULT: WeightEntry

class LeNet_5_Weights(Enum):
    MNIST: WeightEntry
    DEFAULT: WeightEntry

class AlexNet_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_11_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_13_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_16_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_19_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_18_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_34_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_50_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_101_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_152_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class Wide_ResNet_50_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class Wide_ResNet_101_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNeXt_50_32X4D_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNeXt_101_32X8D_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNeXt_101_64X4D_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class DenseNet_121_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class DenseNet_169_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class DenseNet_201_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class MobileNet_V2_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class MobileNet_V3_Small_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class MobileNet_V3_Large_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class BERT_Weights(Enum):
    PRE_TRAIN_BASE: WeightEntry
    DEFAULT: WeightEntry

class MaskFormer_ResNet_50_Weights(Enum):
    ADE20K: WeightEntry
    DEFAULT: WeightEntry

class MaskFormer_ResNet_101_Weights(Enum):
    ADE20K: WeightEntry
    DEFAULT: WeightEntry

class BERTForMaskedLM_Weights(Enum):
    HF_BASE_UNCASED: WeightEntry
    DEFAULT: WeightEntry

class BERTForCausalLM_Weights(Enum):
    HF_BASE_UNCASED: WeightEntry
    DEFAULT: WeightEntry

class BERTForNextSentencePrediction_Weights(Enum):
    HF_BASE_UNCASED: WeightEntry
    DEFAULT: WeightEntry

class BERTForSequenceClassification_Weights(Enum):
    SST2: WeightEntry
    DEFAULT: WeightEntry

class BERTForTokenClassification_Weights(Enum):
    CONLL03: WeightEntry
    DEFAULT: WeightEntry

class BERTForQuestionAnswering_Weights(Enum):
    SQUAD2: WeightEntry
    DEFAULT: WeightEntry

class Swin_Tiny_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class Swin_Base_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class Mask2Former_Swin_Tiny_Weights(Enum):
    ADE20K: WeightEntry
    DEFAULT: WeightEntry

class Mask2Former_Swin_Small_Weights(Enum):
    ADE20K: WeightEntry
    DEFAULT: WeightEntry

class Mask2Former_Swin_Base_Weights(Enum):
    ADE20K: WeightEntry
    DEFAULT: WeightEntry

class Mask2Former_Swin_Large_Weights(Enum):
    ADE20K: WeightEntry
    DEFAULT: WeightEntry

__all__ = [
    "LeNet_1_Weights",
    "LeNet_4_Weights",
    "LeNet_5_Weights",
    "AlexNet_Weights",
    "VGGNet_11_Weights",
    "VGGNet_13_Weights",
    "VGGNet_16_Weights",
    "VGGNet_19_Weights",
    "ResNet_18_Weights",
    "ResNet_34_Weights",
    "ResNet_50_Weights",
    "ResNet_101_Weights",
    "ResNet_152_Weights",
    "Wide_ResNet_50_Weights",
    "Wide_ResNet_101_Weights",
    "ResNeXt_50_32X4D_Weights",
    "ResNeXt_101_32X8D_Weights",
    "ResNeXt_101_64X4D_Weights",
    "DenseNet_121_Weights",
    "DenseNet_169_Weights",
    "DenseNet_201_Weights",
    "MobileNet_V2_Weights",
    "MobileNet_V3_Small_Weights",
    "MobileNet_V3_Large_Weights",
    "BERT_Weights",
    "MaskFormer_ResNet_50_Weights",
    "MaskFormer_ResNet_101_Weights",
    "BERTForMaskedLM_Weights",
    "BERTForCausalLM_Weights",
    "BERTForNextSentencePrediction_Weights",
    "BERTForSequenceClassification_Weights",
    "BERTForTokenClassification_Weights",
    "BERTForQuestionAnswering_Weights",
    "Swin_Tiny_Weights",
    "Swin_Base_Weights",
    "Mask2Former_Swin_Small_Weights",
    "Mask2Former_Swin_Tiny_Weights",
    "Mask2Former_Swin_Large_Weights",
    "Mask2Former_Swin_Base_Weights",
]
