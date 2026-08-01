"""Flow Matching family — Lipman et al., 2023 (simulation-free CNF training)."""

from lucid.models.generative.flow_matching._config import FlowMatchingConfig
from lucid.models.generative.flow_matching._model import (
    FlowMatchingForImageGeneration,
    FlowMatchingModel,
)
from lucid.models.generative.flow_matching._pretrained import (
    flow_matching_cifar,
    flow_matching_cifar_gen,
    flow_matching_imagenet32,
    flow_matching_imagenet32_gen,
    flow_matching_imagenet64,
    flow_matching_imagenet64_gen,
    flow_matching_imagenet128,
    flow_matching_imagenet128_gen,
)

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingModel",
    "FlowMatchingForImageGeneration",
    "flow_matching_cifar",
    "flow_matching_imagenet32",
    "flow_matching_imagenet64",
    "flow_matching_imagenet128",
    "flow_matching_cifar_gen",
    "flow_matching_imagenet32_gen",
    "flow_matching_imagenet64_gen",
    "flow_matching_imagenet128_gen",
]
