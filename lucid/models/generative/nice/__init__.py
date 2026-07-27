"""NICE family — Dinh, Krueger & Bengio, 2014 (additive-coupling flow)."""

from lucid.models.generative.nice._config import NICEConfig
from lucid.models.generative.nice._model import NICEForImageGeneration, NICEModel
from lucid.models.generative.nice._pretrained import (
    nice_cifar,
    nice_cifar_gen,
    nice_mnist,
    nice_mnist_gen,
    nice_svhn,
    nice_svhn_gen,
    nice_tfd,
    nice_tfd_gen,
)

__all__ = [
    "NICEConfig",
    "NICEModel",
    "NICEForImageGeneration",
    "nice_mnist",
    "nice_tfd",
    "nice_svhn",
    "nice_cifar",
    "nice_mnist_gen",
    "nice_tfd_gen",
    "nice_svhn_gen",
    "nice_cifar_gen",
]
