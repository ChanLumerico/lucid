"""Offline weight-conversion pipeline for the Lucid model zoo.

Loads pretrained checkpoints from upstream sources (torchvision / timm /
transformers), remaps their ``state_dict`` keys to Lucid's module
layout, and writes a Lucid-native ``model.safetensors`` + ``config.json``
+ ``README.md`` ready to upload to the Hugging Face Hub under the
``lucid-dl`` org.

This is **dev-only tooling** — it imports torch / torchvision and lives
under ``tools/`` (outside ``lucid/``), so it is exempt from the runtime
no-external-dependency rule.  The produced checkpoints are pure Lucid
safetensors; nothing here ships in the installed package.

Usage
-----
::

    python -m tools.convert_weights resnet_18 --tag IMAGENET1K_V1
    python -m tools.convert_weights resnet_18 --tag IMAGENET1K_V1 --upload

The first form writes to ``./_converted/<repo>/<tag>/`` for inspection;
``--upload`` pushes the folder to the Hub (requires ``huggingface-cli
login`` or ``HF_TOKEN``).
"""

from tools.convert_weights._base import Architecture, ConversionSpec, convert, write

__all__ = ["Architecture", "ConversionSpec", "convert", "write"]
