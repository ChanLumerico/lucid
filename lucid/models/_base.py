"""Base classes for model configuration and pretrained-model contract."""

import json
import os
from abc import ABC
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import ClassVar, Self

import lucid as _lucid
import lucid.nn as nn


class ModelConfig(ABC):
    """Base class for model configuration dataclasses.

    Subclasses must:

    - Use ``@dataclass(frozen=True)`` (immutability + ``fields()`` introspection)
    - Set the ``model_type`` ``ClassVar[str]`` to a unique family identifier
    - Optionally implement ``__post_init__`` for cross-field validation

    The base provides JSON round-trip (``to_dict`` / ``from_dict`` /
    ``save`` / ``load``) so all configs share one persistence format.
    """

    model_type: ClassVar[str] = "base"

    def to_dict(self) -> dict[str, object]:
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be decorated with @dataclass")
        d: dict[str, object] = asdict(self)
        d["model_type"] = self.model_type
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        copy: dict[str, object] = dict(d)
        # ``model_type`` is a ClassVar — never a constructor arg.
        copy.pop("model_type", None)
        # Filter unknown keys so older checkpoints survive new optional fields
        # being added; raise loudly only on missing required ones.
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a @dataclass")
        known = {f.name for f in fields(cls)}
        unknown = set(copy) - known
        if unknown:
            raise ValueError(
                f"{cls.__name__}.from_dict: unknown fields {sorted(unknown)}"
            )
        return cls(**copy)

    def save(self, path: str) -> None:
        """Write a JSON config file to ``path``."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> Self:
        """Read a JSON config file from ``path``."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError(f"{path} did not contain a JSON object")
        return cls.from_dict(d)


class PretrainedModel(nn.Module):
    """Base for models that round-trip through ``from_pretrained`` / ``save_pretrained``.

    Contract for subclasses:

    - Set ``config_class`` (``ClassVar[type[ModelConfig]]``) to the matching
      :class:`ModelConfig` subclass
    - Define ``__init__(self, config: TConfig) -> None`` — single argument,
      no extra kwargs.  All variant differences belong inside ``config``
    - Implement ``forward(...) -> ModelOutput``
    """

    config_class: ClassVar[type[ModelConfig]] = ModelConfig
    base_model_prefix: ClassVar[str] = ""

    config: ModelConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, self.config_class):
            raise TypeError(
                f"{type(self).__name__} expects "
                f"{self.config_class.__name__}, got {type(config).__name__}"
            )
        self.config = config

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, strict: bool = True) -> Self:
        """Load a model.

        Two modes:

        1. Registered name (``"resnet_50"`` / ``"resnet-50"``) — looked up
           in the global registry; the registered factory is invoked with
           ``pretrained=True`` (which downloads weights via :mod:`_hub`).
        2. Local directory containing ``config.json`` + ``weights.lucid``
           — config is loaded via ``cls.config_class.load`` and weights
           via :func:`lucid.load`.
        """
        # Local imports avoid a circular dep (registry imports _base for typing).
        from lucid.models._registry import is_model, model_entrypoint

        if is_model(name_or_path):
            factory = model_entrypoint(name_or_path)
            model = factory(pretrained=True)
            if not isinstance(model, cls):
                raise TypeError(
                    f"Factory {name_or_path!r} produced "
                    f"{type(model).__name__}, not a subclass of {cls.__name__}"
                )
            return model

        path = Path(name_or_path)
        if not path.is_dir():
            raise ValueError(
                f"{name_or_path!r} is neither a registered model nor "
                f"an existing directory"
            )
        config_file = path / "config.json"
        weights_file = path / "weights.lucid"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {path}")
        if not weights_file.exists():
            raise FileNotFoundError(f"weights.lucid not found in {path}")

        config = cls.config_class.load(str(config_file))
        model = cls(config)
        sd = _lucid.load(str(weights_file), weights_only=True)
        if not isinstance(sd, dict):
            raise TypeError(
                f"weights.lucid did not contain a state_dict, "
                f"got {type(sd).__name__}"
            )
        model.load_state_dict(sd, strict=strict)
        return model

    def save_pretrained(self, path: str) -> None:
        """Write ``config.json`` and ``weights.lucid`` to a directory."""
        os.makedirs(path, exist_ok=True)
        self.config.save(os.path.join(path, "config.json"))
        _lucid.save(self.state_dict(), os.path.join(path, "weights.lucid"))

    def num_parameters(self, *, only_trainable: bool = False) -> int:
        """Total parameter count (sum of element counts)."""
        total: int = 0
        for p in self.parameters():
            if only_trainable and not p.requires_grad:
                continue
            sz: int = 1
            for s in p.shape:
                sz *= int(s)
            total += sz
        return total

    def get_input_embeddings(self) -> nn.Module | None:
        """Return the input-embedding submodule, or None if not applicable."""
        return None

    def set_input_embeddings(self, value: nn.Module) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_input_embeddings"
        )
