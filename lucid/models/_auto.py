"""Auto classes — task-aware generic model / config loaders.

Each ``AutoModelFor{Task}`` is a thin shell that delegates to the registry,
filtered by ``_task``.  The same name can resolve to different classes under
different Auto types: ``AutoModel.from_pretrained("resnet_50")`` returns the
backbone, ``AutoModelForImageClassification.from_pretrained("resnet_50")``
returns the classification head.
"""

from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.models._base import ModelConfig, PretrainedModel


class _BaseAutoClass:
    """Common implementation shared by all Auto* classes."""

    _task: ClassVar[str]

    def __init__(self) -> None:
        raise EnvironmentError(
            f"{type(self).__name__} cannot be instantiated. "
            f"Use {type(self).__name__}.from_pretrained(...)."
        )

    @classmethod
    def from_pretrained(
        cls, name_or_path: str, *, strict: bool = True
    ) -> PretrainedModel:
        """Resolve *name_or_path* and return a fully constructed model.

        Accepts either a registered model name (``"resnet_50"``) or a local
        directory that contains ``config.json`` + ``weights.lucid``.
        """
        from lucid.models._registry import _registry_lookup, is_model

        path = Path(name_or_path)
        if path.is_dir():
            return _load_from_directory(cls._task, path, strict=strict)

        if not is_model(name_or_path):
            from lucid.models._registry import _unknown_model_message, _REGISTRY

            raise ValueError(
                _unknown_model_message(name_or_path, list(_REGISTRY.keys()))
            )

        entry = _registry_lookup(name_or_path, task=cls._task)
        return entry.factory(pretrained=True)


def _load_from_directory(task: str, path: Path, *, strict: bool) -> PretrainedModel:
    """Load a model from a ``config.json`` + ``weights.lucid`` directory.

    Strategy:
    1. Parse ``model_type`` from ``config.json``.
    2. Find the matching registry entry (task + model_type).
    3. **Fast path** (model_class provided): instantiate the class directly with
       the saved config — zero redundant parameter allocation.
    4. **Fallback** (no model_class): call factory(pretrained=False) once to
       discover the class, then instantiate with the saved config.  Still only
       one factory call, not two.
    """
    import json

    import lucid as _lucid
    from lucid.models._registry import _REGISTRY, _RegistryEntry

    cfg_file = path / "config.json"
    weights_st = path / "model.safetensors"
    weights_lucid = path / "weights.lucid"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json not found in {path}")
    if weights_st.exists():
        weights_file = weights_st
    elif weights_lucid.exists():
        weights_file = weights_lucid
    else:
        raise FileNotFoundError(
            f"No weights file found in {path}. "
            f"Expected 'model.safetensors' or 'weights.lucid'."
        )

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"{cfg_file} did not contain a JSON object")
    model_type = cfg_dict.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError(f"{cfg_file} must declare 'model_type' (string)")

    matching: list[_RegistryEntry] = [
        entry
        for entry in _REGISTRY.values()
        if entry.task == task and entry.model_type == model_type
    ]
    if not matching:
        raise ValueError(
            f"No model registered for task={task!r}, model_type={model_type!r}"
        )

    entry = matching[0]

    if entry.model_class is not None:
        # Fast path: instantiate directly — no factory call needed.
        model_cls = entry.model_class
        if model_cls.config_class is None:
            raise TypeError(
                f"{model_cls.__name__} must set config_class before loading"
            )
        config = model_cls.config_class.from_dict(cfg_dict)
        model: PretrainedModel = model_cls(config)
    else:
        # Fallback: call factory once to discover the concrete class, then
        # rebuild with the saved config so dimensions match the checkpoint.
        template = entry.factory(pretrained=False)
        model_cls2 = type(template)
        cc2 = model_cls2.config_class
        if cc2 is None:
            raise TypeError(
                f"{model_cls2.__name__} must set config_class before loading"
            )
        saved_config = cc2.from_dict(cfg_dict)
        model = model_cls2(saved_config)

    sd = _lucid.load(str(weights_file), weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(
            f"weights.lucid did not contain a state_dict, got {type(sd).__name__}"
        )
    model.load_state_dict(sd, strict=strict)
    return model


class AutoConfig:
    """Generic config loader — returns the :class:`ModelConfig` for any name."""

    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoConfig cannot be instantiated. Use AutoConfig.from_pretrained(...)."
        )

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> ModelConfig:
        """Return the config for *name_or_path* without allocating model weights.

        Fast path: if the registry entry carries a ``default_config``, it is
        returned directly (O(1), no factory call).  Otherwise the factory is
        called with ``pretrained=False`` and its ``.config`` attribute is
        returned — still no weight download.
        """
        from lucid.models._registry import _REGISTRY, _normalize, is_model

        path = Path(name_or_path)
        if path.is_dir():
            return _load_config_from_directory(path)

        if not is_model(name_or_path):
            from lucid.models._registry import _unknown_model_message

            raise ValueError(
                _unknown_model_message(name_or_path, list(_REGISTRY.keys()))
            )

        # Fast path: default_config pre-registered, no model instantiation.
        entry = _REGISTRY.get(_normalize(name_or_path))
        if entry is not None and entry.default_config is not None:
            return entry.default_config

        # Fallback: build the model (pretrained=False) and return its config.
        from lucid.models._registry import model_entrypoint

        model = model_entrypoint(name_or_path)(pretrained=False)
        return model.config


def _load_config_from_directory(path: Path) -> ModelConfig:
    import json

    from lucid.models._registry import _REGISTRY

    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json not found in {path}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"{cfg_file} did not contain a JSON object")
    model_type = cfg_dict.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError(f"{cfg_file} must declare 'model_type' (string)")

    # Fast path: find an entry with a matching model_type and use its
    # default_config if present.
    for entry in _REGISTRY.values():
        if entry.model_type == model_type and entry.default_config is not None:
            return type(entry.default_config).from_dict(cfg_dict)

    # Fallback: call factory to get the config class, then parse.
    matching = [e for e in _REGISTRY.values() if e.model_type == model_type]
    if not matching:
        raise ValueError(f"No model registered for model_type={model_type!r}")
    template = matching[0].factory(pretrained=False)
    cc_dir = type(template).config_class
    if cc_dir is None:
        raise TypeError(
            f"{type(template).__name__} must set config_class before loading"
        )
    return cc_dir.from_dict(cfg_dict)


class AutoModel(_BaseAutoClass):
    """Backbone — returns the family base class (no task head)."""

    _task: ClassVar[str] = "base"


class AutoModelForImageClassification(_BaseAutoClass):
    _task: ClassVar[str] = "image-classification"


class AutoModelForObjectDetection(_BaseAutoClass):
    _task: ClassVar[str] = "object-detection"


class AutoModelForSemanticSegmentation(_BaseAutoClass):
    _task: ClassVar[str] = "semantic-segmentation"


class AutoModelForCausalLM(_BaseAutoClass):
    _task: ClassVar[str] = "causal-lm"


class AutoModelForMaskedLM(_BaseAutoClass):
    _task: ClassVar[str] = "masked-lm"
