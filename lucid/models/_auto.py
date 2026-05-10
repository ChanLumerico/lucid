"""Auto classes — task-aware generic model / config loaders.

Each ``AutoModelFor{Task}`` is a thin shell that delegates to the registry,
filtered by ``_task``.  Same string can resolve different classes under
different Auto types: ``AutoModel.from_pretrained("resnet_50")`` returns the
backbone, ``AutoModelForImageClassification.from_pretrained("resnet_50")``
returns the classification head.
"""

from pathlib import Path
from typing import ClassVar, TYPE_CHECKING, cast

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
        """Resolve ``name_or_path`` and return a fully constructed model."""
        from lucid.models._registry import _registry_lookup, is_model

        # Local directory takes priority — needed for save_pretrained
        # round-trip from Tier-1 tests.
        path = Path(name_or_path)
        if path.is_dir():
            return _load_from_directory(cls._task, path, strict=strict)

        if not is_model(name_or_path):
            from lucid.models._registry import _unknown_model_message, _REGISTRY

            raise ValueError(_unknown_model_message(name_or_path, list(_REGISTRY.keys())))

        entry = _registry_lookup(name_or_path, task=cls._task)
        return entry.factory(pretrained=True)


def _load_from_directory(task: str, path: Path, *, strict: bool) -> PretrainedModel:
    """Resolve a saved-model directory to the right class via ``model_type``.

    The directory must contain ``config.json`` (with ``model_type`` field)
    and ``weights.lucid``.  We look up which registered family handles that
    ``model_type`` for the current ``task``, then call its factory's class.
    """
    import json

    import lucid as _lucid
    from lucid.models._registry import _REGISTRY

    cfg_file = path / "config.json"
    weights_file = path / "weights.lucid"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json not found in {path}")
    if not weights_file.exists():
        raise FileNotFoundError(f"weights.lucid not found in {path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"{cfg_file} did not contain a JSON object")
    model_type = cfg_dict.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError(f"{cfg_file} must declare 'model_type' (string)")

    matching: list[tuple[str, object]] = [
        (name, entry)
        for name, entry in _REGISTRY.items()
        if entry.task == task and entry.model_type == model_type
    ]
    if not matching:
        raise ValueError(
            f"No model registered for task={task!r}, model_type={model_type!r}"
        )

    # Any matching factory has the same backing class — pick the first.
    _, entry = matching[0]
    # Construct without weights via factory(pretrained=False), then overwrite.
    model = entry.factory(pretrained=False)  # type: ignore[attr-defined]
    config_cls = type(model).config_class
    config = config_cls.from_dict(cfg_dict)
    # Re-instantiate with the saved config (factory may have built default).
    from lucid.models._base import PretrainedModel as _PM
    model = cast(_PM, type(model)(config))
    sd = _lucid.load(str(weights_file), weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(
            f"weights.lucid did not contain a state_dict, got {type(sd).__name__}"
        )
    model.load_state_dict(sd, strict=strict)
    return model


class AutoConfig:
    """Generic config loader."""

    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoConfig cannot be instantiated. Use AutoConfig.from_pretrained(...)."
        )

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> ModelConfig:
        from lucid.models._registry import _REGISTRY, is_model

        path = Path(name_or_path)
        if path.is_dir():
            return _load_config_from_directory(path)

        if not is_model(name_or_path):
            from lucid.models._registry import _unknown_model_message

            raise ValueError(_unknown_model_message(name_or_path, list(_REGISTRY.keys())))

        # For a registered name, build the model with pretrained=False to
        # get its default config — cheap because no weights are touched.
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

    matching = [e for e in _REGISTRY.values() if e.model_type == model_type]
    if not matching:
        raise ValueError(f"No model registered for model_type={model_type!r}")
    factory = matching[0].factory
    template_model = factory(pretrained=False)
    config_cls = type(template_model).config_class
    return config_cls.from_dict(cfg_dict)


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
