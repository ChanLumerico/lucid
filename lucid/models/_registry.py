"""Task-tagged global model registry with name normalization."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lucid.models._base import ModelConfig, PretrainedModel


class ModelFactory(Protocol):
    """Protocol satisfied by every factory function passed to @register_model.

    The ``pretrained`` flag controls whether to download and load weights.
    ``**overrides`` allows callers to override config fields at creation time
    (e.g. ``create_model("resnet_50", num_classes=10)``).
    """

    __name__: str  # every Python function has __name__; Protocol must declare it

    def __call__(
        self,
        pretrained: bool = False,
        **overrides: object,
    ) -> PretrainedModel: ...


@dataclass(frozen=True)
class _RegistryEntry:
    factory: ModelFactory
    task: str
    family: str
    model_type: str
    # Optional fast-path fields: when provided, Auto classes avoid calling the
    # factory just to discover the class or default config.
    model_class: type[PretrainedModel] | None = field(default=None)
    default_config: ModelConfig | None = field(default=None)


_REGISTRY: dict[str, _RegistryEntry] = {}


def _normalize(name: str) -> str:
    """Hyphens are equivalent to underscores; case-insensitive lookup."""
    return name.replace("-", "_").lower()


def register_model(
    *,
    task: str = "base",
    family: str | None = None,
    model_type: str | None = None,
    model_class: type[PretrainedModel] | None = None,
    default_config: ModelConfig | None = None,
) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator that registers a factory function under its ``__name__``.

    Parameters
    ----------
    task:
        Task tag used by Auto classes for typed dispatch
        (``"base"``, ``"image-classification"``, ``"causal-lm"``, …).
    family:
        Architecture family (e.g. ``"resnet"``); inferred from the
        factory's parent module name when omitted.
    model_type:
        ``ModelConfig.model_type`` this factory produces; defaults to
        *family*.
    model_class:
        The concrete :class:`PretrainedModel` subclass this factory
        returns.  When supplied, :class:`AutoModel` directory-loading
        avoids a redundant factory call — recommended for all Phase 1+
        registrations.
    default_config:
        The default :class:`ModelConfig` for this variant (i.e., the
        config the factory uses when ``pretrained=False``).  When
        supplied, :class:`AutoConfig` returns it instantly without
        instantiating the model.
    """

    def decorator(fn: ModelFactory) -> ModelFactory:
        name = _normalize(fn.__name__)
        if name in _REGISTRY:
            raise ValueError(f"Model {name!r} already registered")

        if family is None:
            # Module path looks like ``lucid.models.vision.resnet.pretrained``;
            # the second-to-last component is the family name.
            parts = fn.__module__.split(".")
            family_resolved = parts[-2] if len(parts) >= 2 else fn.__module__
        else:
            family_resolved = family

        mt = model_type if model_type is not None else family_resolved

        _REGISTRY[name] = _RegistryEntry(
            factory=fn,
            task=task,
            family=family_resolved,
            model_type=mt,
            model_class=model_class,
            default_config=default_config,
        )
        return fn

    return decorator


def list_models(*, task: str | None = None, family: str | None = None) -> list[str]:
    """All registered model names, optionally filtered. Sorted alphabetically."""
    out: list[str] = []
    for name, entry in _REGISTRY.items():
        if task is not None and entry.task != task:
            continue
        if family is not None and entry.family != family:
            continue
        out.append(name)
    return sorted(out)


def is_model(name: str) -> bool:
    return _normalize(name) in _REGISTRY


def model_entrypoint(name: str) -> ModelFactory:
    """Return the registered factory for *name* (any task)."""
    norm = _normalize(name)
    entry = _REGISTRY.get(norm)
    if entry is None:
        raise ValueError(_unknown_model_message(name, list(_REGISTRY.keys())))
    return entry.factory


def create_model(
    name: str, *, pretrained: bool = False, **overrides: object
) -> PretrainedModel:
    """timm-style entry point: look up name and call its factory."""
    factory = model_entrypoint(name)
    return factory(pretrained=pretrained, **overrides)


def _registry_lookup(name: str, *, task: str) -> _RegistryEntry:
    """Lookup with task filter — used by Auto classes for typed dispatch."""
    norm = _normalize(name)
    entry = _REGISTRY.get(norm)
    if entry is None:
        raise ValueError(_unknown_model_message(name, list(_REGISTRY.keys())))
    if entry.task != task:
        raise ValueError(
            f"Model {name!r} is registered for task {entry.task!r}, "
            f"but task {task!r} was requested. Use the matching Auto class."
        )
    return entry


def _unknown_model_message(name: str, candidates: list[str]) -> str:
    """Build an error message with up to 3 typo suggestions."""
    norm = _normalize(name)
    nearby = sorted(
        ((_edit_distance(c, norm), c) for c in candidates),
        key=lambda t: t[0],
    )
    suggestions = [c for d, c in nearby[:3] if d <= 3]
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return f"Unknown model {name!r}.{hint}"


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — small dependency-free implementation."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev: list[int] = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr: list[int] = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]
