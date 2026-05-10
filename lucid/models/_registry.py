"""Task-tagged global model registry with name normalization."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.models._base import PretrainedModel


_FactoryFn = Callable[..., "PretrainedModel"]


@dataclass(frozen=True)
class _RegistryEntry:
    factory: _FactoryFn
    task: str
    family: str
    model_type: str


_REGISTRY: dict[str, _RegistryEntry] = {}


def _normalize(name: str) -> str:
    """Hyphens are equivalent to underscores; case-insensitive lookup."""
    return name.replace("-", "_").lower()


def register_model(
    *,
    task: str = "base",
    family: str | None = None,
    model_type: str | None = None,
) -> Callable[[_FactoryFn], _FactoryFn]:
    """Decorator that registers a factory function under its ``__name__``.

    Args:
        task: Task tag the Auto class will filter on (``"base"``,
            ``"image-classification"``, ``"causal-lm"``, etc.)
        family: Family identifier (e.g. ``"resnet"``); inferred from the
            factory's parent module name if omitted.
        model_type: ``ModelConfig.model_type`` value this factory produces;
            defaults to ``family``.
    """

    def decorator(fn: _FactoryFn) -> _FactoryFn:
        name = _normalize(fn.__name__)
        if name in _REGISTRY:
            raise ValueError(f"Model {name!r} already registered")

        if family is None:
            # Module path looks like ``lucid.models.vision.resnet.pretrained``;
            # the parent (``resnet``) is the family.
            parts = fn.__module__.split(".")
            family_resolved = parts[-2] if len(parts) >= 2 else fn.__module__
        else:
            family_resolved = family

        _REGISTRY[name] = _RegistryEntry(
            factory=fn,
            task=task,
            family=family_resolved,
            model_type=model_type if model_type is not None else family_resolved,
        )
        return fn

    return decorator


def list_models(*, task: str | None = None, family: str | None = None) -> list[str]:
    """All registered names, optionally filtered by task / family. Sorted."""
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


def model_entrypoint(name: str) -> _FactoryFn:
    """Return the registered factory for ``name`` (any task)."""
    norm = _normalize(name)
    entry = _REGISTRY.get(norm)
    if entry is None:
        raise ValueError(_unknown_model_message(name, list(_REGISTRY.keys())))
    return entry.factory


def create_model(
    name: str, *, pretrained: bool = False, **overrides: object
) -> "PretrainedModel":
    """timm-style entrypoint: look up name and call its factory."""
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
    candidate_list = candidates
    nearby = sorted(
        ((_edit_distance(c, norm), c) for c in candidate_list),
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
