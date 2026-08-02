"""Enumerating what there is to audit.

Deliberately excludes ``lucid.models``: the zoo is 593 symbols of
composition over the primitives audited here, it has its own contract
test and its own device sweep, and folding it in would triple the run
time while diluting the coverage number this tool exists to report
honestly.

Some callables cannot be surveyed because *calling them changes the
process*.  A first version of this sweep walked names alphabetically,
reached ``set_grad_enabled``, called it with a tensor, and every op after
it failed — 278 of them.  ``STATEFUL`` is that lesson.
"""

import importlib
import inspect
from typing import TYPE_CHECKING, Any

import lucid
import lucid._tensor.tensor as _tensor_module
import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Subsystem name -> (module path, kind).  ``kind`` selects how symbols
#: are probed: ``"op"`` are functions over tensors, ``"module"`` are
#: ``nn.Module`` subclasses, ``"optim"`` are optimizers, ``"other"`` is
#: enumerated for the coverage denominator but has no numeric axis.
SUBSYSTEMS: dict[str, tuple[str, str]] = {
    "lucid": ("lucid", "op"),
    "tensor": ("lucid._tensor.tensor", "op"),
    "nn.functional": ("lucid.nn.functional", "op"),
    "nn": ("lucid.nn", "module"),
    "linalg": ("lucid.linalg", "op"),
    "fft": ("lucid.fft", "op"),
    "special": ("lucid.special", "op"),
    "einops": ("lucid.einops", "op"),
    "signal": ("lucid.signal", "op"),
    "distributions": ("lucid.distributions", "other"),
    "optim": ("lucid.optim", "optim"),
    "diffeq": ("lucid.diffeq", "other"),
    "serialization": ("lucid.serialization", "other"),
    "amp": ("lucid.amp", "other"),
    "quantization": ("lucid.quantization", "other"),
    "compile": ("lucid.compile", "other"),
    "profiler": ("lucid.profiler", "other"),
    "metal": ("lucid.metal", "other"),
    "utils": ("lucid.utils", "other"),
}

#: Excluded from the whole audit, by design.  Stated here rather than
#: silently omitted so the coverage figure means what it says.
EXCLUDED: dict[str, str] = {
    "lucid.models": "the model zoo — 593 symbols with their own contract and device suites",
}

#: Calling these mutates process-wide state, opens files, spawns work or
#: blocks.  They are counted in the denominator and never invoked.
STATEFUL: tuple[str, ...] = (
    "set_",
    "get_",
    "manual_seed",
    "seed",
    "no_grad",
    "enable_grad",
    "inference_mode",
    "save",
    "load",
    "compile",
    "device",
    "dtype",
    "print",
    "config",
    "profile",
    "benchmark",
    "synchronize",
    "empty_cache",
    "use_deterministic",
    "are_deterministic",
    "init_",
    "register_",
    "share_memory",
    "pin_memory",
    "record_stream",
    "backward",
    "retain_",
    "requires_grad_",
    "detach_",
    "apply_",
    "to",
    "cpu",
    "metal",
    "cuda",
    "item",
    "tolist",
    "numpy",
    "storage",
    "data_ptr",
    "set_printoptions",
)

#: Draw fresh random numbers, so two calls never agree.  Probed for shape
#: and dtype but never compared value-for-value.
STOCHASTIC: tuple[str, ...] = (
    "rand",
    "randn",
    "randint",
    "randperm",
    "normal",
    "uniform",
    "bernoulli",
    "multinomial",
    "poisson",
    "dropout",
    "rrelu",
    "gumbel",
    "empty",
    "sample",
    "rsample",
)


class Symbol:
    """One auditable name.

    Attributes
    ----------
    qualname : str
        How a user would write it — ``"F.conv2d"``, ``"lucid.exp"``.
    subsystem : str
        Key into :data:`SUBSYSTEMS`.
    kind : str
        ``"op"`` / ``"module"`` / ``"optim"`` / ``"other"``.
    obj : object
        The callable or class itself.
    """

    __slots__ = ("qualname", "subsystem", "kind", "obj", "flags")

    def __init__(self, qualname: str, subsystem: str, kind: str, obj: Any) -> None:
        self.qualname = qualname
        self.subsystem = subsystem
        self.kind = kind
        self.obj = obj
        self.flags: set[str] = set()
        tail = qualname.rsplit(".", 1)[-1]
        if any(tail.startswith(p) or tail == p.rstrip("_") for p in STATEFUL):
            self.flags.add("stateful")
        if tail.endswith("_") and not tail.startswith("_"):
            self.flags.add("inplace")
        if any(p in tail for p in STOCHASTIC):
            self.flags.add("stochastic")

    @property
    def inert(self) -> bool:
        """bool: Safe to call during a survey."""
        return "stateful" not in self.flags

    @property
    def short(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]

    def __repr__(self) -> str:
        return f"Symbol({self.qualname!r}, {self.kind})"


_PREFIX = {
    "lucid": "lucid.",
    "tensor": "Tensor.",
    "nn.functional": "F.",
    "nn": "nn.",
}


def _names(module: Any) -> list[str]:
    declared = getattr(module, "__all__", None)
    if declared:
        return list(declared)
    return [n for n in dir(module) if not n.startswith("_")]


def enumerate_surface(subsystems: "list[str] | None" = None) -> list[Symbol]:
    """Every public symbol in scope, in a stable order.

    Parameters
    ----------
    subsystems : list of str, optional
        Restrict to these keys of :data:`SUBSYSTEMS`.  ``None`` takes all.

    Returns
    -------
    list of Symbol
    """
    wanted = subsystems or list(SUBSYSTEMS)
    out: list[Symbol] = []

    for key in wanted:
        if key not in SUBSYSTEMS:
            continue
        path, kind = SUBSYSTEMS[key]
        prefix = _PREFIX.get(key, f"lucid.{key}.")

        if key == "tensor":
            for name in sorted(dir(_tensor_module.Tensor)):
                if name.startswith("_"):
                    continue
                attr = inspect.getattr_static(_tensor_module.Tensor, name, None)
                if attr is None:
                    continue
                out.append(Symbol(f"Tensor.{name}", key, "method", attr))
            continue

        try:
            module = importlib.import_module(path)
        except Exception:  # noqa: BLE001 - a missing optional subsystem is not a defect
            continue

        for name in sorted(_names(module)):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            if isinstance(obj, type):
                resolved = (
                    "module"
                    if _is_nn_module(obj)
                    else ("optim" if kind == "optim" else "class")
                )
                out.append(Symbol(prefix + name, key, resolved, obj))
            elif callable(obj):
                out.append(
                    Symbol(prefix + name, key, "op" if kind == "op" else kind, obj)
                )
            else:
                out.append(Symbol(prefix + name, key, "value", obj))
    return out


def _is_nn_module(obj: type) -> bool:
    try:
        return issubclass(obj, nn.Module)
    except TypeError:
        return False


def resolve(symbol: Symbol) -> Any:
    """The callable to invoke for ``symbol``, or ``None``."""
    if symbol.kind == "method":
        return getattr(lucid, symbol.short, None) or symbol.obj
    return symbol.obj


def counterparts(symbol: Symbol) -> "Iterator[tuple[str, Any]]":
    """Every entry point the same operation is reachable through.

    One op with three spellings is one op with three chances to drift —
    which is how scalar coercion came to exist on the operator path only.
    """
    name = symbol.short
    free = getattr(lucid, name, None)
    if callable(free) and not isinstance(free, type):
        yield "free", free
    method = inspect.getattr_static(_tensor_module.Tensor, name, None)
    if callable(method):
        yield "method", method
    functional = getattr(F, name, None)
    if (
        callable(functional)
        and not isinstance(functional, type)
        and functional is not free
    ):
        yield "functional", functional


__all__ = [
    "EXCLUDED",
    "STATEFUL",
    "STOCHASTIC",
    "SUBSYSTEMS",
    "Symbol",
    "counterparts",
    "enumerate_surface",
    "resolve",
]
