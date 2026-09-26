"""Bind the names annotations mention but a module imports only for type checkers.

Under PEP 649 a function's annotations are evaluated in its own module
globals when something asks for them — ``inspect.signature``, ``help()``, an
editor's tooltip, ``typing.get_type_hints``.  The package imports most of the
types it annotates with inside ``if TYPE_CHECKING:``, to keep import cycles
out, so those names are absent at that moment and the asking raises
``NameError``: 297 public callables did, across ``nn``, ``compile``,
``quantization``, ``models`` and the rest.

Once a module has finished executing, a cycle can no longer bite, so the
names are bound then, from the table ``tools/gen_annotation_names.py``
writes (:mod:`lucid._annotation_names`).  How eagerly depends on when:

* a module imported by ``import lucid`` itself binds only what is already
  loaded and waits for the rest — importing a source for it would make
  ``import lucid`` pay for subpackages it deliberately defers (``lucid.nn``
  alone doubles it, 0.18 s to 0.39 s);
* a module imported later — a subpackage the user asked for — also imports
  its own Lucid and standard-library sources, which that import is paying
  for anyway: once the outermost Lucid import in progress has finished, and
  never inside one.  Importing a source while an enclosing Lucid module is
  still executing reaches that module half-built — ``lucid.nn`` loading
  ``lucid.utils`` loading the rollout driver, whose ``RSSM`` source needs
  ``lucid.nn.GELU`` — and the failed import leaves ``lucid.models`` out of
  ``sys.modules`` with its subpackages still in it, which the next plain
  ``import`` of a model family trips over.

A third-party source is never imported for an annotation.  numpy is a
dependency of the bridge, not of the package, and making it required to
read a signature would be the worse trade; those names bind only if numpy
is already loaded.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any, override

from lucid._annotation_names import NAMES

_MISSING = object()

#: source module -> (target module, bound name, attribute) waiting for it.
_waiting: dict[str, list[tuple[str, str, str | None]]] = {}

#: Set once ``import lucid`` has finished: from then on a module binds its
#: sources eagerly.
_eager = False

#: Modules whose code has run to the end.  The import system keeps a module
#: flagged as initialising until ``exec_module`` returns — which is after
#: :func:`_resolve` runs inside it — so the flag alone would make a module
#: wait for itself: ``lucid.weights`` never received ``Module`` from
#: ``lucid.nn``, which imports it.
_done: set[str] = set()

#: Lucid modules executing right now, one inside another.
_depth = 0

#: Sources to import once ``_depth`` is back to zero, in the order asked for.
_deferred: dict[str, None] = {}


def _importable(source: str) -> bool:
    top = source.split(".")[0]
    return top == "lucid" or top in sys.stdlib_module_names


def _ready(name: str) -> ModuleType | None:
    """``name`` if it is loaded and has finished executing."""
    module = sys.modules.get(name)
    if module is None:
        return None
    if name == "lucid" or name in _done:
        # ``lucid`` is still initialising while :func:`install` runs — as its
        # last statement, so everything it defines is already there.
        return module
    spec = getattr(module, "__spec__", None)
    if spec is not None and getattr(spec, "_initializing", False):
        return None
    return module


def _bind(target: ModuleType, bound: str, source: str, attr: str | None) -> None:
    if bound in target.__dict__:
        return
    module = _ready(source)
    if module is None:
        _waiting.setdefault(source, []).append((target.__name__, bound, attr))
        if _eager and _importable(source):
            _deferred[source] = None
        return
    value: Any = module if attr is None else getattr(module, attr, _MISSING)
    if value is _MISSING:
        # ``from package import submodule`` before the submodule is loaded.
        _waiting.setdefault(f"{source}.{attr}", []).append(
            (target.__name__, bound, None)
        )
        return
    target.__dict__[bound] = value


def _resolve(module: ModuleType) -> None:
    for bound, source, attr in NAMES.get(module.__name__, ()):
        _bind(module, bound, source, attr)
    for target_name, bound, attr in _waiting.pop(module.__name__, ()):
        target = sys.modules.get(target_name)
        if target is not None:
            _bind(target, bound, module.__name__, attr)


def _import_deferred() -> None:
    """Import the sources modules asked for, now that none is half-built.

    Each import binds its waiting names as the source finishes; one that
    asks for more sources adds them here, and they are taken in turn.
    """
    while _deferred:
        source = next(iter(_deferred))
        del _deferred[source]
        if source in sys.modules:
            continue
        try:
            importlib.import_module(source)
        except Exception:  # noqa: BLE001 — an annotation must never break an import
            pass


class _ThenResolve(importlib.abc.Loader):
    """The module's own loader, followed by :func:`_resolve`.

    Everything else — ``get_source`` for ``inspect.getsource``, resource
    readers, ``is_package`` — is the wrapped loader's.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @override
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        created: ModuleType | None = self._inner.create_module(spec)
        return created

    @override
    def exec_module(self, module: ModuleType) -> None:
        global _depth
        _depth += 1
        try:
            self._inner.exec_module(module)
        finally:
            _depth -= 1
        _done.add(module.__name__)
        _resolve(module)
        if _depth == 0:
            _import_deferred()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader of every ``lucid`` module but the test suite's."""

    @override
    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("lucid.") or fullname.startswith("lucid.test."):
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec: importlib.machinery.ModuleSpec | None = find_spec(
                fullname, path, target
            )
            if spec is None:
                continue
            if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                spec.loader = _ThenResolve(spec.loader)
            return spec
        return None


def install() -> None:
    """Bind annotation names now, and for every ``lucid`` module loaded later.

    Called once, as the last statement of ``lucid/__init__.py``.  Modules
    already loaded bind what their sources already provide and wait for the
    rest; a finder placed first on :data:`sys.meta_path` then runs the same
    binding as each later ``lucid`` module finishes executing.  A second
    call does nothing.

    Returns
    -------
    None
    """
    global _eager
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        return
    for name in list(NAMES):
        module = _ready(name)
        if module is not None:
            _resolve(module)
    for name in list(_waiting):
        module = _ready(name)
        if module is not None:
            _resolve(module)
    _eager = True
    sys.meta_path.insert(0, _Finder())
