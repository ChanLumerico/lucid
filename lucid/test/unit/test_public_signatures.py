"""Every public callable can say what it takes.

``inspect.signature`` is not a typing nicety. It is what ``help()``
prints, what an editor shows in a tooltip, and what the documentation
build reads to produce a page's parameter table. A callable it cannot
read has no signature anywhere those look.

Under PEP 649 a function's annotations are evaluated, on demand, in that
function's own module globals. A name imported only inside an
``if TYPE_CHECKING:`` block is therefore absent at the moment somebody
asks — and ``Tensor`` is the return type of most of this package. A
third of ``lucid.__all__`` was raising ``NameError`` when asked, silently
enough that nothing noticed until an export needed one factory's
signature.

The rule the package follows is that a module whose annotations mention
``Tensor`` binds it at runtime: at the foot of the file where the import
cycle allows, and from ``lucid/__init__.py`` for the three factory
modules that run while ``Tensor`` is still being defined.
"""

import inspect

import pytest

import lucid

#: Two that stay unreadable, for reasons that are not oversights.
#:
#: ``Generator`` is a pybind11 type whose ``__init__`` is a built-in
#: method, and CPython cannot describe those from Python at all.
#:
#: ``from_numpy`` annotates a numpy type, and numpy is a dependency of
#: the bridge rather than of the package: ``converters`` imports it
#: inside the one function that needs it so a Lucid install without
#: numpy still works. Binding it at module scope to make this signature
#: readable would turn an optional import into a required one, which is
#: the worse trade.
NO_PYTHON_SIGNATURE = {"Generator", "from_numpy"}


def _public_callables() -> list[str]:
    """Names in ``lucid.__all__`` that are callable.

    ``dir(lucid)`` is not the list: the package resolves most of its
    surface through ``__getattr__``, so the names that matter are absent
    from it until something asks for them — which is exactly how this
    class of defect stayed invisible.
    """
    found = []
    for name in lucid.__all__:
        try:
            value = getattr(lucid, name)
        except Exception:  # noqa: BLE001 — an unreachable name is another test's
            continue
        if callable(value):
            found.append(name)
    return found


@pytest.mark.parametrize("name", _public_callables())
def test_a_public_callable_has_a_readable_signature(name: str) -> None:
    if name in NO_PYTHON_SIGNATURE:
        pytest.skip("a pybind11 built-in has no Python-readable signature")
    inspect.signature(getattr(lucid, name))


def test_the_tensor_constructor_is_readable() -> None:
    """The class this package is named for, in particular.

    ``Tensor.__init__`` annotates ``dtype`` and ``device`` — and the
    class body defines properties of those names. PEP 649 evaluates the
    annotations with the class namespace in scope, so the bare names
    resolved to the properties and the signature raised ``TypeError``.
    The parameters are spelled through aliases now.
    """
    signature = inspect.signature(lucid.Tensor)
    assert "dtype" in signature.parameters
    assert "device" in signature.parameters


def test_a_module_that_annotates_tensor_binds_it() -> None:
    """The rule stated as a check, so a new module inherits it.

    A module can annotate ``Tensor`` without importing it at runtime and
    look perfectly correct — until something reads a signature. This
    walks the modules the public surface actually comes from.
    """
    import sys

    missing = []
    for name in _public_callables():
        if name in NO_PYTHON_SIGNATURE:
            continue
        value = getattr(lucid, name)
        module = sys.modules.get(getattr(value, "__module__", ""))
        if module is None or not module.__name__.startswith("lucid."):
            continue
        annotates = "Tensor" in str(getattr(value, "__annotations__", {}))
        if annotates and not hasattr(module, "Tensor"):
            missing.append(f"{module.__name__}.{name}")
    assert not missing, (
        "these modules annotate Tensor but do not bind it at runtime, so "
        f"inspect.signature cannot read them: {sorted(set(missing))}"
    )
