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

That rule once covered ``lucid.__all__`` alone, by binding ``Tensor`` at the
foot of each file; 297 callables across ``nn``, ``compile``,
``quantization``, ``models`` and the rest still raised.  Now
:mod:`lucid._annotation_resolver` binds every name a module imports under
``TYPE_CHECKING`` once the module has finished executing, from a table
``tools/gen_annotation_names.py`` writes — and the whole surface is held
here, not only the top level.
"""

import annotationlib
import inspect
import subprocess
import sys
from pathlib import Path

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
#:
#: Each has a test below that asserts the reason still holds, so an
#: entry cannot outlive it unnoticed.
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


@pytest.mark.parametrize(
    "name", [n for n in _public_callables() if n not in NO_PYTHON_SIGNATURE]
)
def test_a_public_callable_has_a_readable_signature(name: str) -> None:
    inspect.signature(getattr(lucid, name))


def test_the_exemptions_are_still_public_callables() -> None:
    """An entry for a name that left the surface is a stale entry."""
    assert NO_PYTHON_SIGNATURE <= set(_public_callables())


def test_generator_is_a_pybind11_type_python_cannot_describe() -> None:
    """The exemption holds only while ``Generator`` is the bound C++ class.

    Should it gain a Python wrapper, or a ``__signature__``, this fails and
    the name leaves ``NO_PYTHON_SIGNATURE``.
    """
    generator = lucid.Generator
    assert type(generator).__name__ == "pybind11_type"
    assert generator.__module__ == "lucid._C.engine"
    with pytest.raises(ValueError, match="no signature found for builtin"):
        inspect.signature(generator)


def test_from_numpy_is_unreadable_only_through_its_numpy_annotation() -> None:
    """The one thing missing is the lazily imported ``np``.

    Evaluating the annotations fails on that name and nothing else: asked
    for forward references instead of values, the signature reads.
    """
    with pytest.raises(NameError, match="'np' is not defined"):
        inspect.signature(lucid.from_numpy)
    signature = inspect.signature(
        lucid.from_numpy, annotation_format=annotationlib.Format.FORWARDREF
    )
    assert list(signature.parameters) == ["arr"]


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


_REPO = Path(__file__).resolve().parents[3]


def test_every_public_callable_in_every_subpackage_has_a_readable_signature() -> None:
    """The whole surface the audit walks, not only ``lucid.__all__``."""
    from lucid.test.audit import _surface

    unreadable = []
    for symbol in _surface.enumerate_surface():
        value = _surface.resolve(symbol)
        if value is None or not callable(value):
            continue
        if symbol.qualname.rsplit(".", 1)[-1] in NO_PYTHON_SIGNATURE:
            continue
        try:
            inspect.signature(value)
        except NameError as exc:
            unreadable.append(f"{symbol.qualname}: {exc}")
        except ValueError, TypeError:
            pass  # a builtin or C type: not something an annotation fixes
    assert not unreadable, "\n".join(unreadable)


def test_the_annotation_table_is_fresh() -> None:
    """``lucid/_annotation_names.py`` matches the ``TYPE_CHECKING`` blocks."""
    result = subprocess.run(
        [sys.executable, "-m", "tools.gen_annotation_names", "--check"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_binding_names_imports_nothing_the_package_defers() -> None:
    """Readable signatures, at no cost to ``import lucid`` or to numpy.

    ``lucid.nn`` stays unloaded by ``import lucid`` alone, and reading every
    ``lucid.nn`` signature never imports numpy — a dependency of the bridge,
    not of the package.
    """
    code = (
        "import sys, inspect, lucid\n"
        "assert 'lucid.nn' not in sys.modules, 'import lucid now loads lucid.nn'\n"
        "import lucid.nn as nn\n"
        "for name in dir(nn):\n"
        "    obj = getattr(nn, name)\n"
        "    if callable(obj):\n"
        "        try: inspect.signature(obj)\n"
        "        except (ValueError, TypeError): pass\n"
        "assert 'numpy' not in sys.modules, 'reading a signature imported numpy'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO
    )
    assert result.returncode == 0, result.stderr[-2000:]
