"""Every TensorImpl binding outside the op dispatcher tells the tracer (LCD-345).

The op dispatcher records each op for a compile trace or a Core ML export.
Bindings that work on a tensor directly do not go through it. A copy made
there (``transfer_to_device``) was a tensor the trace had never seen, so a
host read of the copy went unnoticed, and an op reading the copy took it as a
feed pinned at its first value. A binding that handed out bytes (``to_bytes``,
which ``pickle``, ``deepcopy`` and ``lucid.save`` use) or a DLPack capsule
read the values without saying so. Either way every later compiled call
answered with the first call's values.

Two hooks in ``lucid/_C/compile/Tracer.h`` close this: ``traced_derive``
for a binding that returns a new tensor made from an existing one, and
``note_host_read`` for one that exposes values. This scan holds every
binding in the files below to them, so a new binding cannot skip them by
accident. The behaviour itself is tested in ``test_host_reads.py``.
"""

import re
from pathlib import Path

import pytest

import lucid

_BINDINGS = Path(lucid.__file__).parent / "_C" / "bindings"
_FILES = ("bind_tensor.cpp", "bind_interop.cpp")

#: Bindings that return a TensorImpl not made from an existing tensor's
#: values. The trace treats each as a constant, which it is.
_FRESH = {
    "from_bytes": "built from a Python bytes object",
    "make_shared_tensor": "a new zero-filled buffer",
    "from_dlpack_metal": "adopts a buffer from outside Lucid",
    "grad_to_tensor": "the gradient, which no forward trace computes",
    "grad_as_impl": "the graph-mode gradient, which no forward trace computes",
}

#: Bindings that hand values to Python but are not host reads of a value the
#: result depends on.
_NOT_A_READ = {
    "to_string": "display only (repr); nothing it returns reaches a result",
    "grad_as_python": "the gradient, which no forward trace computes",
}

#: Bindings to a member function by pointer, whose return type the scan
#: cannot see. ``transfer_to_device`` was one of these. Each is reviewed: none
#: returns a tensor or its values.
_BARE_MEMBERS = {
    "numel",
    "nbytes",
    "is_contiguous",
    "is_dense",
    "storage_offset_bytes",
    "grad_as_python",
    "from_bytes",
    "to_string",
    "copy_from",
    "zero_grad",
    "is_aliased",
}

_START = re.compile(r'\.def(?:_static)?\(\s*"(\w+)"')
_RETURNS_IMPL = re.compile(r"->\s*(?:std::shared_ptr<TensorImpl>|TensorImplPtr)\s*\{")
_EXPOSES = re.compile(r"(?:->|\.)(?:tolist|item|data_as_python|to_bytes)\(\)|metal_to_dlpack\(")
_BARE = re.compile(r'^\.def(?:_static)?\(\s*"\w*",\s*&TensorImpl::')


def _bindings() -> list[tuple[str, str, str]]:
    """``(file, name, source)`` for every ``.def`` / ``.def_static``."""
    found: list[tuple[str, str, str]] = []
    for file in _FILES:
        path = _BINDINGS / file
        if not path.exists():
            pytest.skip("the engine sources are not installed")
        text = path.read_text()
        starts = list(_START.finditer(text))
        ends = [m.start() for m in starts[1:]] + [len(text)]
        for m, end in zip(starts, ends, strict=True):
            found.append((file, m.group(1), _code(text[m.start() : end])))
    return found


def _code(src: str) -> str:
    """``src`` without comments or the text of string literals.

    A block runs to the next binding, so it carries the comment above that
    one, which may well mention ``.item()``.
    """
    src = re.sub(r"//[^\n]*", "", src)
    return re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', src)


def test_the_scan_finds_the_bindings() -> None:
    names = {name for _file, name, _src in _bindings()}
    assert {"transfer_to_device", "to_bytes", "to_dlpack_metal", "tolist"} <= names


def test_a_binding_returning_a_derived_tensor_records_it() -> None:
    missing = [
        f"{file}: {name}"
        for file, name, src in _bindings()
        if _RETURNS_IMPL.search(src) and name not in _FRESH and "traced_derive(" not in src
    ]
    assert not missing, (
        "return the new tensor through compile::traced_derive, or list the "
        f"binding in _FRESH with the reason it is a constant: {missing}"
    )


def test_a_binding_exposing_values_notes_the_read() -> None:
    missing = [
        f"{file}: {name}"
        for file, name, src in _bindings()
        if _EXPOSES.search(src) and name not in _NOT_A_READ and "note_host_read(" not in src
    ]
    assert not missing, (
        "call compile::note_host_read before handing values to Python, or "
        f"list the binding in _NOT_A_READ with the reason: {missing}"
    )


def test_a_binding_by_member_pointer_is_reviewed() -> None:
    unreviewed = [
        f"{file}: {name}"
        for file, name, src in _bindings()
        if _BARE.match(src) and name not in _BARE_MEMBERS
    ]
    assert not unreviewed, (
        "a member bound by pointer may return a tensor or its values without "
        "the trace hooks; bind a lambda that calls them, or add it to "
        f"_BARE_MEMBERS once checked: {unreviewed}"
    )
