"""H5, H6 and H7 hold across the whole ``lucid/`` tree, tests included.

The runtime package was clean, but the test tree was not. It held 14 mentions
of the reference framework's packages by name and 401 string annotations, and
nothing checked either rule. They surfaced only when closed issues were re-read
(Linear CHA-15, CHA-16), so this does the checking.

* **H5** — the reference framework is not named outside
  :mod:`lucid.test._fixtures.ref_framework`, the one file allowed to.  The
  pattern is built from that module, so this file does not name it either.
* **H6** — the other vendor's GPU API is not named anywhere.  The word is
  assembled below so this file passes its own check.
* **H7** — no annotation is a string.  Python 3.14 evaluates annotations
  lazily (PEP 649), so a forward reference needs no quotes.  A name used
  only for typing is imported under ``TYPE_CHECKING``.  ``Literal["..."]``
  arguments and ``Annotated`` metadata are values, not types, and are
  allowed.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from lucid.test._fixtures import ref_framework

ROOT = Path(__file__).resolve().parents[2]  # lucid/
_SOURCE = {".py", ".pyi", ".h", ".hpp", ".cpp", ".mm", ".md"}
_REF_FILE = Path(ref_framework.__file__).resolve()
_REF_WORD = re.compile(re.escape(ref_framework._REF_NAME), re.IGNORECASE)
_GPU_WORD = re.compile(r"\b" + "cu" + "da" + r"\b", re.IGNORECASE)


def _sources(suffixes: set[str]) -> Iterator[Path]:
    for path in ROOT.rglob("*"):
        if path.suffix in suffixes and "__pycache__" not in path.parts:
            yield path


def _lines_matching(pattern: re.Pattern[str], skip: Path | None = None) -> list[str]:
    hits = []
    for path in _sources(_SOURCE):
        if skip is not None and path.resolve() == skip:
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if pattern.search(line):
                hits.append(
                    f"{path.relative_to(ROOT.parent)}:{number}: {line.strip()[:100]}"
                )
    return hits


def test_h5_the_reference_framework_is_named_in_one_file() -> None:
    hits = _lines_matching(_REF_WORD, skip=_REF_FILE)
    assert not hits, "H5 — use the ref fixture or 'reference framework':\n" + "\n".join(
        hits
    )


def test_h6_no_other_gpu_api_is_named() -> None:
    hits = _lines_matching(_GPU_WORD)
    assert not hits, "H6 — Lucid's GPU is 'metal':\n" + "\n".join(hits)


def _string_types(annotation: ast.expr) -> Iterator[ast.Constant]:
    """String constants used as types, skipping ``Literal`` and ``Annotated`` metadata."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        yield annotation
        return
    if isinstance(annotation, ast.Subscript):
        base = annotation.value
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name == "Literal":
            return
        yield from _string_types(base)
        if name == "Annotated" and isinstance(annotation.slice, ast.Tuple):
            yield from _string_types(annotation.slice.elts[0])
            return
        yield from _string_types(annotation.slice)
        return
    for child in ast.iter_child_nodes(annotation):
        if isinstance(child, ast.expr):
            yield from _string_types(child)


def _annotations(tree: ast.AST) -> Iterator[ast.expr]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                args.vararg,
                args.kwarg,
            ]:
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def test_h7_no_annotation_is_a_string() -> None:
    hits = []
    for path in _sources({".py"}):
        tree = ast.parse(path.read_text(), filename=str(path))
        for annotation in _annotations(tree):
            for found in _string_types(annotation):
                hits.append(
                    f"{path.relative_to(ROOT.parent)}:{found.lineno}: {found.value!r}"
                )
    assert not hits, "H7 — drop the quotes (PEP 649):\n" + "\n".join(hits)
