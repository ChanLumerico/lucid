#!/usr/bin/env python3
"""tools/check_source_rules.py — Hard Rules H1, H5, H6 and H7 over the source.

Each rule is a property of one source file, so each is checked per file:

* **H1** (``future-annotations``) — no ``from __future__ import
  annotations``, in ``lucid/`` and ``tools/``.  Python 3.14 evaluates
  annotations lazily (PEP 649); the import turns every annotation back
  into a string.  The rules below hold in ``lucid/`` only: ``tools/``
  converts the reference framework's checkpoints and names it.
* **H5** (``ref-import``, ``ref-name``) — the reference framework is
  imported, and named, only in ``lucid/test/_fixtures/ref_framework.py``.
  Every other test takes the ``ref`` fixture.  The name is read from that
  file's ``_REF_NAME``, so it is written down once.
* **H6** (``gpu-name``) — the other vendor's GPU API is not named; Lucid's
  GPU is ``metal``.
* **H7** (``string-annotation``) — no annotation is a string.  A name used
  only for typing is imported under ``TYPE_CHECKING``.  ``Literal[...]``
  arguments and ``Annotated`` metadata are values, not types.

Usage::

    python3 tools/check_source_rules.py            # the whole lucid/ and tools/
    python3 tools/check_source_rules.py --staged   # the staged blobs (pre-commit)
    python3 tools/check_source_rules.py PATH ...   # these files

Exit 1 on any violation.  ``lucid/test/unit/test_source_rules.py`` runs the
whole tree; ``.githooks/pre-commit`` runs ``--staged``, so a commit can
neither add a violation nor carry one along in a file it touches.  The
script uses the standard library only.
"""

import argparse
import ast
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple, override

REPO_ROOT = Path(__file__).resolve().parent.parent
LUCID = "lucid"
TOOLS = "tools"
REF_FILE = "lucid/test/_fixtures/ref_framework.py"
TEXT_SUFFIXES = frozenset({".py", ".pyi", ".h", ".hpp", ".cpp", ".mm", ".md"})
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
# Assembled so that this file passes its own H6 check.
_GPU_WORD = re.compile(r"\b" + "cu" + "da" + r"\b", re.IGNORECASE)
# Calls whose first argument names a module to import.
_IMPORTERS = frozenset({"import_module", "importorskip", "__import__"})


class Violation(NamedTuple):
    path: str
    line: int
    rule: str
    text: str

    @override
    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.text}"


def ref_name(root: Path = REPO_ROOT) -> str:
    """The reference framework's package name, from ``_REF_NAME`` in its file."""
    tree = ast.parse((root / REF_FILE).read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_REF_NAME" for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise LookupError(f"{REF_FILE} assigns no _REF_NAME string")


class _Rules:
    """The checks, bound to the reference framework's name."""

    def __init__(self, ref: str) -> None:
        self.ref_word = re.compile(re.escape(ref), re.IGNORECASE)
        self.ref_module = re.compile(re.escape(ref) + r"\w*", re.IGNORECASE)

    def _is_ref_module(self, name: str) -> bool:
        return self.ref_module.fullmatch(name.split(".")[0]) is not None

    def text(self, path: str, source: str) -> Iterator[Violation]:
        """H5 ``ref-name`` and H6, line by line."""
        for number, line in enumerate(source.splitlines(), 1):
            if path != REF_FILE and self.ref_word.search(line):
                yield Violation(path, number, "ref-name", line.strip()[:100])
            if _GPU_WORD.search(line):
                yield Violation(path, number, "gpu-name", line.strip()[:100])

    def python(self, path: str, source: str, full: bool) -> Iterator[Violation]:
        """H1, and with ``full`` H5 ``ref-import`` and H7, from the syntax tree."""
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as e:
            yield Violation(path, e.lineno or 0, "syntax", e.msg)
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                if any(alias.name == "annotations" for alias in node.names):
                    yield Violation(
                        path, node.lineno, "future-annotations", ast.unparse(node)
                    )
            if (
                full
                and path != REF_FILE
                and isinstance(node, (ast.Import, ast.ImportFrom, ast.Call))
                and self._imports_ref(node)
            ):
                yield Violation(path, node.lineno, "ref-import", ast.unparse(node))
        if not full:
            return
        for annotation in _annotations(tree):
            for found in _string_types(annotation):
                yield Violation(
                    path, found.lineno, "string-annotation", repr(found.value)
                )

    def _imports_ref(self, node: ast.Import | ast.ImportFrom | ast.Call) -> bool:
        if isinstance(node, ast.Import):
            return any(self._is_ref_module(alias.name) for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return node.level == 0 and self._is_ref_module(node.module or "")
        if not node.args:
            return False
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0]
        return (
            name in _IMPORTERS
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and self._is_ref_module(first.value)
        )


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


def _scope(path: str) -> str | None:
    """``lucid`` or ``tools`` for a repo-relative path the rules read, else None."""
    top, _, _ = path.partition("/")
    suffix = Path(path).suffix
    if top == LUCID and suffix in TEXT_SUFFIXES:
        return LUCID
    if top == TOOLS and suffix in PYTHON_SUFFIXES:
        return TOOLS
    return None


def check(sources: Iterable[tuple[str, str]], ref: str) -> list[Violation]:
    """Every violation in ``(repo-relative path, source text)`` pairs."""
    rules = _Rules(ref)
    found: list[Violation] = []
    for path, source in sources:
        scope = _scope(path)
        if scope == LUCID:
            found.extend(rules.text(path, source))
        if scope is not None and Path(path).suffix in PYTHON_SUFFIXES:
            found.extend(rules.python(path, source, full=scope == LUCID))
    return found


def tree_sources(root: Path = REPO_ROOT) -> Iterator[tuple[str, str]]:
    """Every checked file under ``root/lucid`` and ``root/tools``, as on disk."""
    for top in (LUCID, TOOLS):
        for file in sorted((root / top).rglob("*")):
            path = file.relative_to(root).as_posix()
            if _scope(path) is not None and "__pycache__" not in file.parts:
                yield path, file.read_text(errors="ignore")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def staged_sources(root: Path) -> Iterator[tuple[str, str]]:
    """The staged blob of every added, copied, modified or renamed file.

    ``root`` is any directory of the repository whose index is read — the
    hook's checkout, which is not this script's when a worktree that
    predates the script borrows the main checkout's copy.
    """
    names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    for path in filter(None, names.split("\0")):
        if _scope(path) is not None:
            yield path, _git(root, "show", f":{path}")


def path_sources(
    paths: Iterable[str], root: Path = REPO_ROOT
) -> Iterator[tuple[str, str]]:
    """The given files, as they are on disk; a file outside ``root`` is skipped."""
    for given in paths:
        file = Path(given).resolve()
        if file.is_relative_to(root):
            yield file.relative_to(root).as_posix(), file.read_text(errors="ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Hard Rules H1, H5, H6, H7.")
    parser.add_argument("paths", nargs="*", help="files (default: lucid/, tools/)")
    parser.add_argument("--staged", action="store_true", help="check the staged blobs")
    args = parser.parse_args(argv)
    if args.staged:
        sources = staged_sources(Path.cwd())
    elif args.paths:
        sources = path_sources(args.paths)
    else:
        sources = tree_sources()
    found = check(sources, ref_name())
    for violation in found:
        print(violation, file=sys.stderr)
    if found:
        print(
            f"\n{len(found)} source-rule violation(s) — CLAUDE.md H1 / H5 / H6 / H7.\n"
            "  H5: take the ``ref`` fixture and say 'reference framework'.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
