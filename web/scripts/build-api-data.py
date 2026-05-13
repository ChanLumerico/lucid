"""
build-api-data.py — Griffe → JSON API pipeline for Lucid docs

Reads lucid/ Python source statically (NumPy docstring parser),
outputs one JSON file per module to web/public/api-data/.

Usage:
    python scripts/build-api-data.py [--module MODULE] [--verbose]

    # Build all (default)
    python scripts/build-api-data.py

    # Single module PoC
    python scripts/build-api-data.py --module lucid.fft
"""

# fmt: off

from __future__ import annotations  # allowed in build scripts (not lucid/ source)

import argparse
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.parent.parent  # lucid repo root
LUCID_SRC   = REPO_ROOT                            # lucid/ lives here
WEB_ROOT    = Path(__file__).parent.parent         # web/
OUT_DIR     = WEB_ROOT / "public" / "api-data"
GITHUB_BASE = "https://github.com/ChanLumerico/lucid/blob"

# ---------------------------------------------------------------------------
# Module manifest
# key   → slug used as the JSON filename  (also the URL slug in the site)
# value → griffe module path to load
# ---------------------------------------------------------------------------

MODULE_MANIFEST: dict[str, str] = {
    "lucid":                "lucid",
    "lucid.tensor":         "lucid._tensor.tensor",   # Tensor class only
    "lucid.nn":             "lucid.nn",
    "lucid.nn.functional":  "lucid.nn.functional",
    "lucid.nn.init":        "lucid.nn.init",
    "lucid.nn.utils":       "lucid.nn.utils",
    "lucid.optim":          "lucid.optim",
    "lucid.autograd":       "lucid.autograd",
    "lucid.func":           "lucid.func",
    "lucid.linalg":         "lucid.linalg",
    "lucid.fft":            "lucid.fft",
    "lucid.signal":         "lucid.signal.windows",
    "lucid.special":        "lucid.special",
    "lucid.distributions":  "lucid.distributions",
    "lucid.utils.data":     "lucid.utils.data",
    "lucid.amp":            "lucid.amp",
    "lucid.profiler":       "lucid.profiler",
    "lucid.einops":         "lucid.einops",
    "lucid.serialization":  "lucid.serialization",
}

# ---------------------------------------------------------------------------
# Griffe helpers
# ---------------------------------------------------------------------------

def _get_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "main"


def _source_link(obj: Any, sha: str) -> str | None:
    """Return a GitHub permalink for a griffe object."""
    try:
        if obj.filepath is None:
            return None
        rel = Path(obj.filepath).relative_to(REPO_ROOT)
        line = obj.lineno or 1
        return f"{GITHUB_BASE}/{sha}/{rel}#L{line}"
    except Exception:
        return None


def _annotation_str(ann: Any) -> str | None:
    """Convert a griffe annotation expression to a plain string."""
    if ann is None:
        return None
    try:
        return str(ann)
    except Exception:
        return None


def _rst_to_text(text: str) -> str:
    """Convert reST markup to markdown-friendly text with $...$ math notation.

    Converts:
      - ``.. math::`` blocks           → $$...$$  (display math)
      - ``:math:`expr```               → $expr$   (inline math)
      - ````code````                   → `code`   (inline code)
      - ``:class:`Foo```, etc.         → `Foo`    (cross-references as inline code)
    """
    # Block math: .. math::\n\n   expr  →  $$\nexpr\n$$
    def _block_math(m: re.Match[str]) -> str:
        body = textwrap.dedent(m.group(1)).strip()
        return f"$$\n{body}\n$$"
    text = re.sub(r"\.\. math::\n\n((?:[ \t]+.+\n?)+)", _block_math, text)

    # Inline math: :math:`expr` → $expr$
    text = re.sub(r":math:`([^`]+)`", r"$\1$", text)

    # Double-backtick code: ``code`` → `code`
    text = re.sub(r"``([^`]+)``", r"`\1`", text)

    # Cross-references: :class:`Tensor` → `Tensor`
    text = re.sub(
        r":(?:class|func|meth|attr|mod|ref|data|exc|obj):`~?([^`]+)`",
        r"`\1`",
        text,
    )

    # Remove citations
    text = re.sub(r":cite:t?:`[^`]+`", "", text)

    return text.strip()


def _get_ast_docstring(fn_name: str) -> str | None:
    """Search implementation .py files for fn_name's docstring via AST.

    Fallback for .pyi stubs that declare functions with '...' body and no
    docstring.  The actual .py implementation has the full docstring.

    Uses ast.get_docstring which returns the evaluated string value, so
    raw-string docstrings (r-prefix) have their backslashes preserved —
    LaTeX sequences like backslash-forall arrive intact.
    """
    import ast

    IMPL_DIRS = [
        LUCID_SRC / "lucid" / "_factories",
        LUCID_SRC / "lucid" / "_ops",
        LUCID_SRC / "lucid" / "autograd",
        LUCID_SRC / "lucid" / "serialization",
        LUCID_SRC / "lucid",
    ]
    for impl_dir in IMPL_DIRS:
        for fpath in sorted(impl_dir.glob("*.py")):
            try:
                tree = ast.parse(fpath.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == fn_name
                ):
                    doc = ast.get_docstring(node)
                    if doc and doc.strip():
                        return doc
    return None


def _parse_docstring(obj: Any, parser: Any) -> dict[str, Any]:
    """Parse a griffe object's docstring into structured sections."""
    result: dict[str, Any] = {
        "summary": None,
        "extended": None,
        "parameters": [],
        "returns": None,
        "raises": [],
        "examples": [],
        "notes": [],
        "attributes": [],
        "warns": [],
    }

    # Prefer the Griffe docstring object; fall back to the AST-extracted __doc__
    # when the stub (.pyi) has no docstring but the .py implementation does.
    if not obj.docstring:
        fn_name = getattr(obj, "name", None) or obj.path.split(".")[-1]
        raw_text = _get_ast_docstring(fn_name)
        if not raw_text:
            return result
        from griffe import Docstring as _GDoc
        doc_obj = _GDoc(raw_text)
    else:
        doc_obj = obj.docstring

    try:
        sections = doc_obj.parse(parser)
    except Exception:
        result["summary"] = _rst_to_text(doc_obj.value.split("\n")[0])
        return result

    from griffe import DocstringSectionKind as K

    for section in sections:
        kind = section.kind
        val  = section.value

        if kind == K.text:
            text = _rst_to_text(str(val))
            if result["summary"] is None:
                # Split first paragraph → summary, rest → extended
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                result["summary"] = paragraphs[0] if paragraphs else text
                rest = "\n\n".join(paragraphs[1:])
                if rest:
                    result["extended"] = rest
            else:
                # Second K.text block: append to extended
                if result["extended"]:
                    result["extended"] += "\n\n" + text
                else:
                    result["extended"] = text

        elif kind == K.parameters:
            for item in val:
                result["parameters"].append({
                    "name":        item.name,
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                    "default":     str(item.default) if item.default is not None and str(item.default) != "required" else None,
                })

        elif kind == K.returns:
            items = list(val) if hasattr(val, "__iter__") and not isinstance(val, str) else [val]
            if items:
                first = items[0]
                result["returns"] = {
                    "annotation":  _annotation_str(getattr(first, "annotation", None)) or _annotation_str(obj.returns),
                    "description": _rst_to_text(getattr(first, "description", "") or ""),
                }

        elif kind == K.raises:
            for item in val:
                result["raises"].append({
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                })

        elif kind == K.examples:
            # Griffe NumPy parser yields a list of (DocstringSectionKind, str) tuples
            parts: list[str] = []
            items_iter = val if hasattr(val, "__iter__") and not isinstance(val, str) else [val]
            for ex in items_iter:
                if isinstance(ex, tuple) and len(ex) == 2:
                    # (DocstringSectionKind.examples|text, code_str) tuple
                    parts.append(str(ex[1]).strip())
                elif isinstance(ex, str):
                    parts.append(ex.strip())
                elif hasattr(ex, "value"):
                    parts.append(str(ex.value).strip())
                else:
                    parts.append(str(ex).strip())
            block = "\n".join(p for p in parts if p)
            if not block:
                block = str(val).strip()
            if block:
                result["examples"].append(block)

        elif kind in (K.admonition,):
            # Notes/warnings expressed as admonitions (reST style)
            note_text = _rst_to_text(str(val) if isinstance(val, str) else
                                     getattr(val, "description", str(val)))
            if note_text:
                result["notes"].append(note_text)

        elif hasattr(K, "notes") and kind == K.notes:
            # NumPy "Notes" section → dedicated kind in newer Griffe
            note_text = _rst_to_text(str(val))
            if note_text:
                result["notes"].append(note_text)

        elif kind == K.attributes:
            for item in val:
                result["attributes"].append({
                    "name":        item.name,
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                })

        elif kind == K.warns:
            for item in val:
                result["warns"].append({
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                })

    return result


def _build_signature(obj: Any) -> str | None:
    """Build a human-readable function/class __init__ signature string."""
    try:
        params = obj.parameters
        parts: list[str] = []
        for p in params:
            if p.name in ("self", "cls"):
                continue
            ann = _annotation_str(p.annotation)
            default = str(p.default) if p.default is not None and str(p.default) != "required" else None
            chunk = p.name
            if ann:
                chunk = f"{chunk}: {ann}"
            if default is not None:
                chunk = f"{chunk} = {default}"
            parts.append(chunk)
        name = obj.name
        return f"{name}({', '.join(parts)})"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Object serialisers
# ---------------------------------------------------------------------------

_KNOWN_LABELS = {"property", "staticmethod", "classmethod", "abstractmethod", "writable"}

def _extract_labels(fn: Any) -> list[str]:
    """Collect meaningful labels from griffe labels + decorators."""
    result: set[str] = set()
    # griffe .labels: set of str
    if hasattr(fn, "labels") and fn.labels:
        result |= set(fn.labels) & _KNOWN_LABELS
    # .decorators: list of Decorator objects whose .value is a string/expression
    if hasattr(fn, "decorators") and fn.decorators:
        for d in fn.decorators:
            val = str(d.value).strip().lstrip("@").split("(")[0]
            if val in _KNOWN_LABELS:
                result.add(val)
    return sorted(result)


def _serialise_function(fn: Any, parser: Any, sha: str) -> dict[str, Any]:
    doc = _parse_docstring(fn, parser)
    ret_ann = _annotation_str(fn.returns)
    if doc["returns"] is None and ret_ann:
        doc["returns"] = {"annotation": ret_ann, "description": ""}

    return {
        "name":       fn.name,
        "path":       fn.path,
        "kind":       "function",
        "labels":     _extract_labels(fn),
        "signature":  _build_signature(fn),
        "source":     _source_link(fn, sha),
        **doc,
    }


def _detect_class_kind(cls: Any) -> str:
    """Return 'abstract', 'dataclass', or 'regular'."""
    # @dataclass decorator check
    if hasattr(cls, "decorators") and cls.decorators:
        for d in cls.decorators:
            if "dataclass" in str(d.value):
                return "dataclass"
    # Any *direct* (non-alias) method has abstractmethod label
    from griffe import Function as _GF, Attribute as _GA
    for _name, member in cls.members.items():
        try:
            if isinstance(member, (_GF, _GA)) and hasattr(member, "labels"):
                if "abstractmethod" in member.labels:
                    return "abstract"
        except Exception:
            pass
    # Explicit ABC / ABCMeta base class
    if hasattr(cls, "bases") and cls.bases:
        for base in cls.bases:
            if "ABC" in str(base):
                return "abstract"
    return "regular"


def _extract_bases(cls: Any) -> list[str]:
    """Return simplified base class names, excluding 'object'."""
    if not hasattr(cls, "bases") or not cls.bases:
        return []
    result: list[str] = []
    for base in cls.bases:
        base_str = str(base).strip()
        if base_str in ("object", ""):
            continue
        # Take the last dotted component for display brevity
        result.append(base_str.split(".")[-1])
    return result


def _serialise_property(attr: Any, sha: str) -> dict[str, Any]:
    """Serialise a Griffe Attribute that has the 'property' label as a pseudo-function."""
    from griffe import Parser
    labels = sorted(set(attr.labels) & _KNOWN_LABELS)
    ann = _annotation_str(attr.annotation)
    doc = _parse_docstring(attr, Parser.numpy) if hasattr(attr, "docstring") else {
        "summary": None, "extended": None, "parameters": [], "returns": None,
        "raises": [], "examples": [], "notes": [], "attributes": [], "warns": [],
    }
    if doc["returns"] is None and ann:
        doc["returns"] = {"annotation": ann, "description": ""}
    return {
        "name":      attr.name,
        "path":      attr.path,
        "kind":      "function",
        "labels":    labels,
        "signature": f"{attr.name}: {ann}" if ann else attr.name,
        "source":    _source_link(attr, sha),
        **doc,
    }


def _serialise_class(cls: Any, parser: Any, sha: str) -> dict[str, Any]:
    doc = _parse_docstring(cls, parser)

    # Collect public methods + properties
    # Keep selected dunders that are part of the public interface
    KEEP_DUNDERS = {"__init__", "__call__", "__len__", "__iter__",
                    "__getitem__", "__setitem__", "__repr__", "__str__",
                    "__add__", "__radd__", "__iadd__", "__sub__", "__rsub__",
                    "__mul__", "__rmul__", "__matmul__", "__truediv__",
                    "__floordiv__", "__mod__", "__pow__", "__neg__",
                    "__pos__", "__abs__", "__bool__", "__float__", "__int__",
                    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
                    "__enter__", "__exit__"}

    methods: list[dict[str, Any]] = []
    for name, member in cls.members.items():
        from griffe import Function as GriffeFunction, Attribute as GriffeAttr

        # Skip private names (but keep selected dunders)
        if name.startswith("_") and name not in KEEP_DUNDERS:
            continue

        if isinstance(member, GriffeFunction):
            methods.append(_serialise_function(member, parser, sha))
        elif isinstance(member, GriffeAttr):
            # Include Attribute members that are @property (have 'property' label)
            attr_labels = set(getattr(member, "labels", set()))
            if "property" in attr_labels:
                methods.append(_serialise_property(member, sha))
            # Skip plain class/instance attributes (not properties)

    return {
        "name":       cls.name,
        "path":       cls.path,
        "kind":       "class",
        "class_kind": _detect_class_kind(cls),
        "bases":      _extract_bases(cls),
        "labels":     [],   # classes don't have labels; field kept for schema consistency
        "signature":  _build_signature(cls),
        "source":     _source_link(cls, sha),
        "methods":    methods,
        **doc,
    }


def _resolve_member(member: Any) -> Any:
    """Resolve an Alias to its final target, return None on failure."""
    from griffe import Alias as GriffeAlias
    if isinstance(member, GriffeAlias):
        try:
            return member.final_target
        except Exception:
            return None
    return member


def _parse_dunder_all(mod: Any) -> set[str] | None:
    """Return the set of names in __all__ if it exists, otherwise None."""
    import ast
    from griffe import Attribute as GriffeAttr
    all_attr = mod.members.get("__all__")
    if not isinstance(all_attr, GriffeAttr):
        return None
    try:
        names = ast.literal_eval(str(all_attr.value))
        if isinstance(names, (list, tuple)):
            return set(names)
    except Exception:
        pass
    return None


def _serialise_module(mod: Any, parser: Any, sha: str, *, slug: str) -> dict[str, Any]:
    """Top-level serialiser for a module."""
    from griffe import Function as GriffeFunction, Class as GriffeClass, Module as GriffeModule

    doc = _parse_docstring(mod, parser)
    members: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    # Honour __all__ when present — this is the authoritative public API surface.
    # Without this, Griffe picks up every imported name (e.g. Tensor imported into
    # lucid.nn.init for type-annotation use) and leaks cross-package symbols.
    allowed_names = _parse_dunder_all(mod)

    for name, raw_member in mod.members.items():
        if name.startswith("_"):
            continue
        # __all__ filter: skip names not explicitly exported
        if allowed_names is not None and name not in allowed_names:
            continue

        member = _resolve_member(raw_member)
        if member is None:
            continue
        # Deduplicate — same class re-exported from multiple aliases
        if member.path in seen_paths:
            continue
        seen_paths.add(member.path)

        try:
            if isinstance(member, GriffeFunction):
                members.append(_serialise_function(member, parser, sha))
            elif isinstance(member, GriffeClass):
                members.append(_serialise_class(member, parser, sha))
            elif isinstance(member, GriffeModule):
                pass
        except Exception as exc:
            if VERBOSE:
                print(f"  [warn] skipping {name}: {exc}")

    return {
        "slug":    slug,
        "name":    mod.name,
        "path":    mod.path,
        "kind":    "module",
        "source":  _source_link(mod, sha),
        "members": members,
        **doc,
    }


# ---------------------------------------------------------------------------
# Special handler: lucid.tensor (Tensor class only)
# ---------------------------------------------------------------------------

def _build_tensor_data(loader: Any, parser: Any, sha: str) -> dict[str, Any]:
    """Extract only the Tensor class from lucid._tensor.tensor."""
    from griffe import Class as GriffeClass

    mod = loader.load("lucid._tensor.tensor")
    tensor_cls = mod.members.get("Tensor")
    if tensor_cls is None or not isinstance(tensor_cls, GriffeClass):
        return {"slug": "lucid.tensor", "kind": "module", "name": "Tensor",
                "path": "lucid.Tensor", "members": [], "summary": None}

    serialised = _serialise_class(tensor_cls, parser, sha)
    return {
        "slug":    "lucid.tensor",
        "kind":    "class-module",   # special kind for the site
        "name":    "Tensor",
        "path":    "lucid.Tensor",
        "source":  _source_link(tensor_cls, sha),
        **{k: v for k, v in serialised.items() if k not in ("name", "path", "kind", "source")},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

VERBOSE = False


def build_one(slug: str, griffe_path: str, loader: Any, parser: Any, sha: str) -> None:
    from griffe import Module as GriffeModule, Class as GriffeClass

    out_file = OUT_DIR / f"{slug}.json"
    print(f"  {slug} → {out_file.name}", end="", flush=True)

    try:
        if slug == "lucid.tensor":
            data = _build_tensor_data(loader, parser, sha)
        else:
            obj = loader.load(griffe_path)
            if isinstance(obj, GriffeModule):
                data = _serialise_module(obj, parser, sha, slug=slug)
            elif isinstance(obj, GriffeClass):
                serialised = _serialise_class(obj, parser, sha)
                data = {"slug": slug, **serialised}
            else:
                data = {"slug": slug, "kind": "unknown", "name": griffe_path}

        out_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        member_count = len(data.get("members", data.get("methods", [])))
        print(f"  ✓  ({member_count} members)")

    except Exception as exc:
        print(f"  ✗  {exc}")
        if VERBOSE:
            import traceback
            traceback.print_exc()


def main() -> None:
    global VERBOSE

    parser_args = argparse.ArgumentParser(description="Build Lucid API JSON data")
    parser_args.add_argument("--module", help="Build only this slug (e.g. lucid.fft)")
    parser_args.add_argument("--verbose", action="store_true")
    args = parser_args.parse_args()
    VERBOSE = args.verbose

    # Verify griffe is available
    try:
        from griffe import GriffeLoader, Parser, DocstringSectionKind  # noqa: F401
    except ImportError:
        print("ERROR: griffe not installed. Run: pip install griffe")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = _get_commit_sha()

    from griffe import GriffeLoader, Parser
    loader = GriffeLoader(
        docstring_parser=Parser.numpy,
        search_paths=[str(LUCID_SRC)],
        allow_inspection=False,  # static analysis only — no C extension import needed
    )

    manifest = MODULE_MANIFEST
    if args.module:
        if args.module not in manifest:
            print(f"ERROR: unknown slug '{args.module}'. Valid: {list(manifest)}")
            sys.exit(1)
        manifest = {args.module: manifest[args.module]}

    print(f"Building {len(manifest)} module(s)  [commit: {sha}]")
    for slug, griffe_path in manifest.items():
        build_one(slug, griffe_path, loader, Parser.numpy, sha)

    print(f"\nDone. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
