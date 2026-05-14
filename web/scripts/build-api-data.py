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
# Module manifest — auto-discovered from lucid/ directory structure
# ---------------------------------------------------------------------------
#
# The discovery walk recurses two levels under ``lucid/``, looking for
# directories that contain an ``__init__.py``.  Anything matching is exposed
# as ``lucid.<dir>`` (or ``lucid.<dir>.<subdir>`` at the second level).
#
# - Private packages (leading ``_``) are skipped automatically.
# - Internal-only packages are listed in ``_EXCLUDED_SUBPKGS`` below.
# - Special cases (class-modules, slug-path remappings, .py-file modules)
#   live in ``_MANIFEST_OVERRIDES`` — applied AFTER discovery.
#
# Adding a new public subpackage = create the directory with an
# ``__init__.py``.  No manual manifest edit required.

# Entire subtree excluded — neither this slug nor any descendant is documented.
_EXCLUDED_SUBTREE: set[str] = {
    "lucid.metal",       # GPU backend internals
    "lucid.backends",    # CPU/GPU dispatch internals
    "lucid.test",        # test infrastructure (parity fixtures, etc.)
    "lucid.models",      # zoo — not yet docs-ready
    "lucid.benchmarks",  # internal perf scripts
}

# Self excluded — this slug is dropped but descendants still get discovered.
# Used for umbrella packages whose ``__init__.py`` has no user-facing content
# of its own but whose children DO get documented separately.
_EXCLUDED_SELF: set[str] = {
    "lucid.nn.modules",  # umbrella; content re-exposed via lucid.nn path-grouping
    "lucid.utils",       # umbrella; only lucid.utils.data has docs content
}

# Special-case slug→griffe-path mappings (layered ON TOP of auto-discovery).
_MANIFEST_OVERRIDES: dict[str, str] = {
    "lucid.tensor":  "lucid._tensor.tensor",   # class-module: → Tensor only
    "lucid.signal":  "lucid.signal.windows",   # redirect to subpkg with content
    "lucid.nn.init": "lucid.nn.init",          # .py file at module root, not a dir
}

# Synthesised slugs — these are NOT real Python packages with their own
# discoverable ``__init__.py`` content.  Instead, the build pipeline loads
# the lucid top-level once, partitions its members by the subcategory each
# falls into (via ``_LUCID_NAME_OVERRIDES``), and emits one JSON per
# partition.  Used for groups whose runtime population is dynamic and
# therefore invisible to Griffe's static walker:
#   - ``_factories/``     → registered via lucid's _load_factories hook
#   - ``_ops/``           → bound at import time by ``_populate_free_fns()``
#   - ``_ops/composite/`` → dynamically populated from submodule ``__all__``s
# Each entry maps the public slug to the subcategory tag set by
# ``_LUCID_NAME_OVERRIDES``.
_SYNTH_SLUGS: dict[str, str] = {
    "lucid.creation":      "factories",
    "lucid.ops":           "ops",
    "lucid.ops.composite": "composite",
}


def _discover_manifest() -> dict[str, str]:
    """Walk lucid/ to auto-discover all documentable subpackages.

    Returns a manifest dict ``{slug → griffe_path}``.  Walks two levels of
    public directories under lucid/, then layers the explicit overrides
    above.  The bare ``lucid`` top-level is intentionally NOT included —
    its content is fully re-exported from the private subpackages that
    ``_MANIFEST_OVERRIDES`` already surfaces under friendly aliases
    (``lucid.creation``, ``lucid.ops``, ``lucid.ops.composite``).
    """
    manifest: dict[str, str] = {}
    lucid_dir = LUCID_SRC / "lucid"
    if not lucid_dir.is_dir():
        return manifest

    def _is_public_pkg(p: Path) -> bool:
        return (
            p.is_dir()
            and not p.name.startswith("_")
            and (p / "__init__.py").exists()
        )

    for first in sorted(lucid_dir.iterdir()):
        if not _is_public_pkg(first):
            continue
        slug = f"lucid.{first.name}"
        if slug in _EXCLUDED_SUBTREE:
            continue
        if slug not in _EXCLUDED_SELF:
            manifest[slug] = slug
        for second in sorted(first.iterdir()):
            if not _is_public_pkg(second):
                continue
            sub_slug = f"{slug}.{second.name}"
            if sub_slug in _EXCLUDED_SUBTREE or sub_slug in _EXCLUDED_SELF:
                continue
            manifest[sub_slug] = sub_slug

    manifest.update(_MANIFEST_OVERRIDES)
    return manifest


MODULE_MANIFEST: dict[str, str] = _discover_manifest()

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


# ---------------------------------------------------------------------------
# Source-file → subcategory mapping (used for the lucid top-level sidebar)
# ---------------------------------------------------------------------------

# Order matters: more specific paths first.  Each entry maps a relative path
# fragment under lucid/ to a short subcategory slug.  These rules apply only
# to files in lucid's PRIVATE subpackages — public subpackages like
# ``lucid.autograd`` / ``lucid.serialization`` get per-file grouping via the
# generic fallback inside :func:`_subcategory`.
_SUBCATEGORY_RULES: list[tuple[str, str]] = [
    ("lucid/_ops/composite", "composite"),
    ("lucid/_ops",           "ops"),
    ("lucid/_factories",     "factories"),
    ("lucid/_tensor",        "tensor"),
    ("lucid/_types.py",      "types"),
    ("lucid/_dtype.py",      "dtypes"),
    ("lucid/dtypes.py",      "dtypes"),
    ("lucid/_device.py",     "device"),
    ("lucid/_globals.py",    "globals"),
    ("lucid/_threads.py",    "threads"),
    ("lucid/_dispatch.py",   "dispatch"),
    ("lucid/_vmap_ctx.py",   "vmap"),
]


def _subcategory(
    obj: Any,
    module_root: Path | None = None,
    *,
    is_lucid_toplevel: bool = False,
) -> str | None:
    """Determine the source-directory-based subcategory for a Griffe object.

    Returns a slash-separated path slug based on where the symbol is actually
    defined in the lucid/ source tree, **relative to the module being
    serialised**.  Used by the sidebar to mirror the on-disk directory
    structure rather than the flat re-exported namespace.

    Resolution order (deterministic, applied in this exact order):

      1. **(lucid top-level only)** Name-based override — dynamic loaders
         (``_FACTORY_NAMES`` / ``_OPS_NAMES`` / etc.) that Griffe can't
         statically resolve.  ``is_lucid_toplevel=True`` MUST be set, otherwise
         this step is skipped — preventing names like ``relu`` (which is in
         ``_OPS_NAMES``) from miscategorising members of unrelated modules
         like ``lucid.nn.functional``.
      2. **(lucid top-level only)** Path-based rules for lucid's private
         subpackages (``_factories/``, ``_ops/``, ``_tensor/``, ...).
      3. **Module-root-relative path** — for any module, the member's source
         file path relative to the module's ``__init__.py`` directory.
         ``lucid/nn/modules/conv.py`` (module root ``lucid/nn/``) → subcategory
         ``"modules/conv"``.  Multi-segment slugs let the sidebar render a
         tree that mirrors ``ls lucid/nn/``.
      4. **Fallback** (no module_root context, e.g. Tensor class methods):
         file basename with leading underscores stripped; ``__init__.py`` → None.
    """
    try:
        if is_lucid_toplevel and obj.name in _LUCID_NAME_OVERRIDES:
            return _LUCID_NAME_OVERRIDES[obj.name]
        if obj.filepath is None:
            return None
        p = Path(obj.filepath)
        if is_lucid_toplevel:
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            for path_frag, slug in _SUBCATEGORY_RULES:
                if rel.startswith(path_frag):
                    return slug
        # Module-root-relative path (slash-separated, multi-segment).
        if module_root is not None:
            try:
                rr = p.relative_to(module_root)
                parts = list(rr.parts)
                if not parts:
                    return None
                last = parts[-1]
                if last == "__init__.py":
                    return None    # member is defined in the package's own init
                if last.endswith(".py"):
                    parts[-1] = last[:-3]
                parts = [pp.lstrip("_") for pp in parts]
                parts = [pp for pp in parts if pp]
                if not parts:
                    return None
                return "/".join(parts)
            except ValueError:
                pass
        # Fallback when no module_root context: file basename only.
        if p.name == "__init__.py":
            return None
        name = p.stem.lstrip("_")
        return name or None
    except Exception:
        return None


def _load_lucid_name_overrides() -> dict[str, str]:
    """Parse ``lucid/__init__.py`` to map dynamic-loaded names to subcategories.

    Reads the ``_FACTORY_NAMES``, ``_OPS_NAMES``, and ``_SCATTER_NAMES``
    frozensets defined at module scope, plus ``COMPOSITE_NAMES`` from the
    ``lucid._ops.composite`` package.  These sets are what the runtime
    ``__getattr__`` actually checks, so they are the single source of truth
    for the top-level namespace.
    """
    import ast

    mapping: dict[str, str] = {}
    init_path = LUCID_SRC / "lucid" / "__init__.py"

    NAME_TO_SLUG = {
        "_FACTORY_NAMES":       "factories",
        "_OPS_NAMES":           "ops",
        "_SCATTER_NAMES":       "ops",
        "_METHOD_ALIASES":      "ops",
        "_GRAD_NAMES":          "autograd",
        "_PREDICATE_NAMES":     "predicates",
        "_SERIALIZATION_NAMES": "serialization",
        "_TYPE_ALIAS_NAMES":    "types",
    }

    # Hard-coded overrides for top-level names that aren't in any *_NAMES set
    # but still need a sensible bucket (dtype singletons, deterministic toggles).
    mapping.update({
        "float16": "dtypes", "float32": "dtypes", "float64": "dtypes",
        "bfloat16": "dtypes", "int8": "dtypes", "int16": "dtypes",
        "int32": "dtypes", "int64": "dtypes", "bool_": "dtypes",
        "complex64": "dtypes",
        "use_deterministic_algorithms": "globals",
        "are_deterministic_algorithms_enabled": "globals",
    })

    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except Exception:
        return mapping

    def _iter_assigns(node_iter: Any) -> Any:
        """Yield (target_name, value_node) for both Assign and AnnAssign."""
        for n in node_iter:
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                tgt = n.targets[0]
                if isinstance(tgt, ast.Name):
                    yield tgt.id, n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                yield n.target.id, n.value

    def _extract_string_set(value_node: Any) -> set[str]:
        """Extract a set of string literals from `frozenset([...])`, `{...}`, or `[...]`."""
        try:
            if (
                isinstance(value_node, ast.Call)
                and getattr(value_node.func, "id", None) == "frozenset"
                and value_node.args
            ):
                value_node = value_node.args[0]
            return {n for n in ast.literal_eval(value_node) if isinstance(n, str)}
        except Exception:
            return set()

    for target_id, value_node in _iter_assigns(ast.walk(tree)):
        if target_id in NAME_TO_SLUG:
            slug = NAME_TO_SLUG[target_id]
            for n in _extract_string_set(value_node):
                mapping[n] = slug

    # Composite ops: COMPOSITE_NAMES is built at runtime from each submodule's
    # ``__all__``.  Static analysis: walk every submodule in
    # ``lucid/_ops/composite/*.py`` and union their ``__all__`` lists.
    comp_dir = LUCID_SRC / "lucid" / "_ops" / "composite"
    if comp_dir.exists():
        for comp_file in comp_dir.glob("*.py"):
            if comp_file.name == "__init__.py":
                continue
            try:
                comp_tree = ast.parse(comp_file.read_text(encoding="utf-8"))
                for target_id, value_node in _iter_assigns(ast.walk(comp_tree)):
                    if target_id == "__all__":
                        for n in _extract_string_set(value_node):
                            mapping[n] = "composite"
            except Exception:
                pass

    # OpEntry registry (``lucid/_ops/_registry.py``): the runtime source of
    # truth for which engine ops are exposed.  Extract every ``OpEntry(...)``
    # call's first positional arg (the op name) — these include some names
    # not in ``_OPS_NAMES`` (``dot``, ``inner``, ``outer``) plus all the
    # in-place variants (``add_``, ``sub_``, ``mul_``, ...).
    reg_path = LUCID_SRC / "lucid" / "_ops" / "_registry.py"
    try:
        reg_tree = ast.parse(reg_path.read_text(encoding="utf-8"))
        for node in ast.walk(reg_tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "OpEntry"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                op_name = node.args[0].value
                # Don't clobber a more-specific slug if one is already set.
                mapping.setdefault(op_name, "ops")
    except Exception:
        pass

    return mapping


def _load_op_arity() -> dict[str, int]:
    """Parse ``lucid/_ops/_registry.py`` to map each OpEntry name to its arity.

    OpEntry signature is ``OpEntry(name, callable, arity, ...)`` — the third
    positional arg is the arity.  Used by ``build_synth`` to subcategorise
    the ``lucid.ops`` slug by unary / binary / ternary / etc. so the page
    isn't one giant flat list of 143 ops.
    """
    import ast
    result: dict[str, int] = {}
    reg_path = LUCID_SRC / "lucid" / "_ops" / "_registry.py"
    try:
        tree = ast.parse(reg_path.read_text(encoding="utf-8"))
    except Exception:
        return result
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OpEntry"
            and len(node.args) >= 3
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, int)
        ):
            result[node.args[0].value] = node.args[2].value
    return result


_OP_ARITY: dict[str, int] = _load_op_arity()


_LUCID_NAME_OVERRIDES: dict[str, str] = _load_lucid_name_overrides()


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

    # Inline math: :math:`expr` → $expr$ (join multi-line math to single line)
    def _inline_math(m: re.Match[str]) -> str:
        inner = " ".join(m.group(1).split())   # collapse whitespace/newlines
        return f"${inner}$"
    text = re.sub(r":math:`([^`]+)`", _inline_math, text, flags=re.DOTALL)

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


def _serialise_function(
    fn: Any, parser: Any, sha: str, *,
    module_root: Path | None = None,
    is_lucid_toplevel: bool = False,
) -> dict[str, Any]:
    doc = _parse_docstring(fn, parser)
    ret_ann = _annotation_str(fn.returns)
    if doc["returns"] is None and ret_ann:
        doc["returns"] = {"annotation": ret_ann, "description": ""}

    return {
        "name":        fn.name,
        "path":        fn.path,
        "kind":        "function",
        "labels":      _extract_labels(fn),
        "signature":   _build_signature(fn),
        "source":      _source_link(fn, sha),
        "subcategory": _subcategory(fn, module_root=module_root, is_lucid_toplevel=is_lucid_toplevel),
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


def _serialise_class(
    cls: Any, parser: Any, sha: str, *,
    module_root: Path | None = None,
    is_lucid_toplevel: bool = False,
) -> dict[str, Any]:
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
        "name":        cls.name,
        "path":        cls.path,
        "kind":        "class",
        "class_kind":  _detect_class_kind(cls),
        "bases":       _extract_bases(cls),
        "labels":      [],   # classes don't have labels; field kept for schema consistency
        "signature":   _build_signature(cls),
        "source":      _source_link(cls, sha),
        "subcategory": _subcategory(cls, module_root=module_root, is_lucid_toplevel=is_lucid_toplevel),
        "methods":     methods,
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

    # Compute the module's source directory so members get subcategories
    # relative to it (e.g. ``lucid.nn`` members from ``lucid/nn/modules/conv.py``
    # → subcategory ``"modules/conv"``).
    module_root: Path | None = None
    try:
        if mod.filepath is not None:
            module_root = Path(mod.filepath).parent
    except Exception:
        module_root = None

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

        is_lucid_top = slug == "lucid"
        try:
            if isinstance(member, GriffeFunction):
                members.append(_serialise_function(
                    member, parser, sha,
                    module_root=module_root,
                    is_lucid_toplevel=is_lucid_top,
                ))
            elif isinstance(member, GriffeClass):
                members.append(_serialise_class(
                    member, parser, sha,
                    module_root=module_root,
                    is_lucid_toplevel=is_lucid_top,
                ))
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
    """Extract only the Tensor class from lucid._tensor.tensor.

    Filters the methods list so that operators / factories / composites that
    are already documented as top-level ``lucid.*`` functions are NOT
    duplicated here — the Tensor docs surface only what is genuinely
    Tensor-specific: properties, dunders, in-place mutations, ``new_*``
    constructors, autograd lifecycle (``backward`` / ``retain_grad`` /
    ``register_hook`` / ``requires_grad_``), and bridge methods
    (``item`` / ``numpy`` / ``tolist`` / ``__dlpack__`` / ...).
    """
    from griffe import Class as GriffeClass

    mod = loader.load("lucid._tensor.tensor")
    tensor_cls = mod.members.get("Tensor")
    if tensor_cls is None or not isinstance(tensor_cls, GriffeClass):
        return {"slug": "lucid.tensor", "kind": "module", "name": "Tensor",
                "path": "lucid.Tensor", "members": [], "summary": None}

    serialised = _serialise_class(tensor_cls, parser, sha)

    # Names that have a top-level `lucid.<name>` function — listing them as
    # Tensor methods too is redundant.  Pulled from the same name sets that
    # drive the sidebar subcategory assignment, so the two views stay in sync.
    redundant_names: set[str] = {
        name for name, slug in _LUCID_NAME_OVERRIDES.items()
        if slug in ("ops", "factories", "composite", "predicates")
    }

    def _keep_method(m: dict[str, Any]) -> bool:
        name: str = m["name"]
        labels = m.get("labels") or []
        if "property" in labels:
            return True                       # properties are Tensor-only state
        if name.startswith("__"):
            return True                       # dunders (constructor, repr, ops, ...)
        if name.startswith("new_"):
            return True                       # new_zeros / new_full / ...
        # In-place mutation: filter if its non-mutating counterpart is a
        # top-level op (e.g. add_ → add).  Otherwise keep (zero_, fill_,
        # share_memory_, etc. have no free-function counterpart).
        if name.endswith("_") and not name.startswith("__"):
            base = name[:-1]
            if name in redundant_names or base in redundant_names:
                return False
            return True
        if name in redundant_names:
            return False                      # duplicate of lucid.<name>
        return True

    serialised["methods"] = [m for m in serialised["methods"] if _keep_method(m)]

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


def build_synth(slug: str, filter_subcat: str, lucid_data: dict[str, Any]) -> None:
    """Build a synthesized slug by filtering the lucid top-level by subcategory.

    Special-case for ``lucid.ops``: re-bucket members by arity (unary/binary/
    ternary) so the page isn't a flat 143-item list.  All other synthesised
    slugs drop subcategory entirely so their members render flat.
    """
    out_file = OUT_DIR / f"{slug}.json"
    print(f"  {slug} → {out_file.name}", end="", flush=True)

    filtered = [
        m for m in lucid_data.get("members", [])
        if m.get("subcategory") == filter_subcat
    ]

    def _arity_bucket(name: str) -> str:
        """Map an op name to its arity bucket.  In-place variants share the
        bucket of their base op (so ``add_`` → binary alongside ``add``).
        Arity 0 (variadic; e.g. ``cat``, ``where``, ``meshgrid``) and
        unknown arity fall into ``variadic`` so nothing is left orphaned.
        """
        a = _OP_ARITY.get(name)
        if a is None and name.endswith("_"):
            a = _OP_ARITY.get(name[:-1])
        if a == 1: return "unary"
        if a == 2: return "binary"
        if a == 3: return "ternary"
        return "variadic"

    if slug == "lucid.ops":
        for m in filtered:
            m["subcategory"] = _arity_bucket(m["name"])
    else:
        for m in filtered:
            m["subcategory"] = None

    data = {
        "slug":    slug,
        "name":    slug.split(".")[-1],
        "path":    slug,
        "kind":    "module",
        "source":  None,
        "summary": None,
        "extended": None,
        "parameters": [],
        "returns": None,
        "raises": [],
        "examples": [],
        "notes": [],
        "attributes": [],
        "warns": [],
        "members": filtered,
    }

    out_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  ✓  ({len(filtered)} members)")


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

    from griffe import GriffeLoader, Parser, Module as GriffeModule
    loader = GriffeLoader(
        docstring_parser=Parser.numpy,
        search_paths=[str(LUCID_SRC)],
        allow_inspection=False,  # static analysis only — no C extension import needed
    )

    manifest = MODULE_MANIFEST
    if args.module:
        if args.module not in manifest and args.module not in _SYNTH_SLUGS:
            print(f"ERROR: unknown slug '{args.module}'. Valid: {list(manifest) + list(_SYNTH_SLUGS)}")
            sys.exit(1)

    # Build lucid top-level once (internally) — used as the source for any
    # synthesized slugs.  It's NEVER emitted to disk as ``lucid.json`` because
    # the bare ``lucid`` namespace shouldn't appear in the sidebar.
    synth_slugs = _SYNTH_SLUGS
    if args.module and args.module in _SYNTH_SLUGS:
        synth_slugs = {args.module: _SYNTH_SLUGS[args.module]}
    elif args.module:
        synth_slugs = {}

    lucid_data: dict[str, Any] | None = None
    if synth_slugs:
        try:
            lucid_mod = loader.load("lucid")
            if isinstance(lucid_mod, GriffeModule):
                lucid_data = _serialise_module(lucid_mod, Parser.numpy, sha, slug="lucid")
        except Exception as exc:
            print(f"  [warn] failed to load lucid for synthesis: {exc}")

    # Regular slugs from MODULE_MANIFEST
    if args.module and args.module in MODULE_MANIFEST:
        manifest = {args.module: MODULE_MANIFEST[args.module]}
    elif args.module and args.module in _SYNTH_SLUGS:
        manifest = {}

    print(f"Building {len(manifest) + len(synth_slugs)} module(s)  [commit: {sha}]")
    for slug, griffe_path in manifest.items():
        build_one(slug, griffe_path, loader, Parser.numpy, sha)
    if lucid_data:
        for slug, filter_subcat in synth_slugs.items():
            build_synth(slug, filter_subcat, lucid_data)

    print(f"\nDone. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
