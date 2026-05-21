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
import logging
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:                                          # graceful fallback
    tqdm = None                                              # type: ignore[assignment]

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
    "lucid.benchmarks",  # internal perf scripts
}

# Self excluded — this slug is dropped but descendants still get discovered.
# Used for umbrella packages whose ``__init__.py`` has no user-facing content
# of its own but whose children DO get documented separately.
_EXCLUDED_SELF: set[str] = {
    "lucid.nn.modules",         # umbrella; content re-exposed via lucid.nn path-grouping
    "lucid.utils",              # umbrella; only lucid.utils.data has docs content
    "lucid.models._utils",      # internal helper subpackage of lucid.models
}

# Family-root slugs: their ``__init__.py`` is intentionally minimal but each
# is given its own browsable index page populated with cards for every
# direct sub-family.  The build script post-processes these JSONs to
# inject ``family_groups`` (one entry per child package).
_FAMILY_ROOTS: set[str] = {
    "lucid.models.vision",
    "lucid.models.text",
    "lucid.models.generative",
}


def _is_family_leaf(slug: str) -> bool:
    """A slug like ``lucid.models.vision.resnet`` — depth-4 under a
    family root.  These are the actual model-family pages (resnet, vit,
    bert, …) that own ``ModelConfig._canonical_name`` and the 4-slot
    sidebar structure."""
    parts = slug.split(".")
    if len(parts) != 4:
        return False
    return ".".join(parts[:3]) in _FAMILY_ROOTS

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
            # Family-root walk: lucid.models has a 4-level structure
            # (lucid/models/{vision,text,generative}/{family}/) — every
            # leaf family directory becomes its own page so the family
            # cards on the family-root index are clickable.
            if sub_slug in _FAMILY_ROOTS:
                for third in sorted(second.iterdir()):
                    if not _is_public_pkg(third):
                        continue
                    leaf_slug = f"{sub_slug}.{third.name}"
                    if leaf_slug in _EXCLUDED_SUBTREE or leaf_slug in _EXCLUDED_SELF:
                        continue
                    manifest[leaf_slug] = leaf_slug

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
    # If the body uses `&` (column alignment) or contains an explicit `\\`
    # line break, wrap it in an `aligned` environment — KaTeX requires the
    # explicit environment for these alignment markers to parse, whereas
    # docstrings in the wild tend to omit it.
    def _block_math(m: re.Match[str]) -> str:
        body = textwrap.dedent(m.group(1)).strip()
        needs_aligned = (
            "&" in body
            and "\\begin{" not in body
            and "\\end{" not in body
        )
        if needs_aligned:
            body = f"\\begin{{aligned}}\n{body}\n\\end{{aligned}}"
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


def _get_ast_docstring(fn_name: str, class_name: str | None = None) -> str | None:
    """Search implementation .py files for fn_name's docstring via AST.

    Fallback for .pyi stubs that declare functions with '...' body and no
    docstring.  The actual .py implementation has the full docstring.

    Uses ast.get_docstring which returns the evaluated string value, so
    raw-string docstrings (r-prefix) have their backslashes preserved —
    LaTeX sequences like backslash-forall arrive intact.

    When ``class_name`` is given, the fallback prefers a match nested
    inside a class of that name (or a sibling class with the same name —
    e.g. dunders injected via ``_inject_dunders``).  This avoids
    misattributing common dunders like ``__eq__`` to the first matching
    definition in alphabetical file order (e.g. ``_device.py``'s
    ``__eq__`` outrunning the Tensor's).
    """
    import ast

    IMPL_DIRS = [
        LUCID_SRC / "lucid" / "_factories",
        LUCID_SRC / "lucid" / "_ops",
        LUCID_SRC / "lucid" / "autograd",
        LUCID_SRC / "lucid" / "serialization",
        LUCID_SRC / "lucid",
    ]

    def _matches_class(tree: ast.AST, target: ast.AST) -> bool:
        """Return True if ``target`` lives inside a class whose name matches
        ``class_name`` (or in an injected dunders pattern targeting it)."""
        if class_name is None:
            return True
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Direct member of the class
                for child in ast.walk(node):
                    if child is target:
                        return True
            # Injected dunders pattern: a closure-style helper named e.g.
            # `_inject_dunders` that defines methods at function scope and
            # attaches them to the target class via globals().  Treat any
            # nested function whose file mentions the class name as a
            # plausible match.
        return False

    fallback: str | None = None
    for impl_dir in IMPL_DIRS:
        # rglob: recurse into subpackages.  This lets the lookup find e.g.
        # composite ops in lucid/_ops/composite/{elementwise,blas,shape,...}.py
        # without us having to enumerate every nested directory above.
        for fpath in sorted(impl_dir.rglob("*.py")):
            try:
                tree = ast.parse(fpath.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            file_mentions_class = class_name is not None and class_name in fpath.read_text(encoding="utf-8", errors="ignore")
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == fn_name
                ):
                    doc = ast.get_docstring(node)
                    if not (doc and doc.strip()):
                        continue
                    # Strong match: function lives inside the target class
                    if class_name is not None and _matches_class(tree, node):
                        return doc
                    # Soft match: file mentions the target class name and
                    # the function is at module scope or inside a helper
                    # (e.g. the injected-dunders pattern in _dunders.py)
                    if class_name is not None and file_mentions_class:
                        return doc
                    # No class context — first match wins as before
                    if class_name is None:
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
        # When this method belongs to a class, pass the class name so the
        # AST fallback doesn't pick up a same-named method on a different
        # class (e.g. Tensor.__eq__ vs device.__eq__).  We extract the
        # enclosing-class name from ``obj.path`` (more reliable than
        # ``obj.parent`` which Griffe doesn't always populate the way we
        # expect for stub-loaded methods).
        class_ctx: str | None = None
        path_parts = (obj.path or "").split(".")
        if len(path_parts) >= 2:
            candidate = path_parts[-2]
            # Heuristic: a class name starts with an uppercase letter (PEP 8)
            # and isn't a typical module path segment.
            if candidate and candidate[0].isupper():
                class_ctx = candidate
        raw_text = _get_ast_docstring(fn_name, class_name=class_ctx)
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


_MODEL_SUMMARIES: dict[str, Any] | None = None


def _load_model_summaries() -> dict[str, Any]:
    """Lazy-load the cached layer-summary tree map (built once by
    ``tools/build_model_summaries.py``).  Cached in-process so the
    JSON read happens once per build run."""
    global _MODEL_SUMMARIES
    if _MODEL_SUMMARIES is not None:
        return _MODEL_SUMMARIES
    cache = OUT_DIR / "_summaries.json"
    if cache.is_file():
        try:
            _MODEL_SUMMARIES = json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            _MODEL_SUMMARIES = {}
    else:
        _MODEL_SUMMARIES = {}
    return _MODEL_SUMMARIES


def _extract_register_model_params(fn: Any) -> int | None:
    """Pull the ``params=<int>`` kwarg out of a ``@register_model(...)``
    decorator on a factory function.  Returns ``None`` when the
    decorator is absent, the kwarg is missing, or the value isn't a
    literal int / numeric expression Python can ``literal_eval``.

    Surfaced on the JSON as ``param_count`` so the docs site can render
    factory cards / detail pages with the family's paper-cited model
    size (e.g. ``"61.1M"`` for ``alexnet_cls``).
    """
    import ast as _ast

    if not getattr(fn, "decorators", None):
        return None
    for d in fn.decorators:
        try:
            tree = _ast.parse(str(d.value), mode="eval")
        except SyntaxError:
            continue
        call = tree.body
        if not isinstance(call, _ast.Call):
            continue
        func = call.func
        fname = (
            func.id if isinstance(func, _ast.Name)
            else (func.attr if isinstance(func, _ast.Attribute) else None)
        )
        if fname != "register_model":
            continue
        for kw in call.keywords:
            if kw.arg != "params":
                continue
            try:
                v = _ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(v, int) and v > 0:
                return v
    return None


def _serialise_function(
    fn: Any, parser: Any, sha: str, *,
    module_root: Path | None = None,
    is_lucid_toplevel: bool = False,
) -> dict[str, Any]:
    doc = _parse_docstring(fn, parser)
    ret_ann = _annotation_str(fn.returns)
    if doc["returns"] is None and ret_ann:
        doc["returns"] = {"annotation": ret_ann, "description": ""}

    out = {
        "name":        fn.name,
        "path":        fn.path,
        "kind":        "function",
        "labels":      _extract_labels(fn),
        "signature":   _build_signature(fn),
        "source":      _source_link(fn, sha),
        "subcategory": _subcategory(fn, module_root=module_root, is_lucid_toplevel=is_lucid_toplevel),
        **doc,
    }
    param_count = _extract_register_model_params(fn)
    if param_count is not None:
        out["param_count"] = param_count
    # Layer-summary tree for the docs site's expandable "Model Size"
    # card — generated by ``tools/build_model_summaries.py`` and stored
    # in the ``_summaries.json`` cache.  Keyed by the factory name.
    summary = _load_model_summaries().get(fn.name)
    if summary is not None:
        out["model_summary"] = summary
    return out


def _detect_class_kind(cls: Any) -> str:
    """Return 'protocol', 'abstract', 'dataclass', or 'regular'.

    Resolution order is intentional: Protocol is checked first because a
    Protocol subclass *is also* technically abstract (no concrete bodies);
    dataclass next because ``@dataclass`` is a strong, explicit marker;
    abstract last as the catch-all for ABC-based classes.
    """
    # ``typing.Protocol`` base — checked first so Protocol subclasses
    # surface as their own category rather than as "abstract".
    if hasattr(cls, "bases") and cls.bases:
        for base in cls.bases:
            base_str = str(base)
            # Bare ``Protocol`` or qualified ``typing.Protocol``.  Also
            # accept ``runtime_checkable``-decorated bare Protocols whose
            # bases are other Protocols composed via multiple inheritance.
            if base_str == "Protocol" or base_str.endswith(".Protocol"):
                return "protocol"
    if hasattr(cls, "decorators") and cls.decorators:
        for d in cls.decorators:
            dec_str = str(d.value)
            # ``@runtime_checkable`` is the Protocol idiom — when present
            # alongside a Protocol-derived class it confirms the category.
            if "runtime_checkable" in dec_str:
                return "protocol"
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


_RST_INLINE_MATH = re.compile(r":math:`([^`]+)`")
_RST_BLOCK_MATH = re.compile(
    r"\.\.\s*math::\s*\n((?:[ \t]+[^\n]*\n?|\s*\n)+)",
    re.MULTILINE,
)


def _rst_math_to_markdown(text: str) -> str:
    """Convert rST math directives to markdown / remark-math syntax.

    ``MathText`` (remark-math + KaTeX) expects markdown math syntax —
    ``$x$`` for inline and ``$$...$$`` for block.  Theory bodies on
    family Configs are written in rST style (``:math:`x``` /
    ``.. math::``), so we translate here at build time.
    """
    import textwrap

    def _block_sub(m: "re.Match[str]") -> str:
        body = textwrap.dedent(m.group(1)).strip()
        if not body:
            return ""
        return f"\n\n$$\n{body}\n$$\n\n"

    text = _RST_BLOCK_MATH.sub(_block_sub, text)
    text = _RST_INLINE_MATH.sub(r"$\1$", text)
    return text


# Task-wrapper class suffix → HF-style task identifier.  Each family-leaf's
# supported tasks are inferred at build time from the names of its public
# ``*For*`` classes — no per-Config bookkeeping needed.  Identifiers use
# kebab-case to match Hugging Face Hub conventions so the same tags travel
# unchanged if/when the docs cross-link to external repos.
_TASK_SUFFIX_MAP: dict[str, str] = {
    "ForImageClassification":      "image-classification",
    "ForObjectDetection":          "object-detection",
    "ForInstanceSegmentation":     "instance-segmentation",
    "ForSemanticSegmentation":     "semantic-segmentation",
    "ForPanopticSegmentation":     "panoptic-segmentation",
    "ForImageGeneration":          "image-generation",
    "ForImageToImage":             "image-to-image",
    "ForMaskedImageModeling":      "masked-image-modeling",
    "ForMaskedLM":                 "fill-mask",
    "ForCausalLM":                 "text-generation",
    "ForSeq2SeqLM":                "text2text-generation",
    "ForSequenceClassification":   "text-classification",
    "ForTokenClassification":      "token-classification",
    "ForQuestionAnswering":        "question-answering",
    "ForNextSentencePrediction":   "next-sentence-prediction",
    "ForMultipleChoice":           "multiple-choice",
    "ForPreTraining":              "pretraining",
    # Alternative head-style naming (GPT / GPT-2 follow reference-framework
    # conventions: ``GPTLMHeadModel`` rather than ``GPTForCausalLM``).
    "LMHeadModel":                 "text-generation",
    "DoubleHeadsModel":            "pretraining",
}


def _infer_tasks(members: list[dict[str, Any]]) -> list[str]:
    """Scan a module's serialised members for ``*For*`` task wrappers and
    return the list of distinct task identifiers, preserving first-seen
    order so the JSON output is stable.  Returns an empty list when no
    public task wrapper exists (e.g. raw-backbone-only families)."""
    seen: list[str] = []
    for m in members:
        if m.get("kind") != "class":
            continue
        name = m.get("name", "")
        for suffix, task in _TASK_SUFFIX_MAP.items():
            if name.endswith(suffix) and not name.endswith("Output"):
                if task not in seen:
                    seen.append(task)
                break
    return seen


def _slug_to_family_dir(slug: str) -> Path:
    """Map a family-leaf slug to its source directory.

    e.g. ``lucid.models.vision.resnet`` → ``<repo>/lucid/models/vision/resnet/``
    """
    parts = slug.split(".")
    return REPO_ROOT.joinpath(*parts)


def _extract_family_meta(slug: str) -> dict[str, str]:
    """Parse a family-leaf source to extract ``@model_family_meta``
    decorator arguments.

    Convention (see ``arch-models-canonical-name``): each family's Config
    dataclass is wrapped with ``@model_family_meta(canonical_name=...,
    citation=..., theory=...)`` whose keyword arguments are all string
    literals (or implicit concatenations of literals).  Most families
    own a single ``_config.py``; a few (e.g. ``yolo`` with ``_v1.py``,
    ``_v2.py``, ...) split the configs across several files.  We scan
    every ``.py`` file in the family directory and return the first
    decorator we successfully parse.

    Returns a dict with any of the keys ``canonical_name`` / ``citation``
    / ``theory`` that were found and successfully decoded.  Empty dict
    when no decorator is present in any source file.
    """
    import ast as _ast
    import textwrap

    family_dir = _slug_to_family_dir(slug)
    if not family_dir.is_dir():
        return {}

    wanted = {"canonical_name", "citation", "theory"}

    # Scan order: prefer ``_config.py`` first (the conventional location),
    # then any other ``.py`` files at the family-dir root (handles YOLO's
    # per-version configs and similar splits).  Skip ``__init__.py`` —
    # re-exports won't carry the decorator literally.
    candidates: list[Path] = []
    primary = family_dir / "_config.py"
    if primary.is_file():
        candidates.append(primary)
    for p in sorted(family_dir.iterdir()):
        if p.is_file() and p.suffix == ".py" and p.name not in {"__init__.py", "_config.py"}:
            candidates.append(p)

    for src_file in candidates:
        try:
            tree = _ast.parse(src_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue

        out: dict[str, str] = {}
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, _ast.Call):
                    continue
                func = dec.func
                if isinstance(func, _ast.Name):
                    name = func.id
                elif isinstance(func, _ast.Attribute):
                    name = func.attr
                else:
                    continue
                if name != "model_family_meta":
                    continue
                for kw in dec.keywords:
                    if kw.arg not in wanted:
                        continue
                    try:
                        val = _ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        continue
                    if not isinstance(val, str):
                        continue
                    # ``theory`` is typically a raw-triple-quoted block
                    # with leading indentation; dedent so the rST
                    # renderer sees clean column-0 directives, then
                    # translate rST math syntax to markdown so the
                    # MathText (remark-math) component can render it.
                    if kw.arg == "theory":
                        val = textwrap.dedent(val).strip("\n")
                        val = _rst_math_to_markdown(val)
                    else:
                        val = val.strip()
                    if val:
                        out[kw.arg] = val
                if out:
                    return out
    return {}


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


def _emit(msg: str) -> None:
    """Print a status line without breaking an active tqdm bar."""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


_LUCID_MODELS_TIER1_SUBS = {
    "auto", "base", "hub", "mixins", "output", "protocols", "registry",
}
"""Subcategories kept on the ``lucid.models`` landing page.  Everything
with a slash (e.g. ``vision/resnet/pretrained``) is a family member and
belongs on its own family page — keeping them here would make the index
page ~500 members long and obscure the user-facing dispatch surface."""

_LUCID_MODELS_FAMILY_GROUPS = [
    {"slug": "lucid.models.vision",     "label": "Vision",     "icon": "image"},
    {"slug": "lucid.models.text",       "label": "Text",       "icon": "text"},
    {"slug": "lucid.models.generative", "label": "Generative", "icon": "sparkles"},
]


def _build_family_groups(family_root_slug: str) -> list[dict[str, Any]]:
    """Walk the filesystem under a family-root and emit one entry per
    direct sub-package (model family).  Each entry is a card on the
    family-root's index page linking to that family's detail page."""
    parts = family_root_slug.split(".")  # e.g. ["lucid", "models", "vision"]
    family_dir = LUCID_SRC / Path(*parts)
    if not family_dir.is_dir():
        return []

    groups: list[dict[str, Any]] = []
    for sub in sorted(family_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        if not (sub / "__init__.py").exists():
            continue
        groups.append({
            "slug": f"{family_root_slug}.{sub.name}",
            "label": sub.name,
        })
    return groups


def build_one(slug: str, griffe_path: str, loader: Any, parser: Any, sha: str) -> None:
    from griffe import Module as GriffeModule, Class as GriffeClass

    out_file = OUT_DIR / f"{slug}.json"

    try:
        if slug == "lucid.tensor":
            data = _build_tensor_data(loader, parser, sha)
        else:
            obj = loader.load(griffe_path)
            if isinstance(obj, GriffeModule):
                data = _serialise_module(obj, parser, sha, slug=slug)
                # Family-leaf metadata: each ``lucid.models.<domain>.<family>``
                # Config class is wrapped with ``@model_family_meta(...)``
                # carrying three docs-facing fields (``canonical_name`` /
                # ``citation`` / ``theory``).  Parsed directly from the
                # source file via ``ast`` — does not depend on Griffe.
                # See ``arch-models-canonical-name``.
                if _is_family_leaf(slug):
                    meta = _extract_family_meta(slug)
                    for key in ("canonical_name", "citation", "theory"):
                        if meta.get(key):
                            data[key] = meta[key]
                    # HF-style task tags inferred from the family's
                    # ``*For*`` task-wrapper class names.  Surfaced on the
                    # family-leaf JSON's top level so family-root cards
                    # can render them next to the canonical name.
                    tasks = _infer_tasks(data.get("members", []))
                    if tasks:
                        data["tasks"] = tasks
            elif isinstance(obj, GriffeClass):
                serialised = _serialise_class(obj, parser, sha)
                data = {"slug": slug, **serialised}
            else:
                data = {"slug": slug, "kind": "unknown", "name": griffe_path}

        if slug == "lucid.models":
            members = data.get("members", []) or []
            filtered = [
                m for m in members
                if (m.get("subcategory") or "") in _LUCID_MODELS_TIER1_SUBS
            ]
            data["members"] = filtered
            data["family_groups"] = _LUCID_MODELS_FAMILY_GROUPS

        if slug in _FAMILY_ROOTS:
            data["members"] = []  # __init__.py is intentionally empty
            data["family_groups"] = _build_family_groups(slug)

        out_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        member_count = len(data.get("members", data.get("methods", [])))
        _emit(f"  ✓ {slug:32}  ({member_count} members)")

    except Exception as exc:
        _emit(f"  ✗ {slug:32}  {exc}")
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
    _emit(f"  ✓ {slug:32}  ({len(filtered)} members)")


class _ElapsedSpinner:
    """Print '[mm:ss] msg' on a timer until ``stop()``.  Used during the
    upfront ``loader.load('lucid')`` walk which has no granular progress."""

    def __init__(self, msg: str, interval: float = 2.0) -> None:
        self._msg = msg
        self._interval = interval
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _fmt(self) -> str:
        dt = int(time.monotonic() - self._t0)
        return f"\r[{dt//60:02d}:{dt%60:02d}] {self._msg}"

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            sys.stdout.write(self._fmt())
            sys.stdout.flush()

    def __enter__(self) -> "_ElapsedSpinner":
        sys.stdout.write(self._fmt()); sys.stdout.flush()
        self._thread.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=0.1)
        dt = int(time.monotonic() - self._t0)
        sys.stdout.write(f"\r[{dt//60:02d}:{dt%60:02d}] {self._msg} ✓\n")
        sys.stdout.flush()


def main() -> None:
    global VERBOSE

    parser_args = argparse.ArgumentParser(description="Build Lucid API JSON data")
    parser_args.add_argument("--module", help="Build only this slug (e.g. lucid.fft)")
    parser_args.add_argument("--verbose", action="store_true")
    args = parser_args.parse_args()
    VERBOSE = args.verbose

    # Griffe's docstring parser emits a warning per signature/docstring mismatch.
    # In verbose mode keep them visible; otherwise silence so the progress bar
    # isn't drowned out.
    if not VERBOSE:
        logging.getLogger("griffe").setLevel(logging.ERROR)
        logging.getLogger("mkdocstrings").setLevel(logging.ERROR)

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
            with _ElapsedSpinner("loading lucid/ tree (Griffe AST walk)..."):
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

    total = len(manifest) + len(synth_slugs)
    print(f"Building {total} module(s)  [commit: {sha}]")

    items: list[tuple[str, str, str]] = []
    for slug, griffe_path in manifest.items():
        items.append(("manifest", slug, griffe_path))
    if lucid_data:
        for slug, filter_subcat in synth_slugs.items():
            items.append(("synth", slug, filter_subcat))

    iterator: Any = items
    if tqdm is not None:
        iterator = tqdm(items, total=total, desc="modules", unit="mod",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for kind, slug, payload in iterator:
        if tqdm is not None:
            iterator.set_postfix_str(slug, refresh=False)
        if kind == "manifest":
            build_one(slug, payload, loader, Parser.numpy, sha)
        else:
            assert lucid_data is not None
            build_synth(slug, payload, lucid_data)

    print(f"\nDone. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
