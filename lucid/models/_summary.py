"""Layer-summary tree for the model-zoo docs site.

Walks ``model.named_children()`` recursively and emits a nested
dictionary of the form::

    {
        "name": "ResNet",
        "type": "ResNet",
        "params": 25_557_032,
        "own_params": 0,
        "children": [
            {"name": "stem", "type": "Sequential", "params": 9408, ...},
            {"name": "stage1", "type": "Sequential", "params": 218_368,
             "children": [
                 {"name": "0", "type": "Bottleneck", "params": ...,
                  "repeat": 3, ...},
             ]},
            ...
        ],
    }

``repeat`` is filled in by :func:`compress_repeats` when consecutive
siblings share an *identical* substructure — this turns ResNet's
``Bottleneck-Bottleneck-Bottleneck`` runs into a single
``Bottleneck × 3`` node so the docs tree stays readable.

Consumed by ``tools/build_model_summaries.py`` which calls
:func:`compute_model_summary` once per registered factory and writes
the merged result to a JSON cache that the web build pipeline picks
up.

Convention rationale: see ``arch-models-family-contract`` (model-size
section) and the ``@register_model(summary=...)`` field.
"""

from typing import Any

import lucid.nn as nn

# ---------------------------------------------------------------------------
# Param accounting
# ---------------------------------------------------------------------------


def _own_param_numel(mod: nn.Module) -> int:
    """Number of scalar parameters declared *directly* on ``mod`` —
    excluding parameters owned by child submodules.  This is what makes
    a module's "own" contribution at a node (vs. its descendants')."""
    total = 0
    for p in mod.parameters(recurse=False):
        n = 1
        for s in p.shape:
            n *= int(s)
        total += n
    return total


def _total_param_numel(mod: nn.Module) -> int:
    """Recursive total — own params + everything under it."""
    total = 0
    for p in mod.parameters():
        n = 1
        for s in p.shape:
            n *= int(s)
        total += n
    return total


# ---------------------------------------------------------------------------
# Tree extraction
# ---------------------------------------------------------------------------


def _module_node(name: str, mod: nn.Module) -> dict[str, Any]:
    """Build one tree node for ``mod`` (used recursively)."""
    children: list[dict[str, Any]] = []
    for child_name, child in mod.named_children():
        children.append(_module_node(child_name, child))
    return {
        "name": name or type(mod).__name__,
        "type": type(mod).__name__,
        "params": _total_param_numel(mod),
        "own_params": _own_param_numel(mod),
        "children": children,
    }


# ---------------------------------------------------------------------------
# Repeat-run compression
# ---------------------------------------------------------------------------


def _structural_signature(node: dict[str, Any]) -> tuple:
    """Signature used to decide whether two sibling nodes can collapse
    into a single ``× N`` entry.  Two nodes match iff they have the
    *same type* and the *same param count* and the same recursively
    matching children — i.e. byte-identical structurally.

    The node ``name`` is intentionally **not** included: ResNet's
    sequential stage children are named ``"0"``, ``"1"``, ``"2"``,
    which differ but represent the same block.
    """
    return (
        node["type"],
        node["params"],
        node["own_params"],
        tuple(_structural_signature(c) for c in node.get("children") or ()),
    )


def compress_repeats(node: dict[str, Any]) -> dict[str, Any]:
    """Walk the tree depth-first, collapsing consecutive identical
    siblings into a single node with a ``repeat`` count.  Returns a
    new node — the input is left untouched.

    Example:  ``[Bottleneck, Bottleneck, Bottleneck]`` becomes one
    ``Bottleneck`` node with ``repeat=3``.  A ``Downsample`` block in
    the middle of the run breaks the chain, preserving the underlying
    ResNet stage structure (``Downsample`` + ``Bottleneck × 5`` for
    deeper stages).
    """
    children = node.get("children") or []
    if not children:
        return dict(node)

    compressed = [compress_repeats(c) for c in children]

    out: list[dict[str, Any]] = []
    i = 0
    n = len(compressed)
    while i < n:
        run = 1
        sig = _structural_signature(compressed[i])
        while i + run < n and _structural_signature(compressed[i + run]) == sig:
            run += 1
        first = dict(compressed[i])
        if run >= 2:
            first["repeat"] = run
        out.append(first)
        i += run

    new_node = dict(node)
    new_node["children"] = out
    return new_node


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_model_summary(model: nn.Module) -> dict[str, Any]:
    r"""Build a compressed layer-summary tree for ``model``.

    Parameters
    ----------
    model : nn.Module
        Fully constructed model instance.  Typically the value returned
        by a ``@register_model``-decorated factory.

    Returns
    -------
    dict
        Nested dict with keys ``name`` / ``type`` / ``params`` /
        ``own_params`` / ``children`` (and ``repeat`` on collapsed
        runs).  Top-level ``name`` is the model class name.

    Examples
    --------
    >>> from lucid.models.vision.alexnet import alexnet_cls
    >>> tree = compute_model_summary(alexnet_cls())
    >>> tree["params"]
    61100840
    """
    return compress_repeats(_module_node(type(model).__name__, model))
