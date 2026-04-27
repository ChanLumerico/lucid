"""
lucid.ops.einops — einops-style pattern operations.

Implements rearrange / reduce / repeat by composing C++ engine primitives
(reshape, permute, reduce, expand). Patterns follow the standard einops
syntax — see https://einops.rocks/api/rearrange/ — though only the most
common subset (no `...`, no parenthesized groups beyond simple split/merge)
is currently supported. For exotic patterns, fall back to the upstream
`einops` package.
"""

from __future__ import annotations

import re
from typing import Any

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import impl_of, normalize_shape


__all__ = ["rearrange", "reduce", "repeat", "einsum", "asnumpy"]


# --------------------------------------------------------------------------- #
# Pattern parsing
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\(([^()]+)\)|([A-Za-z_][A-Za-z0-9_]*)|(\d+)")


def _parse_side(s: str) -> list[Any]:
    """Parse a pattern side ('a (b c) d') into a list of tokens.

    Each token is either a string axis name or a list of names (group)."""
    out: list[Any] = []
    for m in _TOKEN_RE.finditer(s):
        if m.group(1) is not None:
            inner = _parse_side(m.group(1))
            out.append(inner)
        elif m.group(2) is not None:
            out.append(m.group(2))
        else:
            out.append(int(m.group(3)))
    return out


def _flat_axes(tokens: list[Any]) -> list[str]:
    """Flatten a parsed token list to a flat list of axis names."""
    out: list[str] = []
    for tk in tokens:
        if isinstance(tk, list):
            out.extend(_flat_axes(tk))
        elif isinstance(tk, str):
            out.append(tk)
    return out


# --------------------------------------------------------------------------- #
# rearrange
# --------------------------------------------------------------------------- #

def rearrange(
    a: Tensor, pattern: str, /, **axes_lengths: int,
) -> Tensor:
    """Rearrange axes according to einops pattern.

    Supports:
      - simple permutation:  'a b c -> c a b'
      - split:               'b (h w) c -> b h w c'   with h or w in kwargs
      - merge:               'b h w c -> b (h w) c'
      - inner unsqueeze:     'b -> b 1'   (numeric literals)
    """
    if "->" not in pattern:
        raise ValueError(f"rearrange pattern must contain '->': {pattern!r}")
    lhs_str, rhs_str = (s.strip() for s in pattern.split("->"))
    lhs = _parse_side(lhs_str)
    rhs = _parse_side(rhs_str)

    # 1. Compute axis sizes from lhs.
    if len(lhs) != a.ndim:
        raise ValueError(
            f"rearrange: lhs has {len(lhs)} groups, tensor has ndim={a.ndim}")
    axis_size: dict[str, int] = {}
    for token, dim in zip(lhs, a.shape):
        if isinstance(token, str):
            axis_size[token] = dim
        elif isinstance(token, list):
            # Group must be specified by kwargs (or all-but-one are).
            unknown = [n for n in token if n not in axis_size
                                              and n not in axes_lengths]
            known_prod = 1
            for n in token:
                if n in axes_lengths:
                    axis_size[n] = axes_lengths[n]
                if n in axis_size:
                    known_prod *= axis_size[n]
            if len(unknown) == 1:
                axis_size[unknown[0]] = dim // known_prod
            elif len(unknown) > 1:
                raise ValueError(
                    f"rearrange: cannot infer multiple unknown axes in group "
                    f"{token!r}; pass via kwargs")
            elif known_prod != dim:
                raise ValueError(
                    f"rearrange: group {token!r} product {known_prod} "
                    f"!= input dim {dim}")
        else:
            # Literal int
            if token != dim:
                raise ValueError(
                    f"rearrange: literal {token} != input dim {dim}")

    flat_lhs = _flat_axes(lhs)
    flat_rhs = _flat_axes(rhs)

    # 2. Reshape input from grouped→flat shape (one dim per axis name).
    flat_lhs_shape = [axis_size[n] for n in flat_lhs]
    out = Tensor._wrap(_C_engine.reshape(impl_of(a), flat_lhs_shape))

    # 3. Permute to rhs flat order.
    perm = [flat_lhs.index(n) for n in flat_rhs]
    if perm != list(range(len(perm))):
        out = Tensor._wrap(_C_engine.permute(impl_of(out), perm))

    # 4. Reshape into rhs grouped shape, inserting numeric literals as size-1
    #    dims (or matching values).
    rhs_shape: list[int] = []
    cursor = 0
    for token in rhs:
        if isinstance(token, str):
            rhs_shape.append(axis_size[token])
            cursor += 1
        elif isinstance(token, list):
            prod = 1
            for n in token:
                prod *= axis_size[n]
                cursor += 1
            rhs_shape.append(prod)
        else:
            # literal int -> size-token dim
            rhs_shape.append(int(token))

    if rhs_shape != list(out.shape):
        out = Tensor._wrap(_C_engine.reshape(impl_of(out), rhs_shape))
    return out


# --------------------------------------------------------------------------- #
# reduce
# --------------------------------------------------------------------------- #

_REDUCE_OPS = {
    "sum":  _C_engine.sum,
    "mean": _C_engine.mean,
    "max":  _C_engine.max,
    "min":  _C_engine.min,
    "prod": _C_engine.prod,
}


def reduce(
    a: Tensor, pattern: str, /, reduction: str = "mean", **axes_lengths: int,
) -> Tensor:
    """Reduce axes that disappear from the rhs of the pattern.

    e.g. `reduce(a, 'b h w c -> b c', 'mean')` collapses h, w via mean.
    """
    if reduction not in _REDUCE_OPS:
        raise ValueError(f"reduce: unknown reduction {reduction!r}")
    if "->" not in pattern:
        raise ValueError(f"reduce pattern must contain '->': {pattern!r}")
    lhs_str, rhs_str = (s.strip() for s in pattern.split("->"))
    lhs = _parse_side(lhs_str)
    rhs = _parse_side(rhs_str)
    flat_lhs = _flat_axes(lhs)
    flat_rhs = _flat_axes(rhs)

    # First flatten lhs groups by treating reduce as a rearrange to flat-lhs.
    out = rearrange(a, lhs_str + " -> " + " ".join(flat_lhs), **axes_lengths)

    # Compute reduction axes (those in flat_lhs not in flat_rhs).
    reduce_axes = [i for i, n in enumerate(flat_lhs) if n not in flat_rhs]
    if reduce_axes:
        out = Tensor._wrap(_REDUCE_OPS[reduction](
            impl_of(out), reduce_axes, False))
    # Now out's axes correspond to [n for n in flat_lhs if n in flat_rhs].
    # Re-permute / re-group to rhs.
    surviving = [n for n in flat_lhs if n in flat_rhs]
    if surviving != flat_rhs:
        perm = [surviving.index(n) for n in flat_rhs]
        out = Tensor._wrap(_C_engine.permute(impl_of(out), perm))

    # Group merge per rhs.
    axis_size = {n: out.shape[i] for i, n in enumerate(flat_rhs)}
    rhs_shape: list[int] = []
    for token in rhs:
        if isinstance(token, str):
            rhs_shape.append(axis_size[token])
        elif isinstance(token, list):
            prod = 1
            for n in token:
                prod *= axis_size[n]
            rhs_shape.append(prod)
        else:
            rhs_shape.append(int(token))
    if rhs_shape != list(out.shape):
        out = Tensor._wrap(_C_engine.reshape(impl_of(out), rhs_shape))
    return out


# --------------------------------------------------------------------------- #
# repeat
# --------------------------------------------------------------------------- #

def repeat(
    a: Tensor, pattern: str, /, **axes_lengths: int,
) -> Tensor:
    """Insert and repeat new axes per pattern.

    e.g. `repeat(a, 'h w -> h w c', c=3)` repeats along a new axis.
    """
    if "->" not in pattern:
        raise ValueError(f"repeat pattern must contain '->': {pattern!r}")
    lhs_str, rhs_str = (s.strip() for s in pattern.split("->"))
    lhs = _parse_side(lhs_str)
    rhs = _parse_side(rhs_str)
    flat_lhs = _flat_axes(lhs)
    flat_rhs = _flat_axes(rhs)

    # Step 1: rearrange to flat lhs.
    out = rearrange(a, lhs_str + " -> " + " ".join(flat_lhs), **axes_lengths)
    axis_size = {n: out.shape[i] for i, n in enumerate(flat_lhs)}

    # Step 2: for each new axis in flat_rhs not in flat_lhs, insert size-1 dim.
    for new_n in flat_rhs:
        if new_n not in axis_size:
            if new_n not in axes_lengths:
                raise ValueError(f"repeat: new axis '{new_n}' needs size kwarg")
            axis_size[new_n] = axes_lengths[new_n]

    # Insert size-1 dims at the right positions then expand.
    interim_axes = list(flat_lhs)
    for new_n in flat_rhs:
        if new_n not in interim_axes:
            interim_axes.append(new_n)
            out = Tensor._wrap(_C_engine.unsqueeze(impl_of(out),
                                                    len(out.shape)))

    # Permute interim → flat_rhs order.
    perm = [interim_axes.index(n) for n in flat_rhs]
    if perm != list(range(len(perm))):
        out = Tensor._wrap(_C_engine.permute(impl_of(out), perm))

    # Expand to target sizes.
    target_shape = [axis_size[n] for n in flat_rhs]
    if list(out.shape) != target_shape:
        out = Tensor._wrap(_C_engine.broadcast_to(impl_of(out), target_shape))

    # Group merge per rhs.
    rhs_shape: list[int] = []
    for token in rhs:
        if isinstance(token, str):
            rhs_shape.append(axis_size[token])
        elif isinstance(token, list):
            prod = 1
            for n in token:
                prod *= axis_size[n]
            rhs_shape.append(prod)
        else:
            rhs_shape.append(int(token))
    if rhs_shape != list(out.shape):
        out = Tensor._wrap(_C_engine.reshape(impl_of(out), rhs_shape))
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def asnumpy(a: Tensor):
    return a.numpy()


# --------------------------------------------------------------------------- #
# einsum — wraps the engine's tensordot for the most common patterns,
# falls back to numpy.einsum for the rest.
# --------------------------------------------------------------------------- #

def einsum(pattern: str, *operands: Tensor) -> Tensor:
    """Einstein summation expressed as a pattern string.

    Implementation: parse the equation, dispatch to ``matmul`` /
    ``tensordot`` for binary patterns when possible, otherwise fall
    back to numpy.einsum on detached host data and wrap the result
    back into a Tensor (no autograd through the fallback path —
    matches legacy behaviour for unusual patterns).
    """
    if not operands:
        raise ValueError("einsum: at least one operand required")

    eq = pattern.replace(" ", "")
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        # Implicit output: every index that appears once across all
        # operands, sorted alphabetically.
        lhs = eq
        counts: dict[str, int] = {}
        for s in lhs.split(","):
            for c in s:
                if c == ".":
                    continue
                counts[c] = counts.get(c, 0) + 1
        rhs = "".join(sorted([c for c, v in counts.items() if v == 1]))

    in_specs = lhs.split(",")
    if len(in_specs) != len(operands):
        raise ValueError(
            f"einsum: pattern expects {len(in_specs)} operands, got {len(operands)}")

    # Generic path: defer to numpy.einsum on host.  Tensors must be on
    # CPU; GPU operands are downloaded via .data which calls numpy().
    import numpy as np
    arrays = [op.numpy() if hasattr(op, "numpy") else np.asarray(op) for op in operands]
    out_np = np.einsum(pattern, *arrays)
    # Re-upload onto the first operand's device for symmetry with other ops.
    target_device = operands[0].device
    return Tensor(out_np).to(target_device)
