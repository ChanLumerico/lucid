import re
import numpy as np

from functools import partial
from types import ModuleType
from typing import Literal

from lucid._tensor import Tensor
from lucid.types import _EinopsPattern, _NumPyArray

from lucid._backend.core import (
    operation,
    unary_func_op,
    _FuncOpReturnType,
    _GradFuncType,
)
from lucid._backend.metal import mx


_ReduceStr = Literal["sum", "mean"]


def _parse_pattern(pattern_side: str) -> list[str | tuple[str, ...]]:
    parts = re.findall(r"\([^)]+\)|\S+", pattern_side)
    tokens: list[str | tuple[str, ...]] = []

    for part in parts:
        if part.startswith("(") and part.endswith(")"):
            inner = part[1:-1].strip()
            if not inner:
                raise ValueError("Empty group in pattern.")
            tokens.append(tuple(inner.split()))
        else:
            tokens.append(part)

    return tokens


def _build_intermediate(
    input_tokens: list[str | tuple[str, ...]], shape: tuple[int, ...], shapes: dict
) -> tuple[list[str], list[int]]:
    inter_tokens: list[str] = []
    inter_shape: list[int] = []

    for token, dim in zip(input_tokens, shape):
        if isinstance(token, tuple):
            prod = 1
            group_sizes = []
            for t in token:
                if t not in shapes:
                    raise ValueError(
                        f"Size for token '{t}' in group {token} must be provided."
                    )
                s = shapes[t]
                group_sizes.append(s)
                prod *= s

            if prod != dim:
                raise ValueError(
                    f"Product of sizes {prod} for grouped tokens "
                    + f"{token} does not match merged axis size {dim}."
                )
            inter_tokens.extend(token)
            inter_shape.extend(group_sizes)

        else:
            if token in shapes and shapes[token] != dim:
                raise ValueError(
                    f"Provided size for token '{token}' ({shapes[token]}) "
                    + f"does not match tensor dimension ({dim})."
                )
            inter_tokens.append(token)
            inter_shape.append(dim)

    return inter_tokens, inter_shape


class rearrange(operation):
    def __init__(self, pattern: _EinopsPattern, t_shape: int, **shapes: int) -> None:
        super().__init__()
        try:
            in_pat, out_pat = map(str.strip, pattern.split("->"))
        except Exception as e:
            raise ValueError(
                "Pattern must contain '->' separating input and output patterns."
            ) from e

        input_tokens = _parse_pattern(in_pat)
        output_tokens = _parse_pattern(out_pat)

        if len(input_tokens) != len(t_shape):
            raise ValueError(
                f"Input pattern has {len(input_tokens)} tokens, "
                + f"but tensor has {len(t_shape)} dimensions."
            )

        self.inter_tokens, self.inter_shape = _build_intermediate(
            input_tokens, t_shape, shapes
        )

        self.perm: list[int] = []
        self.group_splits: list[tuple[int, ...]] = []
        used = set()

        for token in output_tokens:
            if isinstance(token, tuple):
                group_perm, group_dims = [], []
                for t in token:
                    found = None
                    for i, it in enumerate(self.inter_tokens):
                        if it == t and i not in used:
                            found = i
                            break
                    if found is None:
                        raise ValueError(
                            f"Token '{t}' in output group {token} not found in input."
                        )
                    group_perm.append(found)
                    group_dims.append(self.inter_shape[found])
                    used.add(found)

                self.group_splits.append(tuple(group_dims))
                self.perm.extend(group_perm)

            else:
                found = None
                for i, it in enumerate(self.inter_tokens):
                    if it == token and i not in used:
                        found = i
                        break
                if found is None:
                    raise ValueError(
                        f"Token '{token}' from output pattern not found in input."
                    )

                self.perm.append(found)
                self.group_splits.append((self.inter_shape[found],))
                used.add(found)

        self.final_shape = tuple(np.prod(group).item() for group in self.group_splits)

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        inter = a.data.reshape(tuple(self.inter_shape))
        transposed = np.transpose(inter, axes=self.perm)

        self.result = Tensor(transposed.reshape(self.final_shape))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        inter = a.data.reshape(tuple(self.inter_shape))
        transposed = mx.transpose(inter, axes=self.perm)

        self.result = Tensor(transposed.reshape(self.final_shape))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        unmerged_shape = tuple(dim for group in self.group_splits for dim in group)
        grad_unmerged = self.result.grad.reshape(unmerged_shape)

        inv_perm = lib_.argsort(lib_.array(self.perm))
        grad_interm = lib_.transpose(grad_unmerged, axes=inv_perm)

        return grad_interm.reshape(a.shape)


# TODO: Continue from here


@unary_func_op()
def reduce(
    self: Tensor, pattern: _EinopsPattern, reduction: _ReduceStr = "sum", **shapes: int
) -> _FuncOpReturnType:
    try:
        in_pat, out_pat = map(str.strip, pattern.split("->"))
    except Exception as e:
        raise ValueError(
            "Pattern must contain '->' separating input and output patterns."
        ) from e

    input_tokens = _parse_pattern(in_pat)
    output_tokens = _parse_pattern(out_pat)

    if len(input_tokens) != len(self.shape):
        raise ValueError(
            f"Input pattern has {len(input_tokens)} tokens, "
            + f"but tensor has {len(self.shape)} dimensions."
        )

    inter_tokens, inter_shape = _build_intermediate(input_tokens, self.shape, shapes)
    kept_indices: list[int] = []
    kept_groups: list[tuple[int, ...]] = []
    used = set()

    for token in output_tokens:
        if isinstance(token, tuple):
            group_inds, group_dims = [], []
            for t in token:
                found = None
                for i, it in enumerate(inter_tokens):
                    if it == t and i not in used:
                        found = i
                        break
                if found is None:
                    raise ValueError(
                        f"Token '{t}' in output group {token} not found in input."
                    )

                group_inds.append(found)
                group_dims.append(inter_shape[found])
                used.add(found)

            kept_indices.extend(group_inds)
            kept_groups.append(tuple(group_dims))

        else:
            found = None
            for i, it in enumerate(inter_tokens):
                if it == token and i not in used:
                    found = i
                    break
            if found is None:
                raise ValueError(
                    f"Token '{token}' from output pattern not found in input."
                )

            kept_indices.append(found)
            kept_groups.append((inter_shape[found],))
            used.add(found)

    all_indices = set(range(len(inter_tokens)))
    reduced_indices = sorted(list(all_indices - used))

    perm = kept_indices + reduced_indices

    kept_flat_shape = tuple(inter_shape[i] for i in kept_indices)
    reduced_shape = tuple(inter_shape[i] for i in reduced_indices)
    final_shape = tuple(np.prod(group) for group in kept_groups)

    intermediate = self.data.reshape(tuple(inter_shape))
    transposed = np.transpose(intermediate, axes=perm)

    reduce_axes = tuple(
        range(len(kept_flat_shape), len(kept_flat_shape) + len(reduced_shape))
    )

    if reduction == "sum":
        reduced_data = np.sum(transposed, axis=reduce_axes)
    elif reduction == "mean":
        reduced_data = np.mean(transposed, axis=reduce_axes)
    else:
        raise ValueError(f"Unsupported reduction method: {reduction}")

    result_data = reduced_data.reshape(final_shape)
    result = Tensor(result_data)

    def compute_grad() -> _NumPyArray:
        grad_kept = result.grad.reshape(kept_flat_shape)
        new_shape = kept_flat_shape + (1,) * len(reduced_shape)

        grad_expanded = grad_kept.reshape(new_shape)
        grad_broadcast = np.broadcast_to(
            grad_expanded, kept_flat_shape + tuple(reduced_shape)
        )

        if reduction == "mean":
            factor = np.prod(reduced_shape) if reduced_shape else 1
            grad_broadcast = grad_broadcast / factor

        inv_perm = np.argsort(perm)
        grad_intermediate = np.transpose(grad_broadcast, axes=inv_perm)

        return grad_intermediate.reshape(self.shape)

    return result, compute_grad


@unary_func_op()
def repeat(self: Tensor, pattern: _EinopsPattern, **shapes: int) -> _FuncOpReturnType:
    try:
        in_pat, out_pat = map(str.strip, pattern.split("->"))
    except Exception as e:
        raise ValueError(
            "Pattern must contain '->' separating input and output patterns."
        ) from e

    input_tokens = _parse_pattern(in_pat)
    output_tokens = _parse_pattern(out_pat)

    if len(input_tokens) != len(self.shape):
        raise ValueError(
            f"Input pattern has {len(input_tokens)} tokens, "
            + f"but tensor has {len(self.shape)} dimensions."
        )

    intermediate_tokens, intermediate_shape = _build_intermediate(
        input_tokens, self.shape, shapes
    )

    used_indices = set()
    group_splits: list[tuple[int, ...]] = []
    perm: list[int] = []

    for token in output_tokens:
        if isinstance(token, tuple):
            group_perm, group_dims = [], []
            for t in token:
                found = None
                for i, it in enumerate(intermediate_tokens):
                    if it == t and i not in used_indices:
                        found = i
                        break

                if found is None:
                    group_perm.append(-1)
                    group_dims.append(1)
                else:
                    group_perm.append(found)
                    group_dims.append(intermediate_shape[found])
                    used_indices.add(found)

            group_splits.append(tuple(group_dims))
            perm.extend(group_perm)

        else:
            found = None
            for i, it in enumerate(intermediate_tokens):
                if it == token and i not in used_indices:
                    found = i
                    break

            if found is None:
                perm.append(-1)
                group_splits.append((1,))
            else:
                perm.append(found)
                group_splits.append((intermediate_shape[found],))
                used_indices.add(found)

    base_shape = tuple(dim for group in group_splits for dim in group)
    out_shape_list = []
    idx = 0

    for token in output_tokens:
        if isinstance(token, tuple):
            n = len(token)
            group_perm = perm[idx : idx + n]

            if all(p == -1 for p in group_perm):
                prod_val = 1
                for t in token:
                    if t not in shapes:
                        raise ValueError(
                            f"Size for expansion token '{t}' must be provided."
                        )
                    prod_val *= shapes[t]
                out_shape_list.append(prod_val)

            else:
                prod_val = np.prod(base_shape[idx : idx + n])
                out_shape_list.append(prod_val)

            idx += n
        else:
            if perm[idx] == -1:
                out_shape_list.append(shapes[token])
            else:
                out_shape_list.append(base_shape[idx])
            idx += 1

    out_shape = tuple(out_shape_list)
    kept_order = [p for p in perm if p != -1]

    intermediate = self.data.reshape(tuple(intermediate_shape))
    transposed = (
        np.transpose(intermediate, axes=kept_order) if kept_order else intermediate
    )
    base = transposed.reshape(base_shape)
    tile_multiples = tuple(o // b for b, o in zip(base_shape, out_shape))

    result_data = np.tile(base, tile_multiples)
    result = Tensor(result_data)

    def compute_grad() -> _NumPyArray:
        grad = result.grad
        for i, (b, o) in enumerate(zip(base_shape, out_shape)):
            if b == 1 and o != 1:
                grad = np.sum(grad, axis=i, keepdims=True)

        grad = grad.reshape(base_shape)
        return grad.reshape(self.shape)

    return result, compute_grad
