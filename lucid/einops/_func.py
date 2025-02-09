from typing import Iterator, Literal
import numpy as np

from lucid._tensor import Tensor
from lucid._backend import create_ufunc_op, _FuncOpReturnType
from lucid.types import _ShapeLike, _EinopsPattern, _NumPyArray


def _parse_pattern(pattern: _EinopsPattern, shape: _ShapeLike, shapes: dict) -> dict:
    in_pattern, out_pattern = map(str.strip, pattern.split("->"))

    in_axes = in_pattern.split()
    tokens = out_pattern.replace("(", " ( ").replace(")", " ) ").split()

    shape_dict = {name: dim for name, dim in zip(in_axes, shape)}
    shape_dict.update(shapes)

    out_shape = []
    reverse_shape = []
    in_order = []
    expanded_axes = {}

    def _consume_group(indices_iter: Iterator) -> list:
        group_tokens = []
        for j in indices_iter:
            if tokens[j] == ")":
                return group_tokens
            group_tokens.append(tokens[j])

        raise ValueError("Unmatched '(' in pattern.")

    it = iter(range(len(tokens)))
    for i in it:
        token = tokens[i]
        if token == "(":
            merged_tokens = _consume_group(it)
            if not all(t in in_axes or t in shapes for t in merged_tokens):
                missing = [
                    t for t in merged_tokens if t not in in_axes and t not in shapes
                ]
                raise ValueError(
                    f"Tokens {missing} not found in input and "
                    + "not provided as repeat factors."
                )

            input_tokens = [t for t in merged_tokens if t in in_axes]
            repeat_tokens = [
                t for t in merged_tokens if t not in in_axes and t in shapes
            ]

            merged_input = (
                np.prod([shape_dict[t] for t in input_tokens]) if input_tokens else 1
            )
            merged_repeat = (
                np.prod([shape_dict[t] for t in repeat_tokens]) if repeat_tokens else 1
            )

            merged_size = int(merged_input * merged_repeat)
            out_shape.append(merged_size)

            group_tuple = tuple(
                [shape_dict[t] for t in input_tokens]
                + [shape_dict[t] for t in repeat_tokens]
            )
            reverse_shape.append(group_tuple)
            in_order.extend([in_axes.index(t) for t in input_tokens])

        else:
            if token not in in_axes:
                raise ValueError(
                    f"Token '{token}' is not in input and "
                    + "expansions are not supported here."
                )

            out_shape.append(shape_dict[token])
            in_order.append(in_axes.index(token))
            reverse_shape.append((shape_dict[token],))

    used_tokens = {in_axes[i] for i in in_order}
    reduced_axes = [i for i, token in enumerate(in_axes) if token not in used_tokens]

    return {
        "out_shape": tuple(out_shape),
        "in_order": tuple(in_order),
        "reverse_shape": reverse_shape,
        "reduced_axes": reduced_axes,
        "expanded_axes": expanded_axes,
    }


@create_ufunc_op()
def rearrange(
    self: Tensor, pattern: _EinopsPattern, **shapes: int
) -> _FuncOpReturnType:
    parsed = _parse_pattern(pattern, self.shape, shapes)
    if parsed["expanded_axes"]:
        raise ValueError(
            "This rearrange implementation only supports "
            + "pure permutations and merging, not expansions."
        )

    transposed = np.transpose(self.data, axes=parsed["in_order"])
    result_data = transposed.reshape(parsed["out_shape"])
    result = Tensor(result_data)

    def compute_grad() -> _NumPyArray:
        grad = result.grad
        transposed_shape = tuple(
            dim for group in parsed["reverse_shape"] for dim in group
        )
        grad_reshaped = grad.reshape(transposed_shape)

        inv_perm = np.argsort(parsed["in_order"])
        grad_final = np.transpose(grad_reshaped, axes=inv_perm)

        return grad_final

    return result, compute_grad


_ReduceStr = Literal["sum", "mean"]


@create_ufunc_op()
def reduce(
    self: Tensor, pattern: _EinopsPattern, reduction: _ReduceStr = "sum", **shapes: int
) -> _FuncOpReturnType:
    parsed = _parse_pattern(pattern, self.shape, shapes)
    if parsed["expanded_axes"]:
        raise ValueError(
            "This reduce implementation only supports permutations/merging "
            "and dropping (reducing) axes, not expansions."
        )

    in_pattern, out_pattern = pattern.split("->")
    in_tokens = in_pattern.split()
    out_tokens = out_pattern.split()

    kept_indices = [i for i, token in enumerate(in_tokens) if token in out_tokens]
    reduced_indices = parsed["reduced_axes"]

    perm = kept_indices + reduced_indices
    transposed = np.transpose(self.data, axes=perm)

    kept_shape = [self.shape[i] for i in kept_indices]
    reduced_shape = [self.shape[i] for i in reduced_indices]

    full_shape = tuple(kept_shape) + tuple(reduced_shape)
    transposed = transposed.reshape(full_shape)

    reduce_axes = tuple(range(len(kept_shape), len(full_shape)))
    if reduction == "sum":
        reduced_data = np.sum(transposed, axis=reduce_axes)
    elif reduction == "mean":
        reduced_data = np.mean(transposed, axis=reduce_axes)
    else:
        raise ValueError(f"Unsupported reduction method: {reduction}")

    result_data = reduced_data.reshape(parsed["out_shape"])
    result = Tensor(result_data)

    def compute_grad() -> _NumPyArray:
        grad = result.grad
        grad = grad.reshape(kept_shape)

        grad_expanded = np.broadcast_to(grad[..., None], full_shape)
        if reduction == "mean":
            factor = np.prod(reduced_shape)
            grad_expanded /= factor

        grad_transposed = grad_expanded.reshape(transposed.shape)
        inv_perm = np.argsort(perm)

        grad_final = np.transpose(grad_transposed, axes=inv_perm)
        return grad_final

    return result, compute_grad
