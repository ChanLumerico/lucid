import re
import numpy as np

from functools import partial
from types import ModuleType
from typing import Literal

from lucid._tensor import Tensor
from lucid.types import _EinopsPattern, _NumPyArray, _MLXArray, _ShapeLike

from lucid._backend.core import (
    operation,
    unary_func_op,
    poly_func_op,
    _FuncOpReturnType,
    _GradFuncType,
)
from lucid._backend.metal import mx


_ReduceStr = Literal["sum", "mean"]


def _parse_pattern(pattern_side: str) -> list[str | tuple[str, ...]]:
    parts = re.findall(r"\.\.\.|\([^)]+\)|\S+", pattern_side)
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
    def __init__(
        self, pattern: _EinopsPattern, t_shape: _ShapeLike, **shapes: int
    ) -> None:
        super().__init__()
        try:
            in_pat, out_pat = map(str.strip, pattern.split("->"))
        except Exception as e:
            raise ValueError(
                "Pattern must contain '->' separating input and output patterns."
            ) from e

        input_tokens = _parse_pattern(in_pat)
        output_tokens = _parse_pattern(out_pat)

        if "..." in input_tokens:
            if input_tokens.count("...") > 1:
                raise ValueError("Only one ellipsis '...' allowed in input pattern.")
            ell_pos = input_tokens.index("...")
            num_unnamed = len(t_shape) - (len(input_tokens) - 1)
            if num_unnamed < 1:
                raise ValueError("Ellipsis '...' expands to zero dimensions.")
            ell_names = [f"_ellipsis_{i}" for i in range(num_unnamed)]
            input_tokens = (
                input_tokens[:ell_pos] + ell_names + input_tokens[ell_pos + 1 :]
            )
        else:
            ell_names = []

        if "..." in output_tokens:
            if not ell_names:
                raise ValueError("Output pattern has '...' but input pattern did not.")
            if output_tokens.count("...") > 1:
                raise ValueError("Only one ellipsis '...' allowed in output pattern.")
            ell_pos_out = output_tokens.index("...")
            output_tokens = (
                output_tokens[:ell_pos_out]
                + ell_names
                + output_tokens[ell_pos_out + 1 :]
            )

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
        return self.result, partial(self.__grad__, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        inter = a.data.reshape(tuple(self.inter_shape))
        transposed = mx.transpose(inter, axes=self.perm)

        self.result = Tensor(transposed.reshape(self.final_shape))
        return self.result, partial(self.__grad__, a=a, lib_=mx)

    def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        unmerged_shape = tuple(dim for group in self.group_splits for dim in group)
        grad_unmerged = self.result.grad.reshape(unmerged_shape)

        inv_perm = lib_.argsort(lib_.array(self.perm))
        grad_interm = lib_.transpose(grad_unmerged, axes=inv_perm)

        return grad_interm.reshape(a.shape)


class reduce(operation):
    def __init__(
        self,
        pattern: _EinopsPattern,
        reduction: _ReduceStr,
        t_shape: _ShapeLike,
        **shapes: int,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        try:
            in_pat, out_pat = map(str.strip, pattern.split("->"))
        except Exception as e:
            raise ValueError(
                "Pattern must contain '->' separating input and output patterns."
            ) from e

        input_tokens = _parse_pattern(in_pat)
        output_tokens = _parse_pattern(out_pat)

        if "..." in input_tokens:
            if input_tokens.count("...") > 1:
                raise ValueError("Only one ellipsis '...' allowed in input pattern.")
            ell_pos = input_tokens.index("...")
            num_unnamed = len(t_shape) - (len(input_tokens) - 1)
            if num_unnamed < 1:
                raise ValueError("Ellipsis '...' expands to zero dimensions.")
            ell_names = [f"_ellipsis_{i}" for i in range(num_unnamed)]
            input_tokens = (
                input_tokens[:ell_pos] + ell_names + input_tokens[ell_pos + 1 :]
            )
        else:
            ell_names = []

        if "..." in output_tokens:
            if not ell_names:
                raise ValueError("Output pattern has '...' but input pattern did not.")
            if output_tokens.count("...") > 1:
                raise ValueError("Only one ellipsis '...' allowed in output pattern.")
            ell_pos_out = output_tokens.index("...")
            output_tokens = (
                output_tokens[:ell_pos_out]
                + ell_names
                + output_tokens[ell_pos_out + 1 :]
            )

        if len(input_tokens) != len(t_shape):
            raise ValueError(
                f"Input pattern has {len(input_tokens)} tokens, "
                + f"but tensor has {len(t_shape)} dimensions."
            )

        self.inter_tokens, self.inter_shape = _build_intermediate(
            input_tokens, t_shape, shapes
        )
        kept_indices: list[int] = []
        kept_groups: list[tuple[int, ...]] = []
        used = set()

        for token in output_tokens:
            if isinstance(token, tuple):
                group_inds, group_dims = [], []
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

                    group_inds.append(found)
                    group_dims.append(self.inter_shape[found])
                    used.add(found)

                kept_indices.extend(group_inds)
                kept_groups.append(tuple(group_dims))

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

                kept_indices.append(found)
                kept_groups.append((self.inter_shape[found],))
                used.add(found)

        all_indices = set(range(len(self.inter_tokens)))
        reduced_indices = sorted(list(all_indices - used))

        self.perm = kept_indices + reduced_indices

        self.kept_flat_shape = tuple(self.inter_shape[i] for i in kept_indices)
        self.reduced_shape = tuple(self.inter_shape[i] for i in reduced_indices)
        self.final_shape = tuple(np.prod(group) for group in kept_groups)

    def _unified(self, arr: _NumPyArray | _MLXArray) -> _NumPyArray | _MLXArray:
        intermediate = arr.reshape(tuple(self.inter_shape))
        transposed = intermediate.transpose(*self.perm)

        reduce_axes = tuple(
            range(
                len(self.kept_flat_shape),
                len(self.kept_flat_shape) + len(self.reduced_shape),
            )
        )
        if self.reduction == "sum":
            reduced_data = transposed.sum(axis=reduce_axes)
        elif self.reduction == "mean":
            reduced_data = transposed.mean(axis=reduce_axes)
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")

        return reduced_data.reshape(self.final_shape)

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(self._unified(a.data))
        return self.result, partial(self.__grad__, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(self._unified(a.data))
        return self.result, partial(self.__grad__, a=a, lib_=mx)

    def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        grad_kept = self.result.grad.reshape(self.kept_flat_shape)
        new_shape = self.kept_flat_shape + (1,) * len(self.reduced_shape)

        grad_expanded = grad_kept.reshape(new_shape)
        grad_broadcast = lib_.broadcast_to(
            grad_expanded, self.kept_flat_shape + tuple(self.reduced_shape)
        )

        if self.reduction == "mean":
            factor = (
                lib_.prod(lib_.array(self.reduced_shape)) if self.reduced_shape else 1
            )
            grad_broadcast = grad_broadcast / factor

        inv_perm = lib_.argsort(lib_.array(self.perm))
        grad_intermediate = lib_.transpose(grad_broadcast, axes=inv_perm)

        return grad_intermediate.reshape(a.shape)

    def __flops__(self, _) -> int:
        reduced_size = np.prod(self.reduced_shape)
        output_size = np.prod(self.final_shape)

        if self.reduction == "sum":
            return output_size * (reduced_size - 1)
        elif self.reduction == "mean":
            return output_size * reduced_size
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")


class repeat(operation):
    def __init__(
        self, pattern: _EinopsPattern, t_shape: _ShapeLike, **shapes: int
    ) -> None:
        super().__init__()
        try:
            in_pat, out_pat = map(str.strip, pattern.split("->"))
        except Exception as e:
            raise ValueError(
                "Pattern must contain '->' separating input and output patterns."
            ) from e

        input_tokens = _parse_pattern(in_pat)
        output_tokens = _parse_pattern(out_pat)

        if "..." in input_tokens:
            if input_tokens.count("...") > 1:
                raise ValueError("Only one ellipsis '...' allowed in input pattern.")
            ell_pos = input_tokens.index("...")
            num_unnamed = len(t_shape) - (len(input_tokens) - 1)
            if num_unnamed < 1:
                raise ValueError("Ellipsis '...' expands to zero dimensions.")
            ell_names = [f"_ellipsis_{i}" for i in range(num_unnamed)]
            input_tokens = (
                input_tokens[:ell_pos] + ell_names + input_tokens[ell_pos + 1 :]
            )
        else:
            ell_names = []

        if "..." in output_tokens:
            if not ell_names:
                raise ValueError("Output pattern has '...' but input pattern did not.")
            if output_tokens.count("...") > 1:
                raise ValueError("Only one ellipsis '...' allowed in output pattern.")
            ell_pos_out = output_tokens.index("...")
            output_tokens = (
                output_tokens[:ell_pos_out]
                + ell_names
                + output_tokens[ell_pos_out + 1 :]
            )

        if len(input_tokens) != len(t_shape):
            raise ValueError(
                f"Input pattern has {len(input_tokens)} tokens, "
                + f"but tensor has {len(t_shape)} dimensions."
            )

        intermediate_tokens, self.intermediate_shape = _build_intermediate(
            input_tokens, t_shape, shapes
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
                        group_dims.append(self.intermediate_shape[found])
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
                    group_splits.append((self.intermediate_shape[found],))
                    used_indices.add(found)

        self.base_shape = tuple(dim for group in group_splits for dim in group)
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
                    prod_val = np.prod(self.base_shape[idx : idx + n])
                    out_shape_list.append(prod_val)

                idx += n
            else:
                if perm[idx] == -1:
                    out_shape_list.append(shapes[token])
                else:
                    out_shape_list.append(self.base_shape[idx])
                idx += 1

        self.out_shape = tuple(out_shape_list)
        self.kept_order = [p for p in perm if p != -1]

    def _unified(
        self, arr: _NumPyArray | _MLXArray, lib_: ModuleType
    ) -> _NumPyArray | _MLXArray:
        intermediate = arr.reshape(tuple(self.intermediate_shape))
        transposed = (
            lib_.transpose(intermediate, axes=self.kept_order)
            if self.kept_order
            else intermediate
        )
        base = transposed.reshape(self.base_shape)
        tile_multiples = tuple(o // b for b, o in zip(self.base_shape, self.out_shape))

        result_data = lib_.tile(base, tile_multiples)
        return result_data

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(self._unified(a.data, lib_=np))
        return self.result, partial(self.__grad__, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(self._unified(a.data, lib_=mx))
        return self.result, partial(self.__grad__, a=a)

    def __grad__(self, a: Tensor) -> _GradFuncType:
        grad = self.result.grad
        for i, (b, o) in enumerate(zip(self.base_shape, self.out_shape)):
            if b == 1 and o != 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad.reshape(self.base_shape).reshape(a.shape)

    def __flops__(self, _) -> int:
        return int(np.prod(self.out_shape))


class einsum(operation):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern

    def _parse_equation(self, n_in: int) -> tuple[list[str], str]:
        eq = self.pattern.replace(" ", "")
        if "->" in eq:
            lhs, rhs = eq.split("->")
        else:
            lhs, rhs = eq, None

        inputs = lhs.split(",")
        if len(inputs) != n_in:
            raise ValueError(f"Pattern expects {len(inputs)} operands, got {n_in}.")

        if rhs is None:
            counts: dict[str, int] = {}
            for s in inputs:
                for c in s.replace("...", ""):
                    counts[c] = counts.get(c, 0) + 1

            rhs = "".join(sorted([c for c, v in counts.items() if v == 1]))
            if any("..." in s for s in inputs):
                rhs = "..." + rhs

        return inputs, rhs

    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data for a in arr]
        self.in_sub, self.out_sub = self._parse_equation(len(arr))

        self.result = Tensor(np.einsum(self.pattern, *data_arr))
        return self.result, partial(self.__grad__, arr=arr, lib_=np)

    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data.astype(mx.float32) for a in arr]
        self.in_sub, self.out_sub = self._parse_equation(len(arr))

        self.result = Tensor(mx.einsum(self.pattern, *data_arr))
        return self.result, partial(self.__grad__, arr=arr, lib_=mx)

    def __grad__(self, arr: tuple[Tensor, ...], lib_: ModuleType) -> _GradFuncType:
        grads = []
        for i in range(len(arr)):
            other = [t.data for j, t in enumerate(arr) if j != i]

            eq = f"{self.out_sub}," + ",".join(self.in_sub[:i] + self.in_sub[i + 1 :])
            eq += f"->{self.in_sub[i]}"

            grads.append(lib_.einsum(eq, self.result.grad, *other))

        return tuple(grads)
