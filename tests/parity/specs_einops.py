"""Specs for einops: rearrange/reduce/repeat/einsum."""

from __future__ import annotations

import torch

from lucid._C import engine as E

from ._specs import OpSpec


SPECS: list[OpSpec] = [
    OpSpec(
        name="einops_rearrange_permute",
        engine_fn=lambda ts: E.einops.rearrange(ts[0], "a b c -> a c b", {}),
        torch_fn=lambda ts: ts[0].permute(0, 2, 1),
        input_shapes=[(2, 3, 4)],
    ),
    OpSpec(
        name="einops_rearrange_split",
        engine_fn=lambda ts: E.einops.rearrange(ts[0], "(a b) c d -> a b c d", {"a": 2}),
        torch_fn=lambda ts: ts[0].reshape(2, ts[0].shape[0]//2, *ts[0].shape[1:]),
        input_shapes=[(4, 3, 5)],
    ),
    OpSpec(
        name="einops_rearrange_merge",
        engine_fn=lambda ts: E.einops.rearrange(ts[0], "b h w c -> b (h w) c", {}),
        torch_fn=lambda ts: ts[0].reshape(ts[0].shape[0], -1, ts[0].shape[-1]),
        input_shapes=[(2, 3, 4, 5)],
    ),
    OpSpec(
        name="einops_reduce_mean",
        engine_fn=lambda ts: E.einops.reduce(ts[0], "b h w c -> b c", "mean", {}),
        torch_fn=lambda ts: ts[0].mean(dim=(1, 2)),
        input_shapes=[(2, 3, 4, 5)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="einops_reduce_sum",
        engine_fn=lambda ts: E.einops.reduce(ts[0], "a b c -> a", "sum", {}),
        torch_fn=lambda ts: ts[0].sum(dim=(1, 2)),
        input_shapes=[(3, 4, 5)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="einops_repeat_new_axis",
        engine_fn=lambda ts: E.einops.repeat(ts[0], "h w -> h w c", {"c": 3}),
        torch_fn=lambda ts: ts[0].unsqueeze(-1).expand(*ts[0].shape, 3),
        input_shapes=[(4, 5)],
    ),
    OpSpec(
        name="einsum_matmul",
        engine_fn=lambda ts: E.einops.einsum("ij,jk->ik", [ts[0], ts[1]]),
        torch_fn=lambda ts: torch.einsum("ij,jk->ik", ts[0], ts[1]),
        input_shapes=[(3, 4), (4, 5)],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="einsum_batched_matmul",
        engine_fn=lambda ts: E.einops.einsum("bij,bjk->bik", [ts[0], ts[1]]),
        torch_fn=lambda ts: torch.einsum("bij,bjk->bik", ts[0], ts[1]),
        input_shapes=[(2, 3, 4), (2, 4, 5)],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="einsum_full_reduce",
        engine_fn=lambda ts: E.einops.einsum("ij->", [ts[0]]),
        torch_fn=lambda ts: torch.einsum("ij->", ts[0]),
        input_shapes=[(4, 5)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="einsum_3operand_chain",
        engine_fn=lambda ts: E.einops.einsum("ij,jk,kl->il", [ts[0], ts[1], ts[2]]),
        torch_fn=lambda ts: torch.einsum("ij,jk,kl->il", ts[0], ts[1], ts[2]),
        input_shapes=[(3, 4), (4, 5), (5, 6)],
        atol=1e-3, rtol=1e-3,
    ),
]
