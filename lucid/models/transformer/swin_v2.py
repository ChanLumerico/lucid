from typing import Type

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

from .swin import _to_2tuple, window_partition, window_reverse, _MLP


class _WindowAttention_V2(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(lucid.log(10 * lucid.ones(num_heads, 1, 1)))

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, num_heads, bias=False)
        )

        rel_coords_h = lucid.arange(-(self.window_size[0] - 1), self.window_size[0])
        rel_coords_w = lucid.arange(-(self.window_size[1] - 1), self.window_size[1])

        rel_coords_table = (
            lucid.stack(lucid.meshgrid(rel_coords_h, rel_coords_w))
            .transpose((1, 2, 0))
            .unsqueeze(axis=0)
        )
        rel_coords_table[..., 0] /= self.window_size[0] - 1
        rel_coords_table[..., 1] /= self.window_size[1] - 1

        rel_coords_table *= 8
        rel_coords_table = (
            lucid.sign(rel_coords_table)
            * lucid.log2(lucid.abs(rel_coords_table) + 1.0)
            / lucid.log2(8)
        )

        self.rel_coords_table: nn.Buffer
        self.register_buffer("rel_coords_table", rel_coords_table)

        coords_h = lucid.arange(self.window_size[0])
        coords_w = lucid.arange(self.window_size[1])

        coords = lucid.stack(lucid.meshgrid(coords_h, coords_w))
        coords_flatten = coords.reshape(coords.shape[0], -1)

        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel_coords = rel_coords.transpose((1, 2, 0))

        rel_coords[:, :, 0] += self.window_size[0] - 1
        rel_coords[:, :, 1] += self.window_size[1] - 1
        rel_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        rel_pos_index = rel_coords.sum(axis=-1)
        self.rel_pos_index: nn.Buffer
        self.register_buffer("rel_pos_index", rel_pos_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(lucid.zeros(dim))
            self.v_bias = nn.Parameter(lucid.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B_, N, C = x.shape
        qkv_bias = None

        if self.q_bias is not None:
            qkv_bias = lucid.concatenate(
                [self.q_bias, lucid.zeros_like(self.v_bias), self.v_bias]
            )
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).transpose((2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = F.normalize(q, axis=-1) @ F.normalize(k, axis=-1).mT
            logit_scale = lucid.clip(
                self.logit_scale, max_value=lucid.log(100.0).item()
            )
            attn *= lucid.exp(logit_scale)

            rel_pos_bias_table = self.cpb_mlp(self.rel_coords_table).reshape(
                -1, self.num_heads
            )
            rel_pos_bias = rel_pos_bias_table[
                self.rel_pos_index.flatten().astype(int)
            ].reshape(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            rel_pos_bias = rel_pos_bias.transpose((2, 0, 1))
            rel_pos_bias = 16 * F.sigmoid(rel_pos_bias)

            attn += rel_pos_bias.unsqueeze(axis=0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.reshape(
                    B_ // nW, nW, self.num_heads, N, N
                ) + mask.unsqueeze(axis=(0, 1))

                attn = attn.reshape(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x
