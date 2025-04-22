from typing import Any, override
from functools import partial

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "CrossViT",
    "crossvit_tiny",
    "crossvit_small",
    "crossvit_base",
    "crossvit_9",
    "crossvit_15",
    "crossvit_18",
    "crossvit_9_dagger",
    "crossvit_15_dagger",
    "crossvit_18_dagger",
]


def _to_2tuple(val: Any) -> tuple[Any, Any]:
    return (val, val)


def _get_num_patches(img_size: tuple[int, int], patches: tuple[int, int]) -> list[int]:
    return [(i // p) ** 2 for i, p in zip(img_size, patches)]


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        multi_conv: bool = False,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim // 4, 7, 4, 3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 3, 0),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
                )

            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim // 4, 7, 4, 3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
                )
        else:
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[2:]
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Input image size {(H, W)} does not match with {self.img_size}."
            )

        x = self.proj(x).flatten(axis=2).swapaxes(1, 2)
        return x


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(self.norm(x)))

        return x


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * lucid.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv.chunk(3)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = q @ k.mT * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_value: float | None = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        mlp_layer: type[nn.Module] = _MLP,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(
            dim,
            num_heads,
            qkv_bias,
            qk_norm,
            proj_bias,
            attn_drop,
            proj_drop,
            norm_layer,
        )
        self.ls1 = _LayerScale(dim, init_value) if init_value else nn.Identity()
        self.drop_path1 = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = _LayerScale(dim, init_value) if init_value else nn.Identity()
        self.drop_path2 = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x += self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x += self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class _CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q = (
            self.wq(x[:, 0:1, ...])
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .swapaxes(1, 2)
        )
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).swapaxes(1, 2)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).swapaxes(1, 2)

        attn = (q @ k.mT) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        has_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _CrossAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop=drop
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = _MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x += self.drop_path(self.mlp(self.norm2(x)))

        return x


class _MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: list[int],
        depth: list[int],
        num_heads: list[int],
        mlp_ratio: list[float],
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: list[float] | float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        num_branches = len(dim)
        self.num_branches = num_branches

        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    _AttentionBlock(
                        dim[d],
                        num_heads[d],
                        mlp_ratio[d],
                        qkv_bias,
                        proj_drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),
                        norm_layer=norm_layer,
                    )
                )
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches]:
                tmp = [nn.Identity()]
            else:
                tmp = [
                    norm_layer(dim[d]),
                    act_layer(),
                    nn.Linear(dim[d], dim[(d + 1) % num_branches]),
                ]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]

            partial_crossblock = partial(
                _CrossAttentionBlock,
                dim=dim[d_],
                num_heads=nh,
                mlp_ratio=mlp_ratio[d],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            if depth[-1] == 0:
                self.fusion.append(partial_crossblock(has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(partial_crossblock(has_mlp=False))

                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d]:
                tmp = [nn.Identity()]
            else:
                tmp = [
                    norm_layer(dim[(d + 1) % num_branches]),
                    act_layer(),
                    nn.Linear(dim[(d + 1) % num_branches], dim[d]),
                ]
            self.revert_projs.append(nn.Sequential(*tmp))

    @override
    def forward(self, xs: list[Tensor]) -> list[Tensor]:
        outs_b = [block(x_) for x_, block in zip(xs, self.blocks)]
        proj_cls_token = [f_proj(x[:, 0:1]) for x, f_proj in zip(outs_b, self.projs)]

        outs = []
        for i in range(self.num_branches):
            tmp = lucid.concatenate(
                [proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]],
                axis=1,
            )
            tmp = self.fusion[i](tmp)

            revert_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = lucid.concatenate(
                [revert_proj_cls_token, outs_b[i][:, 1:, ...]], axis=1
            )
            outs.append(tmp)

        return outs


class CrossViT(nn.Module):
    def __init__(
        self,
        img_size: int | list[int] = [224, 224],
        patch_size: list[int] = [12, 16],
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: list[int] = [192, 384],
        depth: list[list[int]] = [[1, 3, 1], [1, 3, 1], [1, 3, 1]],
        num_heads: list[int] = [6, 12],
        mlp_ratio: list[float] = [2.0, 2.0, 4.0],
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        multi_conv: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(img_size, (list, tuple)):
            img_size = _to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _get_num_patches(img_size, patch_size)
        self.num_branches = len(num_patches)

        self.pos_embed = nn.ParameterList()
        self.cls_token = nn.ParameterList()

        for i in range(self.num_branches):
            self.pos_embed.append(
                nn.Parameter(lucid.zeros(1, 1 + num_patches[i], embed_dim[i]))
            )
            self.cls_token.append(nn.Parameter(lucid.zeros(1, 1, embed_dim[i])))

        self.patch_embed = nn.ModuleList()
        for im_s, p, d in zip(img_size, patch_size, embed_dim):
            self.patch_embed.append(
                _PatchEmbed(im_s, p, in_channels, d, multi_conv=multi_conv)
            )

        self.pos_drop = nn.Dropout(drop_rate)

        total_depth = sum([sum(d[-2:]) for d in depth])
        dpr = [x.item() for x in lucid.linspace(0, drop_path_rate, total_depth)]
        dpr_ptr = 0

        self.blocks = nn.ModuleList()
        for block_cfg in depth:
            cur_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr : dpr_ptr + cur_depth]

            blk = _MultiScaleBlock(
                dim=embed_dim,
                depth=block_cfg,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_,
                norm_layer=norm_layer,
            )
            dpr_ptr += cur_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList(
            [norm_layer(embed_dim[i]) for i in range(self.num_branches)]
        )
        self.head = nn.ModuleList()
        for i in range(self.num_branches):
            self.head.append(
                nn.Linear(embed_dim[i], num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                nn.init.normal(self.pos_embed[i], std=0.02)
            nn.init.normal(self.cls_token[i], std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant(m.weight, 1.0)
            nn.init.constant(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, _ = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = (
                F.interpolate(x, (self.img_size[i], self.img_size[i]), mode="bilinear")
                if H != self.img_size[i]
                else x
            )
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].repeat(B, axis=0)

            tmp = lucid.concatenate([cls_tokens, tmp], axis=1)
            tmp += self.pos_embed[i]
            tmp = self.pos_drop(tmp)

            xs.append(tmp)

        for block in self.blocks:
            xs = block(xs)

        xs = [self.norm[i](_x) for i, _x in enumerate(xs)]
        out = [_x[:, 0] for _x in xs]

        logit = [self.head[i](_x) for i, _x in enumerate(out)]
        logit = lucid.stack(logit, axis=0).mean(axis=0)

        return logit


@register_model
def crossvit_tiny(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[96, 192],
        depth=[[1, 4, 0] for _ in range(3)],
        num_heads=[3, 3],
        mlp_ratio=[4, 4, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@register_model
def crossvit_small(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[192, 384],
        depth=[[1, 4, 0] for _ in range(3)],
        num_heads=[6, 6],
        mlp_ratio=[4, 4, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@register_model
def crossvit_base(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[384, 768],
        depth=[[1, 4, 0] for _ in range(3)],
        num_heads=[12, 12],
        mlp_ratio=[4, 4, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@register_model
def crossvit_9(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[128, 256],
        depth=[[1, 3, 0] for _ in range(3)],
        num_heads=[4, 4],
        mlp_ratio=[3, 3, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@register_model
def crossvit_15(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[192, 384],
        depth=[[1, 5, 0] for _ in range(3)],
        num_heads=[6, 6],
        mlp_ratio=[3, 3, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@register_model
def crossvit_18(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[224, 448],
        depth=[[1, 6, 0] for _ in range(3)],
        num_heads=[7, 7],
        mlp_ratio=[3, 3, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@register_model
def crossvit_9_dagger(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[128, 256],
        depth=[[1, 3, 0] for _ in range(3)],
        num_heads=[4, 4],
        mlp_ratio=[3, 3, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        multi_conv=True,
        **kwargs,
    )


@register_model
def crossvit_15_dagger(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[192, 384],
        depth=[[1, 5, 0] for _ in range(3)],
        num_heads=[6, 6],
        mlp_ratio=[3, 3, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        multi_conv=True,
        **kwargs,
    )


@register_model
def crossvit_18_dagger(num_classes: int = 1000, **kwargs) -> CrossViT:
    return CrossViT(
        img_size=[240, 224],
        num_classes=num_classes,
        embed_dim=[224, 448],
        depth=[[1, 6, 0] for _ in range(3)],
        num_heads=[7, 7],
        mlp_ratio=[3, 3, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        multi_conv=True,
        **kwargs,
    )
