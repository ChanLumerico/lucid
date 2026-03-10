from dataclasses import dataclass
from typing import ClassVar, Literal, Type, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "SwinTransformer",
    "SwinTransformerConfig",
    "SwinTransformer_V2",
    "SwinTransformerV2Config",
    "swin_tiny",
    "swin_small",
    "swin_base",
    "swin_large",
    "swin_v2_tiny",
    "swin_v2_small",
    "swin_v2_base",
    "swin_v2_large",
    "swin_v2_huge",
    "swin_v2_giant",
]


@dataclass
class SwinTransformerConfig:
    img_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 96
    depths: tuple[int, ...] | list[int] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] | list[int] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: float | None = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Type[nn.Module] = nn.LayerNorm
    abs_pos_emb: bool = False
    patch_norm: bool = True

    def __post_init__(self) -> None:
        self.depths = tuple(self.depths)
        self.num_heads = tuple(self.num_heads)

        if self.img_size <= 0:
            raise ValueError("img_size must be greater than 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")
        if self.img_size < self.patch_size:
            raise ValueError("img_size must be greater than or equal to patch_size")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.num_classes < 0:
            raise ValueError("num_classes must be greater than or equal to 0")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0")
        if len(self.depths) == 0:
            raise ValueError("depths must contain at least one stage")
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if any(depth <= 0 for depth in self.depths):
            raise ValueError("depths must contain positive integers")
        if any(head <= 0 for head in self.num_heads):
            raise ValueError("num_heads must contain positive integers")
        for stage_index, heads in enumerate(self.num_heads):
            if (self.embed_dim * 2**stage_index) % heads != 0:
                raise ValueError(
                    "stage embedding dimensions must be divisible by num_heads"
                )
        if self.img_size // self.patch_size < 2 ** (len(self.depths) - 1):
            raise ValueError(
                "img_size and patch_size do not provide enough resolution for the configured number of stages"
            )
        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be greater than 0")
        if self.drop_rate < 0 or self.drop_rate >= 1:
            raise ValueError("drop_rate must be in the range [0, 1)")
        if self.attn_drop_rate < 0 or self.attn_drop_rate >= 1:
            raise ValueError("attn_drop_rate must be in the range [0, 1)")
        if self.drop_path_rate < 0 or self.drop_path_rate >= 1:
            raise ValueError("drop_path_rate must be in the range [0, 1)")


@dataclass
class SwinTransformerV2Config(SwinTransformerConfig):
    pass


def _to_2tuple(val: int | float) -> tuple[int | float, ...]:
    return (val, val)


def window_partition(x: Tensor, window_size: int) -> Tensor:
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)

    windows = x.swapaxes(2, 3).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.swapaxes(2, 3).reshape(B, H, W, -1)
    return x


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class _WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            lucid.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.normal(self.relative_position_bias_table, std=0.02)

        coords_h = lucid.arange(self.window_size[0])
        coords_w = lucid.arange(self.window_size[1])

        coords = lucid.stack(lucid.meshgrid(coords_h, coords_w))
        coords_flatten = coords.reshape(coords.shape[0], -1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose((1, 2, 0))

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_pos_index = relative_coords.sum(axis=-1)
        self.register_buffer("relative_pos_index", relative_pos_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale
        attn = q @ k.mT

        relative_pos_bias = self.relative_position_bias_table[
            self.relative_pos_index.flatten().astype(lucid.Int)
        ].reshape(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_pos_bias = relative_pos_bias.transpose((2, 0, 1))
        attn += relative_pos_bias.unsqueeze(axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N)
            attn += mask.unsqueeze(axis=1).unsqueeze(axis=0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_res: tuple[int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_res = input_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self._configured_shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_res) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_res)

        if not (0 <= self.shift_size < self.window_size):
            raise ValueError("shift_size must be in [0, window_size).")

        self.norm1 = norm_layer(dim)
        self.attn = _WindowAttention(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = self._build_attn_mask(*self.input_res)

        self.attn_mask: nn.Buffer
        self.register_buffer("attn_mask", attn_mask)

    def _effective_shift_size(self, H: int, W: int) -> int:
        if min(H, W) <= self.window_size:
            return 0
        return self._configured_shift_size

    def _build_attn_mask(
        self,
        H: int,
        W: int,
        device: str | None = None,
        *,
        shift_size: int | None = None,
        window_size: int | None = None,
    ) -> Tensor | None:
        shift_size = self.shift_size if shift_size is None else shift_size
        window_size = self.window_size if window_size is None else window_size

        if shift_size == 0:
            return None

        img_mask = lucid.zeros((1, H, W, 1), device=device or "cpu")
        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size
        if pad_b > 0:
            img_mask = lucid.concatenate(
                [img_mask, lucid.zeros((1, pad_b, W, 1), device=img_mask.device)],
                axis=1,
            )
        if pad_r > 0:
            img_mask = lucid.concatenate(
                [
                    img_mask,
                    lucid.zeros(
                        (1, H + pad_b, pad_r, 1),
                        device=img_mask.device,
                    ),
                ],
                axis=2,
            )
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.reshape(-1, window_size * window_size)

        attn_mask = mask_windows.unsqueeze(axis=1) - mask_windows.unsqueeze(axis=2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "wrong input feature size."
        window_size = self.window_size
        shift_size = self._effective_shift_size(H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size
        if pad_b > 0:
            pad_h = lucid.zeros((B, pad_b, W, C), dtype=x.dtype, device=x.device)
            x = lucid.concatenate([x, pad_h], axis=1)
        if pad_r > 0:
            pad_w = lucid.zeros(
                (B, H + pad_b, pad_r, C), dtype=x.dtype, device=x.device
            )
            x = lucid.concatenate([x, pad_w], axis=2)
        Hp, Wp = H + pad_b, W + pad_r

        if shift_size > 0:
            shifted_x = lucid.roll(x, shifts=(-shift_size, -shift_size), axis=(1, 2))
            use_cached_mask = (
                shift_size == self.shift_size
                and pad_b == 0
                and pad_r == 0
                and self.attn_mask is not None
                and self.attn_mask.shape[0] == (Hp // window_size) * (Wp // window_size)
            )
            attn_mask = (
                self.attn_mask
                if use_cached_mask
                else self._build_attn_mask(
                    Hp,
                    Wp,
                    device=x.device,
                    shift_size=shift_size,
                    window_size=window_size,
                )
            )
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)

        x_windows = x_windows.reshape(-1, window_size * window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.reshape(-1, window_size, window_size, C)

        if shift_size > 0:
            shifted_x = window_reverse(attn_windows, window_size, Hp, Wp)
            x = lucid.roll(shifted_x, shifts=(shift_size, shift_size), axis=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, window_size, Hp, Wp)
            x = shifted_x

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :]

        x = x.reshape(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x += self.drop_path(self.mlp(self.norm2(x)))

        return x


class _PatchMerging(nn.Module):
    def __init__(
        self,
        input_res: tuple[int, int],
        dim: int,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_res = input_res
        self.dim = dim

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "wrong input feature size."

        x = x.reshape(B, H, W, C)
        if H % 2 == 1:
            x = lucid.concatenate(
                [x, lucid.zeros((B, 1, W, C), dtype=x.dtype, device=x.device)], axis=1
            )
            H += 1
        if W % 2 == 1:
            x = lucid.concatenate(
                [x, lucid.zeros((B, H, 1, C), dtype=x.dtype, device=x.device)], axis=2
            )
            W += 1

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = lucid.concatenate([x0, x1, x2, x3], axis=-1)
        x = x.reshape(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class _BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_res: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        _version: Literal["v1", "v2"] = "v1",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_res = input_res
        self.depth = depth

        swin_block_dict = {"v1": _SwinTransformerBlock, "v2": _SwinTransformerBlock_V2}
        blocks = [
            swin_block_dict[_version](
                dim,
                input_res,
                num_heads,
                window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.downsample: nn.Module | None
        if downsample is not None:
            self.downsample = downsample(input_res, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def _init_res_post_norm(self) -> None:
        for block in self.blocks:
            nn.init.constant(block.norm1.bias, 0)
            nn.init.constant(block.norm1.weight, 0)

            nn.init.constant(block.norm2.bias, 0)
            nn.init.constant(block.norm2.weight, 0)


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        patches_res = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_res = patches_res
        self.num_patches = patches_res[0] * patches_res[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            + f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        x = x.reshape(*x.shape[:2], -1).swapaxes(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class SwinTransformer(nn.Module):
    _version: ClassVar[str] = "v1"

    def __init__(self, config: SwinTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.num_layers = len(config.depths)
        self.embed_dim = config.embed_dim
        self.abs_pos_emb = config.abs_pos_emb
        self.patch_norm = config.patch_norm
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = config.mlp_ratio

        self.patch_embed = _PatchEmbed(
            config.img_size,
            config.patch_size,
            config.in_channels,
            config.embed_dim,
            norm_layer=config.norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_res = self.patch_embed.patches_res
        self.patches_res = patches_res

        if self.abs_pos_emb:
            self.absolute_pos_emb = nn.Parameter(
                lucid.zeros(1, num_patches, config.embed_dim)
            )
            nn.init.normal(self.absolute_pos_emb, std=0.02)

        self.pos_drop = nn.Dropout(config.drop_rate)
        dpr = [
            x.item()
            for x in lucid.linspace(0, config.drop_path_rate, sum(config.depths))
        ]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = _BasicLayer(
                dim=int(config.embed_dim * 2**i),
                input_res=(patches_res[0] // (2**i), patches_res[1] // (2**i)),
                depth=config.depths[i],
                num_heads=config.num_heads[i],
                window_size=config.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[
                    sum(config.depths[:i]) : sum(config.depths[: i + 1])
                ],
                norm_layer=config.norm_layer,
                downsample=_PatchMerging if i < self.num_layers - 1 else None,
                _version=self._version,
            )
            self.layers.append(layer)

        self.norm = config.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant(module.bias, 0)
            nn.init.constant(module.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        if self.abs_pos_emb:
            x += self.absolute_pos_emb
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.swapaxes(1, 2))

        x = x.reshape(x.shape[0], -1)
        x = self.head(x)

        return x


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
        rel_coords_table[..., 0] /= max(self.window_size[0] - 1, 1)
        rel_coords_table[..., 1] /= max(self.window_size[1] - 1, 1)

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
        logit_scale = lucid.clip(self.logit_scale, None, lucid.log(100.0).item())
        attn *= lucid.exp(logit_scale)

        rel_pos_bias_table = self.cpb_mlp(self.rel_coords_table).reshape(
            -1, self.num_heads
        )
        rel_pos_bias = rel_pos_bias_table[
            self.rel_pos_index.flatten().astype(lucid.Int)
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
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N)
            attn += mask.unsqueeze(axis=1).unsqueeze(axis=0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _SwinTransformerBlock_V2(_SwinTransformerBlock):
    def __init__(
        self,
        dim: int,
        input_res: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__(
            dim,
            input_res,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )
        self.attn = _WindowAttention_V2(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "wrong input feature size."
        window_size = self.window_size
        shift_size = self._effective_shift_size(H, W)

        shortcut = x
        x = x.reshape(B, H, W, C)

        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size
        if pad_b > 0:
            x = lucid.concatenate(
                [x, lucid.zeros((B, pad_b, W, C), dtype=x.dtype, device=x.device)],
                axis=1,
            )
        if pad_r > 0:
            x = lucid.concatenate(
                [
                    x,
                    lucid.zeros(
                        (B, H + pad_b, pad_r, C), dtype=x.dtype, device=x.device
                    ),
                ],
                axis=2,
            )
        Hp, Wp = H + pad_b, W + pad_r

        if shift_size > 0:
            shifted_x = x.roll((-shift_size, -shift_size), axis=(1, 2))
            attn_mask = self._build_attn_mask(
                Hp,
                Wp,
                device=x.device,
                shift_size=shift_size,
                window_size=window_size,
            )
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.reshape(-1, window_size * window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows, window_size, Hp, Wp)

        if shift_size > 0:
            x = shifted_x.roll((shift_size, shift_size), axis=(1, 2))
        else:
            x = shifted_x

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :]

        x = x.reshape(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x += self.drop_path(self.norm2(self.mlp(x)))

        return x


class SwinTransformer_V2(SwinTransformer):
    _version: ClassVar[str] = "v2"

    def __init__(self, config: SwinTransformerV2Config) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer._init_res_post_norm()


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_swin_config(
    config_cls: Type[SwinTransformerConfig] | Type[SwinTransformerV2Config],
    *,
    img_size: int,
    num_classes: int,
    embed_dim: int,
    depths: tuple[int, ...] | list[int],
    num_heads: tuple[int, ...] | list[int],
    **kwargs,
) -> SwinTransformerConfig | SwinTransformerV2Config:
    return config_cls(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        **kwargs,
    )


@register_model
def swin_tiny(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerConfig,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )
    return SwinTransformer(config)


@register_model
def swin_small(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerConfig,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )
    return SwinTransformer(config)


@register_model
def swin_base(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerConfig,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs,
    )
    return SwinTransformer(config)


@register_model
def swin_large(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerConfig,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs,
    )
    return SwinTransformer(config)


@register_model
def swin_v2_tiny(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerV2Config,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )
    return SwinTransformer_V2(config)


@register_model
def swin_v2_small(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerV2Config,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )
    return SwinTransformer_V2(config)


@register_model
def swin_v2_base(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerV2Config,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs,
    )
    return SwinTransformer_V2(config)


@register_model
def swin_v2_large(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerV2Config,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs,
    )
    return SwinTransformer_V2(config)


@register_model
def swin_v2_huge(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerV2Config,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=352,
        depths=(2, 2, 18, 2),
        num_heads=(11, 22, 44, 88),
        **kwargs,
    )
    return SwinTransformer_V2(config)


@register_model
def swin_v2_giant(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> SwinTransformer_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embed_dim", "depths", "num_heads"},
        "factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    )
    config = _build_swin_config(
        SwinTransformerV2Config,
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=512,
        depths=(2, 2, 42, 4),
        num_heads=(16, 32, 64, 128),
        **kwargs,
    )
    return SwinTransformer_V2(config)
