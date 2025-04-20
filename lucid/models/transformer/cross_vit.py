import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


def _to_2tuple(val: int | float) -> tuple[int | float, ...]:
    return (val, val)


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
