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
    def __init__(self):
        super().__init__()
