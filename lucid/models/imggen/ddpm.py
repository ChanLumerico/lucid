import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


class _ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.activation = nn.Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(self.activation(self.norm1(x)))
        h += self.time_proj(t_emb)[:, :, None, None]
        h = self.covn2(self.activation(self.norm2(h)))
        h = self.dropout(h)

        return h + self.residual(x)


class _AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, axis=1)

        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)

        attn = (q.mT @ k) / C**0.5
        attn = F.softmax(attn, axis=-1)

        out = v @ attn.mT
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


class _UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int] = (1, 2, 2),
        num_res_blocks: int = 2,
        attention_res: tuple[int] = (16,),
        image_size: int = 32,
        time_emb_dim: int = 512,
        dropout: float = 0.1,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        current_res = image_size

        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    _ResBlock(now_channels, out_ch, time_emb_dim, dropout)
                )
                now_channels = out_ch

                if current_res in attention_res:
                    self.downs.append(_AttentionBlock(now_channels))

            if mult != channel_mults[-1]:
                self.downs.append(
                    nn.Conv2d(
                        now_channels, now_channels, kernel_size=3, stride=2, padding=1
                    )
                )
                current_res //= 2
            channels.append(now_channels)

        self.mid_block1 = _ResBlock(now_channels, now_channels, time_emb_dim, dropout)
        self.mid_attn = _AttentionBlock(now_channels)
        self.mid_block2 = _ResBlock(now_channels, now_channels, time_emb_dim, dropout)

        self.ups = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    _ResBlock(
                        now_channels + channels.pop(), out_ch, time_emb_dim, dropout
                    )
                )
                now_channels = out_ch

                if current_res in attention_res:
                    self.ups.append(_AttentionBlock(now_channels))

            if mult != channel_mults[0]:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(now_channels, now_channels, kernel_size=3, padding=1),
                    )
                )
                current_res *= 2

        self.final_norm = nn.GroupNorm(num_groups=32, num_channels=now_channels)
        self.final_act = nn.Swish()
        self.final_conv = nn.Conv2d(
            now_channels, out_channels, kernel_size=3, padding=1
        )
        self.final_sigmoid = nn.Sigmoid()

    def time_embedding(self, t: Tensor) -> Tensor:
        half_dim = self.time_mlp[0].in_features // 2
        emb_scale = lucid.log(10000.0) / (half_dim - 1)

        emb = lucid.exp(lucid.arange(half_dim) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = lucid.concatenate([lucid.sin(emb), lucid.cos(emb)], axis=-1)

        return self.time_mlp(emb.astype(lucid.Float32))


class _GaussianDiffuser(nn.Module):
    def __init__(
        self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
    ) -> None:
        super().__init__()
        ...
        # TODO: Continue from here; first implement `lucid.Tensor.cumprod`
