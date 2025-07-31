import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["DDPM"]


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
        self.dropout = nn.Dropout2d(dropout)

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(self.activation(self.norm1(x)))
        h += self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.activation(self.norm2(h)))
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
        self.timesteps = timesteps
        self.register_buffer("betas", lucid.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)

        self.register_buffer("alphas_cumprod", lucid.cumprod(self.alphas, axis=0))
        self.register_buffer(
            "alphas_cumprod_prev",
            lucid.concatenate([Tensor(1.0), self.alphas_cumprod[:-1]], axis=0),
        )
        self.register_buffer(
            "posterior_var",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )

    def sample_timesteps(self, batch_size: int) -> Tensor:
        return lucid.random.randint(0, self.timesteps, (batch_size,))

    def add_noise(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        shape = (t.shape[0],) + (1,) * (x_start.ndim - 1)
        sqrt_alpha = lucid.sqrt(self.alphas_cumprod)[t].reshape(*shape)
        sqrt_one_minus_alpha = lucid.sqrt(1.0 - self.alphas_cumprod)[t].reshape(*shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def denoise(
        self, model: nn.Module, x: Tensor, t: Tensor, clip_denoised: bool
    ) -> Tensor:
        noise_pred = model(x, t)
        shape = (t.shape[0],) + (1,) * (x.ndim - 1)

        alpha_t = self.alphas[t].reshape(*shape)
        beta_t = self.betas[t].reshape(*shape)
        alpha_bar = self.alphas_cumprod[t].reshape(*shape)

        pred_x0 = (x - lucid.sqrt(1 - alpha_bar) * noise_pred) / lucid.sqrt(alpha_bar)
        if clip_denoised:
            pred_x0 = pred_x0.clip(0.0, 1.0)

        posterior_mean = (beta_t * pred_x0 + (1 - beta_t) * x) / (1.0 - alpha_t)
        posterior_var = self.posterior_var[t].reshape(*shape)

        noise = lucid.random.randn(*x.shape)
        nonzero_mask = (t != 0).astype(x.dtype).reshape(*shape)

        return posterior_mean + nonzero_mask * lucid.sqrt(posterior_var) * noise


class DDPM(nn.Module):
    def __init__(
        self,
        model: nn.Module | None = None,
        image_size: int = 32,
        channels: int = 3,
        timesteps: int = 1000,
        diffuser: nn.Module | None = None,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        self.model = model or _UNet(
            in_channels=channels, out_channels=channels, image_size=image_size
        )
        self.diffuser = diffuser or _GaussianDiffuser(timesteps)

        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised

    def forward(self, x: Tensor) -> Tensor:
        return self.get_loss(x)

    def get_loss(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        t = self.diffuser.sample_timesteps(B)
        noise = lucid.random.randn(*x.shape)

        x_t = self.diffuser.add_noise(x, t, noise)
        noise_pred = self.model(x_t, t)

        return F.mse_loss(noise_pred, noise)

    def sample(self, batch_size: int) -> Tensor:
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = lucid.random.randn(*shape).to(self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = lucid.full((shape[0],), t, dtype=lucid.Int32).to(self.device)
            x = self.diffuser.denoise(self.model, x, t_batch, self.clip_denoised)

        return x
