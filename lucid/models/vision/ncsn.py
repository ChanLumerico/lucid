import math
from typing import Literal, Sequence

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["NCSN"]


class _CondInstanceNorm(nn.Module):
    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=eps)
        self.embed = nn.Embedding(num_classes, num_features * 2)

        nn.init.constant(self.embed.weight[:, :num_features], 1.0)
        nn.init.constant(self.embed.weight[:, num_features:], 0.0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if y.dtype != lucid.Long:
            y = y.long()

        h = self.norm(x)
        gamma_beta = self.embed(y)
        gamma, beta = lucid.chunk(gamma_beta, 2, axis=1)

        gamma = gamma.reshape(-1, self.num_features, 1, 1)
        beta = beta.reshape(-1, self.num_features, 1, 1)

        return h * gamma + beta


class _Conv3x3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _ResidualConvUnit(nn.Module):
    def __init__(self, channels: int, num_classes: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = _CondInstanceNorm(channels, num_classes)
        self.conv1 = _Conv3x3(channels, channels, dilation=dilation)

        self.norm2 = _CondInstanceNorm(channels, num_classes)
        self.conv2 = _Conv3x3(channels, channels, dilation=dilation)

        self.act = nn.ELU()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x, y)))
        h = self.conv2(self.act(self.norm2(h, y)))

        return x + h


class _RCUBlock(nn.Module):
    def __init__(
        self, channels: int, num_classes: int, num_units: int = 2, dilation: int = 1
    ) -> None:
        super().__init__()
        self.units = nn.ModuleList(
            [
                _ResidualConvUnit(channels, num_classes, dilation=dilation)
                for _ in range(num_units)
            ]
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = x
        for unit in self.units:
            h = unit(h, y)
        return h


class _CondAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == out_channels:
            self.norm = None
            self.conv = None
        else:
            self.norm = _CondInstanceNorm(in_channels, num_classes)
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        self.act = nn.ELU()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.in_channels == self.out_channels:
            return x
        return self.conv(self.act(self.norm(x, y)))


class _MultiResFusion(nn.Module):
    def __init__(
        self, in_channels_arr: Sequence[int], out_channels: int, num_classes: int
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.norms = nn.ModuleList(
            [_CondInstanceNorm(c, num_classes) for c in in_channels_arr]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1)
                for c in in_channels_arr
            ]
        )
        self.act = nn.ELU()

    def forward(self, xs: Sequence[Tensor], y: Tensor) -> Tensor:
        if len(xs) != len(self.convs):
            raise ValueError(f"Expected {len(self.convs)} inputs, got {len(xs)}")

        target_h = max(x.shape[-2] for x in xs)
        target_w = max(x.shape[-1] for x in xs)
        fused = None

        for x, norm, conv in zip(xs, self.norms, self.convs):
            h = conv(self.act(norm(x, y)))
            if h.shape[-2:] != (target_h, target_w):
                h = F.interpolate(h, size=(target_h, target_w), mode="nearest")

            fused = h if fused is None else fused + h

        return fused


class _ChainedResPooling(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_stages: int = 4) -> None:
        super().__init__()
        self.norms = nn.ModuleList(
            [_CondInstanceNorm(channels, num_classes) for _ in range(num_stages)]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                for _ in range(num_stages)
            ]
        )
        self.act = nn.ELU()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = x
        out = x
        for norm, conv in zip(self.norms, self.convs):
            h = self.act(norm(h, y))
            h = F.max_pool2d(h, kernel_size=5, stride=1, padding=2)
            h = conv(h)
            out = out + h

        return out


class _RefineBlock(nn.Module):
    def __init__(
        self, in_channels_arr: Sequence[int], out_channels: int, num_classes: int
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [_CondAdapter(c, out_channels, num_classes) for c in in_channels_arr]
        )
        self.rcu_in = nn.ModuleList(
            [_RCUBlock(out_channels, num_classes, num_units=2) for _ in in_channels_arr]
        )
        self.msf = _MultiResFusion(
            [out_channels] * len(in_channels_arr), out_channels, num_classes
        )
        self.crp = _ChainedResPooling(out_channels, num_classes, num_stages=4)
        self.rcu_out = _RCUBlock(out_channels, num_classes, num_units=2)

    def forward(self, xs: Sequence[Tensor], y: Tensor) -> Tensor:
        if len(xs) != len(self.adapters):
            raise ValueError(f"Expected {len(self.adapters)} inputs, got {len(xs)}")

        hs: list[Tensor] = []
        for x, adapter, rcu in zip(xs, self.adapters, self.rcu_in):
            h = adapter(x, y)
            h = rcu(h, y)
            hs.append(h)

        h = hs[0] if len(hs) == 1 else self.msf(hs, y)
        h = self.crp(h, y)
        h = self.rcu_out(h, y)

        return h


@nn.set_state_dict_pass_attr("sigmas")
class NCSN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        nf: int = 128,
        num_classes: int = 10,
        dilations: Sequence[int] = (1, 2, 4, 8),
        scale_by_sigma: bool = True,
    ) -> None:
        super().__init__()
        if len(dilations) != 4:
            raise ValueError("Expected 4 dilation values (for 4 RefineNet stages).")

        self.in_channels = in_channels
        self.nf = nf
        self.num_classes = num_classes
        self.scale_by_sigma = bool(scale_by_sigma)

        self.sigmas: nn.Buffer
        self.register_buffer("sigmas", lucid.empty(num_classes))

        self.begin_conv = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)

        self.stage1 = _RCUBlock(nf, num_classes, num_units=2, dilation=dilations[0])
        self.stage2 = _RCUBlock(nf, num_classes, num_units=2, dilation=dilations[1])
        self.stage3 = _RCUBlock(nf, num_classes, num_units=2, dilation=dilations[2])
        self.stage4 = _RCUBlock(nf, num_classes, num_units=2, dilation=dilations[3])

        self.refine4 = _RefineBlock([nf], nf, num_classes)
        self.refine3 = _RefineBlock([nf, nf], nf, num_classes)
        self.refine2 = _RefineBlock([nf, nf], nf, num_classes)
        self.refine1 = _RefineBlock([nf, nf], nf, num_classes)

        self.end_norm = _CondInstanceNorm(nf, num_classes)
        self.end_act = nn.ELU()
        self.end_conv = nn.Conv2d(nf, in_channels, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 0.0)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        h = self.begin_conv(x)
        h1 = self.stage1(h, labels)
        h2 = self.stage2(h1, labels)
        h3 = self.stage3(h2, labels)
        h4 = self.stage4(h3, labels)

        r4 = self.refine4([h4], labels)
        r3 = self.refine3([h3, r4], labels)
        r2 = self.refine2([h2, r3], labels)
        r1 = self.refine1([h1, r2], labels)

        out = self.end_conv(self.end_act(self.end_norm(r1, labels)))
        if self.scale_by_sigma:
            if self.sigmas.size != self.num_classes:
                raise RuntimeError(
                    f"'sigmas' buffer has shape {self.sigmas.shape}; "
                    f"expected ({self.num_classes},). Call 'set_sigmas(...)'."
                )

            used_sigmas = self.sigmas[labels].reshape(-1, 1, 1, 1)
            out = out / used_sigmas

        return out

    @lucid.no_grad()
    def set_sigmas(self, sigmas: Tensor) -> None:
        if sigmas.ndim != 1:
            raise ValueError("sigmas must be 1D.")
        if sigmas.size != self.num_classes:
            raise ValueError(
                f"sigmas length ({sigmas.size}) must match "
                f"num_classes ({self.num_classes})."
            )
        tmp = sigmas.detach()
        tmp.to(self.sigmas.device)
        tmp.to(self.sigmas.dtype)
        self.sigmas.data = tmp.data

    @staticmethod
    @lucid.no_grad()
    def make_sigmas(sigma_begin: float, sigma_end: float, num_scales: int) -> Tensor:
        if sigma_begin <= 0 or sigma_end <= 0:
            raise ValueError("sigmas must be positive.")
        if sigma_begin <= sigma_end:
            raise ValueError(
                "Expected sigma_begin > sigma_end (descending noise schedule)."
            )
        if num_scales < 2:
            raise ValueError("num_scales must be >= 2.")

        return lucid.exp(
            lucid.linspace(math.log(sigma_begin), math.log(sigma_end), num_scales)
        )

    def get_loss(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        labels = lucid.random.randint(
            0, self.sigmas.shape[0], (batch_size,), device=x.device
        ).long()
        used_sigmas = self.sigmas[labels].reshape(batch_size, 1, 1, 1)

        noise = lucid.random.randn(x.shape, device=x.device)
        perturbed = x + used_sigmas * noise
        score = self.forward(perturbed, labels)

        loss = lucid.sum((score * used_sigmas + noise) ** 2, axis=(1, 2, 3)).mean()
        return loss, labels

    @lucid.no_grad()
    def sample(
        self,
        n_samples: int,
        image_size: int,
        in_channels: int,
        n_steps_each: int,
        step_lr: float,
        clip: bool = True,
        denoise: bool = False,
        init: Tensor | None = None,
        init_dist: Literal["uniform", "normal"] = "uniform",
        verbose: bool = True,
    ) -> Tensor:
        self.eval()
        if init is None:
            if init_dist == "uniform":
                x = lucid.random.uniform(
                    -1.0,
                    1.0,
                    (n_samples, in_channels, image_size, image_size),
                    device=self.device,
                )
            elif init_dist == "normal":
                x = lucid.random.randn(
                    n_samples, in_channels, image_size, image_size, device=self.device
                )
            else:
                raise ValueError("init_dist must be either 'uniform' or 'normal'.")

        else:
            x = init.to(self.device)
            if x.shape != (n_samples, in_channels, image_size, image_size):
                raise ValueError(
                    f"init has shape {x.shape} but expected "
                    f"{(n_samples, in_channels, image_size, image_size)}."
                )

        from tqdm import tqdm

        total = int(self.sigmas.shape[0]) * int(n_steps_each)
        pbar = (
            tqdm(total=total, desc="Sampling", dynamic_ncols=True) if verbose else None
        )
        for i, sigma in enumerate(self.sigmas):
            labels = lucid.full(n_samples, i, device=self.device, dtype=lucid.Long)
            step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

            for j in range(n_steps_each):
                grad = self.forward(x, labels)
                noise = lucid.random.randn(x.shape, device=self.device)

                x = x + step_size * grad + lucid.sqrt(2.0 * step_size) * noise
                if clip:
                    x = x.clip(-1.0, 1.0)
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix(sigma=f"{sigma.item():.4f}", l=i, t=j)

        if denoise:
            last_label = lucid.full(
                n_samples,
                self.sigmas.shape[0] - 1,
                device=self.device,
                dtype=lucid.Long,
            )
            x = x + (self.sigmas[-1] ** 2) * self.forward(x, last_label)
            if clip:
                x = x.clip(-1.0, 1.0)

        if verbose:
            pbar.close()
        return x
