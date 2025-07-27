from typing import Callable, Literal

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["VAE"]


class VAE(nn.Module):
    def __init__(
        self,
        encoders: list[nn.Module],
        decoders: list[nn.Module],
        priors: list[nn.Module] | None,
        reconstruction_loss: Literal["mse", "bce"] = "mse",
        kl_weight: float = 1.0,
        beta_schedule: Callable[[int], float] | None = None,
        hierarchical_kl: bool = True,
        depth: int | None = None,
    ) -> None:
        super().__init__()
        self.depth = depth or len(encoders)
        assert (
            self.depth == len(encoders) == len(decoders)
        ), "Mismatch in component lengths"

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.priors = nn.ModuleList(priors) if priors is not None else None

        self.reconstruction_loss = reconstruction_loss
        self.base_kl_weight = kl_weight
        self.beta_schedule = beta_schedule
        self.hierarchical_kl = hierarchical_kl

        self._training_step = 0

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = lucid.exp(0.5 * logvar)
        eps = lucid.random.randn(std.shape, device=std.device)
        return mu + eps * std

    def encode(self, x: Tensor) -> tuple[list[Tensor], ...]:
        mus: list[Tensor] = []
        logvars: list[Tensor] = []
        zs: list[Tensor] = []
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            mu, logvar = lucid.split(h, 2, axis=1)
            z = self.reparameterize(mu, logvar)

            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)
            h = z

        return zs, mus, logvars

    def decode(self, zs: list[Tensor]) -> Tensor:
        h = zs[-1]
        for decoder in reversed(self.decoders):
            h = decoder(h)
        return h

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        zs, mus, logvars = self.encode(x)
        recon = self.decode(zs)
        return recon, mus, logvars, zs

    def compute_kl(
        self, mu_q: Tensor, logvar_q: Tensor, mu_p: Tensor, logvar_p: Tensor
    ) -> Tensor:
        total_kl = 0.5 * (
            logvar_p
            - logvar_q
            + (lucid.exp(logvar_q) + (mu_q - mu_p) ** 2) / lucid.exp(logvar_p)
            - 1
        )
        return total_kl.sum(axis=1).mean()

    def current_beta(self) -> float:
        if self.beta_schedule is not None:
            return self.beta_schedule(self._training_step)
        return self.base_kl_weight

    def get_loss(
        self,
        x: Tensor,
        recon: Tensor,
        mus: list[Tensor],
        logvars: list[Tensor],
        zs: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        beta = self.current_beta()
        self._training_step += 1

        if self.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(recon, x)
        elif self.reconstruction_loss == "bce":
            recon = recon.clip(1e-7, 1 - 1e-7)
            recon_loss = F.binary_cross_entropy(recon, x)
        else:
            raise ValueError(
                f"Unsupported reconstruction loss: {self.reconstruction_loss}"
            )

        kl = Tensor(0.0)
        for l in range(self.depth):
            mu_q, logvar_q = mus[l], logvars[l]
            if l == self.depth - 1:
                mu_p = lucid.zeros_like(mu_q)
                logvar_p = lucid.zeros_like(logvar_q)
            elif self.priors is not None:
                prior_out = self.priors[l](zs[l + 1])
                mu_p, logvar_p = lucid.split(prior_out, 2, axis=1)
            else:
                mu_p = lucid.zeros_like(mu_q)
                logvar_p = lucid.zeros_like(logvar_q)

            scale = beta / self.depth if self.hierarchical_kl else beta
            kl += scale * self.compute_kl(mu_q, logvar_q, mu_p, logvar_p)

        return recon_loss + kl, recon_loss, kl
