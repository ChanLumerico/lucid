VAEConfig
=========

.. autoclass:: lucid.models.VAEConfig

`VAEConfig` stores the encoder stack, decoder stack, optional hierarchical
priors, and KL scheduling options used by :class:`lucid.models.VAE`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class VAEConfig:
        encoders: list[nn.Module]
        decoders: list[nn.Module]
        priors: list[nn.Module] | None = None
        reconstruction_loss: Literal["mse", "bce"] = "mse"
        kl_weight: float = 1.0
        beta_schedule: Callable[[int], float] | None = None
        hierarchical_kl: bool = True
        depth: int | None = None

Parameters
----------

- **encoders** (*list[nn.Module]*):
  Encoder modules that each emit concatenated mean and log-variance tensors.
- **decoders** (*list[nn.Module]*):
  Decoder modules applied from the deepest latent back to the reconstruction.
- **priors** (*list[nn.Module] | None*):
  Optional hierarchical prior modules. The current implementation expects
  exactly `depth - 1` modules when provided.
- **reconstruction_loss** (*Literal["mse", "bce"]*):
  Reconstruction loss mode.
- **kl_weight** (*float*):
  Base KL multiplier.
- **beta_schedule** (*Callable[[int], float] | None*):
  Optional function that overrides KL weight per training step.
- **hierarchical_kl** (*bool*):
  Whether KL weight is distributed over latent levels.
- **depth** (*int | None*):
  Number of latent levels. Defaults to the encoder count.

Validation
----------

- `encoders` and `decoders` must be non-empty and contain only `nn.Module`.
- `depth` must be positive and must match both encoder and decoder counts.
- `priors`, when provided, must contain only `nn.Module` and must have length `depth - 1`.
- `reconstruction_loss` must be `"mse"` or `"bce"`.
- `kl_weight` must be non-negative.
- `beta_schedule` must be callable or `None`.
- `hierarchical_kl` must be a boolean.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    config = models.VAEConfig(
        encoders=[nn.Sequential(nn.Linear(784, 128))],
        decoders=[nn.Sequential(nn.Linear(64, 784))],
        reconstruction_loss="bce",
    )
    model = models.VAE(config)
