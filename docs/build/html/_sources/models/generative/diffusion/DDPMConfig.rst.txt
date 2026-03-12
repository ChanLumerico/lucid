DDPMConfig
==========

.. autoclass:: lucid.models.DDPMConfig

`DDPMConfig` stores the optional denoiser module, diffusion step count, and
sampling settings used by :class:`lucid.models.DDPM`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class DDPMConfig:
        model: nn.Module | None = None
        image_size: int = 32
        channels: int = 3
        timesteps: int = 1000
        diffuser: nn.Module | None = None
        clip_denoised: bool = True

Parameters
----------

- **model** (*nn.Module | None*):
  Optional noise prediction network. If omitted, `DDPM` builds the default U-Net.
- **image_size** (*int*):
  Side length of the square image.
- **channels** (*int*):
  Number of image channels.
- **timesteps** (*int*):
  Number of diffusion steps.
- **diffuser** (*nn.Module | None*):
  Optional diffusion process module. If omitted, `DDPM` builds the default
  Gaussian diffuser.
- **clip_denoised** (*bool*):
  Whether reverse-process outputs are clipped to `[0, 1]`.

Validation
----------

- `model` and `diffuser` must be `nn.Module` or `None`.
- `image_size`, `channels`, and `timesteps` must be greater than `0`.
- `clip_denoised` must be a boolean.
- If `diffuser` defines `timesteps`, it must match `timesteps`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.DDPMConfig(image_size=32, channels=3, timesteps=1000)
    model = models.DDPM(config)
