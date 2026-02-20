DDPM
====
|diffusion-badge|

.. autoclass:: lucid.models.DDPM

The `DDPM` class implements a Denoising Diffusion Probabilistic Model, 
following the original formulation by Ho et al. (2020). It is designed for 
image generation through iterative denoising of Gaussian-noised data.

This implementation is modular and supports custom noise prediction models 
and diffusion schedules, while defaulting to a U-Net and linear Gaussian 
:math:`\beta`-schedule.

**Total Parameters**: 20,907,649 (Default)

.. mermaid::
    :name: DDPM

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>DDPM</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_UNet"]
        direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m2["time_mlp"]
            direction TB;
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,512) → (1,2048)</span>"];
            m4["Swish"];
            m5["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,512)</span>"];
        end
        m6["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,128,224,224)</span>"];
        subgraph sg_m7["down_resblocks"]
            direction TB;
        style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8(["ModuleList x 3<br/><span style='font-size:11px;font-weight:400'>(1,128,224,224)x2 → (1,128,224,224)</span>"]);
        end
        subgraph sg_m9["downsample"]
            direction TB;
        style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m10(["Conv2d x 2<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,224,224) → (1,128,112,112)</span>"]);
        end
        m11["ModuleDict"];
        subgraph sg_m12["_ResBlock"]
            direction TB;
        style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m13["GroupNorm"];
            m14["Swish"];
            m15["Conv2d"];
            m16["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,512) → (1,256)</span>"];
            m17["GroupNorm"];
            m18["Conv2d"];
            m19["Dropout2d"];
            m20["Identity"];
        end
        subgraph sg_m21["_AttentionBlock"]
            direction TB;
        style sg_m21 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m22["GroupNorm"];
            m23(["Conv2d x 2<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,256,56,56) → (1,768,56,56)</span>"]);
        end
        subgraph sg_m24["_ResBlock"]
            direction TB;
        style sg_m24 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m25["GroupNorm"];
            m26["Swish"];
            m27["Conv2d"];
            m28["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,512) → (1,256)</span>"];
            m29["GroupNorm"];
            m30["Conv2d"];
            m31["Dropout2d"];
            m32["Identity"];
        end
        subgraph sg_m33["upsample"]
            direction TB;
        style sg_m33 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m34(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,256,112,112)</span>"]);
        end
        subgraph sg_m35["up_resblocks"]
            direction TB;
        style sg_m35 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m36(["ModuleList x 3<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56)x2 → (1,256,56,56)</span>"]);
        end
        m37["ModuleDict"];
        m38["GroupNorm"];
        m39["Swish"];
        m40["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,224,224) → (1,3,224,224)</span>"];
        m41["Sigmoid"];
        end
        m42["_GaussianDiffuser"];
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>()</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m3 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m5 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m6 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m10 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m13 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m14 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m15 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m16 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m17 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m18 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m19 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m20 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m22 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m23 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m25 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m26 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m27 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m28 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m29 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m30 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m31 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m32 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m38 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m39 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m40 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m41 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    input --> m3;
    m10 -.-> m8;
    m13 -.-> m14;
    m14 --> m15;
    m14 --> m18;
    m15 --> m16;
    m16 --> m17;
    m17 -.-> m14;
    m18 --> m19;
    m19 --> m20;
    m20 --> m22;
    m22 --> m23;
    m23 --> m25;
    m25 -.-> m26;
    m26 --> m27;
    m26 --> m30;
    m27 --> m28;
    m28 --> m29;
    m29 -.-> m26;
    m3 --> m4;
    m30 --> m31;
    m31 --> m32;
    m32 -.-> m36;
    m34 -.-> m36;
    m36 --> m34;
    m36 --> m38;
    m38 --> m39;
    m39 --> m40;
    m4 --> m5;
    m40 --> m41;
    m41 --> output;
    m5 --> m6;
    m6 -.-> m8;
    m8 --> m10;
    m8 --> m13;

Class Signature
---------------

.. code-block:: python

    class DDPM(
        model: nn.Module | None = None,
        image_size: int = 32,
        channels: int = 3,
        timesteps: int = 1000,
        diffuser: nn.Module | None = None,
        clip_denoised: bool = True,
    )

Parameters
----------

- **model** (*nn.Module | None*):  
  The noise prediction model :math:`\epsilon_\theta(\mathbf{x}_t, t)`. If not given, a default `UNet` is used.

  Required **forward** signature:

  .. code-block:: python

      def forward(self, x: Tensor, t: Tensor) -> Tensor

  where `x` is of shape `(N, C, H, W)` and `t` of shape `(N,)`.

- **image_size** (*int*):  
  Size of the (square) input image.

- **channels** (*int*):  
  Number of channels in the input image.

- **timesteps** (*int*):  
  Total number of diffusion steps. Controls the length of the forward/reverse process.

- **diffuser** (*nn.Module | None*):  
  Module implementing the diffusion process. If not provided, defaults to `GaussianDiffuser`.

  Required methods in `diffuser`:

  .. code-block:: python

      def sample_timesteps(self, batch_size: int) -> Tensor
      def add_noise(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor
      def denoise(self, model: nn.Module, x: Tensor, t: Tensor, clip_denoised: bool) -> Tensor

- **clip_denoised** (*bool*):  
  Whether to clip final denoised outputs to [0, 1].

Returns
-------

Use the `forward` method for training loss:

.. code-block:: python

    loss = ddpm(x)

- **loss** (*Tensor*):  
  MSE loss between true and predicted noise in the denoising process.

Sampling is performed using:

.. code-block:: python

    samples = ddpm.sample(batch_size)

- **samples** (*Tensor*):  
  A tensor of shape `(N, C, H, W)` containing generated images.

Forward Noise Process
---------------------

To diffuse a clean image :math:`\mathbf{x}_0`, noise is incrementally added:

.. math::

    \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})

where :math:`\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s`, and each :math:`\alpha_t = 1 - \beta_t`.

The function :py:meth:`diffuser.add_noise` implements this process.

Reverse Denoising Process
-------------------------

The model predicts the added noise :math:`\epsilon_\theta(\mathbf{x}_t, t)` and reconstructs an estimate of the original image:

.. math::

    \hat{\mathbf{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon_\theta(\mathbf{x}_t, t) \right)

Then, a new sample :math:`\mathbf{x}_{t-1}` is drawn:

.. math::

    \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}

with :math:`\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})` and :math:`\sigma_t^2 = \text{posterior\_var}_t`.

Training Objective
------------------

DDPM is trained using a noise prediction loss:

.. math::

    \mathcal{L}_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \right\|^2 \right]

Implemented via the :py:meth:`DDPM.get_loss` method.

Methods
-------

.. automethod:: lucid.models.DDPM.get_loss
.. automethod:: lucid.models.DDPM.sample

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn
    from lucid.models import DDPM

    model = DDPM(image_size=32, timesteps=1000)

    # Training
    x = lucid.rand(32, 3, 32, 32)
    loss = model(x)
    loss.backward()

    # Sampling
    with lucid.no_grad():
        samples = model.sample(batch_size=16)

.. tip::

    `model` can be any neural network predicting noise from `(x_t, t)`. 
    It must broadcast `t` into `(N, 1, 1, 1)` to match `x`'s spatial dims.

.. warning::

    Diffusion is sensitive to the beta schedule. Linear schedules like 
    :math:`\beta_t \in [1e-4, 0.02]` work well, but others (e.g. cosine) may improve 
    sample quality at fewer steps.
