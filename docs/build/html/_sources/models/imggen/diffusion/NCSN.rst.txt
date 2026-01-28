NCSN
====
|diffusion-badge| |score-diffusion-badge| |imggen-badge|

.. autoclass:: lucid.models.imggen.NCSN

The `NCSN` class implements a Noise Conditional Score Network (NCSN), a score-based
generative model trained with annealed denoising score matching.

Given a noise scale :math:`\sigma`, the model learns the score
:math:`\nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x})` and generates samples using
annealed Langevin dynamics over a descending noise schedule.

**Total Parameters**: 12,471,555 (Default)

.. mermaid::
    :name: NCSN

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>NCSN</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m1["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,64,64) → (1,128,64,64)</span>"];
        subgraph sg_m2["_RCUBlock x 4"]
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2_in(["Input"]);
          m2_out(["Output"]);
      style m2_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m2_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m3["units"]
            direction TB;
          style sg_m3 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m4["_ResidualConvUnit x 2"]
              direction TB;
            style sg_m4 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m4_in(["Input"]);
              m4_out(["Output"]);
      style m4_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m4_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
              m5["_CondInstanceNorm<br/><span style='font-size:11px;font-weight:400'>(1,128,64,64)x2 → (1,128,64,64)</span>"];
              m6["_Conv3x3"];
              m7["_CondInstanceNorm<br/><span style='font-size:11px;font-weight:400'>(1,128,64,64)x2 → (1,128,64,64)</span>"];
              m8["_Conv3x3"];
              m9["ELU"];
            end
          end
        end
        subgraph sg_m10["_RefineBlock x 4"]
        style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m10_in(["Input"]);
          m10_out(["Output"]);
      style m10_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m10_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m11["adapters"]
            direction TB;
          style sg_m11 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m12["_CondAdapter"]
              direction TB;
            style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m13["ELU"];
            end
          end
          subgraph sg_m14["rcu_in"]
            direction TB;
          style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m15["_RCUBlock"]
              direction TB;
            style sg_m15 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m16["ModuleList<br/><span style='font-size:11px;font-weight:400'>(1,128,64,64)x2 → (1,128,64,64)</span>"];
            end
          end
          subgraph sg_m17["_MultiResFusion"]
            direction TB;
          style sg_m17 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m18["norms"]
              direction TB;
            style sg_m18 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m19["_CondInstanceNorm"];
            end
            subgraph sg_m20["convs"]
              direction TB;
            style sg_m20 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m21["Conv2d"];
            end
            m22["ELU"];
          end
          subgraph sg_m23["_ChainedResPooling"]
            direction TB;
          style sg_m23 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m24["norms"]
              direction TB;
            style sg_m24 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m25(["_CondInstanceNorm x 4<br/><span style='font-size:11px;font-weight:400'>(1,128,64,64)x2 → (1,128,64,64)</span>"]);
            end
            subgraph sg_m26["convs"]
              direction TB;
            style sg_m26 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m27(["Conv2d x 4"]);
            end
            m28["ELU"];
          end
          subgraph sg_m29["_RCUBlock"]
            direction TB;
          style sg_m29 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m30["units"]
              direction TB;
            style sg_m30 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m31(["_ResidualConvUnit x 2<br/><span style='font-size:11px;font-weight:400'>(1,128,64,64)x2 → (1,128,64,64)</span>"]);
            end
          end
        end
        subgraph sg_m32["_CondInstanceNorm"]
        style sg_m32 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m33["InstanceNorm2d"];
          m34["Embedding<br/><span style='font-size:11px;color:#475569;font-weight:400'>(1) → (1,256)</span>"];
        end
        m35["ELU"];
        m36["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,64,64) → (1,3,64,64)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,64,64)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,64,64)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m1 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m9 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m13 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m21 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m22 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m27 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m28 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m33 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m34 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
      style m35 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m36 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      input --> m1;
      m1 -.-> m5;
      m10_in --> m13;
      m10_out -.-> m10_in;
      m10_out --> m33;
      m16 -.-> m25;
      m25 --> m28;
      m27 -.-> m25;
      m27 --> m31;
      m28 --> m27;
      m2_in -.-> m5;
      m2_out --> m16;
      m2_out -.-> m2_in;
      m31 -.-> m10_in;
      m31 --> m10_out;
      m33 --> m34;
      m34 --> m35;
      m35 --> m36;
      m36 --> output;
      m4_in -.-> m5;
      m4_out -.-> m2_in;
      m5 -.-> m9;
      m6 --> m7;
      m7 -.-> m9;
      m8 --> m4_in;
      m9 --> m2_out;
      m9 --> m4_out;
      m9 --> m6;
      m9 --> m8;

Class Signature
---------------

.. code-block:: python

    class NCSN(
        in_channels: int = 3,
        nf: int = 128,
        num_classes: int = 10,
        dilations: Sequence[int] = (1, 2, 4, 8),
        scale_by_sigma: bool = True,
    )

Parameters
----------

- **in_channels** (*int*):
  Number of channels in the input image.

- **nf** (*int*):
  Base feature width used by the internal RefineNet-style backbone.

- **num_classes** (*int*):
  Number of noise levels (i.e., the length of the noise schedule). This is also the
  number of class labels used for conditional normalization inside the network.

- **dilations** (*Sequence[int]*):
  Four dilation values used in the four RCU stages.

- **scale_by_sigma** (*bool*):
  If `True`, the network output is divided by the per-sample noise level
  :math:`\sigma` (matching the original NCSN formulation).

Returns
-------

Use :py:meth:`NCSN.get_loss` to compute the annealed DSM loss and sampled noise labels:

.. code-block:: python

    loss, labels = ncsn.get_loss(x)

- **loss** (*Tensor*):
  Scalar training loss.
- **labels** (*Tensor*):
  Integer tensor of shape `(N,)` indicating the sampled noise level index per batch item.

Training Objective (Annealed DSM)
---------------------------------

Let :math:`\epsilon \sim \mathcal{N}(0, I)` and :math:`\mathbf{x}_\sigma = \mathbf{x} + \sigma \epsilon`.
With the score network :math:`s_\theta(\mathbf{x}_\sigma, \sigma)`, the annealed denoising score matching loss is:

.. math::

    \mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}, \sigma, \epsilon}\left[\left\| \sigma\, s_\theta(\mathbf{x}_\sigma, \sigma) + \epsilon \right\|^2\right]

Implemented via :py:meth:`NCSN.get_loss`.

Sampling (Annealed Langevin Dynamics)
-------------------------------------

Sampling iterates over noise levels :math:`\sigma_1 > \sigma_2 > \cdots > \sigma_K` and performs Langevin updates:

.. math::

    \mathbf{x} \leftarrow \mathbf{x} + \eta\, s_\theta(\mathbf{x}, \sigma_i) + \sqrt{2\eta}\,\mathbf{z}, \quad \mathbf{z}\sim\mathcal{N}(0, I)

In this implementation, the per-level step size is scaled as:

.. math::

    \eta_i = \text{step\_lr}\left(\frac{\sigma_i}{\sigma_K}\right)^2

Implemented via :py:meth:`NCSN.sample`.

Methods
-------

.. automethod:: lucid.models.imggen.NCSN.make_sigmas
.. automethod:: lucid.models.imggen.NCSN.set_sigmas
.. automethod:: lucid.models.imggen.NCSN.get_loss
.. automethod:: lucid.models.imggen.NCSN.sample

Examples
--------

.. code-block:: python

    import lucid
    from lucid.models.imggen import NCSN

    model = NCSN(in_channels=3, nf=128, num_classes=10)
    model.set_sigmas(NCSN.make_sigmas(sigma_begin=50.0, sigma_end=0.01, num_scales=10))

    # Training
    x = lucid.random.randn((32, 3, 32, 32))
    loss, labels = model.get_loss(x)
    loss.backward()

    # Sampling
    with lucid.no_grad():
        samples = model.sample(
            n_samples=16,
            image_size=32,
            in_channels=3,
            n_steps_each=100,
            step_lr=2e-5,
            clip=True,
        )

.. tip::

    `set_sigmas(...)` must be called before training or sampling, otherwise the model
    does not know which noise levels to use.

