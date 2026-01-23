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

.. image:: ncsn.png
    :width: 600
    :alt: NCSN architecture
    :align: center

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

