InceptionNeXt
=============

.. toctree::
    :maxdepth: 1
    :hidden:

    inception_next_atto.rst
    inception_next_tiny.rst
    inception_next_small.rst
    inception_next_base.rst

.. autoclass:: lucid.models.InceptionNeXt

The `InceptionNeXt` class implements a modernized version of the Inception architecture,
leveraging depthwise convolutions, multi-layer perceptrons (MLPs), and advanced
token mixing techniques for enhanced performance and efficiency.

.. image:: inception_next.png
    :width: 600
    :alt: InceptionNeXt architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class InceptionNeXt(nn.Module):
        def __init__(
            self,
            num_classes: int = 1000,
            depths: list[int] = [3, 3, 9, 3],
            dims: list[int] = [96, 192, 384, 768],
            token_mixers: Type[nn.Module] = nn.Identity,
            mlp_ratios: list[int] = [4, 4, 4, 3],
            head_fn: Type[nn.Module] = _MLPHead,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            ls_init_value: float = 1e-6,
        ) -> None

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final classification layer. Defaults to 1000.

- **depths** (*list[int]*, optional):
  Number of blocks at each stage of the model. Defaults to [3, 3, 9, 3].

- **dims** (*list[int]*, optional):
  Dimensionality of feature maps at different stages. Defaults to [96, 192, 384, 768].

- **token_mixers** (*Type[nn.Module]*, optional):
  Type of token mixer module applied in each block. Defaults to `nn.Identity`.

- **mlp_ratios** (*list[int]*, optional):
  Expansion ratios for the MLP layers at different stages. Defaults to [4, 4, 4, 3].

- **head_fn** (*Type[nn.Module]*, optional):
  Function to construct the classification head. Defaults to `_MLPHead`.

- **drop_rate** (*float*, optional):
  Dropout rate applied to MLP layers. Defaults to 0.0.

- **drop_path_rate** (*float*, optional):
  Stochastic depth drop path rate. Defaults to 0.0.

- **ls_init_value** (*float*, optional):
  Initial value for layer scale. Defaults to 1e-6.

Examples
--------

.. code-block:: python

    from lucid.models import InceptionNeXt

    # Instantiate InceptionNeXt-Tiny
    model_tiny = InceptionNeXt(num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])

    # Forward pass with a random input
    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Batch size of 1, ImageNet resolution
    output = model_tiny(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
