EfficientNet
============

.. toctree::
    :maxdepth: 1
    :hidden:

    efficientnet_b0.rst
    efficientnet_b1.rst
    efficientnet_b2.rst
    efficientnet_b3.rst
    efficientnet_b4.rst
    efficientnet_b5.rst
    efficientnet_b6.rst
    efficientnet_b7.rst

.. autoclass:: lucid.models.EfficientNet

The `EfficientNet` class implements a scalable and efficient convolutional 
neural network architecture that can be configured to encompass all EfficientNet-B0 
to B7 variants.

.. image:: efficientnet.png
    :width: 600
    :alt: EfficientNet architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class EfficientNet(nn.Module):
        def __init__(
            self,
            num_classes: int = 1000,
            width_coef: float = 1.0,
            depth_coef: float = 1.0,
            scale: float = 1.0,
            dropout: float = 0.2,
            se_scale: int = 4,
            stochastic_depth: bool = False,
            p: float = 0.5,
        ) -> None

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final classification layer. 
  Defaults to 1000 (e.g., for ImageNet).

- **width_coef** (*float*, optional):
  Coefficient to scale the width (number of channels) of the network. Defaults to 1.0.

- **depth_coef** (*float*, optional):
  Coefficient to scale the depth (number of layers) of the network. Defaults to 1.0.

- **scale** (*float*, optional):
  Global scaling factor applied to the input resolution. Defaults to 1.0.

- **dropout** (*float*, optional):
  Dropout rate applied to the final fully connected layer. Defaults to 0.2.

- **se_scale** (*int*, optional):
  Reduction ratio for the squeeze-and-excitation (SE) block. Defaults to 4.

- **stochastic_depth** (*bool*, optional):
  Whether to use stochastic depth regularization. Defaults to False.

- **p** (*float*, optional):
  Probability for stochastic depth when enabled. Defaults to 0.5.

Configurations
--------------

The following table summarizes the configurations for EfficientNet variants B0 to B7:

.. list-table:: EfficientNet Configurations
   :header-rows: 1

   * - Variant
     - Width Coefficient
     - Depth Coefficient
     - Input Resolution
     - Dropout Rate
   
   * - B0
     - 1.0
     - 1.0
     - 224x224
     - 0.2
   
   * - B1
     - 1.0
     - 1.1
     - 240x240
     - 0.2
   
   * - B2
     - 1.1
     - 1.2
     - 260x260
     - 0.3
   
   * - B3
     - 1.2
     - 1.4
     - 300x300
     - 0.3
   
   * - B4
     - 1.4
     - 1.8
     - 380x380
     - 0.4
   
   * - B5
     - 1.6
     - 2.2
     - 456x456
     - 0.4
   
   * - B6
     - 1.8
     - 2.6
     - 528x528
     - 0.5
   
   * - B7
     - 2.0
     - 3.1
     - 600x600
     - 0.5

Examples
--------

.. code-block:: python

    from lucid.models import EfficientNet

    # Instantiate EfficientNet-B0
    model_b0 = EfficientNet(num_classes=1000, width_coef=1.0, depth_coef=1.0, scale=1.0)

    # Forward pass with a random input
    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Batch size of 1, ImageNet resolution
    output = model_b0(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)

    # Instantiate EfficientNet-B7
    model_b7 = EfficientNet(num_classes=1000, width_coef=2.0, depth_coef=3.1, scale=2.0)

    # Forward pass
    input_tensor = lucid.random.randn(1, 3, 600, 600)  # Larger resolution for B7
    output = model_b7(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
