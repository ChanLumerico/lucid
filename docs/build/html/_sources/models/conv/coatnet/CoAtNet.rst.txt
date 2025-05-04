CoAtNet
=======

.. toctree::
    :maxdepth: 1
    :hidden:

    coatnet_0.rst
    coatnet_1.rst
    coatnet_2.rst
    coatnet_3.rst
    coatnet_4.rst
    coatnet_5.rst
    coatnet_6.rst
    coatnet_7.rst

.. raw:: html

   <span
     style="
       display: inline-block; padding: 0.15em 0.6em;
       border-radius: 999px; border: 1px solid #ffa600;
       color: #ffa600; background-color: transparent;
       font-size: 0.72em; font-weight: 500;
     "
   >
     ConvNet
   </span>

   <span
     style="
       display: inline-block; padding: 0.15em 0.6em;
       border-radius: 999px; border: 1px solid #707070;
       color: #707070; background-color: transparent;
       font-size: 0.72em; font-weight: 500;
     "
   >
     Image Classification
   </span>

.. autoclass:: lucid.models.CoAtNet

The `CoAtNet` module in `lucid.nn` implements the CoAtNet architecture, 
a hybrid model combining convolutional and attention-based mechanisms. 
It leverages the strengths of both convolutional neural networks (CNNs) 
and vision transformers (ViTs), making it highly efficient for image 
classification tasks. 

CoAtNet utilizes depthwise convolutions, relative position encoding, 
and pre-normalization to enhance training stability and performance.

.. image:: coatnet.png
    :width: 600
    :alt: CoAtNet architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class CoAtNet(nn.Module):
        def __init__(
            img_size: tuple[int, int],
            in_channels: int,
            num_blocks: list[int],
            channels: list[int],
            num_classes: int = 1000,
            num_heads: int = 32,
            block_types: list[str] = ["C", "C", "T", "T"],
        )

Parameters
----------

- **img_size** (*tuple[int, int]*):
  The spatial resolution of the input image (height, width).

- **in_channels** (*int*):
  The number of input channels, typically 3 for RGB images.

- **num_blocks** (*list[int]*):
  Number of blocks in each stage, defining the depth of each phase.

- **channels** (*list[int]*):
  Number of channels in each stage of the network, controlling the model width.

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **num_heads** (*int*, optional):
  Number of attention heads in the transformer-based blocks. Default is 32.

- **block_types** (*list[str]*, optional):
  Defines whether each stage uses convolution (`C`) or transformer (`T`) blocks. 
  Default is `["C", "C", "T", "T"]`.

Hybrid Architecture
-------------------

The CoAtNet model employs a hybrid structure that fuses convolutional 
and transformer blocks for enhanced representation learning:

1. **Early Convolutional Blocks**:

   - The initial stages use convolution-based feature extraction (`C` blocks).
   - These layers focus on capturing local patterns efficiently.
   - Convolutions perform feature extraction using:
     
     .. math::

         Y = W * X + b

2. **Transformer-Based Blocks**:

   - Later stages transition into transformer blocks (`T` blocks).
   - These layers incorporate self-attention to capture long-range dependencies.
   - Self-attention is computed as:
     
     .. math::

         \mathbf{A} = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V

3. **Pre-Normalization**:

   - Each transformer block applies Layer Normalization before the attention mechanism.
   - Helps improve gradient flow and stability during training.
   - The normalization step follows:
     
     .. math::

         \hat{x} = \frac{x - \mu}{\sigma + \epsilon}

4. **Relative Position Encoding**:

   - Unlike absolute position encoding in traditional ViTs, 
     CoAtNet leverages relative position encoding.
   - The attention mechanism incorporates positional information dynamically:
     
     .. math::

         A_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}} + B_{ij}

   - The relative position bias matrix \( B_{ij} \) is learnable and helps in 
     modeling spatial relationships.

5. **Depthwise Convolutions**:

   - Used to reduce computational complexity while maintaining strong feature 
     extraction capabilities.
   - Reduces the number of parameters compared to traditional convolutional layers.
   - The depthwise convolution operation is:
     
     .. math::
         Y_{i,j} = \sum_{k} X_{i+k, j+k} W_k

6. **Scaling Strategy**:

   - CoAtNet scales efficiently across depth (D), width (W), and resolution (R), 
     making it highly versatile for various image sizes and computational constraints.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create CoAtNet with default settings
    model = models.CoAtNet(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=[2, 2, 6, 2],
        channels=[64, 128, 256, 512],
        num_classes=1000,
        num_heads=32,
        block_types=["C", "C", "T", "T"]
    )

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)
    print(output.shape)  # Shape: (1, 1000)
