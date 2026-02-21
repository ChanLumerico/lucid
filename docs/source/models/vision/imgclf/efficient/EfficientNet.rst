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

|convnet-badge| 

.. autoclass:: lucid.models.EfficientNet

The `EfficientNet` class implements a scalable and efficient convolutional 
neural network architecture that can be configured to encompass all EfficientNet-B0 
to B7 variants.

.. mermaid::
    :name: EfficientNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>efficientnet_b0</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m1["Upsample"];
        subgraph sg_m2["stage1"]
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,112,112)</span>"];
          m4["BatchNorm2d"];
        end
        subgraph sg_m5["stage2 x 7"]
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m5_in(["Input"]);
          m5_out(["Output"]);
      style m5_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m5_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          m6["_MBConv<br/><span style='font-size:11px;font-weight:400'>(1,32,112,112) → (1,16,112,112)</span>"];
        end
        subgraph sg_m7["stage9"]
        style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m8["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,320,7,7) → (1,1280,7,7)</span>"];
          m9["BatchNorm2d"];
          m10["Swish"];
        end
        m11["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1280,7,7) → (1,1280,1,1)</span>"];
        m12["Dropout"];
        m13["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1280) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m1 fill:#fdf2f8,stroke:#b83280,stroke-width:1px;
      style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m8 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m9 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m10 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m11 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m12 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m13 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m1;
      m1 --> m3;
      m10 --> m11;
      m11 --> m12;
      m12 --> m13;
      m13 --> output;
      m3 --> m4;
      m4 -.-> m6;
      m5_in -.-> m6;
      m5_out -.-> m5_in;
      m5_out --> m8;
      m6 -.-> m5_in;
      m6 --> m5_out;
      m8 --> m9;
      m9 --> m10;

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
