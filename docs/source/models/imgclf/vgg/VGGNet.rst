VGGNet
======

.. toctree::
    :maxdepth: 1
    :hidden:

    vggnet_11.rst
    vggnet_13.rst
    vggnet_16.rst
    vggnet_19.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.VGGNet

The `VGGNet` module in `lucid.nn` serves as a base class for creating 
VGG network variants (e.g., VGG-11, VGG-13, VGG-16, VGG-19). 
It provides a flexible architecture defined by a configurable list of convolutional 
and pooling layers.

.. mermaid::
    :name: VGGNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>vggnet_11</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["conv"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,224,224)</span>"];
          m3["ReLU"];
          m4["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,224,224) → (1,64,112,112)</span>"];
          m5["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,112,112) → (1,128,112,112)</span>"];
          m6["ReLU"];
          m7["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,128,112,112) → (1,128,56,56)</span>"];
          m8["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,56,56) → (1,256,56,56)</span>"];
          m9["ReLU"];
          m10["Conv2d"];
          m11["ReLU"];
          m12["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,256,56,56) → (1,256,28,28)</span>"];
          m13["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,256,28,28) → (1,512,28,28)</span>"];
          m14["ReLU"];
          m15["Conv2d"];
          m16["ReLU"];
          m17["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,512,28,28) → (1,512,14,14)</span>"];
          m18["Conv2d"];
          m19["ReLU"];
          m20["Conv2d"];
          m21["ReLU"];
          m22["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,512,14,14) → (1,512,7,7)</span>"];
        end
        m23["AdaptiveAvgPool2d"];
        subgraph sg_m24["fc"]
        style sg_m24 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m25["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,25088) → (1,4096)</span>"];
          m26["ReLU"];
          m27["Dropout"];
          m28["Linear"];
          m29["ReLU"];
          m30["Dropout"];
          m31["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,4096) → (1,1000)</span>"];
        end
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m4 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m5 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m6 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m7 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m8 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m9 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m10 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m12 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m13 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m14 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m15 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m16 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m17 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m18 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m19 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m20 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m21 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m22 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m23 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m25 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m26 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m27 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m28 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m29 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m30 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m31 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m11;
      m11 --> m12;
      m12 --> m13;
      m13 --> m14;
      m14 --> m15;
      m15 --> m16;
      m16 --> m17;
      m17 --> m18;
      m18 --> m19;
      m19 --> m20;
      m2 --> m3;
      m20 --> m21;
      m21 --> m22;
      m22 --> m23;
      m23 --> m25;
      m25 --> m26;
      m26 --> m27;
      m27 --> m28;
      m28 --> m29;
      m29 --> m30;
      m3 --> m4;
      m30 --> m31;
      m31 --> output;
      m4 --> m5;
      m5 --> m6;
      m6 --> m7;
      m7 --> m8;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class VGGNet(nn.Module):
        def __init__(self, config: List[Union[int, str]], num_classes: int = 1000)

Parameters
----------

- **config** (*List[Union[int, str]]*):
  A list defining the architecture of the network. Each integer specifies the number 
  of channels in a convolutional layer, and 'M' indicates a max-pooling layer.

- **num_classes** (*int*, optional):
  The number of output classes for the classifier. Default is 1000.

Attributes
----------

- **features** (*nn.Sequential*):
  A sequential container of convolutional and pooling layers as defined by the configuration.

- **avgpool** (*nn.AdaptiveAvgPool2d*):
  Adaptive average pooling layer that reduces spatial dimensions to (7, 7).

- **classifier** (*nn.Sequential*):
  The fully connected layers for classification, including dropout and ReLU activations.

Methods
-------

- **_make_layers(config: List[Union[int, str]]) -> nn.Sequential**:
  Converts the configuration list into a sequential container of layers.

- **forward(x: torch.Tensor) -> torch.Tensor**:
  Performs the forward pass of the network.

Examples
--------

**Defining a Custom VGG Configuration**

.. code-block:: python

    import lucid.nn as nn

    # Custom configuration
    custom_config = [64, 'M', 128, 'M', 256, 256, 'M']

    # Create a VGGNet with the custom configuration
    model = nn.VGGNet(config=custom_config, num_classes=10)

    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Shape: (1, 10)

**Explanation**

The `custom_config` specifies two convolutional layers with 64 and 128 channels, 
respectively, followed by max-pooling layers, and two consecutive convolutional 
layers with 256 channels.
