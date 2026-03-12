VGGNet
======

.. toctree::
    :maxdepth: 1
    :hidden:

    VGGNetConfig.rst
    vggnet_11.rst
    vggnet_13.rst
    vggnet_16.rst
    vggnet_19.rst

|convnet-badge| 

.. autoclass:: lucid.models.VGGNet

The `VGGNet` module in `lucid.models` serves as a base class for creating
VGG network variants (e.g., VGG-11, VGG-13, VGG-16, VGG-19). 
It provides a flexible architecture defined by `VGGNetConfig`, which controls
the convolutional layout, input channels, classifier widths, and dropout rate.

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
        def __init__(self, config: VGGNetConfig)

Parameters
----------

- **config** (*VGGNetConfig*):
  A configuration object describing the convolutional layout, output class count,
  input channels, classifier hidden dimensions, and dropout probability.

Attributes
----------

- **config** (*VGGNetConfig*):
  The configuration used to build the model.

- **conv** (*nn.Sequential*):
  A sequential container of convolutional and pooling layers as defined by the configuration.

- **avgpool** (*nn.AdaptiveAvgPool2d*):
  Adaptive average pooling layer that reduces spatial dimensions to (7, 7).

- **fc** (*nn.Sequential*):
  The fully connected layers for classification, including dropout and ReLU activations.

Methods
-------

- **_make_layers(config: VGGNetConfig) -> nn.Sequential**:
  Converts the configuration list into a sequential container of layers.

- **forward(x: torch.Tensor) -> torch.Tensor**:
  Performs the forward pass of the network.

Examples
--------

**Defining a Custom VGG Configuration**

.. code-block:: python

    import lucid.models as models

    custom_config = models.VGGNetConfig(
        conv_config=[64, "M", 128, "M", 256, 256, "M"],
        num_classes=10,
        in_channels=1,
        classifier_hidden_features=(512, 256),
        dropout=0.25,
    )

    model = models.VGGNet(custom_config)

    input_tensor = torch.randn(1, 1, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Shape: (1, 10)

**Explanation**

The `custom_config` specifies the convolution and pooling schedule together with
the classifier head dimensions for a custom VGG-style model.
