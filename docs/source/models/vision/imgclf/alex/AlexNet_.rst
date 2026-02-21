AlexNet
=======

.. toctree::
    :maxdepth: 1
    :hidden:

    alexnet.rst

|convnet-badge|

.. autoclass:: lucid.models.AlexNet

The `AlexNet` module in `lucid.nn` implements the AlexNet architecture, 
a convolutional neural network designed for image classification tasks. 
It consists of multiple convolutional and fully connected layers with ReLU 
activations and dropout for regularization.

.. mermaid::
    :name: AlexNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50},"themeCSS":".nodeLabel, .edgeLabel, .cluster text, .node text { fill: #000000 !important; } .node foreignObject *, .cluster foreignObject * { color: #000000 !important; }"} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>alexnet</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["conv"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,3,224,224) → (1,64,55,55)</span>"];
          m3["ReLU"];
          m4["MaxPool2d<br/><span style='font-size:11px;font-weight:400'>(1,64,55,55) → (1,64,27,27)</span>"];
          m5["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,64,27,27) → (1,192,27,27)</span>"];
          m6["ReLU"];
          m7["MaxPool2d<br/><span style='font-size:11px;font-weight:400'>(1,192,27,27) → (1,192,13,13)</span>"];
          m8["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,192,13,13) → (1,384,13,13)</span>"];
          m9["ReLU"];
          m10["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,384,13,13) → (1,256,13,13)</span>"];
          m11["ReLU"];
          m12["Conv2d"];
          m13["ReLU"];
          m14["MaxPool2d<br/><span style='font-size:11px;font-weight:400'>(1,256,13,13) → (1,256,6,6)</span>"];
        end
        m15["AdaptiveAvgPool2d"];
        subgraph sg_m16["fc"]
        style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m17["Dropout"];
          m18["Linear<br/><span style='font-size:11px;font-weight:400'>(1,9216) → (1,4096)</span>"];
          m19["ReLU"];
          m20["Dropout"];
          m21["Linear"];
          m22["ReLU"];
          m23["Linear<br/><span style='font-size:11px;font-weight:400'>(1,4096) → (1,1000)</span>"];
        end
      end
      input["Input<br/><span style='font-size:11px;color:#000000;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#000000;font-weight:400'>(1,1000)</span>"];
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
      style m12 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m13 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m14 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m15 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m17 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m19 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m20 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m21 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m22 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m23 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m11;
      m11 --> m12;
      m12 --> m13;
      m13 --> m14;
      m14 --> m15;
      m15 --> m17;
      m17 --> m18;
      m18 --> m19;
      m19 --> m20;
      m2 --> m3;
      m20 --> m21;
      m21 --> m22;
      m22 --> m23;
      m23 --> output;
      m3 --> m4;
      m4 --> m5;
      m5 --> m6;
      m6 --> m7;
      m7 --> m8;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class AlexNet(nn.Module):
        def __init__(self, num_classes: int = 1000)

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

Attributes
----------

- **features** (*nn.Sequential*):
  The convolutional layers, including pooling and ReLU activations.

- **avgpool** (*nn.AdaptiveAvgPool2d*):
  Adaptive average pooling layer that reduces the spatial dimensions to (6, 6).

- **classifier** (*nn.Sequential*):
  The fully connected layers with dropout and ReLU activations for classification.

Architecture
------------

The architecture of AlexNet is as follows:

1. **Convolutional Layers**:
   - 5 convolutional layers with ReLU activations.
   - MaxPooling after the 1st, 2nd, and 5th convolutional layers.

2. **Fully Connected Layers**:
   - 2 hidden fully connected layers, each with 4096 units and ReLU activations.
   - Output layer with `num_classes` units for classification.

3. **Regularization**:
   - Dropout is applied to fully connected layers to reduce overfitting.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.nn as nn

    # Create AlexNet with default 1000 classes
    model = nn.AlexNet(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = Tensor.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)

**Explanation**

The model processes the input through its convolutional and fully connected layers, 
producing logits for 1000 classes.

**Custom Number of Classes**

.. code-block:: python

    # Create AlexNet with custom 10 classes
    model = nn.AlexNet(num_classes=10)

    input_ = Tensor.randn(1, 3, 224, 224)

    output = model(input_)
    print(output.shape)  # Shape: (1, 10)
