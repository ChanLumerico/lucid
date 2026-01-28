InceptionResNet
===============

.. toctree::
    :maxdepth: 1
    :hidden:

    inception_resnet_v1.rst
    inception_resnet_v2.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.InceptionResNet

Overview
--------

The `InceptionResNet` base class defines a flexible architecture that combines
Inception-style modules with residual connections. This approach improves optimization
and gradient flow in deep neural networks, making it suitable for a variety of image
classification tasks.

This class serves as a foundation for specific versions like Inception-ResNet v1 and v2
by providing essential components such as a stem network, convolutional layers, and
fully connected layers.

.. mermaid::
    :name: Inception-ResNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>inception_resnet_v2</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_InceptionStem_V4"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,3,224,224) → (1,64,109,109)</span>"];
          m3["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,109,109) → (1,64,54,54)</span>"];
          m4["ConvBNReLU2d<br/><span style='font-size:11px;font-weight:400'>(1,64,109,109) → (1,96,54,54)</span>"];
          m5(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,160,54,54) → (1,96,52,52)</span>"]);
          m6["ConvBNReLU2d<br/><span style='font-size:11px;font-weight:400'>(1,192,52,52) → (1,192,25,25)</span>"];
          m7["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,192,52,52) → (1,192,25,25)</span>"];
        end
        subgraph sg_m8["conv"]
        style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m9(["_InceptionResModule_A x 5"]);
          m10["_InceptionReduce_V4A<br/><span style='font-size:11px;font-weight:400'>(1,384,25,25) → (1,1152,12,12)</span>"];
          m11(["_InceptionResModule_B x 10"]);
          m12["_InceptionResReduce<br/><span style='font-size:11px;font-weight:400'>(1,1152,12,12) → (1,2144,5,5)</span>"];
          m13(["_InceptionResModule_C x 5"]);
        end
        m14["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2144,5,5) → (1,2144,1,1)</span>"];
        m15["Dropout"];
        m16["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2144) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m3 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m7 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m14 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m15 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m16 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m11;
      m11 --> m12;
      m12 --> m13;
      m13 --> m14;
      m14 --> m15;
      m15 --> m16;
      m16 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 --> m5;
      m5 --> m7;
      m6 --> m9;
      m7 --> m6;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

   class InceptionResNet(nn.Module):
       def __init__(self, num_classes: int) -> None

Parameters
----------

- **num_classes** (*int*):
  The number of output classes for the final classification layer.

Attributes
----------

- **stem** (*nn.Module*):
  The initial stem module that extracts low-level features from the input.

- **conv** (*nn.Sequential*):
  A sequential container for the main convolutional and residual blocks.

- **fc** (*nn.Sequential*):
  A sequential container for the fully connected layers that perform classification.

Methods
-------

- **forward(x: Tensor) -> Tensor**
  Performs the forward pass through the stem, convolutional blocks, 
  and fully connected layers.

  .. code-block:: python

      def forward(self, x):
          x = self.stem(x)
          x = self.conv(x)
          x = x.view(x.shape[0], -1)  # Flatten
          x = self.fc(x)
          return x
