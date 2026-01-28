SENet
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    se_resnet_18.rst
    se_resnet_34.rst
    se_resnet_50.rst
    se_resnet_101.rst
    se_resnet_152.rst

    se_resnext_50_32x4d.rst
    se_resnext_101_32x4d.rst
    se_resnext_101_32x8d.rst
    se_resnext_101_64x4d.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.SENet

The `SENet` class serves as a foundational implementation for creating Squeeze-and-Excitation 
Network architectures.

.. mermaid::
    :name: SENet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>se_resnet_101</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
          m3["BatchNorm2d"];
          m4["ReLU"];
        end
        m5["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
        subgraph sg_m6["layer1 x 4"]
          direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m6_in(["Input"]);
          m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m7["_Bottleneck"]
            direction TB;
          style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8(["ConvBNReLU2d x 2"]);
            m9["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
            m10["SEModule"];
            m11["ReLU"];
            m12["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
          end
          subgraph sg_m13["_Bottleneck x 2"]
            direction TB;
          style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m13_in(["Input"]);
            m13_out(["Output"]);
      style m13_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m13_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m14(["ConvBNReLU2d x 2<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,64,56,56)</span>"]);
            m15["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
            m16["SEModule"];
            m17["ReLU"];
          end
        end
        m18["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2048,7,7) → (1,2048,1,1)</span>"];
        m19["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m5 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m17 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m18 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m19 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m12;
      m11 -.-> m14;
      m12 --> m11;
      m13_in -.-> m14;
      m13_out -.-> m6_in;
      m14 --> m15;
      m15 --> m16;
      m16 --> m17;
      m17 --> m13_in;
      m17 --> m13_out;
      m17 --> m6_out;
      m18 --> m19;
      m19 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 --> m5;
      m5 -.-> m8;
      m6_in -.-> m8;
      m6_out --> m18;
      m6_out -.-> m6_in;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class SENet(
        block: Type[nn.Module], layers: List[int], num_classes: int = 1000, reduction: int = 16
    ):

Parameters
----------

- **block** (*Type[nn.Module]*):
  The block type to use, incorporating the squeeze-and-excitation mechanism.

- **layers** (*List[int]*):
  A list of integers specifying the number of blocks at each stage of the SENet.

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.

- **reduction** (*int*, optional):
  Reduction ratio for the squeeze-and-excitation mechanism. Default is 16.

Attributes
----------

- **conv** (*Sequential*):
  Initial convolutional block including a convolutional layer, batch normalization, 
  ReLU activation, and max pooling.

- **stage1, stage2, stage3, stage4** (*Sequential*):
  Stages of the SENet, each containing multiple blocks as specified in the `layers` parameter, 
  with squeeze-and-excitation operations.

- **avgpool** (*AdaptiveAvgPool2d*):
  Global average pooling layer before the fully connected layer.

- **fc** (*Linear*):
  Fully connected layer for classification.

Forward Calculation
--------------------

The forward method defines the computation flow:

.. code-block:: python

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
