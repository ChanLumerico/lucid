Fast R-CNN
==========
|convnet-badge| |two-stage-det-badge| |objdet-badge|

.. autoclass:: lucid.models.FastRCNN

`FastRCNN` implements the Fast Region-based Convolutional Neural Network 
architecture for object detection, building upon the R-CNN approach by 
introducing a more efficient detection pipeline. It replaces per-region 
feature extraction with RoI pooling and integrates classification and 
bounding box regression into a single forward pass.

.. mermaid::
    :name: FastRCNN

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>FastRCNN</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["backbone"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m2["Sequential"]
            direction TB;
          style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,64,64) → (1,32,32,32)</span>"];
            m4["BatchNorm2d"];
            m5["ReLU"];
            m6["Conv2d"];
            m7["BatchNorm2d"];
            m8["ReLU"];
            m9["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,32,32,32) → (1,64,32,32)</span>"];
          end
          m10["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,32,32) → (1,64,16,16)</span>"];
          subgraph sg_m11["Sequential"]
            direction TB;
          style sg_m11 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m12["_ResNeStBottleneck"]
              direction TB;
            style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m13["ConvBNReLU2d"];
              m14["_SplitAttention"];
              m15["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,16,16) → (1,256,16,16)</span>"];
              m16["BatchNorm2d"];
              m17["ReLU"];
              m18["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,16,16) → (1,256,16,16)</span>"];
            end
          end
          subgraph sg_m19["Sequential x 2"]
            direction TB;
          style sg_m19 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m19_in(["Input"]);
            m19_out(["Output"]);
      style m19_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m19_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            subgraph sg_m20["_ResNeStBottleneck"]
              direction TB;
            style sg_m20 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m21["ConvBNReLU2d<br/><span style='font-size:11px;font-weight:400'>(1,256,16,16) → (1,128,16,16)</span>"];
              m22["_SplitAttention"];
              m23["AvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,128,16,16) → (1,128,8,8)</span>"];
              m24["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,8,8) → (1,512,8,8)</span>"];
              m25["BatchNorm2d"];
              m26["ReLU"];
              m27["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,256,16,16) → (1,512,8,8)</span>"];
            end
          end
        end
        m28["ROIAlign<br/><span style='font-size:11px;font-weight:400'>(1,1024,4,4)x3 → (1,1024,7,7)</span>"];
        m29["SelectiveSearch<br/><span style='font-size:11px;font-weight:400'>(3,64,64) → (1,4)</span>"];
        m30(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,50176) → (1,4096)</span>"]);
        m31(["Dropout x 2"]);
        m32(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,4096) → (1,100)</span>"]);
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,64,64)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,100)x2</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m5 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m6 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m7 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m8 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m9 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m10 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m15 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m16 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m17 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m23 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m24 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m25 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m26 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m30 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m31 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m32 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m29;
      m10 --> m13;
      m13 --> m14;
      m14 --> m15;
      m15 --> m16;
      m16 --> m18;
      m17 -.-> m21;
      m18 --> m17;
      m19_in -.-> m21;
      m19_out --> m28;
      m21 --> m22;
      m22 --> m23;
      m23 --> m24;
      m24 --> m25;
      m25 --> m27;
      m26 --> m19_in;
      m27 --> m19_out;
      m27 --> m26;
      m28 -.-> m30;
      m29 --> m3;
      m3 --> m4;
      m30 --> m31;
      m31 -.-> m30;
      m31 --> m32;
      m32 --> output;
      m4 --> m5;
      m5 --> m6;
      m6 --> m7;
      m7 --> m8;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class FastRCNN(nn.Module):
        def __init__(
            self,
            backbone: nn.Module,
            feat_channels: int,
            num_classes: int,
            pool_size: tuple[int, int] = (7, 7),
            hidden_dim: int = 4096,
            bbox_reg_means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
            bbox_reg_stds: tuple[float, ...] = (0.1, 0.1, 0.2, 0.2),
            dropout: float = 0.5,
            proposal_generator: Callable[..., Tensor] | None = None,
        )

Parameters
----------

- **backbone** (*nn.Module*):  
  Convolutional feature extractor applied over the entire image once.

- **feat_channels** (*int*):  
  Number of output channels from the backbone feature map.

- **num_classes** (*int*):  
  Number of object categories (excluding background).

- **pool_size** (*tuple[int, int]*, optional):  
  Output size of the spatial pooling operation (typically RoIAlign or RoIPool). 
  Default is `(7, 7)`.

- **hidden_dim** (*int*, optional):  
  Number of hidden units in the fully connected layers after pooling. Default is `4096`.

- **bbox_reg_means** (*tuple[float, ...]*, optional):  
  Normalization means for bounding box regression targets. Default is `(0.0, 0.0, 0.0, 0.0)`.

- **bbox_reg_stds** (*tuple[float, ...]*, optional):  
  Normalization stds for bounding box regression targets. Default is `(0.1, 0.1, 0.2, 0.2)`.

- **dropout** (*float*, optional):  
  Dropout probability used in the fully connected layers. Default is `0.5`.

- **proposal_generator** (*Callable[..., Tensor] | None*, optional):  
  Custom region proposal function. If `None`, uses precomputed proposals.

Architecture
------------

Fast R-CNN improves over the original R-CNN by computing the CNN feature map once 
per image and classifying object proposals directly on this shared map:

1. **Full-Image Feature Map**:
   
   - The input image is passed through the `backbone` to extract a dense feature map.

2. **Region of Interest (RoI) Pooling**:
   
   - Region proposals are projected onto the feature map and cropped to a fixed size 
     using RoI pooling (size defined by `pool_size`).

3. **Two-Stream Head**:
   
   - Each pooled region is passed through a set of fully connected layers.
   - One stream performs classification over `num_classes`.
   - The other stream regresses bounding box adjustments per class.

4. **Bounding Box Normalization**:
   
   - Regression outputs are scaled using `bbox_reg_means` and `bbox_reg_stds`.

Examples
--------

**Basic Usage**

.. code-block:: python

    import lucid.nn as nn
    import lucid.random
    import lucid

    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

        def forward(self, x):
            return self.net(x)

    # Instantiate model
    backbone = ToyBackbone()
    model = nn.FastRCNN(backbone, feat_channels=64, num_classes=4)

    # Dummy input
    image = lucid.random.randn(1, 3, 512, 512)
    output = model.predict(image)
    print(output["boxes"].shape, output["scores"].shape, output["labels"].shape)

**Explanation**

Fast R-CNN accelerates inference by removing redundant computation. 
A single backbone pass generates features, which are then reused for each region proposal. 
RoI pooling ensures a fixed-size input for classification and regression heads.

**Custom Configuration**

.. code-block:: python

    class MiniBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

        def forward(self, x):
            return self.conv(x)

    backbone = MiniBackbone()
    model = nn.FastRCNN(
        backbone=backbone,
        feat_channels=32,
        num_classes=3,
        pool_size=(5, 5),
        hidden_dim=1024,
        dropout=0.3,
    )

    image = lucid.random.randn(1, 3, 256, 256)
    output = model.predict(image)
    print(output["boxes"].shape, output["scores"].shape, output["labels"].shape)
