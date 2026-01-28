Faster R-CNN
============
|convnet-badge| |two-stage-det-badge| |objdet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    faster_rcnn_resnet_50_fpn.rst
    faster_rcnn_resnet_101_fpn.rst

.. autoclass:: lucid.models.FasterRCNN

`FasterRCNN` implements the Faster Region-based Convolutional Neural Network,
an improvement over Fast R-CNN that introduces a learnable Region Proposal Network (RPN).
This architecture eliminates the need for external proposal methods by jointly learning
region proposals and object classification in a unified network.

.. mermaid::
    :name: FasterRCNN

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>faster_rcnn_resnet_50_fpn</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_ResNetFPNBackbone"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m2["resnet_50"]
            direction TB;
          style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
            m12["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
            m5(["Sequential x 4<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"]);
            m6["AdaptiveAvgPool2d"];
            m7["Linear"];
          end
          subgraph sg_m8["stem"]
            direction TB;
          style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m9["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
            m10["BatchNorm2d"];
            m11["ReLU"];
          end
          m12["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
          subgraph sg_m13["FPN"]
            direction TB;
          style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m14(["ModuleList x 2<br/><span style='font-size:11px;font-weight:400'>(1,2048,7,7) → (1,256,7,7)</span>"]);
          end
        end
        m19["_AnchorGenerator"];
        subgraph sg_m16["_RegionProposalNetwork"]
          direction TB;
        style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m17["_RPNHead"]
            direction TB;
          style sg_m17 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m18(["Conv2d x 3"]);
          end
          m19["_AnchorGenerator"];
        end
        subgraph sg_m20["MultiScaleROIAlign"]
          direction TB;
        style sg_m20 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m21["ROIAlign<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56)x3 → (44,256,7,7)</span>"];
        end
        m22(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(44,12544) → (44,1024)</span>"]);
        m23(["Dropout x 2"]);
        m24(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(44,1024) → (44,21)</span>"]);
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(44,21)x2</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m12 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m6 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m7 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m9 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m10 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m12 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m18 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m22 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m23 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m24 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m9;
      m10 --> m11;
      m11 --> m12;
      m12 --> m5;
      m14 --> m18;
      m18 --> m21;
      m21 -.-> m22;
      m22 --> m23;
      m23 -.-> m22;
      m23 --> m24;
      m24 --> output;
      m5 --> m14;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class FasterRCNN(nn.Module):
        def __init__(
            self,
            backbone: nn.Module,
            feat_channels: int,
            num_classes: int,
            *,
            use_fpn: bool = False,
            anchor_sizes: tuple[int, ...] = (128, 256, 512),
            aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
            anchor_stride: int = 16,
            pool_size: tuple[int, int] = (7, 7),
            hidden_dim: int = 4096,
            dropout: float = 0.5,
        )

Parameters
----------

- **backbone** (*nn.Module*):  
  Feature extraction network applied once per image to produce a feature map.

- **feat_channels** (*int*):  
  Number of output channels from the backbone's final feature map.

- **num_classes** (*int*):  
  Number of object categories (excluding background).

- **use_fpn** (*bool*, optional):
  Whether to use Feature Pyramid Network (FPN) for backbone feature extraction.
  Default is `False`. When set to `True`, the user should attach backbone model that
  supports FPN feature returns.

- **anchor_sizes** (*tuple[int, ...]*, optional):  
  Set of anchor box scales used by the RPN. Default is `(128, 256, 512)`.

- **aspect_ratios** (*tuple[float, ...]*, optional):  
  Set of aspect ratios for the anchors. Default is `(0.5, 1.0, 2.0)`.

- **anchor_stride** (*int*, optional):  
  Stride of the anchor generation relative to the backbone feature map. Default is `16`.

- **pool_size** (*tuple[int, int]*, optional):  
  Spatial size of each RoI feature after pooling. Default is `(7, 7)`.

- **hidden_dim** (*int*, optional):  
  Number of units in the fully connected head. Default is `4096`.

- **dropout** (*float*, optional):  
  Dropout probability used in the classification and regression head. Default is `0.5`.

Architecture
------------

Faster R-CNN enhances Fast R-CNN by replacing external proposal mechanisms with a 
learnable RPN:

1. **Feature Map Extraction**:
   
   - The image is processed by the `backbone` to produce a dense feature map.

2. **Region Proposal Network (RPN)**:
   
   - Anchors are placed over the feature map.
   - The RPN classifies whether each anchor contains an object and regresses 
     its bounding box.

3. **RoI Pooling**:
   
   - High-confidence proposals are selected and pooled to a fixed size (`pool_size`).

4. **Detection Head**:
   
   - Each RoI is processed by fully connected layers for classification and bounding 
     box refinement.

5. **Loss Output**:
   
   - The model provides a `.get_loss()` method, returning a structured loss dictionary.

Loss Dictionary
---------------

.. code-block:: python

    class _FasterRCNNLoss(TypedDict):
        rpn_cls_loss: Tensor
        rpn_reg_loss: Tensor
        roi_cls_loss: Tensor
        roi_reg_loss: Tensor
        total_loss: Tensor

Returned by `FasterRCNN.get_loss()`, this dictionary provides detailed loss breakdowns 
for both RPN and RoI heads.

Examples
--------

**Basic Usage**

.. code-block:: python

    import lucid.nn as nn
    import lucid.random

    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

        def forward(self, x):
            return self.conv(x)

    backbone = SimpleBackbone()
    model = nn.FasterRCNN(backbone, feat_channels=128, num_classes=5)

    image = lucid.random.randn(1, 3, 512, 512)
    output = model.predict(image)
    print(output["boxes"].shape, output["scores"].shape, output["labels"].shape)

**Custom Configuration**

.. code-block:: python

    backbone = SimpleBackbone()
    model = nn.FasterRCNN(
        backbone=backbone,
        feat_channels=128,
        num_classes=5,
        anchor_sizes=(64, 128, 256),
        aspect_ratios=(0.5, 1.0),
        pool_size=(5, 5),
        hidden_dim=2048,
        dropout=0.4,
    )

    image = lucid.random.randn(1, 3, 384, 384)
    output = model.predict(image)
    print(output["boxes"].shape, output["scores"].shape, output["labels"].shape)

.. tip::

   For training, use `model.get_loss()` with the predicted and ground-truth targets
   to compute total and component-wise loss terms.

