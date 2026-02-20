R-CNN
=====
|convnet-badge| |two-stage-det-badge| 

.. autoclass:: lucid.models.RCNN

The `RCNN` implements the classic Region-based Convolutional Neural Network architecture 
for object detection. It integrates region proposal extraction, feature warping, CNN feature extraction, and 
classification, following the original R-CNN pipeline introduced by Ross Girshick et al.

.. mermaid::
    :name: R-CNN

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>RCNN</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["backbone"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m2["Sequential"]
            direction TB;
          style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,112,112)</span>"];
            m4["BatchNorm2d"];
            m5["ReLU"];
            m6["Conv2d"];
            m7["BatchNorm2d"];
            m8["ReLU"];
            m9["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,32,112,112) → (1,64,112,112)</span>"];
          end
          m10["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
          subgraph sg_m11["Sequential"]
            direction TB;
          style sg_m11 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m12["_ResNeStBottleneck"]
              direction TB;
            style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m13["ConvBNReLU2d"];
              m14["_SplitAttention"];
              m15["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
              m16["BatchNorm2d"];
              m17["ReLU"];
              m18["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
            end
          end
          subgraph sg_m19["Sequential x 3"]
            direction TB;
          style sg_m19 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m19_in(["Input"]);
            m19_out(["Output"]);
      style m19_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m19_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            subgraph sg_m20["_ResNeStBottleneck"]
              direction TB;
            style sg_m20 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m21["ConvBNReLU2d<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,128,56,56)</span>"];
              m22["_SplitAttention"];
              m23["AvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,128,56,56) → (1,128,28,28)</span>"];
              m24["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,28,28) → (1,512,28,28)</span>"];
              m25["BatchNorm2d"];
              m26["ReLU"];
              m27["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,512,28,28)</span>"];
            end
          end
          m28["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2048,7,7) → (1,2048,1,1)</span>"];
        end
        m29["SelectiveSearch<br/><span style='font-size:11px;font-weight:400'>(3,64,64) → (1,4)</span>"];
        m30["_RegionWarper<br/><span style='font-size:11px;font-weight:400'>(1,3,64,64)x2 → (1,3,224,224)</span>"];
        subgraph sg_m31["_LinearSVM"]
        style sg_m31 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m32["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,100)</span>"];
        end
        subgraph sg_m33["_BBoxRegressor"]
        style sg_m33 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m34["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,400)</span>"];
        end
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
      style m28 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m32 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m34 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m29;
      m10 --> m13;
      m13 --> m14;
      m14 --> m15;
      m15 --> m16;
      m16 --> m18;
      m17 -.-> m21;
      m18 --> m17;
      m19_in -.-> m21;
      m19_out -.-> m19_in;
      m19_out --> m28;
      m21 --> m22;
      m22 --> m23;
      m23 --> m24;
      m24 --> m25;
      m25 --> m27;
      m26 -.-> m19_in;
      m27 --> m19_out;
      m27 --> m26;
      m28 --> m32;
      m29 --> m30;
      m3 --> m4;
      m30 --> m3;
      m32 --> m34;
      m34 --> output;
      m4 --> m5;
      m5 --> m6;
      m6 --> m7;
      m7 --> m8;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class RCNN(nn.Module):
        def __init__(
            backbone: nn.Module,
            feat_dim: int,
            num_classes: int,
            *,
            image_means: tuple[float, float, float] = (0.485, 0.456, 0.406),
            pixel_scale: float = 1.0,
            warper_output_size: tuple[int, int] = (224, 224),
            nms_iou_thresh: float = 0.3,
            score_thresh: float = 0.0,
            add_one: bool = True,
        )

Parameters
----------

- **backbone** (*nn.Module*):
  A convolutional feature extractor shared across all region proposals.

- **feat_dim** (*int*):
  Dimensionality of the backbone’s feature output.

- **num_classes** (*int*):
  Number of object classes to classify (excluding background unless `add_one=True`).

- **image_means** (*tuple[float, float, float]*, optional):
  Per-channel RGB means for input image normalization. Default is ImageNet mean.

- **pixel_scale** (*float*, optional):
  Pixel value scaling factor applied to input images. Default is `1.0`.

- **warper_output_size** (*tuple[int, int]*, optional):
  The fixed resolution to which proposed regions are resized (e.g., 224x224).

- **nms_iou_thresh** (*float*, optional):
  IoU threshold for Non-Maximum Suppression during prediction. Default is `0.3`.

- **score_thresh** (*float*, optional):
  Classification score threshold below which detections are discarded.

- **add_one** (*bool*, optional):
  If `True`, adds background class at index 0, shifting all class indices.

Architecture
------------

The R-CNN architecture consists of the following components:

1. **Region Proposal (Selective Search)**:
   
   - Uses category-independent heuristics to propose candidate bounding boxes.
   - Selective Search merges superpixels based on texture, color, size, and fill similarity.
   - Produces ~2k regions per image.

2. **Feature Extraction**:
   
   - Each proposed region is warped (cropped and resized) to `warper_output_size`.
   - These warped patches are passed through the shared `backbone` CNN.
   - Features are then pooled and flattened for classification.

3. **Classification and Localization**:
   
   - Each region is classified using a fully-connected head.
   - Optionally, bounding box regression can be applied for precise localization.

4. **Inference and NMS**:
   
   - Predictions below `score_thresh` are filtered out.
   - For each class, Non-Maximum Suppression is applied using `nms_iou_thresh` 
     to remove redundant overlapping detections.

Examples
--------

**Basic Usage**

.. code-block:: python

    import lucid.nn as nn
    import lucid.random
    import lucid

    # Define a simple convolutional backbone
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        def forward(self, x):
            x = self.net(x)
            return x.flatten(1)  # (N, C)

    # Instantiate backbone and RCNN
    backbone = ToyBackbone()
    model = nn.RCNN(backbone, feat_dim=64, num_classes=3)

    # Dummy input tensor
    input_ = lucid.random.randn(1, 3, 600, 800)

    # Predict
    output = model.predict(input_)
    print(output["boxes"].shape, output["scores"].shape, output["labels"].shape)

**Explanation**

The image is first processed to generate region proposals using Selective Search. 
Each proposed region is resized and passed through the shared CNN (`backbone`), 
followed by a classification head that predicts object categories. Non-Maximum Suppression 
removes redundant overlapping boxes, and final predictions are returned.

**Custom Configuration**

.. code-block:: python

    class SmallBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        def forward(self, x):
            return self.conv(x).flatten(1)

    # Smaller model with adjusted feature dimension
    backbone = SmallBackbone()
    model = nn.RCNN(
        backbone=backbone,
        feat_dim=32,
        num_classes=2,
        warper_output_size=(112, 112),
        nms_iou_thresh=0.5,
        score_thresh=0.1,
        add_one=False
    )

    input_ = lucid.random.randn(1, 3, 512, 512)
    output = model.predict(input_)
    print(output["boxes"].shape, output["scores"].shape, output["labels"].shape)
