YOLO-v4
=======
|convnet-badge| |one-stage-det-badge| |objdet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    yolo_v4.rst

.. autoclass:: lucid.models.YOLO_V4

The `YOLO_V4` class implements the YOLO-v4 object detection model, 
extending YOLO-v3 with architectural and training improvements for better 
accuracy and speed. It includes CSP-DarkNet-53 as a backbone, SPP and PANet as necks, 
and enhancements like Mish activation and label smoothing.

.. mermaid::
    :name: YOLO-V4

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>yolo_v4</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_DefaultCSPDarkNet53"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m2["csp_darknet_53"]
            direction TB;
          style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,3,448,448) → (1,32,224,224)</span>"]);
            m4["Identity"];
            m5["AdaptiveAvgPool2d"];
            m6["Sequential"];
          end
        end
        subgraph sg_m7["_PANetNeck"]
          direction TB;
        style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m8["_SPPBlock"]
            direction TB;
          style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m9(["_ConvBNAct x 3<br/><span style='font-size:11px;font-weight:400'>(1,1024,14,14) → (1,512,14,14)</span>"]);
            m10["ModuleList"];
            m11(["_ConvBNAct x 3<br/><span style='font-size:11px;font-weight:400'>(1,2048,14,14) → (1,512,14,14)</span>"]);
          end
          subgraph sg_m12["_ConvBNAct x 2"]
            direction TB;
          style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m12_in(["Input"]);
            m12_out(["Output"]);
      style m12_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m12_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m13["Conv2d"];
            m14["BatchNorm2d"];
            m15["LeakyReLU"];
          end
          subgraph sg_m16["_FiveConv"]
            direction TB;
          style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m17["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,1024,28,28) → (1,512,28,28)</span>"];
          end
          subgraph sg_m18["_ConvBNAct x 2"]
            direction TB;
          style sg_m18 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m18_in(["Input"]);
            m18_out(["Output"]);
      style m18_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m18_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m19["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,512,28,28) → (1,256,28,28)</span>"];
            m20["BatchNorm2d"];
            m21["LeakyReLU"];
          end
          subgraph sg_m22["_FiveConv"]
            direction TB;
          style sg_m22 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m23["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,512,56,56) → (1,256,56,56)</span>"];
          end
          subgraph sg_m24["_ConvBNAct"]
            direction TB;
          style sg_m24 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m25["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,256,56,56) → (1,512,28,28)</span>"];
            m26["BatchNorm2d"];
            m27["LeakyReLU"];
          end
          subgraph sg_m28["_FiveConv"]
            direction TB;
          style sg_m28 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m29["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,1024,28,28) → (1,512,28,28)</span>"];
          end
          subgraph sg_m30["_ConvBNAct"]
            direction TB;
          style sg_m30 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m31["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,512,28,28) → (1,512,14,14)</span>"];
            m32["BatchNorm2d"];
            m33["LeakyReLU"];
          end
          subgraph sg_m34["_FiveConv"]
            direction TB;
          style sg_m34 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m35["Sequential"];
          end
        end
        subgraph sg_m36["_YOLOHead"]
          direction TB;
        style sg_m36 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m37["detect"]
            direction TB;
          style sg_m37 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m38(["Sequential x 3<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,318,56,56)</span>"]);
          end
        end
        subgraph sg_m39["_ConvBNAct x 3"]
          direction TB;
        style sg_m39 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m39_in(["Input"]);
          m39_out(["Output"]);
      style m39_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m39_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          m40["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,56,56) → (1,256,56,56)</span>"];
          m41["BatchNorm2d"];
          m42["LeakyReLU"];
        end
        m43["BCEWithLogitsLoss"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,448,448)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,318,56,56)x3</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m4 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m5 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m13 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m14 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m15 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m19 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m20 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m21 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m25 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m26 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m27 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m31 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m32 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m33 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m40 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m41 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m42 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m43 fill:#fffbeb,stroke:#d97706,stroke-width:1px;
      input --> m3;
      m10 --> m11;
      m11 -.-> m13;
      m12_in -.-> m13;
      m12_out --> m17;
      m13 --> m14;
      m14 --> m15;
      m15 --> m12_in;
      m15 --> m12_out;
      m17 -.-> m19;
      m18_in -.-> m19;
      m18_out --> m23;
      m19 --> m20;
      m20 --> m21;
      m21 --> m18_in;
      m21 --> m18_out;
      m23 --> m25;
      m25 --> m26;
      m26 --> m27;
      m27 --> m29;
      m29 --> m31;
      m3 -.-> m40;
      m31 --> m32;
      m32 --> m33;
      m33 --> m35;
      m35 --> m38;
      m38 --> output;
      m39_in -.-> m40;
      m39_out -.-> m39_in;
      m39_out --> m9;
      m40 --> m41;
      m41 --> m42;
      m42 -.-> m39_in;
      m42 --> m39_out;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class YOLO_V4(
        num_classes: int,
        anchors: list[list[tuple[int, int]]] | None = None,
        strides: list[int] | None = None,
        backbone: nn.Module | None = None,
        backbone_out_channels: tuple[int, int, int] | None = None,
        in_channels: tuple[int, int, int] = (256, 512, 1024),
        pos_iou_thr: float = 0.25,
        ignore_iou_thr: float = 0.5,
        obj_balance: tuple[float, float, float] = (4.0, 1.0, 0.4),
        cls_label_smoothing: float = 0.0,
        iou_aware_alpha: float = 0.5,
        iou_branch_weight: float = 1.0,
    )

Parameters
----------
- **num_classes** (*int*):  
  Number of object classes for detection.

- **anchors** (*list[list[tuple[int, int]]]*, optional):  
  3-scale list of anchor box groups. Each inner list holds 3 anchor tuples. 
  Defaults to YOLO-v4 anchors if `None`.

- **strides** (*list[int]*, optional):  
  Strides for the 3 detection heads (typically `[8, 16, 32]`).

- **backbone** (*nn.Module*, optional):  
  Optional feature extractor (default: CSP-DarkNet-53).

- **backbone_out_channels** (*tuple[int, int, int]*, optional):  
  Output channels from backbone corresponding to 3 detection scales.

- **in_channels** (*tuple[int, int, int]*):  
  Input channels to the necks (SPP, PANet).

- **pos_iou_thr** (*float*):  
  Positive label threshold for anchor assignment.

- **ignore_iou_thr** (*float*):  
  IOU threshold above which anchor is ignored in loss.

- **obj_balance** (*tuple[float, float, float]*):  
  Weights for objectness loss at different scales.

- **cls_label_smoothing** (*float*):  
  Smoothing factor for classification targets.

- **iou_aware_alpha** (*float*):  
  Weight for combining IOU and objectness predictions.

- **iou_branch_weight** (*float*):  
  Loss weight for IOU-aware branch.

Attributes
----------
- **backbone** (*nn.Module*):  
  Backbone network for feature extraction.

Methods
-------
.. automethod:: lucid.models.objdet.YOLO_V4.forward  
.. automethod:: lucid.models.objdet.YOLO_V4.get_loss  
.. automethod:: lucid.models.objdet.YOLO_V4.predict  

Architectural Improvements
--------------------------
YOLO-v4 introduces major changes in both **backbone** and **neck** compared to YOLO-v3:

**Backbone**:

- Uses **CSP-DarkNet-53** instead of Darknet-53 for better feature reuse and 
  reduced computation.

**Necks**:

- **SPP (Spatial Pyramid Pooling)**:
  
  - Expands receptive field by applying multiple max-pooling operations in parallel.
  - Concatenates pooled features with original feature map for enhanced context.

- **PAN (Path Aggregation Network)**:
  
  - Strengthens bottom-up path to better propagate low-level localization features.
  - Improves multi-scale feature fusion compared to YOLO-v3’s basic FPN.

**Other Enhancements**:

- **Mish** activation in early layers.
- **DropBlock**, **CIoU Loss**, and **Self-Adversarial Training** for regularization.
- **Label Smoothing**, **Cosine Annealing**, and **IoU-aware objectness**.

Multi-Scale Detection
---------------------
YOLO-v4 detects objects at 3 spatial resolutions:

- **13x13 (stride=32)**: large objects  
- **26x26 (stride=16)**: medium objects  
- **52x52 (stride=8)**: small objects

Each scale predicts 3 bounding boxes per grid cell using separate anchors.

Input Format
------------
Targets should be a tuple of 3 tensors (one per scale):

.. code-block:: python

    (N, Hs, Ws, B * (5 + C))

Where:

- `Hs`, `Ws`: grid size (13, 26, 52)
- `B`: number of anchors per scale (typically 3)
- `C`: number of classes

Each vector contains:

- :math:`(t_x, t_y, t_w, t_h, obj, cls_1, \dots, cls_C)`

Where:

- :math:`t_{x,y}`: cell-relative offsets in [0,1]
- :math:`t_{w,h}`: log-scale of width/height to anchor
- :math:`obj`: 1 if responsible, else 0
- :math:`cls`: one-hot encoding of class

YOLO-v4 Loss
------------
YOLO-v4 applies scale-weighted composite loss:

.. math::

    \begin{aligned}
    \mathcal{L} &=
    \sum_{i,j,b} \mathbb{1}_{ijb}^{obj} \alpha_{ijb} \left[
        (\sigma(\hat{t}_{x,ijb}) - t_{x,ijb})^2 +
        (\sigma(\hat{t}_{y,ijb}) - t_{y,ijb})^2 +
        (\hat{t}_{w,ijb} - t_{w,ijb})^2 +
        (\hat{t}_{h,ijb} - t_{h,ijb})^2
    \right] \\\\
    &\quad+ \sum_{i,j,b} \left[
        \mathbb{1}_{ijb}^{obj}(\hat{C}_{ijb} - 1)^2 +
        \mathbb{1}_{ijb}^{noobj}\hat{C}_{ijb}^2
    \right] \\\\
    &\quad+ \sum_{i,j,b} \mathbb{1}_{ijb}^{obj} \sum_c \text{BCE}(\hat{p}_{ijb}(c), p_{ijb}(c)) \\\\
    &\quad+ \lambda_{iou} \sum_{i,j,b} \text{BCE}(\hat{iou}_{ijb}, iou_{ijb})
    \end{aligned}

Where:

- :math:`\hat{C}`: objectness score after sigmoid  
- :math:`\hat{p}(c)`: predicted class probability  
- :math:`\hat{iou}`: predicted IOU confidence (optional)

Prediction Output
-----------------
The `predict` method returns a list of detections per image:

- **"box"**: `[x1, y1, x2, y2]` in pixels  
- **"score"**: objectness x class probability  
- **"class_id"**: predicted class index

Example Usage
-------------
.. admonition:: Using YOLO-V4 with default neck and backbone

    .. code-block:: python

        >>> from lucid.models import YOLO_V4
        >>> model = YOLO_V4(num_classes=80)
        >>> x = lucid.random.rand(1, 3, 416, 416)
        >>> detections = model.predict(x)
        >>> print(detections[0][0])

Backward Propagation
--------------------
YOLO-V4 supports end-to-end gradient training:

.. code-block:: python

    >>> x = lucid.random.rand(1, 3, 416, 416, requires_grad=True)
    >>> targets = (...)  # Ground truth tuple for 3 scales
    >>> loss = model.get_loss(x, targets)
    >>> loss.backward()
    >>> print(x.grad)
