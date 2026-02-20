YOLO-v1
=======
|convnet-badge| |one-stage-det-badge| 

.. toctree::
    :maxdepth: 1
    :hidden:

    yolo_v1.rst
    yolo_v1_tiny.rst

.. autoclass:: lucid.models.YOLO_V1

The `YOLO_V1` class implements the original YOLO (You Only Look Once) model 
for real-time object detection, as proposed by Redmon et al. (2016).

It divides the input image into an :math:`S \times S` grid and predicts 
bounding boxes, objectness scores, and class probabilities for each cell.

.. mermaid::
    :name: YOLO-v1

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>yolo_v1</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["darknet"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m2["_ConvBlock"]
            direction TB;
          style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,448,448) → (1,64,224,224)</span>"];
            m4["BatchNorm2d"];
            m5["LeakyReLU"];
          end
          m6["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,224,224) → (1,64,112,112)</span>"];
          subgraph sg_m7["_ConvBlock"]
            direction TB;
          style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,112,112) → (1,192,112,112)</span>"];
            m9["BatchNorm2d"];
            m10["LeakyReLU"];
          end
          m11["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,192,112,112) → (1,192,56,56)</span>"];
          subgraph sg_m12["_ConvBlock x 4"]
            direction TB;
          style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m12_in(["Input"]);
            m12_out(["Output"]);
      style m12_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m12_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m13["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,192,56,56) → (1,128,56,56)</span>"];
            m14["BatchNorm2d"];
            m15["LeakyReLU"];
          end
          m16["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,512,56,56) → (1,512,28,28)</span>"];
          subgraph sg_m17["_ConvBlock x 10"]
            direction TB;
          style sg_m17 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m17_in(["Input"]);
            m17_out(["Output"]);
      style m17_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m17_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m18["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,512,28,28) → (1,256,28,28)</span>"];
            m19["BatchNorm2d"];
            m20["LeakyReLU"];
          end
          m21["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1024,28,28) → (1,1024,14,14)</span>"];
          subgraph sg_m22["_ConvBlock x 8"]
            direction TB;
          style sg_m22 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m22_in(["Input"]);
            m22_out(["Output"]);
      style m22_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m22_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m23["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,1024,14,14) → (1,512,14,14)</span>"];
            m24["BatchNorm2d"];
            m25["LeakyReLU"];
          end
        end
        subgraph sg_m26["fcs"]
          direction TB;
        style sg_m26 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m27["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,50176) → (1,4096)</span>"];
          m28["LeakyReLU"];
          m29["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,4096) → (1,5390)</span>"];
        end
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,448,448)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,5390)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m5 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m6 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m8 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m9 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m10 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m11 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m13 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m14 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m15 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m16 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m18 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m19 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m20 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m21 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m23 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m24 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m25 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m27 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m28 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m29 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m3;
      m10 --> m11;
      m11 -.-> m13;
      m12_in -.-> m13;
      m12_out -.-> m12_in;
      m12_out --> m16;
      m13 --> m14;
      m14 --> m15;
      m15 -.-> m12_in;
      m15 --> m12_out;
      m16 -.-> m18;
      m17_in -.-> m18;
      m17_out -.-> m17_in;
      m17_out --> m21;
      m18 --> m19;
      m19 --> m20;
      m20 -.-> m17_in;
      m20 --> m17_out;
      m21 -.-> m23;
      m22_in -.-> m23;
      m22_out -.-> m22_in;
      m22_out --> m27;
      m23 --> m24;
      m24 --> m25;
      m25 -.-> m22_in;
      m25 --> m22_out;
      m27 --> m28;
      m28 --> m29;
      m29 --> output;
      m3 --> m4;
      m4 --> m5;
      m5 --> m6;
      m6 --> m8;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class YOLO_V1(
        in_channels: int,
        split_size: int,
        num_boxes: int,
        num_classes: int,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    )

Parameters
----------

- **in_channels** (*int*):  
  Number of input image channels, typically `3` for RGB.

- **split_size** (*int*):  
  Number of grid divisions per side (S), meaning the input is divided into :math:`S \times S` cells.

- **num_boxes** (*int*):  
  Number of bounding boxes (B) predicted per grid cell.

- **num_classes** (*int*):  
  Number of object classes to predict (C).

- **lambda_coord** (*float*):  
  Weight for the coordinate loss (default: 5.0).

- **lambda_noobj** (*float*):  
  Weight for the no-object confidence loss (default: 0.5).

Input Format
------------

The target tensor from the dataset should have shape:

.. code-block:: python

    (N, S, S,  *)

Where:

- `S` is `split_size` (grid size),
- `B` is `num_boxes` (bounding boxes per cell),
- `C` is `num_classes`.

Each vector at `(i, j)` of shape `(5 * B + C)` contains:

- For each box (B): `(x, y, w, h, conf)`
- For the cell: one-hot class vector of length `C`

Returns
-------

Use the `forward` method for predictions:

.. code-block:: python

    preds = model(x)

- **preds** (*Tensor*):  
  Tensor of shape `(N, S, S, B * 5 + C)` containing all bounding box and classification predictions.

Loss is computed using:

.. code-block:: python

    loss = model.get_loss(x, target)

- **loss** (*Tensor*):  
  Scalar total loss for object detection, including coordinate, confidence, and classification losses.

Loss Formula
------------

The total YOLO loss is defined as:

.. math::

    \begin{aligned}
    \mathcal{L} &= \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}}
    \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
    &+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} 
      \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
    &+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 \\
    &+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} 
      (C_i - \hat{C}_i)^2 \\
    &+ \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c=1}^{C} (p_i(c) - \hat{p}_i(c))^2
    \end{aligned}

Where:

- :math:`\mathbb{1}_{ij}^{\text{obj}}` indicates that object exists in cell i for box j,
- :math:`\hat{C}_i` is the predicted confidence,
- :math:`(x_i, y_i, w_i, h_i)` are bounding box values,
- :math:`p_i(c)` is the class probability.

Methods
-------

.. automethod:: lucid.models.YOLO_V1.forward
.. automethod:: lucid.models.YOLO_V1.get_loss

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn
    from lucid.models import YOLO_V1

    model = YOLO_V1(
        in_channels=3,
        split_size=7,
        num_boxes=2,
        num_classes=20
    )

    # Forward pass
    x = lucid.rand(16, 3, 448, 448)
    preds = model(x)

    # Compute loss
    target = lucid.rand(16, 7, 7, 5 * 2 + 20)
    loss = model.get_loss(x, target)

    loss.backward()

.. tip::

    YOLO expects bounding boxes as relative coordinates:  
    
    - `x`, `y` are center positions relative to the grid cell.  
    - `w`, `h` are normalized by image width and height.

.. warning::

    The most confident bounding box (highest IoU with ground truth) is chosen for loss computation.  
    Make sure your dataset follows the expected shape: `(N, S, S, 5 * B + C)`.
