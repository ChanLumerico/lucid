EfficientDet
============
|convnet-badge| |one-stage-det-badge| |objdet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    efficientdet_d0.rst
    efficientdet_d1.rst
    efficientdet_d2.rst
    efficientdet_d3.rst
    efficientdet_d4.rst
    efficientdet_d5.rst
    efficientdet_d6.rst
    efficientdet_d7.rst

.. autoclass:: lucid.models.EfficientDet

The `EfficientDet` class implements a family of **Efficient Object Detection** models, 
based on the architecture proposed by Tan et al. (2020).

It combines EfficientNet backbones with **BiFPN (Bidirectional Feature Pyramid Networks)** 
for scalable and efficient multi-scale detection.

.. mermaid::
    :name: EfficientDet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>efficientdet_d0</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m1(["Conv2d x 4<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,40,28,28) → (1,64,28,28)</span>"]);
        subgraph sg_m2["conv7"]
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m3["ReLU"];
        m4["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,4,4) → (1,64,2,2)</span>"];
        end
        subgraph sg_m5["bifpn"]
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m6["_BiFPN x 2"]
            direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
    style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            subgraph sg_m7["convs"]
            direction TB;
            style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8(["_ConvBlock x 8"]);
            end
            subgraph sg_m9["ups"]
            direction TB;
            style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m10(["Upsample x 4<br/><span style='font-size:11px;color:#b83280;font-weight:400'>(1,64,14,14) → (1,64,28,28)</span>"]);
            end
            subgraph sg_m11["downs"]
            direction TB;
            style sg_m11 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m12(["AvgPool2d x 4<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,28,28) → (1,64,14,14)</span>"]);
            end
            m13["ParameterDict"];
            subgraph sg_m14["acts"]
            direction TB;
            style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m15(["Swish x 8"]);
            end
        end
        end
        subgraph sg_m16["_BBoxRegressor"]
        style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m17["layers"]
            direction TB;
        style sg_m17 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m18["DepthSeparableConv2d"]
            direction TB;
            style sg_m18 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m19(["Conv2d x 2"]);
            end
            m20["BatchNorm2d"];
            m21["Swish"];
            subgraph sg_m22["DepthSeparableConv2d"]
            direction TB;
            style sg_m22 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m23(["Conv2d x 2"]);
            end
            m24["BatchNorm2d"];
            m25["Swish"];
            subgraph sg_m26["DepthSeparableConv2d"]
            direction TB;
            style sg_m26 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m27(["Conv2d x 2"]);
            end
            m28["BatchNorm2d"];
            m29["Swish"];
        end
        m30["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,2,2) → (1,36,2,2)</span>"];
        end
        subgraph sg_m31["_Classifier"]
        style sg_m31 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m32["layers"]
            direction TB;
        style sg_m32 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m33["DepthSeparableConv2d"]
            direction TB;
            style sg_m33 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m34(["Conv2d x 2"]);
            end
            m35["BatchNorm2d"];
            m36["Swish"];
            subgraph sg_m37["DepthSeparableConv2d"]
            direction TB;
            style sg_m37 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m38(["Conv2d x 2"]);
            end
            m39["BatchNorm2d"];
            m40["Swish"];
            subgraph sg_m41["DepthSeparableConv2d"]
            direction TB;
            style sg_m41 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m42(["Conv2d x 2"]);
            end
            m43["BatchNorm2d"];
            m44["Swish"];
        end
        m45["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,2,2) → (1,720,2,2)</span>"];
        m46["Sigmoid"];
        end
        m47["_Anchors<br/><span style='font-size:11px;font-weight:400'>(1,3,224,224) → (1,9441,4)</span>"];
        m48["_BBoxTransform"];
        m49["_ClipBoxes"];
        m50["_FocalLoss"];
        subgraph sg_m51["_EfficientNetBackbone"]
        style sg_m51 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m52["model"]
            direction TB;
        style sg_m52 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m53["Sequential"]
            direction TB;
            style sg_m53 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m54["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,112,112)</span>"];
            m55["BatchNorm2d"];
            end
            subgraph sg_m56["Sequential x 6"]
            direction TB;
            style sg_m56 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m56_in(["Input"]);
            m56_out(["Output"]);
    style m56_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m56_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m57["_MBConv<br/><span style='font-size:11px;font-weight:400'>(1,32,112,112) → (1,16,112,112)</span>"];
            end
        end
        end
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,9441,80)x3</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m1 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m3 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m4 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m10 fill:#fdf2f8,stroke:#b83280,stroke-width:1px;
    style m12 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m15 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m19 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m20 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m21 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m23 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m24 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m25 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m27 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m28 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m29 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m30 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m34 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m35 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m36 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m38 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m39 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m40 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m42 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m43 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m44 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m45 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m46 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m54 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m55 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    input --> m54;
    m1 --> m3;
    m10 -.-> m8;
    m12 -.-> m8;
    m15 --> m10;
    m15 --> m12;
    m15 --> m6_out;
    m19 --> m20;
    m20 --> m21;
    m21 --> m23;
    m23 --> m24;
    m24 --> m25;
    m25 --> m27;
    m27 --> m28;
    m28 --> m29;
    m29 --> m30;
    m3 --> m4;
    m30 -.-> m19;
    m30 --> m47;
    m34 --> m35;
    m35 --> m36;
    m36 --> m38;
    m38 --> m39;
    m39 --> m40;
    m4 -.-> m15;
    m40 --> m42;
    m42 --> m43;
    m43 --> m44;
    m44 --> m45;
    m45 --> m46;
    m46 -.-> m19;
    m46 -.-> m34;
    m47 --> output;
    m54 --> m55;
    m55 -.-> m57;
    m56_in -.-> m57;
    m56_out --> m1;
    m56_out -.-> m56_in;
    m57 -.-> m56_in;
    m57 --> m56_out;
    m6_in -.-> m8;
    m6_out -.-> m34;
    m8 -.-> m15;
    m8 --> m6_in;

Class Signature
---------------

.. code-block:: python

    class EfficientDet(
        compound_coef: Literal[0, 1, 2, 3, 4, 5, 6, 7] = 0,
        num_anchors: int = 9,
        num_classes: int = 80,
    )

Parameters
----------

- **compound_coef** (*Literal[0-7]*):  
  Compound scaling coefficient controlling backbone depth, width, and input resolution.

- **num_anchors** (*int*):  
  Number of anchor boxes per feature level, typically `9`.

- **num_classes** (*int*):  
  Number of target object classes.

Backbone Network
----------------

The backbone network of EfficientDet, a truncated EfficientNet model can be accessed via
`.backbone` attribute of an instance of the class `EfficientDet`.

.. warning::

    EfficientNet backbone for EfficientDet model is **not** pre-trained by default.
    
    The user should pre-train the corresponding separate EfficientNet variant model
    for image classification task and then migrate the weights of **stage 1-7** to
    `EfficientDet.backbone.model`.

    Weight migration can be easily done with `state_dict` and `load_state_dict` methods.

    .. code-block:: python
        
        pretrained = efficientnet_b0(...)  # Pre-trained for Image Classification
        model = EfficientDet(...)

        # Migrate Weights Stage-by-Stage
        for i in range(7):
            # Save state-dict of individual pre-trained model stage
            st_dict = getattr(pretrained, f"stage{i + 1}").state_dict()

            # model.backbone.model is an nn.Sequential module with 7 stages
            model.backbone.model[i].load_state_dict(st_dict)

Input Format
------------

The model expects a 4D input tensor of shape:

.. code-block:: python

    (N, 3, H, W)

Where:

- `N` is the batch size,
- `3` represents the RGB image channels,
- `H`, `W` are the input height and width (variable depending on `compound_coef`).

Target Format
--------------

Targets should be provided as a **list of Tensors**, one per image:

.. code-block:: python

    targets = [Tensor_i of shape (Ni, 5)]

Where:

- `Ni` is the number of objects in the *i-th* image,
- each row of shape `(5,)` corresponds to `[x_min, y_min, x_max, y_max, class_id]`.

.. note::

    Bounding box coordinates are expected in **absolute pixel units**, not normalized.

Loss Computation
----------------

During training, the total loss combines **classification** and **box regression** terms:

.. math::

    \mathcal{L}_{total} = \mathcal{L}_{cls} + \mathcal{L}_{box}

Where:

- :math:`\mathcal{L}_{cls}` is the **Focal Loss** for classification,  
- :math:`\mathcal{L}_{box}` is the **Smooth L1** or **IoU-based** regression loss.

Focal Loss Explanation
----------------------

The **Focal Loss** addresses the issue of class imbalance by down-weighting easy examples 
and focusing training on hard, misclassified samples.

It is defined as:

.. math::

    \mathcal{L}_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)

Where:

- :math:`p_t` is the predicted probability for the true class:
  
  .. math::

      p_t = 
      \begin{cases}
      p, & \text{if } y = 1 \\
      1 - p, & \text{otherwise}
      \end{cases}

- :math:`\alpha_t` is a weighting factor balancing positive and negative samples,
- :math:`\gamma` (the focusing parameter) controls how much the easy examples are down-weighted.

The gradient of the Focal Loss with respect to the input probability :math:`p` is:

.. math::

    \frac{\partial \mathcal{L}_{\text{focal}}}{\partial p} 
    = \alpha_t \gamma (1 - p_t)^{\gamma - 1} p_t \log(p_t) 
      - \alpha_t (1 - p_t)^{\gamma} \frac{1}{p_t}

As :math:`p_t \to 1`, the gradient approaches zero, reducing the contribution of 
well-classified examples and allowing the model to focus on harder ones.

.. tip::

    - Typical values: :math:`\alpha_t = 0.25`, :math:`\gamma = 2.0`.  
    - For detection, Focal Loss is applied to every anchor across all pyramid levels.

.. note::

    The **Focal Loss** was first introduced in *Lin et al., 2017 (RetinaNet)*, 
    and EfficientDet adopted it to stabilize classification across feature scales.

Methods
-------

.. automethod:: lucid.models.EfficientDet.forward
.. automethod:: lucid.models.EfficientDet.predict
.. automethod:: lucid.models.EfficientDet.get_loss

.. tip::

    EfficientDet uses **compound scaling** to balance model accuracy and efficiency:  
    larger `compound_coef` values correspond to deeper and wider networks 
    (D0-D7).

.. warning::

    The target list length **must match** the batch size, 
    and each Tensor inside should contain `[x_min, y_min, x_max, y_max, class_id]` 
    for all objects in that image.
