Inception
=========

.. toctree::
    :maxdepth: 1
    :hidden:

    inception_v1.rst
    inception_v3.rst
    inception_v4.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.Inception

Overview
--------

The `Inception` base class provides a flexible implementation for defining 
various versions of the Inception architecture, including Inception v1, v3, and v4. 

It facilitates the configuration of the feature extraction and classification components 
through arguments, making it adaptable for different versions of the Inception series.

.. mermaid::
    :name: Inception

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>inception_v4</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["conv"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m2["_InceptionStem_V4<br/><span style='font-size:11px;font-weight:400'>(1,3,224,224) → (1,384,25,25)</span>"];
        m3(["_InceptionModule_V4A x 4"]);
        m4["_InceptionReduce_V4A<br/><span style='font-size:11px;font-weight:400'>(1,384,25,25) → (1,1024,12,12)</span>"];
        m5(["_InceptionModule_V4B x 7"]);
        m6["_InceptionReduce_V4B<br/><span style='font-size:11px;font-weight:400'>(1,1024,12,12) → (1,1536,5,5)</span>"];
        m7(["_InceptionModule_V4C x 3"]);
        end
        m8["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1536,5,5) → (1,1536,1,1)</span>"];
        m9["Dropout"];
        m10["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1536) → (1,1000)</span>"];
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m8 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m9 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m10 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m2;
    m10 --> output;
    m2 --> m3;
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

   class Inception(nn.Module):
       def __init__(self, num_classes: int, use_aux: bool = True) -> None

Parameters
----------

- **num_classes** (*int*)
  The number of output classes for the final classification layer.

- **use_aux** (*bool*, optional)
  Whether to include auxiliary classifiers. Auxiliary classifiers are 
  additional branches used during training to assist optimization. Default is `True`.
