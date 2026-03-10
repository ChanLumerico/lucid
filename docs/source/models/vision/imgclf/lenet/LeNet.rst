LeNet
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    LeNetConfig.rst
    lenet_1.rst
    lenet_4.rst
    lenet_5.rst

|convnet-badge| 

.. autoclass:: lucid.models.LeNet

Overview
--------

The `LeNet` base class provides a flexible implementation for defining
various versions of the LeNet architecture, including LeNet-1, LeNet-4, and LeNet-5.

It takes a `LeNetConfig` object that centralizes the convolution, classifier,
and activation settings for a concrete LeNet variant or a custom architecture.

.. mermaid::
    :name: LeNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>lenet_5</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["conv1 x 2"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m1_in(["Input"]);
          m1_out(["Output"]);
      style m1_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m1_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,1,32,32) → (1,6,28,28)</span>"];
          m3["Tanh"];
          m4["AvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,6,28,28) → (1,6,14,14)</span>"];
        end
        m5["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,400) → (1,120)</span>"];
        m6["Tanh"];
        m7["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,120) → (1,84)</span>"];
        m8["Tanh"];
        m9["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,84) → (1,10)</span>"];
        m10["Tanh"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1,32,32)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,10)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m4 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m5 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m6 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m7 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m8 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m9 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m10 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      input -.-> m2;
      m10 --> output;
      m1_in -.-> m2;
      m1_out --> m5;
      m2 --> m3;
      m3 --> m4;
      m4 --> m1_in;
      m4 --> m1_out;
      m5 --> m6;
      m6 --> m7;
      m7 --> m8;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

   class LeNet(nn.Module):
       def __init__(self, config: LeNetConfig) -> None

Parameters
----------

- **config** (*LeNetConfig*)
  A configuration object describing the two convolution stages, the classifier
  layer sizes, the flattened input size for the first linear layer, and the
  activation module class used between layers.

Attributes
----------

- **config** (*LeNetConfig*)
  The configuration used to build the model.

- **conv1**, **conv2** (*nn.Sequential*)
  Sequential blocks containing convolution, activation, and average pooling.

- **fc1**, **fc2**, ...
  Dynamically added fully connected classifier layers.

Methods
-------

- **forward(x: Tensor) -> Tensor**
  Performs the forward pass through the convolution blocks, flattens the
  features, and applies the classifier stack.

  .. code-block:: python

      def forward(self, x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = x.reshape(x.shape[0], -1)
          ...
          return x

Example Usage
-------------

Below is an example of defining and using a custom LeNet architecture with
`LeNetConfig`:

.. code-block:: python

   import lucid.models as models
   import lucid.nn as nn

   config = models.LeNetConfig(
       conv_layers=[
           {"out_channels": 6},
           {"out_channels": 16},
       ],
       clf_layers=[120, 84, 10],
       clf_in_features=16 * 5 * 5,
       base_activation=nn.Tanh,
   )
   custom_lenet = models.LeNet(config)

   # Sample input tensor (e.g., 32x32 grayscale image)
   input_tensor = lucid.Tensor([...])

   # Forward pass
   output = custom_lenet(input_tensor)
   print(output)
