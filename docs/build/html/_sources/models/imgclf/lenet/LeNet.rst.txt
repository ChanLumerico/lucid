LeNet
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    lenet_1.rst
    lenet_4.rst
    lenet_5.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.LeNet

Overview
--------

The `LeNet` base class provides a flexible implementation for defining 
various versions of the LeNet architecture, including LeNet-1, LeNet-4, and LeNet-5. 

It allows the configuration of convolutional and fully connected layers through arguments, 
making it adaptable for different use cases.

.. mermaid::
  :name: LeNet

    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>lenet_5</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05
        subgraph sg_m1["conv1 x 2"]
        style sg_m1 fill:#000000,fill-opacity:0.05
          m1_in(["Input"]);
          m1_out(["Output"]);
      style m1_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m1_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          m2["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,1,32,32) → (1,6,28,28)</span>"];
          m3["Tanh"];
          m4["AvgPool2d<br/><span style='font-size:11px;font-weight:400'>(1,6,28,28) → (1,6,14,14)</span>"];
        end
        m5["Linear<br/><span style='font-size:11px;font-weight:400'>(1,400) → (1,120)</span>"];
        m6["Tanh"];
        m7["Linear<br/><span style='font-size:11px;font-weight:400'>(1,120) → (1,84)</span>"];
        m8["Tanh"];
        m9["Linear<br/><span style='font-size:11px;font-weight:400'>(1,84) → (1,10)</span>"];
        m10["Tanh"];
      end
      input["Input<br/><span style='font-size:11px;color:#000000;font-weight:400'>(1,1,32,32)</span>"];
      output["Output<br/><span style='font-size:11px;color:#000000;font-weight:400'>(1,10)</span>"];
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
       def __init__(
           self, 
           conv_layers: list[dict], 
           clf_layers: list[int], 
           clf_in_features: int,
        ) -> None

Parameters
----------

- **conv_layers** (*list[dict]*)
  A list of dictionaries specifying the configuration of the convolutional layers. 
  Each dictionary should define the number of output channels (`out_channels`) and optionally other parameters such as kernel size, stride, and padding.

- **clf_layers** (*list[int]*)
  A list specifying the sizes of fully connected (classifier) layers. 
  Each entry represents the number of units in the respective layer.

- **clf_in_features** (*int*)
  The number of input features for the first fully connected layer. 
  This is determined by the output size of the feature extractor.

Attributes
----------

- **feature_extractor** (*nn.Sequential*)
  A sequential model containing the convolutional and pooling layers.

- **classifier** (*nn.Sequential*)
  A sequential model containing the fully connected layers.

Methods
-------

- **forward(x: Tensor) -> Tensor**
  Performs the forward pass through the feature extractor and classifier.

  .. code-block:: python

      def forward(self, x):
          x = self.feature_extractor(x)
          x = x.view(x.shape[0], -1)  # Flatten
          x = self.classifier(x)
          return x

Example Usage
-------------

Below is an example of defining and using a LeNet-based architecture:

.. code-block:: python

   import lucid.models as models

   # Define a custom LeNet architecture
   custom_lenet = models.LeNet(
       conv_layers=[
           {"out_channels": 6},
           {"out_channels": 16},
       ],
       clf_layers=[120, 84, 10],
       clf_in_features=16 * 5 * 5,
   )

   # Sample input tensor (e.g., 32x32 grayscale image)
   input_tensor = lucid.Tensor([...])

   # Forward pass
   output = custom_lenet(input_tensor)
   print(output)
