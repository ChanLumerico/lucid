lucid.nn
========

.. currentmodule:: lucid.nn

Base classes
------------

.. autoclass:: Module
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Parameter
   :members:
   :undoc-members:
   :show-inheritance:

Linear layers
-------------

.. autoclass:: Linear
.. autoclass:: Bilinear
.. autoclass:: LazyLinear

Convolutional layers
--------------------

.. autoclass:: Conv1d
.. autoclass:: Conv2d
.. autoclass:: Conv3d
.. autoclass:: ConvTranspose1d
.. autoclass:: ConvTranspose2d
.. autoclass:: ConvTranspose3d

Pooling layers
--------------

.. autoclass:: MaxPool1d
.. autoclass:: MaxPool2d
.. autoclass:: MaxPool3d
.. autoclass:: AvgPool1d
.. autoclass:: AvgPool2d
.. autoclass:: AvgPool3d
.. autoclass:: AdaptiveMaxPool1d
.. autoclass:: AdaptiveMaxPool2d
.. autoclass:: AdaptiveAvgPool1d
.. autoclass:: AdaptiveAvgPool2d

Normalization layers
--------------------

.. autoclass:: BatchNorm1d
.. autoclass:: BatchNorm2d
.. autoclass:: BatchNorm3d
.. autoclass:: LayerNorm
.. autoclass:: GroupNorm
.. autoclass:: InstanceNorm1d
.. autoclass:: InstanceNorm2d
.. autoclass:: InstanceNorm3d
.. autoclass:: RMSNorm

Recurrent layers
----------------

.. autoclass:: RNN
.. autoclass:: LSTM
.. autoclass:: GRU

Attention
---------

.. autoclass:: MultiheadAttention

Transformer
-----------

.. autoclass:: Transformer
.. autoclass:: TransformerEncoder
.. autoclass:: TransformerDecoder
.. autoclass:: TransformerEncoderLayer
.. autoclass:: TransformerDecoderLayer

Activation layers
-----------------

.. autoclass:: ReLU
.. autoclass:: LeakyReLU
.. autoclass:: PReLU
.. autoclass:: ELU
.. autoclass:: GELU
.. autoclass:: SiLU
.. autoclass:: Mish
.. autoclass:: Sigmoid
.. autoclass:: Tanh
.. autoclass:: Softmax
.. autoclass:: LogSoftmax
.. autoclass:: Softplus
.. autoclass:: Hardswish
.. autoclass:: Hardsigmoid

Dropout layers
--------------

.. autoclass:: Dropout
.. autoclass:: Dropout1d
.. autoclass:: Dropout2d
.. autoclass:: Dropout3d
.. autoclass:: AlphaDropout

Sparse layers
-------------

.. autoclass:: Embedding
.. autoclass:: EmbeddingBag

Padding layers
--------------

.. autoclass:: ZeroPad1d
.. autoclass:: ZeroPad2d
.. autoclass:: ZeroPad3d
.. autoclass:: ConstantPad1d
.. autoclass:: ConstantPad2d
.. autoclass:: ConstantPad3d
.. autoclass:: ReflectionPad1d
.. autoclass:: ReflectionPad2d
.. autoclass:: ReplicationPad1d
.. autoclass:: ReplicationPad2d

Upsampling
----------

.. autoclass:: Upsample

Flatten / unflatten
-------------------

.. autoclass:: Flatten
.. autoclass:: Unflatten

Container modules
-----------------

.. autoclass:: Sequential
.. autoclass:: ModuleList
.. autoclass:: ModuleDict
.. autoclass:: ParameterList
.. autoclass:: ParameterDict

Loss functions
--------------

.. autoclass:: MSELoss
.. autoclass:: L1Loss
.. autoclass:: SmoothL1Loss
.. autoclass:: HuberLoss
.. autoclass:: CrossEntropyLoss
.. autoclass:: NLLLoss
.. autoclass:: BCELoss
.. autoclass:: BCEWithLogitsLoss
.. autoclass:: KLDivLoss
.. autoclass:: CTCLoss
.. autoclass:: TripletMarginLoss
.. autoclass:: CosineEmbeddingLoss
.. autoclass:: MarginRankingLoss

Hooks
-----

.. autofunction:: register_module_forward_hook
.. autofunction:: register_module_backward_hook
