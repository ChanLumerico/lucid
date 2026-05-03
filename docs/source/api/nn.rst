lucid.nn
========

.. currentmodule:: lucid.nn

Base classes
------------

.. autoclass:: Module
   :members:
   :special-members: __call__

.. autoclass:: Parameter
   :members:

Containers
----------

.. autoclass:: Sequential
   :members:
.. autoclass:: ModuleList
   :members:
.. autoclass:: ModuleDict
   :members:
.. autoclass:: ParameterList
   :members:
.. autoclass:: ParameterDict
   :members:

Linear layers
-------------

.. autoclass:: Linear
   :members:
.. autoclass:: Bilinear
   :members:
.. autoclass:: Identity
   :members:

Convolutional layers
--------------------

.. autoclass:: Conv1d
   :members:
.. autoclass:: Conv2d
   :members:
.. autoclass:: Conv3d
   :members:
.. autoclass:: ConvTranspose1d
   :members:
.. autoclass:: ConvTranspose2d
   :members:
.. autoclass:: ConvTranspose3d
   :members:

Recurrent layers
----------------

.. autoclass:: LSTM
   :members:
.. autoclass:: GRU
   :members:
.. autoclass:: RNN
   :members:
.. autoclass:: LSTMCell
   :members:
.. autoclass:: GRUCell
   :members:
.. autoclass:: RNNCell
   :members:

Transformer layers
------------------

.. autoclass:: Transformer
   :members:
.. autoclass:: TransformerEncoder
   :members:
.. autoclass:: TransformerEncoderLayer
   :members:
.. autoclass:: TransformerDecoder
   :members:
.. autoclass:: TransformerDecoderLayer
   :members:
.. autoclass:: MultiheadAttention
   :members:

Normalization layers
--------------------

.. autoclass:: LayerNorm
   :members:
.. autoclass:: BatchNorm1d
   :members:
.. autoclass:: BatchNorm2d
   :members:
.. autoclass:: BatchNorm3d
   :members:
.. autoclass:: GroupNorm
   :members:
.. autoclass:: RMSNorm
   :members:
.. autoclass:: InstanceNorm1d
   :members:
.. autoclass:: InstanceNorm2d
   :members:
.. autoclass:: InstanceNorm3d
   :members:

Activation functions
--------------------

.. autoclass:: ReLU
   :members:
.. autoclass:: LeakyReLU
   :members:
.. autoclass:: ELU
   :members:
.. autoclass:: GELU
   :members:
.. autoclass:: SiLU
   :members:
.. autoclass:: PReLU
   :members:
.. autoclass:: Sigmoid
   :members:
.. autoclass:: Tanh
   :members:
.. autoclass:: Softmax
   :members:
.. autoclass:: LogSoftmax
   :members:
.. autoclass:: Hardtanh
   :members:
.. autoclass:: GLU
   :members:

Pooling layers
--------------

.. autoclass:: MaxPool1d
   :members:
.. autoclass:: MaxPool2d
   :members:
.. autoclass:: AvgPool1d
   :members:
.. autoclass:: AvgPool2d
   :members:
.. autoclass:: AdaptiveAvgPool1d
   :members:
.. autoclass:: AdaptiveAvgPool2d
   :members:
.. autoclass:: AdaptiveMaxPool2d
   :members:

Dropout layers
--------------

.. autoclass:: Dropout
   :members:
.. autoclass:: Dropout2d
   :members:
.. autoclass:: AlphaDropout
   :members:

Padding layers
--------------

.. autoclass:: ZeroPad2d
   :members:
.. autoclass:: ConstantPad1d
   :members:
.. autoclass:: ConstantPad2d
   :members:
.. autoclass:: ConstantPad3d
   :members:
.. autoclass:: ReflectionPad1d
   :members:
.. autoclass:: ReflectionPad2d
   :members:
.. autoclass:: ReplicationPad1d
   :members:
.. autoclass:: ReplicationPad2d
   :members:

Upsampling
----------

.. autoclass:: Upsample
   :members:
.. autoclass:: PixelShuffle
   :members:
.. autoclass:: PixelUnshuffle
   :members:

Sparse layers
-------------

.. autoclass:: Embedding
   :members:

Reshape layers
--------------

.. autoclass:: Flatten
   :members:
.. autoclass:: Unflatten
   :members:

Loss functions
--------------

.. autoclass:: MSELoss
   :members:
.. autoclass:: L1Loss
   :members:
.. autoclass:: CrossEntropyLoss
   :members:
.. autoclass:: NLLLoss
   :members:
.. autoclass:: BCELoss
   :members:
.. autoclass:: BCEWithLogitsLoss
   :members:
.. autoclass:: HuberLoss
   :members:

Initialisation
--------------

.. automodule:: lucid.nn.init
   :members:
