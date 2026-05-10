lucid.nn.functional
===================

.. currentmodule:: lucid.nn.functional

Stateless functional equivalents of the ``nn`` module layers.
All functions accept :class:`~lucid.Tensor` inputs and return
:class:`~lucid.Tensor` outputs without maintaining internal state.

Activations
-----------

.. autofunction:: relu
.. autofunction:: leaky_relu
.. autofunction:: prelu
.. autofunction:: elu
.. autofunction:: selu
.. autofunction:: gelu
.. autofunction:: silu
.. autofunction:: mish
.. autofunction:: sigmoid
.. autofunction:: tanh
.. autofunction:: softmax
.. autofunction:: log_softmax
.. autofunction:: softplus
.. autofunction:: hardswish
.. autofunction:: hardsigmoid

Linear
------

.. autofunction:: linear

Convolution
-----------

.. autofunction:: conv1d
.. autofunction:: conv2d
.. autofunction:: conv3d
.. autofunction:: conv_transpose1d
.. autofunction:: conv_transpose2d
.. autofunction:: conv_transpose3d

Pooling
-------

.. autofunction:: max_pool1d
.. autofunction:: max_pool2d
.. autofunction:: max_pool3d
.. autofunction:: avg_pool1d
.. autofunction:: avg_pool2d
.. autofunction:: avg_pool3d
.. autofunction:: adaptive_max_pool1d
.. autofunction:: adaptive_max_pool2d
.. autofunction:: adaptive_avg_pool1d
.. autofunction:: adaptive_avg_pool2d

Normalization
-------------

.. autofunction:: batch_norm
.. autofunction:: layer_norm
.. autofunction:: group_norm
.. autofunction:: instance_norm

Dropout
-------

.. autofunction:: dropout
.. autofunction:: dropout1d
.. autofunction:: dropout2d
.. autofunction:: dropout3d
.. autofunction:: alpha_dropout

Padding
-------

.. autofunction:: pad

Loss
----

.. autofunction:: mse_loss
.. autofunction:: l1_loss
.. autofunction:: smooth_l1_loss
.. autofunction:: cross_entropy
.. autofunction:: nll_loss
.. autofunction:: binary_cross_entropy
.. autofunction:: binary_cross_entropy_with_logits
.. autofunction:: kl_div

Attention
---------

.. autofunction:: scaled_dot_product_attention
.. autofunction:: multi_head_attention_forward

Embeddings
----------

.. autofunction:: embedding

Upsampling
----------

.. autofunction:: interpolate
