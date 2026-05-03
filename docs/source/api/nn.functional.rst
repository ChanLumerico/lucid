lucid.nn.functional
===================

.. currentmodule:: lucid.nn.functional

Activations
-----------

.. autofunction:: relu
.. autofunction:: leaky_relu
.. autofunction:: elu
.. autofunction:: selu
.. autofunction:: gelu
.. autofunction:: silu
.. autofunction:: sigmoid
.. autofunction:: tanh
.. autofunction:: softmax
.. autofunction:: log_softmax
.. autofunction:: softmin
.. autofunction:: glu
.. autofunction:: prelu
.. autofunction:: normalize
.. autofunction:: cosine_similarity

Linear
------

.. autofunction:: linear
.. autofunction:: bilinear

Convolution
-----------

.. autofunction:: conv1d
.. autofunction:: conv2d
.. autofunction:: conv3d
.. autofunction:: conv_transpose1d
.. autofunction:: conv_transpose2d
.. autofunction:: conv_transpose3d

Normalization
-------------

.. autofunction:: batch_norm
.. autofunction:: layer_norm
.. autofunction:: group_norm
.. autofunction:: rms_norm
.. autofunction:: instance_norm

Pooling
-------

.. autofunction:: max_pool1d
.. autofunction:: max_pool2d
.. autofunction:: avg_pool1d
.. autofunction:: avg_pool2d
.. autofunction:: adaptive_avg_pool1d
.. autofunction:: adaptive_avg_pool2d
.. autofunction:: adaptive_max_pool2d

Attention
---------

.. autofunction:: scaled_dot_product_attention

Dropout
-------

.. autofunction:: dropout
.. autofunction:: dropout2d

Loss functions
--------------

.. autofunction:: mse_loss
.. autofunction:: l1_loss
.. autofunction:: cross_entropy
.. autofunction:: nll_loss
.. autofunction:: binary_cross_entropy
.. autofunction:: binary_cross_entropy_with_logits
.. autofunction:: huber_loss
.. autofunction:: kl_div

Sparse
------

.. autofunction:: embedding
.. autofunction:: one_hot

Sampling / Padding
------------------

.. autofunction:: interpolate
.. autofunction:: pad
.. autofunction:: unfold
.. autofunction:: grid_sample
.. autofunction:: affine_grid
