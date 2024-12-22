lucid.models
============

The `lucid.models` package provides a collection of predefined neural network 
architectures that are ready to use for various tasks, such as image classification 
and feature extraction. These models are designed to demonstrate key deep learning 
concepts while leveraging the modular and educational nature of the `lucid` framework.


ConvNets
--------

.. rubric:: LeNet

LeNet is a pioneering CNN by Yann LeCun for digit recognition, 
combining convolutional, pooling, and fully connected layers. 
It introduced concepts like weight sharing and local receptive fields, 
shaping modern CNNs.

.. list-table::
    :header-rows: 1
    :align: left

    * - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - :func:`lucid.models.lenet_1`
      - :math:`(N,1,28,28)`
      - 3,246
      - ✅
    
    * - :func:`lucid.models.lenet_4`
      - :math:`(N,1,28,28)`
      - 18,378
      - ✅
    
    * - :func:`lucid.models.lenet_5`
      - :math:`(N,1,32,32)`
      - 61,706
      - ✅

