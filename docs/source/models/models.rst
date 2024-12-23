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

 Lecun, Yann, et al. "Gradient-Based Learning Applied to Document Recognition." 
 *Proceedings of the IEEE*, vol. 86, no. 11, Nov. 1998, pp. 2278-2324. 
 doi:10.1109/5.726791.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - LeNet-1
      - `lenet_1 <convnets/lenet/lenet_1>`_
      - :math:`(N,1,28,28)`
      - 3,246
      - ✅
    
    * - LeNet-4
      - `lenet_4 <convnets/lenet/lenet_4>`_
      - :math:`(N,1,28,28)`
      - 18,378
      - ✅
    
    * - LeNet-5
      - `lenet_5 <convnets/lenet/lenet_5>`_
      - :math:`(N,1,32,32)`
      - 61,706
      - ✅

.. rubric:: AlexNet

AlexNet is a pioneering convolutional neural network introduced in 2012, 
known for its deep architecture and use of ReLU activations, dropout, and GPU acceleration. 
It achieved groundbreaking performance in the ImageNet Large Scale Visual Recognition 
Challenge (ILSVRC) in 2012, popularizing deep learning for computer vision.

 Krizhevsky, Alex, et al. "ImageNet Classification with Deep Convolutional Neural Networks." 
 *Advances in Neural Information Processing Systems*, vol. 25, 2012, pp. 1097-1105.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - AlexNet
      - `alexnet <convnets/alex/alexnet>`_
      - :math:`(N,3,224,224)`
      - 61,100,840
      - ✅

.. rubric:: ZFNet

ZFNet (Zeiler and Fergus Net) is a convolutional neural network that improved upon 
AlexNet by using smaller convolutional filters and visualizing learned features to 
better understand network behavior. It achieved state-of-the-art results in object 
recognition and provided insights into deep learning interpretability.

 Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding Convolutional Networks." 
 *European Conference on Computer Vision (ECCV)*, Springer, Cham, 2014, pp. 818-833. 
 doi:10.1007/978-3-319-10590-1_53.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - ZFNet
      - `zfnet <convnets/zfnet/zfnet>`_
      - :math:`(N,3,224,224)`
      - 62,357,608
      - ✅

*To be implemented...🔮*
