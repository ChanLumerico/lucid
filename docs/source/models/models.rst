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
      - `lenet_1 <conv/lenet/lenet_1>`_
      - :math:`(N,1,28,28)`
      - 3,246
      - ✅
    
    * - LeNet-4
      - `lenet_4 <conv/lenet/lenet_4>`_
      - :math:`(N,1,28,28)`
      - 18,378
      - ✅
    
    * - LeNet-5
      - `lenet_5 <conv/lenet/lenet_5>`_
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
      - `alexnet <conv/alex/alexnet>`_
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
      - `zfnet <conv/zfnet/zfnet>`_
      - :math:`(N,3,224,224)`
      - 62,357,608
      - ✅

.. rubric:: VGGNet

VGGNet is a deep convolutional neural network known for its simplicity and use of 
small 3x3 convolutional filters, which significantly improved object recognition accuracy.

 Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for 
 Large-Scale Image Recognition." *arXiv preprint arXiv:1409.1556*, 2014.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - VGGNet-11
      - `vggnet_11 <conv/vgg/vggnet_11>`_
      - :math:`(N,3,224,224)`
      - 132,863,336
      - ✅
    
    * - VGGNet-13
      - `vggnet_13 <conv/vgg/vggnet_13>`_
      - :math:`(N,3,224,224)`
      - 133,047,848
      - ✅
    
    * - VGGNet-16
      - `vggnet_16 <conv/vgg/vggnet_16>`_
      - :math:`(N,3,224,224)`
      - 138,357,544
      - ✅
    
    * - VGGNet-19
      - `vggnet_19 <conv/vgg/vggnet_19>`_
      - :math:`(N,3,224,224)`
      - 143,667,240
      - ✅

.. rubric:: Inception

The Inception architecture, introduced in the GoogLeNet model, is a deep convolutional 
neural network designed for efficient feature extraction using parallel convolutional and 
pooling branches, reducing computational cost. It achieves this by combining multi-scale 
feature processing within each module, making it highly effective for image classification 
tasks.

 Szegedy, Christian, et al. "Going Deeper with Convolutions." *Proceedings of the IEEE 
 Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1-9.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - Inception-v1 (GoogLeNet)
      - `inception_v1 <conv/inception/inception_v1>`_
      - :math:`(N,3,224,224)`
      - 13,393,352
      - ✅
    
    * - Inception-v3
      - `inception_v3 <conv/inception/inception_v3>`_
      - :math:`(N,3,299,299)`
      - 30,817,392
      - ✅
    
    * - Inception-v4
      - `inception_v4 <conv/inception/inception_v4>`_
      - :math:`(N,3,299,299)`
      - 40,586,984
      - ✅

.. rubric:: Inception-ResNet

The Inception-ResNet architecture builds upon the Inception model by integrating 
residual connections, which improve gradient flow and training stability in very 
deep networks. This combination of Inception's multi-scale feature processing with 
ResNet's efficient backpropagation allows for a powerful and scalable design, suitable 
for a wide range of image classification tasks.

 Szegedy, Christian, et al. "Inception-v4, Inception-ResNet and the Impact of Residual 
 Connections on Learning." *Proceedings of the AAAI Conference on Artificial Intelligence*, 
 2017, pp. 4278-4284.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - Inception-ResNet-v1
      - `inception_resnet_v1 <conv/inception_res/inception_resnet_v1>`_
      - :math:`(N,3,299,299)`
      - 22,739,128
      - ✅
    
    * - Inception-ResNet-v2
      - `inception_resnet_v2 <conv/inception_res/inception_resnet_v2>`_
      - :math:`(N,3,299,299)`
      - 35,847,512
      - ✅

.. rubric:: ResNet

ResNets (Residual Networks) are deep neural network architectures that use skip 
connections (residual connections) to alleviate the vanishing gradient problem, 
enabling the training of extremely deep models. They revolutionized deep learning 
by introducing identity mappings, allowing efficient backpropagation and improved 
accuracy in tasks like image classification and object detection.

 He, Kaiming, et al. "Deep Residual Learning for Image Recognition." 
 *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 
 2016, pp. 770-778.

 He, Kaiming, et al. "Identity Mappings in Deep Residual Networks." 
 *European Conference on Computer Vision (ECCV)*, Springer, 2016, pp. 630-645.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented

    * - ResNet-18
      - `resnet_18 <conv/resnet/resnet_18>`_
      - :math:`(N,3,224,224)`
      - 11,689,512
      - ✅
    
    * - ResNet-34
      - `resnet_34 <conv/resnet/resnet_34>`_
      - :math:`(N,3,224,224)`
      - 21,797,672
      - ✅
    
    * - ResNet-50
      - `resnet_50 <conv/resnet/resnet_50>`_
      - :math:`(N,3,224,224)`
      - 25,557,032
      - ✅
    
    * - ResNet-101
      - `resnet_101 <conv/resnet/resnet_101>`_
      - :math:`(N,3,224,224)`
      - 44,549,160
      - ✅
    
    * - ResNet-152
      - `resnet_152 <conv/resnet/resnet_152>`_
      - :math:`(N,3,224,224)`
      - 60,192,808
      - ✅
    
    * - ResNet-200
      - `resnet_200 <conv/resnet/resnet_200>`_
      - :math:`(N,3,224,224)`
      - 64,669,864
      - ✅
    
    * - ResNet-269
      - `resnet_269 <conv/resnet/resnet_269>`_
      - :math:`(N,3,224,224)`
      - 102,069,416
      - ✅
    
    * - ResNet-1001
      - `resnet_1001 <conv/resnet/resnet_1001>`_
      - :math:`(N,3,224,224)`
      - 149,071,016
      - ✅

.. rubric:: SENet

SENets (Squeeze-and-Excitation Networks) are deep neural network architectures that enhance t
he representational power of models by explicitly modeling channel interdependencies. 
They introduce a novel "squeeze-and-excitation" block, which adaptively recalibrates channel-wise 
feature responses. 

 Hu, Jie, et al. "Squeeze-and-Excitation Networks." *Proceedings of the IEEE Conference on 
 Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132-7141.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented

    * - SE-ResNet-18
      - `se_resnet_18 <conv/senet/se_resnet_18>`_
      - :math:`(N,3,224,224)`
      - 11,778,592
      - ✅
    
    * - SE-ResNet-34
      - `se_resnet_34 <conv/senet/se_resnet_34>`_
      - :math:`(N,3,224,224)`
      - 21,958,868
      - ✅
    
    * - SE-ResNet-50
      - `se_resnet_50 <conv/senet/se_resnet_50>`_
      - :math:`(N,3,224,224)`
      - 28,088,024
      - ✅
    
    * - SE-ResNet-101
      - `se_resnet_101 <conv/senet/se_resnet_101>`_
      - :math:`(N,3,224,224)`
      - 49,326,872
      - ✅
    
    * - SE-ResNet-152
      - `se_resnet_152 <conv/senet/se_resnet_152>`_
      - :math:`(N,3,224,224)`
      - 66,821,848
      - ✅

*To be implemented...🔮*
