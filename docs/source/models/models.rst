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

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - WideResNet-50
      - `wide_resnet_50 <conv/resnet/wide_resnet_50>`_
      - :math:`(N,3,224,224)`
      - 78,973,224
      - ✅
    
    * - WideResNet-101
      - `wide_resnet_101 <conv/resnet/wide_resnet_101>`_
      - :math:`(N,3,224,224)`
      - 126,886,696
      - ✅

.. rubric:: ResNeXt

ResNeXt is an extension of the ResNet architecture that introduces a cardinality dimension 
to the model, improving its performance and efficiency by allowing flexible aggregation of 
transformations. ResNeXt builds on residual blocks by incorporating grouped convolutions, 
enabling parallel pathways for feature learning.

 Xie, Saining, et al. "Aggregated Residual Transformations for Deep Neural Networks." 
 *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 
 2017, pp. 5987-5995.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - ResNeXt-50-32x4d
      - `resnext_50_32x4d <conv/resnext/resnext_50_32x4d>`_
      - :math:`(N,3,224,224)`
      - 25,028,904
      - ✅
    
    * - ResNeXt-101-32x4d
      - `resnext_101_32x4d <conv/resnext/resnext_101_32x4d>`_
      - :math:`(N,3,224,224)`
      - 44,177,704
      - ✅
    
    * - ResNeXt-101-32x8d
      - `resnext_101_32x8d <conv/resnext/resnext_101_32x8d>`_
      - :math:`(N,3,224,224)`
      - 88,791,336
      - ✅
    
    * - ResNeXt-101-32x16d
      - `resnext_101_32x16d <conv/resnext/resnext_101_32x16d>`_
      - :math:`(N,3,224,224)`
      - 194,026,792
      - ✅
    
    * - ResNeXt-101-32x32d
      - `resnext_101_32x32d <conv/resnext/resnext_101_32x32d>`_
      - :math:`(N,3,224,224)`
      - 468,530,472
      - ✅
    
    * - ResNeXt-101-64x4d
      - `resnext_101_64x4d <conv/resnext/resnext_101_64x4d>`_
      - :math:`(N,3,224,224)`
      - 83,455,272
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

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - SE-ResNeXt-50-32x4d
      - `se_resnext_50_32x4d <conv/senet/se_resnext_50_32x4d>`_
      - :math:`(N,3,224,224)`
      - 27,559,896
      - ✅
    
    * - SE-ResNeXt-101-32x4d
      - `se_resnext_101_32x4d <conv/senet/se_resnext_101_32x4d>`_
      - :math:`(N,3,224,224)`
      - 48,955,416
      - ✅
    
    * - SE-ResNeXt-101-32x8d
      - `se_resnext_101_32x8d <conv/senet/se_resnext_101_32x8d>`_
      - :math:`(N,3,224,224)`
      - 93,569,048
      - ✅
    
    * - SE-ResNeXt-101-64x4d
      - `se_resnext_101_64x4d <conv/senet/se_resnext_101_64x4d>`_
      - :math:`(N,3,224,224)`
      - 88,232,984
      - ✅

.. rubric:: SKNet

SKNet (Selective Kernel Networks) is a deep learning architecture that enhances the 
representational capacity of neural networks by enabling dynamic selection of kernel sizes 
in convolutional layers. It introduces the concept of a "selective kernel" module, 
which allows the network to adaptively choose the most appropriate receptive field for 
each spatial location in an image, improving its ability to capture multi-scale features.

 Li, X., Zhang, S., & Wang, X. (2019). "Selective Kernel Networks." Proceedings of the 
 IEEE International Conference on Computer Vision (ICCV), 2019, pp. 510-519.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - SK-ResNet-18
      - `sk_resnet_18 <conv/sknet/sk_resnet_18>`_
      - :math:`(N,3,224,224)`
      - 25,647,368
      - ✅
    
    * - SK-ResNet-34
      - `sk_resnet_34 <conv/sknet/sk_resnet_34>`_
      - :math:`(N,3,224,224)`
      - 45,895,512
      - ✅
    
    * - SK-ResNet-50
      - `sk_resnet_50 <conv/sknet/sk_resnet_50>`_
      - :math:`(N,3,224,224)`
      - 57,073,368
      - ✅
    
.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented

    * - SK-ResNeXt-50-32x4d
      - `sk_resnext_50_32x4d <conv/sknet/sk_resnext_50_32x4d>`_
      - :math:`(N,3,224,224)`
      - 29,274,760
      - ✅

.. rubric:: DenseNet

A deep learning architecture designed to improve the flow of information and gradients 
in neural networks by introducing dense connectivity between layers. It leverages the 
concept of "dense blocks," where each layer is directly connected to all preceding layers 
within the block. This dense connectivity pattern enhances feature reuse, reduces the number 
of parameters, and improves the efficiency of gradient propagation during training.

 Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). 
 "Densely Connected Convolutional Networks." *Proceedings of the IEEE Conference on 
 Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4700-4708.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - DenseNet-121
      - `densenet_121 <conv/dense/densenet_121>`_
      - :math:`(N,3,224,224)`
      - 7,978,856
      - ✅
    
    * - DenseNet-169
      - `densenet_169 <conv/dense/densenet_169>`_
      - :math:`(N,3,224,224)`
      - 14,149,480
      - ✅
    
    * - DenseNet-201
      - `densenet_201 <conv/dense/densenet_201>`_
      - :math:`(N,3,224,224)`
      - 20,013,928
      - ✅
    
    * - DenseNet-264
      - `densenet_264 <conv/dense/densenet_264>`_
      - :math:`(N,3,224,224)`
      - 33,337,704
      - ✅

.. rubric:: Xception

A deep learning architecture that introduces depthwise separable convolutions 
to enhance efficiency and accuracy in convolutional neural networks. It builds 
on the idea that spatial and channel-wise information can be decoupled, significantly 
reducing computational cost while maintaining performance.

 Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions." 
 *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 
 2017, pp. 1251-1258.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - Xception
      - `xception <conv/xception/xception>`_
      - :math:`(N,3,224,224)`
      - 22,862,096
      - ✅

.. rubric:: MobileNet

A deep learning architecture that introduces depthwise separable convolutions 
to enhance efficiency and accuracy in convolutional neural networks. It builds 
on the idea that spatial and channel-wise information can be decoupled, significantly 
reducing computational cost while maintaining performance.

 *MobileNet*

 Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., 
 & Adam, H. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile 
 Vision Applications." *arXiv preprint arXiv:1704.04861.*

 *MobileNet-v2*
 
 Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). 
 "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *Proceedings of the IEEE 
 Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4510-4520.

 *MobileNet-v3*

 Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., 
 Pang, R., Vasudevan, V., Le, Q., & Adam, H. (2019). "Searching for MobileNetV3." 
 *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 1314-1324.

 *MobileNet-v4*

 Zhang, Wei, et al. “MobileNet-v4: Advancing Efficiency for Mobile Vision.” 
 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 
 2024, pp. 5720-5730.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - MobileNet
      - `mobilenet <conv/mobile/mobilenet>`_
      - :math:`(N,3,224,224)`
      - 4,232,008
      - ✅
    
    * - MobileNet-v2
      - `mobilenet_v2 <conv/mobile/mobilenet_v2>`_
      - :math:`(N,3,224,224)`
      - 3,504,872
      - ✅
    
    * - MobileNet-v3-Small
      - `mobilenet_v3_small <conv/mobile/mobilenet_v3_small>`_
      - :math:`(N,3,224,224)`
      - 2,537,238
      - ✅
    
    * - MobileNet-v3-Large
      - `mobilenet_v3_large <conv/mobile/mobilenet_v3_large>`_
      - :math:`(N,3,224,224)`
      - 5,481,198
      - ✅

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - MobileNet-v4-Conv-Small
      - `mobilenet_v4_conv_small <conv/mobile/mobilenet_v4_conv_small>`_
      - :math:`(N,3,224,224)`
      - 3,774,024
      - ✅
    
    * - MobileNet-v4-Conv-Medium
      - `mobilenet_v4_conv_medium <conv/mobile/mobilenet_v4_conv_medium>`_
      - :math:`(N,3,224,224)`
      - 9,715,512
      - ✅
    
    * - MobileNet-v4-Conv-Large
      - `mobilenet_v4_conv_large <conv/mobile/mobilenet_v4_conv_large>`_
      - :math:`(N,3,224,224)`
      - 32,590,864
      - ✅
    
    * - MobileNet-v4-Hybrid-Medium
      - `mobilenet_v4_hybrid_medium <conv/mobile/mobilenet_v4_hybrid_medium>`_
      - :math:`(N,3,224,224)`
      - 11,070,136
      - ✅
    
    * - MobileNet-v4-Hybrid-Large
      - `mobilenet_v4_hybrid_large <conv/mobile/mobilenet_v4_hybrid_large>`_
      - :math:`(N,3,224,224)`
      - 37,755,152
      - ✅

.. rubric:: EfficientNet

EfficientNet is a family of convolutional neural networks optimized for 
scalability and performance by systematically balancing network depth, width, 
and resolution. It achieves state-of-the-art accuracy with fewer parameters and 
computational resources compared to previous architectures.

 *EfficientNet*

 Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling for 
 Convolutional Neural Networks." *Proceedings of the 36th International Conference 
 on Machine Learning*, 2019, pp. 6105-6114.

 *EfficientNet-v2*

 Tan, Mingxing, and Quoc V. Le. "EfficientNetV2: Smaller Models and Faster Training." 
 *Proceedings of the 38th International Conference on Machine Learning*, 2021, 
 pp. 10096-10106.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - EfficientNet-B0
      - `efficientnet_b0 <conv/efficient/efficientnet_b0>`_
      - :math:`(N,3,224,224)`
      - 5,289,636
      - ✅
    
    * - EfficientNet-B1
      - `efficientnet_b1 <conv/efficient/efficientnet_b1>`_
      - :math:`(N,3,240,240)`
      - 7,795,560
      - ✅
    
    * - EfficientNet-B2
      - `efficientnet_b2 <conv/efficient/efficientnet_b2>`_
      - :math:`(N,3,260,260)`
      - 9,111,370
      - ✅
    
    * - EfficientNet-B3
      - `efficientnet_b3 <conv/efficient/efficientnet_b3>`_
      - :math:`(N,3,300,300)`
      - 12,235,536
      - ✅
    
    * - EfficientNet-B4
      - `efficientnet_b4 <conv/efficient/efficientnet_b4>`_
      - :math:`(N,3,380,380)`
      - 19,344,640
      - ✅
    
    * - EfficientNet-B5
      - `efficientnet_b5 <conv/efficient/efficientnet_b5>`_
      - :math:`(N,3,456,456)`
      - 30,393,432
      - ✅
    
    * - EfficientNet-B6
      - `efficientnet_b6 <conv/efficient/efficientnet_b6>`_
      - :math:`(N,3,528,528)`
      - 43,046,128
      - ✅
    
    * - EfficientNet-B7
      - `efficientnet_b7 <conv/efficient/efficientnet_b7>`_
      - :math:`(N,3,600,600)`
      - 66,355,448
      - ✅

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - EfficientNet-v2-S
      - `efficientnet_v2_s <conv/efficient/efficientnet_v2_s>`_
      - :math:`(N,3,224,224)`
      - 21,136,440
      - ✅
    
    * - EfficientNet-v2-M
      - `efficientnet_v2_m <conv/efficient/efficientnet_v2_m>`_
      - :math:`(N,3,224,224)`
      - 55,302,108
      - ✅
    
    * - EfficientNet-v2-L
      - `efficientnet_v2_l <conv/efficient/efficientnet_v2_l>`_
      - :math:`(N,3,224,224)`
      - 120,617,032
      - ✅
    
    * - EfficientNet-v2-XL
      - `efficientnet_v2_xl <conv/efficient/efficientnet_v2_xl>`_
      - :math:`(N,3,224,224)`
      - 210,221,568
      - ✅

.. rubric:: ResNeSt

ResNeSt introduces Split Attention Blocks, which divide feature maps into groups, 
compute attention for each group, and reassemble them to enhance representational power. 
It extends ResNet by integrating these blocks, achieving improved performance in image 
recognition tasks with minimal computational overhead.

 Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." arXiv preprint 
 arXiv:2004.08955, 2020.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - ResNeSt-14
      - `resnest_14 <conv/resnest/resnest_14>`_
      - :math:`(N,3,224,224)`
      - 10,611,560
      - ✅
    
    * - ResNeSt-26
      - `resnest_26 <conv/resnest/resnest_26>`_
      - :math:`(N,3,224,224)`
      - 17,069,320
      - ✅
    
    * - ResNeSt-50
      - `resnest_50 <conv/resnest/resnest_50>`_
      - :math:`(N,3,224,224)`
      - 27,483,112
      - ✅
    
    * - ResNeSt-101
      - `resnest_101 <conv/resnest/resnest_101>`_
      - :math:`(N,3,224,224)`
      - 48,274,760
      - ✅
    
    * - ResNeSt-200
      - `resnest_200 <conv/resnest/resnest_200>`_
      - :math:`(N,3,224,224)`
      - 70,201,288
      - ✅
    
    * - ResNeSt-269
      - `resnest_269 <conv/resnest/resnest_269>`_
      - :math:`(N,3,224,224)`
      - 110,929,224
      - ✅
    
    * - ResNeSt-50-4s2x40d
      - `resnest_50_4s2x40d <conv/resnest/resnest_50_4s2x40d>`_
      - :math:`(N,3,224,224)`
      - 30,417,464
      - ✅
    
    * - ResNeSt-50_1s4x24d
      - `resnest_50_1s4x24d <conv/resnest/resnest_50_1s4x24d>`_
      - :math:`(N,3,224,224)`
      - 25,676,872
      - ✅

.. rubric:: ConvNeXt

ConvNeXt reimagines CNNs using principles inspired by vision transformers, 
streamlining architectural design while preserving the efficiency of traditional CNNs. 
It introduces design elements like simplified stem stages, inverted bottlenecks, 
and expanded kernel sizes to enhance feature extraction.

 *ConvNeXt*

 Liu, Zhuang, et al. "A ConvNet for the 2020s." *arXiv preprint arXiv:2201.03545*, 2022.

 *ConvNeXt-v2*

 Liu, Ze, et al. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." 
 *arXiv preprint arXiv:2301.00808*, 2023.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - ConvNeXt-Tiny
      - `convnext_tiny <conv/convnext/convnext_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,589,128
      - ✅
    
    * - ConvNeXt-Small
      - `convnext_small <conv/convnext/convnext_small>`_
      - :math:`(N,3,224,224)`
      - 46,884,148
      - ✅
    
    * - ConvNeXt-Base
      - `convnext_base <conv/convnext/convnext_base>`_
      - :math:`(N,3,224,224)`
      - 88,591,464
      - ✅
    
    * - ConvNeXt-Large
      - `convnext_large <conv/convnext/convnext_large>`_
      - :math:`(N,3,224,224)`
      - 197,767,336
      - ✅
    
    * - ConvNeXt-XLarge
      - `convnext_xlarge <conv/convnext/convnext_xlarge>`_
      - :math:`(N,3,224,224)`
      - 350,196,968
      - ✅

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - ConvNeXt-v2-Atto
      - `convnext_v2_atto <conv/convnext/convnext_v2_atto>`_
      - :math:`(N,3,224,224)`
      - 3,708,400
      - ✅
    
    * - ConvNeXt-v2-Femto
      - `convnext_v2_femto <conv/convnext/convnext_v2_femto>`_
      - :math:`(N,3,224,224)`
      - 5,233,240
      - ✅
    
    * - ConvNeXt-v2-Pico
      - `convnext_v2_pico <conv/convnext/convnext_v2_pico>`_
      - :math:`(N,3,224,224)`
      - 9,066,280
      - ✅
    
    * - ConvNeXt-v2-Nano
      - `convnext_v2_nano <conv/convnext/convnext_v2_nano>`_
      - :math:`(N,3,224,224)`
      - 15,623,800
      - ✅
    
    * - ConvNeXt-v2-Tiny
      - `convnext_v2_tiny <conv/convnext/convnext_v2_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,635,496
      - ✅
    
    * - ConvNeXt-v2-Base
      - `convnext_v2_base <conv/convnext/convnext_v2_base>`_
      - :math:`(N,3,224,224)`
      - 88,717,800
      - ✅
    
    * - ConvNeXt-v2-Large
      - `convnext_v2_large <conv/convnext/convnext_v2_large>`_
      - :math:`(N,3,224,224)`
      - 197,956,840
      - ✅
    
    * - ConvNeXt-v2-Huge
      - `convnext_v2_huge <conv/convnext/convnext_v2_huge>`_
      - :math:`(N,3,224,224)`
      - 660,289,640
      - ✅

.. rubric:: InceptionNeXt

InceptionNeXt extends the Inception architecture by incorporating modern design 
principles inspired by vision transformers. It refines multi-scale feature extraction 
through dynamic kernel selection, depthwise convolutions, and enhanced normalization 
techniques, preserving computational efficiency while improving performance across 
diverse vision tasks.

 Yu, Weihao, et al. "InceptionNeXt: When Inception Meets ConvNeXt." 
 *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 
 2024, pp. 5672-5683.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Implemented
    
    * - InceptionNeXt-Atto
      - `inception_next_atto <conv/inception_next/inception_next_atto>`_
      - :math:`(N,3,224,224)`
      - 4,156,520
      - ✅
    
    * - InceptionNeXt-Tiny
      - `inception_next_tiny <conv/inception_next/inception_next_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,083,832
      - ✅
    
    * - InceptionNeXt-Small
      - `inception_next_small <conv/inception_next/inception_next_small>`_
      - :math:`(N,3,224,224)`
      - 49,431,544
      - ✅
    
    * - InceptionNeXt-Base
      - `inception_next_base <conv/inception_next/inception_next_base>`_
      - :math:`(N,3,224,224)`
      - 86,748,840
      - ✅

*To be implemented...🔮*
