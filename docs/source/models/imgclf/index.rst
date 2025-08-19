Image Classification
====================

.. toctree::
    :maxdepth: 1
    :hidden:
    
    LeNet <lenet/LeNet.rst>
    AlexNet <alex/AlexNet_.rst>
    ZFNet <zfnet/ZFNet_.rst>
    VGGNet <vgg/VGGNet.rst>
    Inception <inception/Inception.rst>
    Inception-ResNet <inception_res/InceptionResNet.rst>
    ResNet <resnet/ResNet.rst>
    ResNeXt <resnext/ResNeXt.rst>
    ResNeSt <resnest/ResNeSt.rst>
    SENet <senet/SENet.rst>
    SKNet <sknet/SKNet.rst>
    DenseNet <dense/DenseNet.rst>
    Xception <xception/Xception_.rst>
    MobileNet <mobile/MobileNet_.rst>
    MobileNet-v2 <mobile/MobileNet_V2_.rst>
    MobileNet-v3 <mobile/MobileNet_V3.rst>
    MobileNet-v4 <mobile/MobileNet_V4.rst>
    EfficientNet <efficient/EfficientNet.rst>
    EfficientNet-v2 <efficient/EfficientNet_V2.rst>
    ConvNeXt <convnext/ConvNeXt.rst>
    ConvNeXt-v2 <convnext/ConvNeXt_V2.rst>
    InceptionNeXt <inception_next/InceptionNeXt.rst>
    CoAtNet <coatnet/CoAtNet.rst>

    ViT <vit/ViT.rst>
    Swin Transformer <swin/SwinTransformer.rst>
    Swin Transformer-v2 <swin/SwinTransformer_V2.rst>
    CvT <cvt/CvT.rst>
    PVT <pvt/PVT.rst>
    PVT-v2 <pvt/PVT_V2.rst>
    CrossViT <crossvit/CrossViT.rst>
    MaxViT <maxvit/MaxViT.rst>
    EfficientFormer <efficientformer/EfficientFormer.rst>

LeNet
-----
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - LeNet-1
      - `lenet_1 <lenet/lenet_1>`_
      - :math:`(N,1,28,28)`
      - 3,246
      - 167.13K
      - ✅
    
    * - LeNet-4
      - `lenet_4 <lenet/lenet_4>`_
      - :math:`(N,1,28,28)`
      - 18,378
      - 182.93K
      - ✅
    
    * - LeNet-5
      - `lenet_5 <lenet/lenet_5>`_
      - :math:`(N,1,32,32)`
      - 61,706
      - 481.49K
      - ✅

AlexNet
-------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - AlexNet
      - `alexnet <alex/alexnet>`_
      - :math:`(N,3,224,224)`
      - 61,100,840
      - 715.21M
      - ✅

ZFNet
-----
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - ZFNet
      - `zfnet <zfnet/zfnet>`_
      - :math:`(N,3,224,224)`
      - 62,357,608
      - 1.20B
      - ❌

VGGNet
------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - VGGNet-11
      - `vggnet_11 <vgg/vggnet_11>`_
      - :math:`(N,3,224,224)`
      - 132,863,336
      - 7.62B
      - ✅
    
    * - VGGNet-13
      - `vggnet_13 <vgg/vggnet_13>`_
      - :math:`(N,3,224,224)`
      - 133,047,848
      - 11.33B
      - ✅
    
    * - VGGNet-16
      - `vggnet_16 <vgg/vggnet_16>`_
      - :math:`(N,3,224,224)`
      - 138,357,544
      - 15.50B
      - ✅
    
    * - VGGNet-19
      - `vggnet_19 <vgg/vggnet_19>`_
      - :math:`(N,3,224,224)`
      - 143,667,240
      - 19.66B
      - ✅

Inception
---------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - Inception-v1 (GoogLeNet)
      - `inception_v1 <inception/inception_v1>`_
      - :math:`(N,3,224,224)`
      - 13,393,352
      - 1.62B
      - ❌
    
    * - Inception-v3
      - `inception_v3 <inception/inception_v3>`_
      - :math:`(N,3,299,299)`
      - 30,817,392
      - 3.20B
      - ❌
    
    * - Inception-v4
      - `inception_v4 <inception/inception_v4>`_
      - :math:`(N,3,299,299)`
      - 40,586,984
      - 5.75B
      - ❌

Inception-ResNet
----------------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - Inception-ResNet-v1
      - `inception_resnet_v1 <inception_res/inception_resnet_v1>`_
      - :math:`(N,3,299,299)`
      - 22,739,128
      - 3.16B
      - ❌
    
    * - Inception-ResNet-v2
      - `inception_resnet_v2 <inception_res/inception_resnet_v2>`_
      - :math:`(N,3,299,299)`
      - 35,847,512
      - 4.54B
      - ❌

ResNet
------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained

    * - ResNet-18
      - `resnet_18 <resnet/resnet_18>`_
      - :math:`(N,3,224,224)`
      - 11,689,512
      - 1.84B
      - ✅
    
    * - ResNet-34
      - `resnet_34 <resnet/resnet_34>`_
      - :math:`(N,3,224,224)`
      - 21,797,672
      - 3.70B
      - ✅
    
    * - ResNet-50
      - `resnet_50 <resnet/resnet_50>`_
      - :math:`(N,3,224,224)`
      - 25,557,032
      - 4.20B
      - ✅
    
    * - ResNet-101
      - `resnet_101 <resnet/resnet_101>`_
      - :math:`(N,3,224,224)`
      - 44,549,160
      - 7.97B
      - ✅
    
    * - ResNet-152
      - `resnet_152 <resnet/resnet_152>`_
      - :math:`(N,3,224,224)`
      - 60,192,808
      - 11.75B
      - ✅
    
    * - ResNet-200
      - `resnet_200 <resnet/resnet_200>`_
      - :math:`(N,3,224,224)`
      - 64,669,864
      - 15.35B
      - ❌
    
    * - ResNet-269
      - `resnet_269 <resnet/resnet_269>`_
      - :math:`(N,3,224,224)`
      - 102,069,416
      - 20.46B
      - ❌
    
    * - ResNet-1001
      - `resnet_1001 <resnet/resnet_1001>`_
      - :math:`(N,3,224,224)`
      - 149,071,016
      - 43.94B
      - ❌

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - Wide-ResNet-50
      - `wide_resnet_50 <resnet/wide_resnet_50>`_
      - :math:`(N,3,224,224)`
      - 78,973,224
      - 11.55B
      - ✅
    
    * - Wide-ResNet-101
      - `wide_resnet_101 <resnet/wide_resnet_101>`_
      - :math:`(N,3,224,224)`
      - 126,886,696
      - 22.97B
      - ✅

ResNeXt
-------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - ResNeXt-50-32x4d
      - `resnext_50_32x4d <resnext/resnext_50_32x4d>`_
      - :math:`(N,3,224,224)`
      - 25,028,904
      - 4.38B
      - ❌
    
    * - ResNeXt-101-32x4d
      - `resnext_101_32x4d <resnext/resnext_101_32x4d>`_
      - :math:`(N,3,224,224)`
      - 44,177,704
      - 8.19B
      - ❌
    
    * - ResNeXt-101-32x8d
      - `resnext_101_32x8d <resnext/resnext_101_32x8d>`_
      - :math:`(N,3,224,224)`
      - 88,791,336
      - 16.73B
      - ❌
    
    * - ResNeXt-101-32x16d
      - `resnext_101_32x16d <resnext/resnext_101_32x16d>`_
      - :math:`(N,3,224,224)`
      - 194,026,792
      - 36.68B
      - ❌
    
    * - ResNeXt-101-32x32d
      - `resnext_101_32x32d <resnext/resnext_101_32x32d>`_
      - :math:`(N,3,224,224)`
      - 468,530,472
      - 88.03B
      - ❌
    
    * - ResNeXt-101-64x4d
      - `resnext_101_64x4d <resnext/resnext_101_64x4d>`_
      - :math:`(N,3,224,224)`
      - 83,455,272
      - 15.78B
      - ❌

SENet
-----
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained

    * - SE-ResNet-18
      - `se_resnet_18 <senet/se_resnet_18>`_
      - :math:`(N,3,224,224)`
      - 11,778,592
      - 1.84B
      - ❌
    
    * - SE-ResNet-34
      - `se_resnet_34 <senet/se_resnet_34>`_
      - :math:`(N,3,224,224)`
      - 21,958,868
      - 3.71B
      - ❌
    
    * - SE-ResNet-50
      - `se_resnet_50 <senet/se_resnet_50>`_
      - :math:`(N,3,224,224)`
      - 28,088,024
      - 4.22B
      - ❌
    
    * - SE-ResNet-101
      - `se_resnet_101 <senet/se_resnet_101>`_
      - :math:`(N,3,224,224)`
      - 49,326,872
      - 8.00B
      - ❌
    
    * - SE-ResNet-152
      - `se_resnet_152 <senet/se_resnet_152>`_
      - :math:`(N,3,224,224)`
      - 66,821,848
      - 11.80B
      - ❌

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - SE-ResNeXt-50-32x4d
      - `se_resnext_50_32x4d <senet/se_resnext_50_32x4d>`_
      - :math:`(N,3,224,224)`
      - 27,559,896
      - 4.40B
      - ❌
    
    * - SE-ResNeXt-101-32x4d
      - `se_resnext_101_32x4d <senet/se_resnext_101_32x4d>`_
      - :math:`(N,3,224,224)`
      - 48,955,416
      - 8.22B
      - ❌
    
    * - SE-ResNeXt-101-32x8d
      - `se_resnext_101_32x8d <senet/se_resnext_101_32x8d>`_
      - :math:`(N,3,224,224)`
      - 93,569,048
      - 16.77B
      - ❌
    
    * - SE-ResNeXt-101-64x4d
      - `se_resnext_101_64x4d <senet/se_resnext_101_64x4d>`_
      - :math:`(N,3,224,224)`
      - 88,232,984
      - 15.81B
      - ❌

SKNet
-----
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - SK-ResNet-18
      - `sk_resnet_18 <sknet/sk_resnet_18>`_
      - :math:`(N,3,224,224)`
      - 25,647,368
      - 3.92B
      - ❌
    
    * - SK-ResNet-34
      - `sk_resnet_34 <sknet/sk_resnet_34>`_
      - :math:`(N,3,224,224)`
      - 45,895,512
      - 7.64B
      - ❌
    
    * - SK-ResNet-50
      - `sk_resnet_50 <sknet/sk_resnet_50>`_
      - :math:`(N,3,224,224)`
      - 57,073,368
      - 9.35B
      - ❌
    
.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained

    * - SK-ResNeXt-50-32x4d
      - `sk_resnext_50_32x4d <sknet/sk_resnext_50_32x4d>`_
      - :math:`(N,3,224,224)`
      - 29,274,760
      - 5.04B
      - ❌

DenseNet
--------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - DenseNet-121
      - `densenet_121 <dense/densenet_121>`_
      - :math:`(N,3,224,224)`
      - 7,978,856
      - 2.99B
      - ❌
    
    * - DenseNet-169
      - `densenet_169 <dense/densenet_169>`_
      - :math:`(N,3,224,224)`
      - 14,149,480
      - 3.55B
      - ❌
    
    * - DenseNet-201
      - `densenet_201 <dense/densenet_201>`_
      - :math:`(N,3,224,224)`
      - 20,013,928
      - 4.54B
      - ❌
    
    * - DenseNet-264
      - `densenet_264 <dense/densenet_264>`_
      - :math:`(N,3,224,224)`
      - 33,337,704
      - 6.09B
      - ❌

Xception
--------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - Xception
      - `xception <xception/xception>`_
      - :math:`(N,3,224,224)`
      - 22,862,096
      - 4.67B
      - ❌

MobileNet
---------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - MobileNet
      - `mobilenet <mobile/mobilenet>`_
      - :math:`(N,3,224,224)`
      - 4,232,008
      - 584.08M
      - ❌
    
    * - MobileNet-v2
      - `mobilenet_v2 <mobile/mobilenet_v2>`_
      - :math:`(N,3,224,224)`
      - 3,504,872
      - 367.39M
      - ❌
    
    * - MobileNet-v3-Small
      - `mobilenet_v3_small <mobile/mobilenet_v3_small>`_
      - :math:`(N,3,224,224)`
      - 2,537,238
      - 73.88M
      - ❌
    
    * - MobileNet-v3-Large
      - `mobilenet_v3_large <mobile/mobilenet_v3_large>`_
      - :math:`(N,3,224,224)`
      - 5,481,198
      - 266.91M
      - ❌

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - MobileNet-v4-Conv-Small
      - `mobilenet_v4_conv_small <mobile/mobilenet_v4_conv_small>`_
      - :math:`(N,3,224,224)`
      - 3,774,024
      - 265.15M
      - ❌
    
    * - MobileNet-v4-Conv-Medium
      - `mobilenet_v4_conv_medium <mobile/mobilenet_v4_conv_medium>`_
      - :math:`(N,3,224,224)`
      - 9,715,512
      - 944.48M
      - ❌
    
    * - MobileNet-v4-Conv-Large
      - `mobilenet_v4_conv_large <mobile/mobilenet_v4_conv_large>`_
      - :math:`(N,3,224,224)`
      - 32,590,864
      - 2.32B
      - ❌
    
    * - MobileNet-v4-Hybrid-Medium
      - `mobilenet_v4_hybrid_medium <mobile/mobilenet_v4_hybrid_medium>`_
      - :math:`(N,3,224,224)`
      - 11,070,136
      - 1.09B
      - ❌
    
    * - MobileNet-v4-Hybrid-Large
      - `mobilenet_v4_hybrid_large <mobile/mobilenet_v4_hybrid_large>`_
      - :math:`(N,3,224,224)`
      - 37,755,152
      - 2.72B
      - ❌

EfficientNet
------------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - EfficientNet-B0
      - `efficientnet_b0 <efficient/efficientnet_b0>`_
      - :math:`(N,3,224,224)`
      - 5,289,636
      - 463.32M
      - ❌
    
    * - EfficientNet-B1
      - `efficientnet_b1 <efficient/efficientnet_b1>`_
      - :math:`(N,3,240,240)`
      - 7,795,560
      - 849.06M
      - ❌
    
    * - EfficientNet-B2
      - `efficientnet_b2 <efficient/efficientnet_b2>`_
      - :math:`(N,3,260,260)`
      - 9,111,370
      - 1.20B
      - ❌
    
    * - EfficientNet-B3
      - `efficientnet_b3 <efficient/efficientnet_b3>`_
      - :math:`(N,3,300,300)`
      - 12,235,536
      - 2.01B
      - ❌
    
    * - EfficientNet-B4
      - `efficientnet_b4 <efficient/efficientnet_b4>`_
      - :math:`(N,3,380,380)`
      - 19,344,640
      - 4.63B
      - ❌
    
    * - EfficientNet-B5
      - `efficientnet_b5 <efficient/efficientnet_b5>`_
      - :math:`(N,3,456,456)`
      - 30,393,432
      - 12.17B
      - ❌
    
    * - EfficientNet-B6
      - `efficientnet_b6 <efficient/efficientnet_b6>`_
      - :math:`(N,3,528,528)`
      - 43,046,128
      - 21.34B
      - ❌
    
    * - EfficientNet-B7
      - `efficientnet_b7 <efficient/efficientnet_b7>`_
      - :math:`(N,3,600,600)`
      - 66,355,448
      - 40.31B
      - ❌

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - EfficientNet-v2-S
      - `efficientnet_v2_s <efficient/efficientnet_v2_s>`_
      - :math:`(N,3,224,224)`
      - 21,136,440
      - 789.91M
      - ❌
    
    * - EfficientNet-v2-M
      - `efficientnet_v2_m <efficient/efficientnet_v2_m>`_
      - :math:`(N,3,224,224)`
      - 55,302,108
      - 1.42B
      - ❌
    
    * - EfficientNet-v2-L
      - `efficientnet_v2_l <efficient/efficientnet_v2_l>`_
      - :math:`(N,3,224,224)`
      - 120,617,032
      - 3.17B
      - ❌
    
    * - EfficientNet-v2-XL
      - `efficientnet_v2_xl <efficient/efficientnet_v2_xl>`_
      - :math:`(N,3,224,224)`
      - 210,221,568
      - 4.12B
      - ❌

ResNeSt
-------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - ResNeSt-14
      - `resnest_14 <resnest/resnest_14>`_
      - :math:`(N,3,224,224)`
      - 10,611,560
      - 2.82B
      - ❌
    
    * - ResNeSt-26
      - `resnest_26 <resnest/resnest_26>`_
      - :math:`(N,3,224,224)`
      - 17,069,320
      - 3.72B
      - ❌
    
    * - ResNeSt-50
      - `resnest_50 <resnest/resnest_50>`_
      - :math:`(N,3,224,224)`
      - 27,483,112
      - 5.52B
      - ❌
    
    * - ResNeSt-101
      - `resnest_101 <resnest/resnest_101>`_
      - :math:`(N,3,224,224)`
      - 48,274,760
      - 10.43B
      - ❌
    
    * - ResNeSt-200
      - `resnest_200 <resnest/resnest_200>`_
      - :math:`(N,3,224,224)`
      - 70,201,288
      - 17.85B
      - ❌
    
    * - ResNeSt-269
      - `resnest_269 <resnest/resnest_269>`_
      - :math:`(N,3,224,224)`
      - 110,929,224
      - 22.98B
      - ❌
    
    * - ResNeSt-50-4s2x40d
      - `resnest_50_4s2x40d <resnest/resnest_50_4s2x40d>`_
      - :math:`(N,3,224,224)`
      - 30,417,464
      - 5.41B
      - ❌
    
    * - ResNeSt-50_1s4x24d
      - `resnest_50_1s4x24d <resnest/resnest_50_1s4x24d>`_
      - :math:`(N,3,224,224)`
      - 25,676,872
      - 5.14B
      - ❌

ConvNeXt
--------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - ConvNeXt-Tiny
      - `convnext_tiny <convnext/convnext_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,589,128
      - 4.73B
      - ❌
    
    * - ConvNeXt-Small
      - `convnext_small <convnext/convnext_small>`_
      - :math:`(N,3,224,224)`
      - 46,884,148
      - 8.46B
      - ❌
    
    * - ConvNeXt-Base
      - `convnext_base <convnext/convnext_base>`_
      - :math:`(N,3,224,224)`
      - 88,591,464
      - 15.93B
      - ❌
    
    * - ConvNeXt-Large
      - `convnext_large <convnext/convnext_large>`_
      - :math:`(N,3,224,224)`
      - 197,767,336
      - 35.23B
      - ❌
    
    * - ConvNeXt-XLarge
      - `convnext_xlarge <convnext/convnext_xlarge>`_
      - :math:`(N,3,224,224)`
      - 350,196,968
      - 62.08B
      - ❌

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - ConvNeXt-v2-Atto
      - `convnext_v2_atto <convnext/convnext_v2_atto>`_
      - :math:`(N,3,224,224)`
      - 3,708,400
      - 641.87M
      - ❌
    
    * - ConvNeXt-v2-Femto
      - `convnext_v2_femto <convnext/convnext_v2_femto>`_
      - :math:`(N,3,224,224)`
      - 5,233,240
      - 893.05M
      - ❌
    
    * - ConvNeXt-v2-Pico
      - `convnext_v2_pico <convnext/convnext_v2_pico>`_
      - :math:`(N,3,224,224)`
      - 9,066,280
      - 1.52B
      - ❌
    
    * - ConvNeXt-v2-Nano
      - `convnext_v2_nano <convnext/convnext_v2_nano>`_
      - :math:`(N,3,224,224)`
      - 15,623,800
      - 2.65B
      - ❌
    
    * - ConvNeXt-v2-Tiny
      - `convnext_v2_tiny <convnext/convnext_v2_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,635,496
      - 4.79B
      - ❌
    
    * - ConvNeXt-v2-Base
      - `convnext_v2_base <convnext/convnext_v2_base>`_
      - :math:`(N,3,224,224)`
      - 88,717,800
      - 16.08B
      - ❌
    
    * - ConvNeXt-v2-Large
      - `convnext_v2_large <convnext/convnext_v2_large>`_
      - :math:`(N,3,224,224)`
      - 197,956,840
      - 35.64B
      - ❌
    
    * - ConvNeXt-v2-Huge
      - `convnext_v2_huge <convnext/convnext_v2_huge>`_
      - :math:`(N,3,224,224)`
      - 660,289,640
      - 120.89B
      - ❌

InceptionNeXt
-------------
|convnet-badge| |imgclf-badge|

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
      - FLOPs
      - Pre-Trained
    
    * - InceptionNeXt-Atto
      - `inception_next_atto <inception_next/inception_next_atto>`_
      - :math:`(N,3,224,224)`
      - 4,156,520
      - 582.25M
      - ❌
    
    * - InceptionNeXt-Tiny
      - `inception_next_tiny <inception_next/inception_next_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,083,832
      - 4.48B
      - ❌
    
    * - InceptionNeXt-Small
      - `inception_next_small <inception_next/inception_next_small>`_
      - :math:`(N,3,224,224)`
      - 49,431,544
      - 8.82B
      - ❌
    
    * - InceptionNeXt-Base
      - `inception_next_base <inception_next/inception_next_base>`_
      - :math:`(N,3,224,224)`
      - 86,748,840
      - 15.47B
      - ❌

CoAtNet
-------
|convnet-badge| |imgclf-badge|

CoAtNet extends the hybrid architecture paradigm by integrating convolutional 
and transformer-based designs. It enhances representation learning through 
hierarchical feature extraction, leveraging early-stage depthwise convolutions 
for locality and later-stage self-attention for global context. With relative 
position encoding, pre-normalization, and an optimized scaling strategy, 
CoAtNet achieves superior efficiency and performance across various vision tasks.

 Dai, Zihang, et al. "CoAtNet: Marrying Convolution and Attention for All Data Sizes." 
 *Advances in Neural Information Processing Systems*, 2021, pp. 3965-3977.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - CoAtNet-0
      - `coatnet_0 <coatnet/coatnet_0>`_
      - :math:`(N,3,224,224)`
      - 27,174,944
      - 5.52B
      - ❌
    
    * - CoAtNet-1
      - `coatnet_1 <coatnet/coatnet_1>`_
      - :math:`(N,3,224,224)`
      - 53,330,240
      - 12.32B
      - ❌
    
    * - CoAtNet-2
      - `coatnet_2 <coatnet/coatnet_2>`_
      - :math:`(N,3,224,224)`
      - 82,516,096
      - 19.72B
      - ❌
    
    * - CoAtNet-3
      - `coatnet_3 <coatnet/coatnet_3>`_
      - :math:`(N,3,224,224)`
      - 157,790,656
      - 37.17B
      - ❌
    
    * - CoAtNet-4
      - `coatnet_4 <coatnet/coatnet_4>`_
      - :math:`(N,3,224,224)`
      - 277,301,632
      - 66.79B
      - ❌
    
    * - CoAtNet-5
      - `coatnet_5 <coatnet/coatnet_5>`_
      - :math:`(N,3,224,224)`
      - 770,124,608
      - 189.34B
      - ❌
    
    * - CoAtNet-6
      - `coatnet_6 <coatnet/coatnet_6>`_
      - :math:`(N,3,224,224)`
      - 2,011,558,336
      - 293.51B
      - ❌
    
    * - CoAtNet-7
      - `coatnet_7 <coatnet/coatnet_7>`_
      - :math:`(N,3,224,224)`
      - 3,107,978,688
      - 364.71B
      - ❌

Visual Transformer (ViT)
------------------------
|transformer-badge| |vision-transformer-badge| |imgclf-badge|

The Vision Transformer (ViT) is a deep learning architecture introduced by 
Dosovitskiy et al. in 2020, designed for image recognition tasks using self-attention 
mechanisms. Unlike traditional convolutional neural networks (CNNs), ViT splits an 
image into fixed-size patches, processes them as a sequence, and applies Transformer 
layers to capture global dependencies.

 Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for 
 Image Recognition at Scale." *International Conference on Learning Representations* 
 (ICLR), 2020.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - ViT-Ti
      - `vit_tiny <vit/vit_tiny>`_
      - :math:`(N,3,224,224)`
      - 5,717,416
      - 1.36B
      - ❌
    
    * - ViT-S
      - `vit_small <vit/vit_small>`_
      - :math:`(N,3,224,224)`
      - 22,050,664
      - 4.81B
      - ❌
    
    * - ViT-B
      - `vit_base <vit/vit_base>`_
      - :math:`(N,3,224,224)`
      - 86,567,656
      - 17.99B
      - ❌
    
    * - ViT-L
      - `vit_large <vit/vit_large>`_
      - :math:`(N,3,224,224)`
      - 304,326,632
      - 62.69B
      - ❌
    
    * - ViT-H
      - `vit_huge <vit/vit_huge>`_
      - :math:`(N,3,224,224)`
      - 632,199,400
      - 169.45B
      - ❌

Swin Transformer
----------------
|transformer-badge| |vision-transformer-badge| |imgclf-badge|

The Swin Transformer is a hierarchical vision transformer introduced by 
Liu et al. in 2021, designed for image recognition and dense prediction 
tasks using self-attention mechanisms within shifted local windows. 
Unlike traditional convolutional neural networks (CNNs) and the original 
Vision Transformer (ViT)—which splits an image into fixed-size patches 
and processes them as a flat sequence—the Swin Transformer divides the 
image into non-overlapping local windows and computes self-attention 
within each window.

 *Swin Transformer*

 Liu, Ze, et al. "Swin Transformer: Hierarchical Vision Transformer using 
 Shifted Windows." arXiv preprint arXiv:2103.14030 (2021).

 *Swin Transformer-v2*

 Liu, Ze, et al. "Swin Transformer V2: Scaling Up Capacity and Resolution." 
 arXiv preprint arXiv:2111.09883 (2021).

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - Swin-T
      - `swin_tiny <swin/swin_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,288,354
      - 4.95B
      - ❌
    
    * - Swin-S
      - `swin_small <swin/swin_small>`_
      - :math:`(N,3,224,224)`
      - 49,606,258
      - 9.37B
      - ❌
    
    * - Swin-B
      - `swin_base <swin/swin_base>`_
      - :math:`(N,3,224,224)`
      - 87,768,224
      - 16.35B
      - ❌
    
    * - Swin-L
      - `swin_large <swin/swin_large>`_
      - :math:`(N,3,224,224)`
      - 196,532,476
      - 36.08B
      - ❌

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - Swin-v2-T
      - `swin_v2_tiny <swin/swin_v2_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,349,842
      - 5.01B
      - ❌
    
    * - Swin-v2-S
      - `swin_v2_small <swin/swin_v2_small>`_
      - :math:`(N,3,224,224)`
      - 49,731,106
      - 9.48B
      - ❌
    
    * - Swin-v2-B
      - `swin_v2_base <swin/swin_v2_base>`_
      - :math:`(N,3,224,224)`
      - 87,922,400
      - 16.49B
      - ❌
    
    * - Swin-v2-L
      - `swin_v2_large <swin/swin_v2_large>`_
      - :math:`(N,3,224,224)`
      - 196,745,308
      - 36.29B
      - ❌
    
    * - Swin-v2-H
      - `swin_v2_huge <swin/swin_v2_huge>`_
      - :math:`(N,3,224,224)`
      - 657,796,668
      - 119.42B
      - ❌
    
    * - Swin-v2-G
      - `swin_v2_giant <swin/swin_v2_giant>`_
      - :math:`(N,3,224,224)`
      - 3,000,869,564
      - 531.67B
      - ❌

Convolutional Transformer (CvT)
-------------------------------
|transformer-badge| |vision-transformer-badge| |imgclf-badge|

CvT (Convolutional Vision Transformer) combines self-attention with depthwise 
convolutions to improve local feature extraction and computational efficiency. 
This hybrid design retains the global modeling capabilities of Vision Transformers 
while enhancing inductive biases, making it effective for image classification and 
dense prediction tasks.

 Wu, Haiping, et al. "CvT: Introducing Convolutions to Vision Transformers." 
 *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 
 2021, pp. 22-31.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - CvT-13
      - `cvt_13 <cvt/cvt_13>`_
      - :math:`(N,3,224,224)`
      - 19,997,480
      - 4.83B
      - ❌
    
    * - CvT-21
      - `cvt_21 <cvt/cvt_21>`_
      - :math:`(N,3,224,224)`
      - 31,622,696
      - 7.57B
      - ❌
    
    * - CvT-W24
      - `cvt_w24 <cvt/cvt_w24>`_
      - :math:`(N,3,384,384)`
      - 277,196,392
      - 62.29B
      - ❌

Pyramid Vision Transformer (PVT)
--------------------------------
|transformer-badge| |vision-transformer-badge| |imgclf-badge|

The **Pyramid Vision Transformer (PVT)** combines CNN-like pyramidal structures 
with Transformer attention, capturing multi-scale features efficiently. It reduces 
spatial resolution progressively and uses **spatial-reduction attention (SRA)** 
to enhance performance in dense prediction tasks like detection and segmentation.

 *PVT*

 Wang, Wenhai, et al. Pyramid Vision Transformer: A Versatile Backbone for Dense 
 Prediction without Convolutions. arXiv, 2021, arXiv:2102.12122.

 *PVT-v2*

 Wang, Wenhai, et al. PVTv2: Improved baselines with pyramid vision transformer.
 *Computational Visual Media* 8.3 (2022): 415-424.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - PVT-Tiny
      - `pvt_tiny <pvt/pvt_tiny>`_
      - :math:`(N,3,224,224)`
      - 12,457,192
      - 2.02B
      - ❌
    
    * - PVT-Small
      - `pvt_small <pvt/pvt_small>`_
      - :math:`(N,3,224,224)`
      - 23,003,048
      - 3.93B
      - ❌
    
    * - PVT-Medium
      - `pvt_medium <pvt/pvt_medium>`_
      - :math:`(N,3,224,224)`
      - 41,492,648
      - 6.66B
      - ❌
    
    * - PVT-Large
      - `pvt_large <pvt/pvt_large>`_
      - :math:`(N,3,224,224)`
      - 55,359,848
      - 8.71B
      - ❌
    
    * - PVT-Huge
      - `pvt_huge <pvt/pvt_huge>`_
      - :math:`(N,3,224,224)`
      - 286,706,920
      - 48.63B
      - ❌

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - PVT-v2-B0
      - `pvt_v2_b0 <pvt/pvt_v2_b0>`_
      - :math:`(N,3,224,224)`
      - 3,666,760
      - 677.67M
      - ❌
    
    * - PVT-v2-B1
      - `pvt_v2_b1 <pvt/pvt_v2_b1>`_
      - :math:`(N,3,224,224)`
      - 14,009,000
      - 2.32B
      - ❌
    
    * - PVT-v2-B2
      - `pvt_v2_b2 <pvt/pvt_v2_b2>`_
      - :math:`(N,3,224,224)`
      - 25,362,856
      - 4.39B
      - ❌
    
    * - PVT-v2-B2-Linear
      - `pvt_v2_b2_li <pvt/pvt_v2_b2_li>`_
      - :math:`(N,3,224,224)`
      - 22,553,512
      - 4.27B
      - ❌
    
    * - PVT-v2-B3
      - `pvt_v2_b3 <pvt/pvt_v2_b3>`_
      - :math:`(N,3,224,224)`
      - 45,238,696
      - 7.39B
      - ❌
    
    * - PVT-v2-B4
      - `pvt_v2_b4 <pvt/pvt_v2_b4>`_
      - :math:`(N,3,224,224)`
      - 62,556,072
      - 10.80B
      - ❌
    
    * - PVT-v2-B5
      - `pvt_v2_b5 <pvt/pvt_v2_b5>`_
      - :math:`(N,3,224,224)`
      - 82,882,984
      - 13.47B
      - ❌

CrossViT
--------
|transformer-badge| |vision-transformer-badge| |imgclf-badge|

CrossViT is a vision transformer architecture that combines multi-scale 
tokenization by processing input images at different resolutions in parallel, 
enabling it to capture both fine-grained and coarse-grained visual features. 
It uses a novel cross-attention mechanism to fuse information across these scales, 
improving performance on image recognition tasks.

 Chen, Chun-Fu, Quanfu Fan, and Rameswar Panda. CrossViT: Cross-Attention Multi-Scale 
 Vision Transformer for Image Classification. arXiv, 2021. arXiv:2103.14899.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - CrossViT-Ti
      - `crossvit_tiny <crossvit/crossvit_tiny>`_
      - :math:`(N,3,224,224)`
      - 7,014,800
      - 1.73B
      - ❌
    
    * - CrossViT-S
      - `crossvit_small <crossvit/crossvit_small>`_
      - :math:`(N,3,224,224)`
      - 26,856,272
      - 5.94B
      - ❌
    
    * - CrossViT-B
      - `crossvit_base <crossvit/crossvit_base>`_
      - :math:`(N,3,224,224)`
      - 105,025,232
      - 21.85B
      - ❌
    
    * - CrossViT-9
      - `crossvit_9 <crossvit/crossvit_9>`_
      - :math:`(N,3,224,224)`
      - 8,553,296
      - 2.01B
      - ❌
    
    * - CrossViT-15
      - `crossvit_15 <crossvit/crossvit_15>`_
      - :math:`(N,3,224,224)`
      - 27,528,464
      - 6.13B
      - ❌
    
    * - CrossViT-18
      - `crossvit_18 <crossvit/crossvit_18>`_
      - :math:`(N,3,224,224)`
      - 43,271,408
      - 9.48B
      - ❌
    
    * - CrossViT-9†
      - `crossvit_9_dagger <crossvit/crossvit_9_dagger>`_
      - :math:`(N,3,224,224)`
      - 8,776,592
      - 2.15B
      - ❌
    
    * - CrossViT-15†
      - `crossvit_15_dagger <crossvit/crossvit_15_dagger>`_
      - :math:`(N,3,224,224)`
      - 28,209,008
      - 6.45B
      - ❌
    
    * - CrossViT-18†
      - `crossvit_18_dagger <crossvit/crossvit_18_dagger>`_
      - :math:`(N,3,224,224)`
      - 44,266,976
      - 9.93B
      - ❌

MaxViT
------
|transformer-badge| |vision-transformer-badge| |imgclf-badge|

MaxViT is a hybrid vision architecture that combines convolution, windowed attention, 
and grid-based attention in a multi-axis design. This hierarchical structure enables 
MaxViT to efficiently capture both local and global dependencies, making it effective 
for various vision tasks with high accuracy and scalability.

 Tu, Zihang, et al. *MaxViT: Multi-Axis Vision Transformer*. arXiv, 2022, arXiv:2204.01697.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - MaxViT-T
      - `maxvit_tiny <maxvit/maxvit_tiny>`_
      - :math:`(N,3,224,224)`
      - 25,081,416
      - 5.60B
      - ❌
    
    * - MaxViT-S
      - `maxvit_small <maxvit/maxvit_small>`_
      - :math:`(N,3,224,224)`
      - 55,757,304
      - 10.59B
      - ❌
    
    * - MaxViT-B
      - `maxvit_base <maxvit/maxvit_base>`_
      - :math:`(N,3,224,224)`
      - 96,626,776
      - 21.83B
      - ❌
    
    * - MaxViT-L
      - `maxvit_large <maxvit/maxvit_large>`_
      - :math:`(N,3,224,224)`
      - 171,187,880
      - 38.51B
      - ❌
    
    * - MaxViT-XL
      - `maxvit_xlarge <maxvit/maxvit_xlarge>`_
      - :math:`(N,3,224,224)`
      - 383,734,024
      - 83.74B
      - ❌

EfficientFormer
---------------
|transformer-badge| |vision-transformer-badge| |imgclf-badge|

EfficientFormer is a lightweight and efficient vision transformer architecture designed 
for mobile and edge devices. By combining the strengths of convolutional inductive biases 
with self-attention in a hybrid structure, EfficientFormer achieves a strong balance 
between accuracy and computational efficiency.

 Li, Yanyu, et al. EfficientFormer: Vision Transformers at MobileNet Speed. 
 arXiv, 2022, arXiv:2206.01191.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
      - Pre-Trained
    
    * - EfficientFormer-L1
      - `efficientformer_l1 <efficientformer/efficientformer_l1>`_
      - :math:`(N,3,224,224)`
      - 11,840,928
      - 316.47M
      - ❌
    
    * - EfficientFormer-L3
      - `efficientformer_l3 <efficientformer/efficientformer_l3>`_
      - :math:`(N,3,224,224)`
      - 30,893,000
      - 1.07B
      - ❌

    * - EfficientFormer-L7
      - `efficientformer_l7 <efficientformer/efficientformer_l7>`_
      - :math:`(N,3,224,224)`
      - 81,460,328
      - 3.44B
      - ❌

*To be implemented...🔮*
