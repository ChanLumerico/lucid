Transformers
============

.. toctree::
    :maxdepth: 1
    :hidden:

    Transformer <base/Transformer.rst>
    ViT <vit/ViT.rst>
    Swin Transformer <swin/SwinTransformer.rst>
    Swin Transformer-v2 <swin/SwinTransformer_V2.rst>
    CvT <cvt/CvT.rst>
    PVT <pvt/PVT.rst>
    PVT-v2 <pvt/PVT_V2.rst>
    CrossViT <cross_vit/CrossViT.rst>

Transformer
-----------

.. versionadded:: 1.18

The Transformer is a deep learning architecture introduced by Vaswani et al. in 2017, 
designed for handling sequential data with self-attention mechanisms. It replaces 
traditional recurrent layers with attention-based mechanisms, enabling highly 
parallelized training and capturing long-range dependencies effectively.

 Vaswani, Ashish, et al. "Attention Is All You Need." 
 *Advances in Neural Information Processing Systems*, 2017, pp. 5998-6008.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
    
    * - Transformer-Base
      - `transformer_base <base/transformer_base>`_
      - :math:`(N, L_{src})`, :math:`(N, L_{tgt})`
      - 62,584,544
      - 
    
    * - Transformer-Big
      - `transformer_big <base/transformer_big>`_
      - :math:`(N, L_{src})`, :math:`(N, L_{tgt})`
      - 213,237,472
      - 

Visual Transformer (ViT)
------------------------

.. versionadded:: 1.19

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
    
    * - ViT-Ti
      - `vit_tiny <vit/vit_tiny>`_
      - :math:`(N,3,224,224)`
      - 5,717,416
      - 
    
    * - ViT-S
      - `vit_small <vit/vit_small>`_
      - :math:`(N,3,224,224)`
      - 22,050,664
      - 
    
    * - ViT-B
      - `vit_base <vit/vit_base>`_
      - :math:`(N,3,224,224)`
      - 86,567,656
      - 
    
    * - ViT-L
      - `vit_large <vit/vit_large>`_
      - :math:`(N,3,224,224)`
      - 304,326,632
      - 
    
    * - ViT-H
      - `vit_huge <vit/vit_huge>`_
      - :math:`(N,3,224,224)`
      - 632,199,400
      - 

Swin Transformer
----------------

.. versionadded:: 1.20
    Swin Transformer

.. versionadded:: 1.20.5
    Swin Transformer-v2

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
    
    * - Swin-T
      - `swin_tiny <swin/swin_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,288,354
      - 
    
    * - Swin-S
      - `swin_small <swin/swin_small>`_
      - :math:`(N,3,224,224)`
      - 49,606,258
      - 
    
    * - Swin-B
      - `swin_base <swin/swin_base>`_
      - :math:`(N,3,224,224)`
      - 87,768,224
      - 
    
    * - Swin-L
      - `swin_large <swin/swin_large>`_
      - :math:`(N,3,224,224)`
      - 196,532,476
      - 

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
    
    * - Swin-v2-T
      - `swin_v2_tiny <swin/swin_v2_tiny>`_
      - :math:`(N,3,224,224)`
      - 28,349,842
      - 
    
    * - Swin-v2-S
      - `swin_v2_small <swin/swin_v2_small>`_
      - :math:`(N,3,224,224)`
      - 49,731,106
      - 
    
    * - Swin-v2-B
      - `swin_v2_base <swin/swin_v2_base>`_
      - :math:`(N,3,224,224)`
      - 87,922,400
      - 
    
    * - Swin-v2-L
      - `swin_v2_large <swin/swin_v2_large>`_
      - :math:`(N,3,224,224)`
      - 196,745,308
      - 
    
    * - Swin-v2-H
      - `swin_v2_huge <swin/swin_v2_huge>`_
      - :math:`(N,3,224,224)`
      - 657,796,668
      - 
    
    * - Swin-v2-G
      - `swin_v2_giant <swin/swin_v2_giant>`_
      - :math:`(N,3,224,224)`
      - 3,000,869,564
      - 

Convolutional Transformer (CvT)
-------------------------------

.. versionadded:: 1.21

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
    
    * - CvT-13
      - `cvt_13 <cvt/cvt_13>`_
      - :math:`(N,3,224,224)`
      - 19,997,480
      - 
    
    * - CvT-21
      - `cvt_21 <cvt/cvt_21>`_
      - :math:`(N,3,224,224)`
      - 31,622,696
      - 
    
    * - CvT-W24
      - `cvt_w24 <cvt/cvt_w24>`_
      - :math:`(N,3,384,384)`
      - 277,196,392
      - 

Pyramid Vision Transformer (PVT)
--------------------------------

.. versionadded:: 1.22.5
    PVT

.. versionadded:: 2.0.7
    PVT-v2

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
    
    * - PVT-Tiny
      - `pvt_tiny <pvt/pvt_tiny>`_
      - :math:`(N,3,224,224)`
      - 12,457,192
      - 
    
    * - PVT-Small
      - `pvt_small <pvt/pvt_small>`_
      - :math:`(N,3,224,224)`
      - 23,003,048
      - 
    
    * - PVT-Medium
      - `pvt_medium <pvt/pvt_medium>`_
      - :math:`(N,3,224,224)`
      - 41,492,648
      - 
    
    * - PVT-Large
      - `pvt_large <pvt/pvt_large>`_
      - :math:`(N,3,224,224)`
      - 55,359,848
      - 
    
    * - PVT-Huge
      - `pvt_huge <pvt/pvt_huge>`_
      - :math:`(N,3,224,224)`
      - 286,706,920
      - 

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
    
    * - PVT-v2-B0
      - `pvt_v2_b0 <pvt/pvt_v2_b0>`_
      - :math:`(N,3,224,224)`
      - 3,666,760
      - 
    
    * - PVT-v2-B1
      - `pvt_v2_b1 <pvt/pvt_v2_b1>`_
      - :math:`(N,3,224,224)`
      - 14,009,000
      - 
    
    * - PVT-v2-B2
      - `pvt_v2_b2 <pvt/pvt_v2_b2>`_
      - :math:`(N,3,224,224)`
      - 25,362,856
      - 
    
    * - PVT-v2-B2-Linear
      - `pvt_v2_b2_li <pvt/pvt_v2_b2_li>`_
      - :math:`(N,3,224,224)`
      - 22,553,512
      - 
    
    * - PVT-v2-B3
      - `pvt_v2_b3 <pvt/pvt_v2_b3>`_
      - :math:`(N,3,224,224)`
      - 45,238,696
      - 
    
    * - PVT-v2-B4
      - `pvt_v2_b4 <pvt/pvt_v2_b4>`_
      - :math:`(N,3,224,224)`
      - 62,556,072
      - 
    
    * - PVT-v2-B5
      - `pvt_v2_b5 <pvt/pvt_v2_b5>`_
      - :math:`(N,3,224,224)`
      - 82,882,984
      - 

CrossViT
--------

.. versionadded:: 2.0.17

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
    
    * - CrossViT-Ti
      - `crossvit_tiny <cross_vit/crossvit_tiny>`_
      - :math:`(N,3,224,224)`
      - 7,014,800
      - 
    
    * - CrossViT-S
      - `crossvit_small <cross_vit/crossvit_small>`_
      - :math:`(N,3,224,224)`
      - 26,856,272
      - 
    
    * - CrossViT-B
      - `crossvit_base <cross_vit/crossvit_base>`_
      - :math:`(N,3,224,224)`
      - 105,025,232
      - 
    
    * - CrossViT-9
      - `crossvit_9 <cross_vit/crossvit_9>`_
      - :math:`(N,3,224,224)`
      - 8,553,296
      - 
    
    * - CrossViT-15
      - `crossvit_15 <cross_vit/crossvit_15>`_
      - :math:`(N,3,224,224)`
      - 27,528,464
      - 
    
    * - CrossViT-18
      - `crossvit_18 <cross_vit/crossvit_18>`_
      - :math:`(N,3,224,224)`
      - 43,271,408
      - 
    
    * - CrossViT-9†
      - `crossvit_9_dagger <cross_vit/crossvit_9_dagger>`_
      - :math:`(N,3,224,224)`
      - 8,776,592
      - 
    
    * - CrossViT-15†
      - `crossvit_15_dagger <cross_vit/crossvit_15_dagger>`_
      - :math:`(N,3,224,224)`
      - 28,209,008
      - 
    
    * - CrossViT-18†
      - `crossvit_18_dagger <cross_vit/crossvit_18_dagger>`_
      - :math:`(N,3,224,224)`
      - 44,266,976
      - 

*To be FLOPs...🔮*
