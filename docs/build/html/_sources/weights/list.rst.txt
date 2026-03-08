Pre-Trained Weights
===================

LeNet
-----
|convnet-badge|

.. code-block:: python

    class lucid.weights.LeNet_1_Weights(Enum)
    class lucid.weights.LeNet_4_Weights(Enum)
    class lucid.weights.LeNet_5_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
    
    * - *LeNet_1_Weights*
      - MNIST
      - mnist
      - 3,246
      - 13.4 KB
      - 0.7937
    
    * - *LeNet_4_Weights*
      - MNIST
      - mnist
      - 18,378
      - 74.1 KB
      - 0.8210
    
    * - *LeNet_5_Weights*
      - MNIST
      - mnist
      - 61,706
      - 248 KB
      - 0.9981

AlexNet
-------
|convnet-badge|

.. code-block:: python

    class lucid.weights.AlexNet_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *AlexNet_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 61,100,840
      - 244 MB
      - 0.5652
      - 0.7906

VGGNet
------
|convnet-badge|

.. code-block:: python

    class lucid.weights.VGGNet_11_Weights(Enum)
    class lucid.weights.VGGNet_13_Weights(Enum)
    class lucid.weights.VGGNet_16_Weights(Enum)
    class lucid.weights.VGGNet_19_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *VGGNet_11_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 132,863,336
      - 531 MB
      - 0.6902
      - 0.8863

    * - *VGGNet_13_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 133,047,848
      - 532 MB
      - 0.6992
      - 0.8924
    
    * - *VGGNet_16_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 138,357,544
      - 553 MB
      - 0.7159
      - 0.9038
    
    * - *VGGNet_19_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 1436,67,240
      - 575 MB
      - 0.7237
      - 0.9087

ResNet
------
|convnet-badge|

.. code-block:: python

    class lucid.weights.ResNet_18_Weights(Enum)
    class lucid.weights.ResNet_34_Weights(Enum)
    class lucid.weights.ResNet_50_Weights(Enum)
    class lucid.weights.ResNet_101_Weights(Enum)
    class lucid.weights.ResNet_152_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *ResNet_18_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 11,689,512
      - 44.7 MB
      - 0.6976
      - 0.8907
    
    * - *ResNet_34_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 21,797,672
      - 83.3 MB
      - 0.7331
      - 0.9124
    
    * - *ResNet_50_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 25,557,032
      - 97.8 MB
      - 0.7613
      - 0.9286
    
    * - *ResNet_101_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 44,549,160
      - 170 MB
      - 0.7737
      - 0.9534
    
    * - *ResNet_152_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 60,192,808
      - 230 MB
      - 0.7831
      - 0.9600

.. code-block:: python

    class lucid.weights.Wide_ResNet_50_Weights(Enum)
    class lucid.weights.Wide_ResNet_101_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *Wide_ResNet_50_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 68,883,240
      - 131 MB
      - 0.7846
      - 0.9409
    
    * - *Wide_ResNet_101_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 126,886,696
      - 242 MB
      - 0.7885
      - 0.9248

ResNeXt
-------
|convnet-badge|

.. code-block:: python

    class lucid.weights.ResNeXt_50_32X4D_Weights(Enum)
    class lucid.weights.ResNeXt_101_32X8D_Weights(Enum)
    class lucid.weights.ResNeXt_101_64X4D_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *ResNeXt_50_32X4D_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 25,028,904
      - 95.7 MB
      - 0.7761
      - 0.9370
    
    * - *ResNeXt_101_32X8D_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 88,791,336
      - 339 MB
      - 0.7931
      - 0.9457
    
    * - *ResNeXt_101_64X4D_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 83,455,272
      - 319 MB
      - 0.8325
      - 0.9645

DenseNet
--------
|convnet-badge|

.. code-block:: python

    class lucid.weights.DenseNet_121_Weights(Enum)
    class lucid.weights.DenseNet_169_Weights(Enum)
    class lucid.weights.DenseNet_201_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *DenseNet_121_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 7,978,856
      - 30.8 MB
      - 0.7434
      - 0.9197
    
    * - *DenseNet_169_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 14,149,480
      - 54.7 MB
      - 0.7560
      - 0.9281
    
    * - *DenseNet_201_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 20,013,928
      - 77.3 MB
      - 0.7690
      - 0.9337

MobileNet
---------
|convnet-badge|

.. code-block:: python

    class lucid.weights.MobileNet_V2_Weights(Enum)
    class lucid.weights.MobileNet_V3_Small_Weights(Enum)
    class lucid.weights.MobileNet_V3_Large_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *MobileNet_V2_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 3,504,872
      - 13.5 MB
      - 0.7188
      - 0.9029
    
    * - *MobileNet_V3_Small_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 2,542,856
      - 9.82 MB
      - 0.6767
      - 0.8740
    
    * - *MobileNet_V3_Large_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 5,483,032
      - 21.11 MB
      - 0.7404
      - 0.9134

Swin Transformer
----------------
|transformer-badge| |vision-transformer-badge|

.. code-block:: python

    class lucid.weights.Swin_Tiny_Weights(Enum)
    class lucid.weights.Swin_Base_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *Swin_Tiny_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 28,288,354
      - 115 MB

    * - *Swin_Base_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 87,768,224
      - 354 MB

MaskFormer
----------
|transformer-badge| |segmentation-transformer-badge|

.. code-block:: python

    class lucid.weights.MaskFormer_ResNet_50_Weights(Enum)
    class lucid.weights.MaskFormer_ResNet_101_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *MaskFormer_ResNet_50_Weights*
      - ADE20K
      - ADE20K
      - 41,307,863
      - 166 MB

    * - *MaskFormer_ResNet_101_Weights*
      - ADE20K
      - ADE20K
      - 60,299,991
      - 242 MB

Mask2Former
-----------
|transformer-badge| |segmentation-transformer-badge|

.. code-block:: python

    class lucid.weights.Mask2Former_Swin_Tiny_Weights(Enum)
    class lucid.weights.Mask2Former_Swin_Small_Weights(Enum)
    class lucid.weights.Mask2Former_Swin_Base_Weights(Enum)
    class lucid.weights.Mask2Former_Swin_Large_Weights(Enum)
  
Mask2Former-Swin-Tiny
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *Mask2Former_Swin_Tiny_Weights*
      - ADE20K_SEMANTIC
      - ade20k_semantic
      - 47,439,633
      - 192 MB

    * - *Mask2Former_Swin_Tiny_Weights*
      - COCO_INSTANCE
      - coco_instance
      - 47,421,643
      - 192 MB

    * - *Mask2Former_Swin_Tiny_Weights*
      - COCO_PANOPTIC
      - coco_panoptic
      - 47,435,264
      - 192 MB

    * - *Mask2Former_Swin_Tiny_Weights*
      - CITYSCAPES_SEMANTIC
      - cityscapes_semantic
      - 47,405,966
      - 192 MB

    * - *Mask2Former_Swin_Tiny_Weights*
      - CITYSCAPES_INSTANCE
      - cityscapes_instance
      - 47,403,139
      - 192 MB

    * - *Mask2Former_Swin_Tiny_Weights*
      - CITYSCAPES_PANOPTIC
      - cityscapes_panoptic
      - 47,405,966
      - 192 MB

Mask2Former-Swin-Small
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *Mask2Former_Swin_Small_Weights*
      - ADE20K_SEMANTIC
      - ade20k_semantic
      - 68,757,537
      - 278 MB

    * - *Mask2Former_Swin_Small_Weights*
      - COCO_INSTANCE
      - coco_instance
      - 68,739,547
      - 278 MB

    * - *Mask2Former_Swin_Small_Weights*
      - COCO_PANOPTIC
      - coco_panoptic
      - 68,753,168
      - 278 MB

    * - *Mask2Former_Swin_Small_Weights*
      - CITYSCAPES_SEMANTIC
      - cityscapes_semantic
      - 68,723,870
      - 278 MB

    * - *Mask2Former_Swin_Small_Weights*
      - CITYSCAPES_INSTANCE
      - cityscapes_instance
      - 68,721,043
      - 278 MB

    * - *Mask2Former_Swin_Small_Weights*
      - CITYSCAPES_PANOPTIC
      - cityscapes_panoptic
      - 68,723,870
      - 278 MB

Mask2Former-Swin-Base
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *Mask2Former_Swin_Base_Weights*
      - ADE20K_SEMANTIC
      - ade20k_semantic
      - 106,922,191
      - 451 MB

    * - *Mask2Former_Swin_Base_Weights*
      - COCO_INSTANCE
      - coco_instance
      - 106,904,201
      - 451 MB

    * - *Mask2Former_Swin_Base_Weights*
      - COCO_PANOPTIC
      - coco_panoptic
      - 106,917,822
      - 451 MB

    * - *Mask2Former_Swin_Base_Weights*
      - CITYSCAPES_SEMANTIC
      - cityscapes_semantic
      - 106,888,524
      - 451 MB

    * - *Mask2Former_Swin_Base_Weights*
      - CITYSCAPES_INSTANCE
      - cityscapes_instance
      - 106,885,697
      - 451 MB

    * - *Mask2Former_Swin_Base_Weights*
      - CITYSCAPES_PANOPTIC
      - cityscapes_panoptic
      - 106,888,524
      - 451 MB

Mask2Former-Swin-Large
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *Mask2Former_Swin_Large_Weights*
      - ADE20K_SEMANTIC
      - ade20k_semantic
      - 215,488,779
      - 885 MB

    * - *Mask2Former_Swin_Large_Weights*
      - ADE20K_PANOPTIC
      - ade20k_panoptic
      - 215,539,979
      - 885 MB

    * - *Mask2Former_Swin_Large_Weights*
      - COCO_INSTANCE
      - coco_instance
      - 215,521,989
      - 885 MB

    * - *Mask2Former_Swin_Large_Weights*
      - COCO_PANOPTIC
      - coco_panoptic
      - 215,535,610
      - 885 MB

    * - *Mask2Former_Swin_Large_Weights*
      - CITYSCAPES_SEMANTIC
      - cityscapes_semantic
      - 215,455,112
      - 885 MB

    * - *Mask2Former_Swin_Large_Weights*
      - CITYSCAPES_INSTANCE
      - cityscapes_instance
      - 215,503,485
      - 885 MB

    * - *Mask2Former_Swin_Large_Weights*
      - CITYSCAPES_PANOPTIC
      - cityscapes_panoptic
      - 215,506,312
      - 885 MB

----

BERT
----
|transformer-badge| |encoder-only-transformer-badge|

.. code-block:: python

    class lucid.weights.BERT_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *BERT_Weights*
      - PRE_TRAIN_BASE
      - BookCorpus + English Wikipedia
      - 110,106,428
      - 628 MB

.. code-block:: python

    class lucid.weights.BERTForMaskedLM_Weights(Enum)
    class lucid.weights.BERTForCausalLM_Weights(Enum)
    class lucid.weights.BERTForNextSentencePrediction_Weights(Enum)
    class lucid.weights.BERTForSequenceClassification_Weights(Enum)
    class lucid.weights.BERTForTokenClassification_Weights(Enum)
    class lucid.weights.BERTForQuestionAnswering_Weights(Enum)

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size

    * - *BERTForMaskedLM_Weights*
      - HF_BASE_UNCASED
      - bert-base-uncased
      - 110,104,890
      - 599 MB

    * - *BERTForCausalLM_Weights*
      - HF_BASE_UNCASED
      - bert-base-uncased
      - 110,104,890
      - 599 MB

    * - *BERTForNextSentencePrediction_Weights*
      - HF_BASE_UNCASED
      - bert-base-uncased
      - 109,483,778
      - 417.7 MB

    * - *BERTForSequenceClassification_Weights*
      - SST2
      - GLUE SST-2
      - 109,483,778
      - 417.7 MB

    * - *BERTForTokenClassification_Weights*
      - CONLL03
      - CoNLL-2003
      - 108,317,193
      - 413.2 MB

    * - *BERTForQuestionAnswering_Weights*
      - SQUAD2
      - SQuAD2.0
      - 108,311,810
      - 413.2 MB
