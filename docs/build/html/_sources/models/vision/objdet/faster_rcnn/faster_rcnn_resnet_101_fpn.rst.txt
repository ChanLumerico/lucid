faster_rcnn_resnet_101_fpn
==========================

.. autofunction:: lucid.models.faster_rcnn_resnet_101_fpn

The `faster_rcnn_resnet_101_fpn` function constructs a two-stage object detection model 
based on the Faster R-CNN architecture, using a deeper ResNet-101 backbone and Feature Pyramid Network (FPN).  
This variant offers improved accuracy over ResNet-50 variants, particularly for more 
complex or large-scale detection tasks.

**Total Parameters**: 62,508,030

Function Signature
------------------

.. code-block:: python

    @register_model
    def faster_rcnn_resnet_101_fpn(
        num_classes: int = 21,
        backbone_num_classes: int = 1000,
        **kwargs
    ) -> FasterRCNN

Parameters
----------

- **num_classes** (*int*, optional):  
  The number of object categories to detect. Default is 21 (e.g., PASCAL VOC).  
  Used in the classification head of the detection model.

- **backbone_num_classes** (*int*, optional):  
  Number of output classes for the ResNet-101 backbone used during initialization.  
  This should match the classification task used for pretraining. Default is 1000.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments for configuring the `FasterRCNN` model, 
  such as anchor settings, thresholds, etc.

Returns
-------

- **FasterRCNN**:  
  A Faster R-CNN model with a ResNet-101 backbone and FPN.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import faster_rcnn_resnet_101_fpn

    # Construct model for 21 object classes
    model = faster_rcnn_resnet_101_fpn(num_classes=21)

    # Dummy input image
    x = lucid.random.randn(1, 3, 224, 224)

    # Inference
    cls_logits, bbox_deltas = model(x)

    print(cls_logits.shape)   # (1, 300, 21)
    print(bbox_deltas.shape)  # (1, 300, 4)

Training Notes
--------------

Lucid does **not** provide pretrained weights for ResNet-101 detection backbones.  
You are expected to train the backbone separately and reuse it for Faster R-CNN training.

1. **Pretrain ResNet-101 Backbone**

   Access and train the internal backbone from the detector:

   .. code-block:: python

       from lucid.models import faster_rcnn_resnet_101_fpn

       model = faster_rcnn_resnet_101_fpn(num_classes=21)
       backbone = model.backbone.net  # ResNet-101

   You can use this as a standard classifier and train it using a dataset like ImageNet:

   .. code-block:: python

       # Train backbone here
       # Save weights:
       state_dict = backbone.state_dict()

2. **Use Pretrained Backbone for Detection**

   Load the pretrained weights back before training object detection:

   .. code-block:: python

       model.backbone.net.load_state_dict(state_dict)

   Then proceed to train the full `FasterRCNN` model.

.. tip::

   Consider using a lower learning rate for the backbone compared to the detection heads  
   when fine-tuning the entire model.

.. warning::

   The `backbone_num_classes` argument must match the number of classes used when 
   pretraining the backbone. A mismatch may lead to incompatible final layers or 
   parameter shapes.
