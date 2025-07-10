faster_rcnn_resnet_50_fpn
=========================

.. autofunction:: lucid.models.faster_rcnn_resnet_50_fpn

The `faster_rcnn_resnet_50_fpn` function constructs a two-stage object detection model 
based on the Faster R-CNN architecture, with a ResNet-50 backbone and Feature Pyramid Network (FPN). 
This configuration is suitable for a wide range of object detection tasks 
with a balance of speed and accuracy.

**Total Parameters**: 43,515,902

Function Signature
------------------

.. code-block:: python

    @register_model
    def faster_rcnn_resnet_50_fpn(
        num_classes: int = 21,
        backbone_num_classes: int = 1000,
        **kwargs
    ) -> FasterRCNN

Parameters
----------

- **num_classes** (*int*, optional):  
  The number of target object classes to detect. Default is 21 (e.g., PASCAL VOC).  
  This affects the final classification layer of the detection head.

- **backbone_num_classes** (*int*, optional):  
  The number of classes used when initializing the ResNet-50 backbone (e.g., 1000 for ImageNet).  
  This affects the initial backbone setup and should match the pretraining phase if any.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments to customize `FasterRCNN`, such as anchor sizes, 
  NMS thresholds, etc.

Returns
-------

- **FasterRCNN**:  
  An instance of the `FasterRCNN` model with a ResNet-50 + FPN backbone.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import faster_rcnn_resnet_50_fpn

    # Create the model with 21 object classes (VOC-style)
    model = faster_rcnn_resnet_50_fpn(num_classes=21)

    # Input tensor: 1 image with 3 channels and 224x224 spatial dimensions
    x = lucid.random.randn(1, 3, 224, 224)

    # Forward pass
    cls_logits, bbox_deltas = model(x)

    print(cls_logits.shape)   # (N, R, C), e.g., (1, 300, 21)
    print(bbox_deltas.shape)  # (N, R, 4),   e.g., (1, 300, 4)

Training Notes
--------------

Lucid does **not** provide pre-trained weights for object detection backbones.  
To train `faster_rcnn_resnet_50_fpn` properly, follow this two-stage strategy:

1. **Pretrain the Backbone as a Classifier**
   
   Extract the internal `resnet_50` backbone via:

   .. code-block:: python

       from lucid.models import faster_rcnn_resnet_50_fpn

       detector = faster_rcnn_resnet_50_fpn(num_classes=21)
       backbone_net = detector.backbone.net  # This is a ResNet-50 model

   You can train `backbone_net` on an image classification task (e.g., ImageNet or CIFAR)  
   using standard training pipelines (`nn.CrossEntropyLoss`, `nn.SGD`, etc.).

   Save the weights via:

   .. code-block:: python

       state_dict = backbone_net.state_dict()
       # Save using your own serialization logic

2. **Fine-tune the Detection Model**
   
   Once pretrained, reload the saved `state_dict` into the `backbone.net` 
   and train the detection head:

   .. code-block:: python

       detector.backbone.net.load_state_dict(state_dict)
       # Now proceed to train the FasterRCNN model on your object detection dataset

.. tip::

   When training from scratch, reduce learning rates for the backbone using parameter groups  
   to stabilize optimization.

.. warning::

   Ensure the backbone's `backbone_num_classes` matches the number of output classes 
   used during pretraining. Otherwise, classifier weights will mismatch and must be discarded.

