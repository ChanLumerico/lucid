mask_rcnn_resnet_50_fpn
=======================

.. autofunction:: lucid.models.mask_rcnn_resnet_50_fpn

The `mask_rcnn_resnet_50_fpn` function builds a Mask R-CNN instance segmentation
model with a ResNet-50 backbone and FPN feature pyramid.

**Total Parameters**: 46,037,607

Function Signature
------------------

.. code-block:: python

    @register_model
    def mask_rcnn_resnet_50_fpn(
        num_classes: int = 21,
        backbone_num_classes: int = 1000,
        **kwargs
    ) -> MaskRCNN

Parameters
----------

- **num_classes** (*int*, optional):
  Number of segmentation/detection classes. Default is `21`.

- **backbone_num_classes** (*int*, optional):
  Number of classes used to initialize the internal ResNet-50 backbone.

- **kwargs** (*dict*, optional):
  Additional arguments passed to `MaskRCNN` (for example mask head size,
  anchor settings, pooling sizes, and hidden dimensions).

Returns
-------

- **MaskRCNN**:
  Mask R-CNN model configured with ResNet-50 + FPN.

Example Usage
-------------

.. code-block:: python

    from lucid.models.vision.mask_rcnn import mask_rcnn_resnet_50_fpn
    import lucid

    model = mask_rcnn_resnet_50_fpn(num_classes=21)
    x = lucid.random.randn(1, 3, 512, 512)

    cls_logits, bbox_deltas, mask_logits = model(x)
    print(cls_logits.shape, bbox_deltas.shape, mask_logits.shape)

    out = model.predict(x)
    print(out[0]["boxes"].shape, out[0]["masks"].shape)
