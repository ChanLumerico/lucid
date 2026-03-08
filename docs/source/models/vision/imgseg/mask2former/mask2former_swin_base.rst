mask2former_swin_base
=====================

.. autofunction:: lucid.models.mask2former_swin_base

The `mask2former_swin_base` function builds a `Mask2Former` model with a
Swin-Base backbone preset.

**Total Parameters** (`num_labels=150`): 106,922,191

Function Signature
------------------

.. code-block:: python

    @register_model
    def mask2former_swin_base(
        num_labels: int,
        *,
        pretrained_backbone: bool = False,
        **config_kwargs
    ) -> Mask2Former

Parameters
----------

- **num_labels** (*int*):
  Number of segmentation classes.

- **pretrained_backbone** (*bool*, optional):
  If `True`, initializes the backbone from available pretrained weights.

- **config_kwargs** (*dict*, optional):
  Additional overrides applied to `Mask2FormerConfig`.

Returns
-------

- **Mask2Former**:
  Mask2Former model configured with a Swin-Base backbone.

Example Usage
-------------

.. code-block:: python

    from lucid.models.vision.mask2former import mask2former_swin_base
    import lucid

    model = mask2former_swin_base(num_labels=150)
    x = lucid.random.randn(1, 3, 384, 384)
    pred = model.predict(x)
    print(pred.shape)

Pretrained Weights
------------------

.. code-block:: python

    import lucid.models as models
    import lucid.weights as W

    model = models.mask2former_swin_base(
        num_labels=150,
        weights=W.Mask2Former_Swin_Base_Weights.ADE20K_SEMANTIC,
    )
