detr_r50
========

.. autofunction:: lucid.models.detr_r50

The `detr_r50` function returns an instance of the `DETR` model
configured with a **ResNet-50** backbone.

**Total Parameters**: 41,578,400 (MS-COCO)

Function Signature
------------------
.. code-block:: python

    @register_model
    def detr_r50(
        num_classes: int = 91,
        num_queries: int = 100,
        pretrained_backbone: bool = False,
        **kwargs,
    ) -> DETR

Parameters
----------
- **num_classes** (*int*, default=91): 
  Number of foreground categories. 
  For COCO this is 91; the model internally adds one **“no-object”** class.

- **num_queries** (*int*, default=100): 
  Number of learned object queries (maximum detections per image). 
  This sets the second dimension of the outputs.

- **pretrained_backbone** (*bool*, default=False): 
  If `True`, initializes the ResNet-50 backbone with pretrained weights and (by default) 
  **fine-tunes** it.

- **kwargs**:  
  Additional keyword arguments forwarded to the underlying :class:`DETR` 
  constructor (e.g., `aux_loss`, loss coefficients, custom matcher, etc.).

Returns
-------
- **DETR**:
  
  A DETR-R50 detector with class and box heads. 
  In eval mode with `aux_loss=False`, a forward pass returns:

  - **pred_logits**: `(B, num_queries, num_classes + 1)` - raw class scores; 
    the last logit is **no-object**.

  - **pred_boxes**:  `(B, num_queries, 4)` - normalized boxes in **(cx, cy, w, h)** 
    with values in **[0, 1]**.

Example Usage
-------------
.. code-block:: python

    >>> from lucid.models import detr_r50
    >>> import lucid
    >>>
    >>> # COCO-style config (91 classes), 100 queries
    >>> model = detr_r50(num_classes=91, num_queries=100, pretrained_backbone=True, aux_loss=False)
    >>> print(model)
    >>>
    >>> # Dummy input
    >>> x = lucid.random.randn(1, 3, 800, 800)
    >>> logits, boxes = model(x)  # aux_loss=False returns a tuple
    >>> print(logits.shape, boxes.shape)
    (1, 100, 92) (1, 100, 4)
    >>>
    >>> # Convert logits to probabilities (drop no-object column)
    >>> probs = lucid.softmax(logits, axis=-1)[..., :-1]
    >>> scores = probs.max(axis=-1)  # (B, num_queries)
