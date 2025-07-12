util.MultiScaleROIAlign
=======================

.. autoclass:: lucid.models.objdet.util.MultiScaleROIAlign

The `MultiScaleROIAlign` module performs RoI Align across multiple feature map levels, 
such as those used in Feature Pyramid Networks (FPN). It dynamically selects the 
appropriate feature level for each RoI based on its scale.

Constructor
-----------

.. code-block:: python

    def __init__(
        output_size: tuple[int, int],
        canonical_level: int = 2,
        canocical_scale: int = 224,
    ) -> None

Parameters
----------

- **output_size** (*tuple[int, int]*): 
  Desired spatial output size :math:`(H_{\text{out}}, W_{\text{out}})` for each region.
- **canonical_level** (*int*, optional): 
  Base pyramid level (usually `2` for FPN) that corresponds to the canonical object scale. 
  Defaults to `2`.
- **canocical_scale** (*int*, optional): 
  Reference object size that maps to the `canonical_level`. Defaults to `224`.

Returns
-------

- **MultiScaleROIAlign** (*nn.Module*): 
  A module that aligns input RoIs to multiple feature levels based on their 
  size and returns uniformly resized features.

Input & Output
--------------

.. code-block:: python

    def forward(
        features: list[Tensor],
        rois: Tensor,
        spatial_scales: list[float],
        sampling_ratio: int = -1
    ) -> Tensor

- **features** (*list[Tensor]*): 
  List of feature maps at different pyramid levels. Each has shape :math:`(N, C, H_l, W_l)`.
- **rois** (*Tensor*): 
  RoIs of shape :math:`(K, 5)` with format :math:`(batch_idx, x_1, y_1, x_2, y_2)`.
- **spatial_scales** (*list[float]*): 
  Scale factors for each feature level.
- **sampling_ratio** (*int*, optional): 
  Grid resolution used in bilinear sampling. Defaults to `-1` (auto).

Returns:

- **Tensor**: 
  A tensor of shape :math:`(K, C, H_{\text{out}}, W_{\text{out}})` 
  containing RoI-aligned feature crops.

.. important::

    The appropriate feature level :math:`l` for each RoI is selected via:

    .. math::

        l = l_0 + \log_2 \left( \frac{\sqrt{w \cdot h}}{s_0} + \epsilon \right)

    where :math:`s_0` is the `canocical_scale` and :math:`l_0` is `canonical_level`.

Example
-------

.. code-block:: python

    >>> from lucid.models.objdet.util import MultiScaleROIAlign
    >>> features = [lucid.random.randn(1, 256, s, s) for s in [64, 32, 16, 8]]
    >>> rois = lucid.Tensor([[0, 10, 10, 40, 40]])
    >>> align = MultiScaleROIAlign(output_size=(7, 7))
    >>> spatial_scales = [1/4, 1/8, 1/16, 1/32]
    >>> pooled = align(features, rois, spatial_scales)
    >>> print(pooled.shape)
    (1, 256, 7, 7)
