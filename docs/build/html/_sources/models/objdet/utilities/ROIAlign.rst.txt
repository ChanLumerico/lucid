util.ROIAlign
=============

.. autoclass:: lucid.models.utils.ROIAlign

The `ROIAlign` module performs Region of Interest (RoI) Align, 
which extracts fixed-size feature maps from input feature tensors based on 
given bounding box regions. Unlike RoI Pooling, RoI Align avoids quantization 
artifacts using bilinear interpolation.

Constructor
-----------

.. code-block:: python

    def __init__(self, output_size: tuple[int, int]) -> None

Parameters
----------

- **output_size** (*tuple[int, int]*): 
  The target spatial size :math:`(H_{\text{out}}, W_{\text{out}})` for each cropped region.

Returns
-------

- **ROIAlign** (*nn.Module*): 
  A module that, when called, takes a feature tensor and region boxes 
  and returns aligned feature crops.

Input & Output
--------------

.. code-block:: python

    def forward(
        features: Tensor,
        rois: Tensor,
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1
    ) -> Tensor

- **features** (*Tensor*): 
  Input feature map of shape :math:`(N, C, H, W)`.
- **rois** (*Tensor*): 
  Boxes of shape :math:`(K, 5)` where each row is :math:`(batch_idx, x_1, y_1, x_2, y_2)`.
- **spatial_scale** (*float*, optional): 
  Scale factor applied to RoI coordinates to match the input feature map size. 
  Defaults to `1.0`.
- **sampling_ratio** (*int*, optional): 
  Number of sampling points in the interpolation grid. `-1` means automatic. 
  Defaults to `-1`.

Returns:

- **Tensor**: 
  A tensor of shape :math:`(K, C, H_{\text{out}}, W_{\text{out}})` 
  containing the aligned region features.

.. note::

    - Gradient is preserved through bilinear interpolation.
    - Batch index must be included in the RoI tensor.

Example
-------

.. code-block:: python

    >>> from lucid.models.utils import ROIAlign
    >>> roi_align = ROIAlign(output_size=(7, 7))
    >>> features = lucid.random.randn(1, 256, 32, 32)
    >>> rois = lucid.Tensor([[0, 4, 4, 24, 24]])
    >>> crops = roi_align(features, rois)
    >>> print(crops.shape)
    (1, 256, 7, 7)
