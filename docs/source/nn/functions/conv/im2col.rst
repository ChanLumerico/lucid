nn.functional.im2col
====================

.. autofunction:: lucid.nn.functional.im2col

The `im2col` function extracts sliding local blocks from a tensor and
rearranges them into columns, a common preprocessing step in
convolutional operations.

Function Signature
------------------

.. code-block:: python

    def im2col(
        input_: Tensor,
        filter_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C, H, W), where N is the batch size,
    C is the number of channels, and H, W are the height and width of
    the input.

- **filter_size** (*tuple[int, ...]*): 
    The size of the sliding filter (kernel) as (filter_height, filter_width).

- **stride** (*tuple[int, ...]*): 
    The stride of the sliding filter as (stride_height, stride_width).

- **padding** (*tuple[int, ...]*): 
    The padding applied to the input tensor as (padding_height, padding_width).

Returns
-------

- **Tensor**: 
    A tensor of shape (N, C * filter_height * filter_width, output_height * output_width), 
    where output_height and output_width are computed based on the input size, 
    filter size, stride, and padding.

Forward Calculation
-------------------

The `im2col` operation rearranges local patches of the input tensor 
into columns for efficient computation in convolution operations. 
Given an input tensor and the specified filter size, stride, and padding:

1. Apply padding to the input tensor.
2. Extract sliding patches of size `filter_size` from the input tensor with a stride `stride`.
3. Rearrange these patches into columns.

The result is a matrix where each column corresponds to a flattened 
local patch of the input tensor.

Example
-------

Using `im2col` to preprocess input data for a convolution operation:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], requires_grad=True)  # Shape: (1, 1, 3, 3)
    >>> filter_size = (2, 2)  # Kernel size: 2x2
    >>> stride = (1, 1)  # Stride: 1
    >>> padding = (0, 0)  # No padding

    >>> col = F.im2col(input_, filter_size, stride, padding)
    >>> print(col)
    Tensor([
        [1, 2, 4, 5],
        [2, 3, 5, 6],
        [4, 5, 7, 8],
        [5, 6, 8, 9]
    ], grad=None)  # Shape: (1, 4, 4)

The resulting tensor contains columns of sliding patches.

.. note::

    This function is essential in implementing convolution operations 
    and is often used in conjunction with matrix multiplication 
    for optimized computation of convolutions.
