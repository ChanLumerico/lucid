lucid.meshgrid
==============

.. autofunction:: lucid.meshgrid

The `meshgrid` function creates coordinate grids from two 1D coordinate tensors, 
allowing for the generation of X and Y coordinate matrices. It supports both 'xy' 
and 'ij' indexing, making it versatile for use in Cartesian and matrix-based indexing systems.

Function Signature
------------------

.. code-block:: python

    def meshgrid(x: Tensor, y: Tensor, indexing: str = 'xy') -> Tuple[Tensor, Tensor]

Parameters
----------

- **x** (*Tensor*):
  A 1D tensor representing the x-coordinates.
  
- **y** (*Tensor*):
  A 1D tensor representing the y-coordinates.
  
- **indexing** (*str*, optional):
  The type of indexing to use. Can be either `'xy'` (Cartesian indexing) or `'ij'` (matrix indexing). 
  Default is `'xy'`.

Returns
-------

- **Tuple[Tensor, Tensor]**: 
  A tuple of two 2D tensors `(X, Y)` containing the coordinate grid. 
  The shape of both `X` and `Y` is `(len(y), len(x))`.

Forward Calculation
-------------------

The forward calculation for `meshgrid` involves repeating the x and y coordinates 
to form two 2D grids, `X` and `Y`.

Backward Gradient Calculation
-----------------------------

To compute the gradient of `X` and `Y` with respect to the inputs `x` and `y`, 
we use the following calculations:

1. **Gradient with respect to** :math:`x`:

   Since each element of `x` is repeated along the rows of `X`, the gradient of `X` 
   with respect to `x` is the sum of gradients along the first axis (rows):
   
   .. math::
      
      \frac{\partial X}{\partial x} = \sum_{i} X_{i,:}

2. **Gradient with respect to** :math:`y`:

   Since each element of `y` is repeated along the columns of `Y`, the gradient of `Y` 
   with respect to `y` is the sum of gradients along the second axis (columns):
   
   .. math::
      
      \frac{\partial Y}{\partial y} = \sum_{j} Y_{:,j}

Usage
-----

The `meshgrid` function is used to generate coordinate grids for image processing, plotting, 
and other spatial transformations. It converts two 1D tensors into 2D coordinate matrices.

Examples
--------

**Example 1: Cartesian ('xy') indexing**

.. code-block:: python

    import lucid
    
    x = lucid.tensor([1, 2, 3])  # x-coordinates
    y = lucid.tensor([4, 5])  # y-coordinates
    X, Y = lucid.meshgrid(x, y, indexing='xy')
    
    print(X)
    # Tensor([[1, 2, 3],
    #         [1, 2, 3]])
    
    print(Y)
    # Tensor([[4, 4, 4],
    #         [5, 5, 5]])

**Example 2: Matrix ('ij') indexing**

.. code-block:: python

    import lucid
    
    x = lucid.tensor([1, 2, 3])  # x-coordinates
    y = lucid.tensor([4, 5])  # y-coordinates
    X, Y = lucid.meshgrid(x, y, indexing='ij')
    
    print(X)
    # Tensor([[1, 1, 1],
    #         [2, 2, 2],
    #         [3, 3, 3]])
    
    print(Y)
    # Tensor([[4, 5],
    #         [4, 5],
    #         [4, 5]])

.. note::

    - **Indexing Choice**: Use `'xy'` for Cartesian-style (image-based) grids and `'ij'` 
      for matrix-based (row, column) grids.
    - **Shape**: The output tensors `X` and `Y` have shape `(len(y), len(x))`.
    - **Gradient Propagation**: If `x` or `y` requires gradients, they are automatically 
      propagated through the resulting coordinate grids `X` and `Y`.
