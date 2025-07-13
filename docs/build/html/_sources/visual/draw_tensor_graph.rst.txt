visual.draw_tensor_graph
========================

.. autofunction:: lucid.visual.draw_tensor_graph

Visualizes the computational graph of a given `Tensor` object, 
showing the flow of operations and tensor dependencies. This function 
is useful for understanding how gradients propagate through a model 
during backpropagation.

Function Signature
------------------

.. code-block:: python

    def draw_tensor_graph(
        tensor: Tensor,
        horizontal: bool = False,
        title: str | None = None,
        start_id: int | None = None,
    ) -> plt.Figure

Parameters
----------

- **tensor** (*Tensor*):  
  The root output tensor from which to start the graph traversal.

- **horizontal** (*bool*, optional):  
  If `True`, the graph is drawn left-to-right. Defaults to top-down.

- **title** (*str or None*, optional):  
  Optional title for the graph plot.

- **start_id** (*int or None*, optional):  
  If provided, highlights the tensor with the specified ID in blue.

Returns
-------

- **plt.Figure**:  
  The matplotlib Figure object containing the plotted graph.

Example
-------

.. code-block:: python

    import lucid
    import lucid.nn.functional as F
    from lucid.visual import draw_tensor_graph

    x = lucid.random.rand(1, 3, 8, 8, requires_grad=True)
    w = lucid.random.randn(4, 3, 3, 3, requires_grad=True)
    b = lucid.random.randn(4, requires_grad=True)

    out = F.conv2d(x, w, b, stride=1, padding=1)

    fig = draw_tensor_graph(out, horizontal=True, title="Conv2D Output Graph")
    fig.show()

.. note::

    The visualization shows Tensor shapes and operations with color coding:

    - **lightgreen**: operations
    - **lightblue**: intermediate tensors requiring grad
    - **lightgray**: intermediate tensors not requiring grad
    - **violet**: `Parameter` tensors
    - **red**: the output tensor
    - **yellow**: tensor marked by `start_id`
