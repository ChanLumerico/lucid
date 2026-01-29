visual.build_tensor_mermaid_chart
=================================

.. autofunction:: lucid.visual.build_tensor_mermaid_chart

Generates a Mermaid **flowchart** diagram from a computation graph rooted at a
`Tensor`. This is useful for inspecting how tensors and operations connect during
forward computation and for embedding lightweight graphs in docs or notebooks.

Basic Example
-------------

.. code-block:: python

   import lucid
   import lucid.nn.functional as F
   from lucid.visual import build_tensor_mermaid_chart

   x = lucid.random.rand(1, 3, 8, 8, requires_grad=True)
   w = lucid.random.randn(4, 3, 3, 3, requires_grad=True)
   b = lucid.random.randn(4, requires_grad=True)

   out = F.conv2d(x, w, b, stride=1, padding=1)
   chart = build_tensor_mermaid_chart(out, horizontal=True, copy_to_clipboard=True)
   print(chart)

Key Parameters
--------------

- **tensor**:
  The root output tensor from which to traverse the computation graph.
- **horizontal**:
  If `True`, the graph is left-to-right; default is top-down.
- **title**:
  Optional Mermaid comment title (rendering depends on viewer).
- **start_id**:
  Highlights a tensor node by ID (useful for pinpointing a specific tensor).
- **copy_to_clipboard**:
  If `True`, copies the Mermaid text to the system clipboard.
- **use_class_defs** and color args:
  Control node color coding for ops, params, outputs, leaves, grads, and start node.
