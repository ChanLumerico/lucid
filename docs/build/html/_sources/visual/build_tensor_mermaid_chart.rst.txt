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

.. mermaid::

    flowchart LR;
    t_4418730848["(1, 3, 8, 8)"];
    t_4418525520["(4, 3, 3, 3)"];
    op_4419339664(("conv_nd_kernel"));
    t_4299709040["(1, 4, 8, 8)"];
    t_4418526480["(4,)"];
    op_4419340000(("_reshape_immediate"));
    t_4418992160["(1, 4, 1, 1)"];
    op_4419340336(("add"));
    t_4418559376["(1, 4, 8, 8)"];
    op_4419339664 --> t_4299709040;
    t_4418730848 --> op_4419339664;
    t_4418525520 --> op_4419339664;
    op_4419340000 --> t_4418992160;
    t_4418526480 --> op_4419340000;
    op_4419340336 --> t_4418559376;
    t_4299709040 --> op_4419340336;
    t_4418992160 --> op_4419340336;
    classDef op fill:lightgreen,stroke:#666,stroke-width:1px;
    classDef param fill:plum,stroke:#666,stroke-width:1px;
    classDef result fill:lightcoral,stroke:#666,stroke-width:1px;
    classDef leaf fill:lightgray,stroke:#666,stroke-width:1px;
    classDef grad fill:lightblue,stroke:#666,stroke-width:1px;
    classDef start fill:gold,stroke:#666,stroke-width:1px;
    class t_4418730848 grad;
    class t_4418525520 grad;
    class op_4419339664 op;
    class t_4299709040 grad;
    class t_4418526480 grad;
    class op_4419340000 op;
    class t_4418992160 grad;
    class op_4419340336 op;
    class t_4418559376 result;

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
