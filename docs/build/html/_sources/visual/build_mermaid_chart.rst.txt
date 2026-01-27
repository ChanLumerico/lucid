visual.build_mermaid_chart
==========================

.. autofunction:: lucid.visual.build_mermaid_chart

Generates a Mermaid **flowchart** diagram from a `lucid.nn.Module` by running a
forward pass (using the provided inputs or a randomly generated tensor matching
`input_shape`) and recording the execution or dataflow between modules.

This is intended for quick architecture inspection and for embedding lightweight
model diagrams in documentation.

Basic Example
-------------

.. code-block:: python

   import lucid
   import lucid.nn as nn
   from lucid.visual import build_mermaid_chart

   class Tiny(nn.Module):
       def __init__(self):
           super().__init__()
           self.net = nn.Sequential(
               nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
               nn.ReLU(),
               nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
           )

       def forward(self, x):
           return self.net(x)

   model = Tiny()
   chart = build_mermaid_chart(
       model,
       input_shape=(1, 3, 32, 32),
       depth=3,
       edge_mode="execution",
       show_shapes=True,
       collapse_repeats=True,
   )

   print(chart)

Key Parameters
--------------

- **inputs / input_shape**:
  Provide explicit inputs (`Tensor` or iterable of `Tensor`) or let the function
  generate random inputs from `input_shape`.
- **depth**:
  Limits how deep the module tree is expanded for grouping (subgraphs).
- **edge_mode**:
  `"execution"` records sequential execution order; `"dataflow"` attempts to connect
  producers and consumers via intermediate tensors.
- **show_shapes**:
  Adds input/output shape hints to node labels (when a shape change is detected).
- **collapse_repeats / repeat_min**:
  Collapses repeated sibling structures into a single node like `Layer x N`.
- **hide_subpackages / hide_module_names**:
  Allows filtering out modules by Lucid builtin subpackage (e.g. `activation`, `drop`)
  or by class name (e.g. `ReLU`).

