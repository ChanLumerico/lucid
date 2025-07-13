lucid.visual
============

The `lucid.visual` package provides visualization utilities for inspecting tensors, 
operations, and neural networks built with Lucid. These tools are designed to support 
intuitive understanding of model structures and training behaviors, especially 
in educational or debugging contexts.

The goal of this module is to make abstract concepts like computation graphs 
and data flows more accessible through visual representations.

Currently Available
-------------------

- **draw_tensor_graph**:  
  Renders the computation graph of a Lucid tensor, helping users understand the 
  flow of operations and how gradients propagate.

Future extensions to this module may include tools for:

- Visualizing weights and activations across layers  
- Showing parameter distributions and gradient norms  
- Generating architecture diagrams for Lucid models  

.. tip::

   Visual tools are best used in combination with interactive environments such as 
   Jupyter Notebooks for quick feedback and inspection. 
   (*will be supported in future releases*)
