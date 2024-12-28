models.summarize
================

.. autofunction:: lucid.models.summarize

The `summarize` function generates a detailed summary of a neural network model, including 
information about its layers, input and output shapes, and the number of trainable parameters. 
This function is useful for debugging and understanding the architecture of complex models.

Function Signature
------------------

.. code-block:: python

    def summarize(
        model: nn.Module,
        input_shape: _ShapeLike,
        recurse: bool = True,
        truncate_from: int | None = None,
    ) -> None

Parameters
----------

- **model** (*nn.Module*): 
  The neural network model to summarize. Must be an instance of `nn.Module`.

- **input_shape** (*_ShapeLike*): 
  Shape of the input tensor to the model. This is used to create a dummy input 
  and simulate a forward pass.

- **recurse** (*bool*, optional): 
  If set to `True`, includes submodules in the summary. Default is `True`.

- **truncate_from** (*int | None*, optional): 
  If specified, limits the number of layers displayed in the summary. Default is `None`, 
  meaning no truncation.

Returns
-------

- **None**: 
  The function does not return a value. Instead, it prints a formatted table 
  summarizing the model's layers.

Summary Table
-------------

The output of the function is a table with the following columns:

- **Layer**: The name of the layer.
- **Input Shape**: Shape of the input tensor for the layer.
- **Output Shape**: Shape of the output tensor from the layer.
- **Parameter Size**: Number of trainable parameters in the layer.
- **Layer Count**: Number of sublayers within the module.

.. note::

   The function also prints the total number of layers and trainable parameters in the model.
   If the summary is truncated using the `truncate_from` parameter, it displays the number 
   of truncated layers.

Examples
--------

**Basic Example**

Summarizing a simple feedforward neural network:

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid.models import summarize
    >>> class SimpleModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10, 5)
    ...         self.fc2 = nn.Linear(5, 2)
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.fc2(x)
    ...         return x
    >>> model = SimpleModel()
    >>> summarize(model, input_shape=(1, 10))

This will produce a table summarizing the layers of the model.

**Using Submodules**

If a model contains nested submodules:

.. code-block:: python

    >>> class NestedModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.block1 = nn.Sequential(
    ...             nn.Linear(10, 8),
    ...             nn.ReLU()
    ...         )
    ...         self.block2 = nn.Sequential(
    ...             nn.Linear(8, 4),
    ...             nn.ReLU()
    ...         )
    ...     def forward(self, x):
    ...         x = self.block1(x)
    ...         x = self.block2(x)
    ...         return x
    >>> model = NestedModel()
    >>> summarize(model, input_shape=(1, 10), recurse=True)

This will include the layers within the submodules in the summary.

**Truncating the Summary**

If a model has many layers, you can truncate the output to display only the first few:

.. code-block:: python

    >>> summarize(model, input_shape=(1, 10), truncate_from=5)

This will display the first 5 layers and indicate the number of truncated layers.

.. note::

    - The function creates a dummy input tensor using `lucid.zeros` to simulate 
      a forward pass.
    - The summary is printed to the console and is not returned as a value.
    - Use this function to understand and debug model architectures before training.
