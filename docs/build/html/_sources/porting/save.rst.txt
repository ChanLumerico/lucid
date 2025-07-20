lucid.save
==========

.. autofunction:: lucid.save

The `save` function serializes and saves a Lucid object to disk. 
It supports saving `Tensor` objects and raw `OrderedDict`-based state_dicts. 
The data is stored using Python's `pickle` module.

Function Signature
------------------

.. code-block:: python

    def save(obj: Tensor | Module | OrderedDict, path: Path | str) -> Path

Parameters
----------

- **obj** (`Tensor` | `Module` | `OrderedDict`):
  The object to be saved. Must be one of the following:
    
  - `Tensor`: saved as a NumPy array into a `.lct` file.
  - `Module`: converted into a `state_dict` and saved as `.lcd`.
  - `OrderedDict`: treated as a raw state_dict and saved as `.lcd`.

- **path** (`Path` | `str`):
  File path to save the object to. If no suffix is provided, 
  an appropriate one will be inferred based on the object type.

Returns
-------

- **Path**: The absolute path to the saved file.

Behavior
--------

- If the suffix is missing:
  
  - `.lct` is used for `Tensor`
  - `.lcd` is used for `OrderedDict` or `Module`

- For `Tensor`, only its raw data (as NumPy array) is saved. Autograd-related properties are discarded.

.. warning::

   `Tensor` objects are saved without gradients, computation graphs, or hooks.

Examples
--------

.. code-block:: python

    import lucid
    from lucid import save
    from pathlib import Path

    t = lucid.Tensor([[1.0, 2.0]], requires_grad=True)
    path = save(t, "tensor")
    print(path.name)  # tensor.lct

    model = lucid.nn.Linear(2, 3)
    path = save(model.state_dict(), "weights")
    print(path.name)  # weights.lcd

.. note::

   This function uses `pickle` for serialization. 
   The saved files are not guaranteed to be portable across major Python versions or environments.
