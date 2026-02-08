lucid.load
==========

.. autofunction:: lucid.load

The `load` function deserializes and loads a previously saved Lucid object 
from disk. It supports loading `Tensor` objects from `.lct` files and 
`OrderedDict` state_dicts from `.lcd` files.

Function Signature
------------------

.. code-block:: python

    def load(path: Path | str) -> Tensor | OrderedDict

Parameters
----------

- **path** (`Path` | `str`):
  Path to the file containing the saved object. The file must have a `.lct` or `.lcd` suffix.

Returns
-------

- **Tensor**: If the file contains a saved `Tensor` (from `.lct` file).
- **OrderedDict**: If the file contains a saved `state_dict` (from `.lcd` file).

Behavior
--------

- If the file has a `.lct` extension, the function loads the raw NumPy array and wraps it as a `Tensor`.
- If the file has a `.lcd` extension, the function returns the `OrderedDict` representing the model's `state_dict`.

.. warning::

   The function assumes that the file was created using `lucid.save`. 
   Files not conforming to Lucid's format will raise an error.

Examples
--------

.. code-block:: python

    from lucid import load

    # Load a tensor
    t = load("tensor.lct")
    print(t)  # Tensor([[1.0, 2.0]], grad=None)

    # Load a state_dict
    state = load("weights.lcd")
    print(state.keys())  # dict_keys([...])

.. note::

   This function relies on `pickle` for deserialization. 
   Ensure that the file was created in a compatible Python environment.
