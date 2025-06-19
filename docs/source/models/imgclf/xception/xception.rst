xception
========

.. autofunction:: lucid.models.xception

The `xception` function is a registered factory method for creating an 
instance of the Xception model. It simplifies the process of initializing 
the Xception architecture with the desired number of output classes and 
additional parameters.

**Total Parameters**: 22,862,096

Function Signature
------------------

.. code-block:: python

    @register_model
    def xception(num_classes: int = 1000, **kwargs) -> Xception:

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional arguments for customizing the model. 
  These can include hyperparameters or modifications to the model architecture.

Returns
-------

- **Xception**: 
  An instance of the `Xception` model with the specified number of output 
  classes and any additional configuration provided through `kwargs`.

Examples
--------

**Creating a default Xception model:**

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.xception()
    ...

**Creating an Xception model with custom classes:**

.. code-block:: python

    >>> model = models.xception(num_classes=1000)
    >>> print(model)
    ...
