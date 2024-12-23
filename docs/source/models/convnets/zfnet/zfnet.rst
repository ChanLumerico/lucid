models.zfnet
============

.. autofunction:: lucid.models.zfnet

The `zfnet` function in `lucid.models` provides a convenient way to 
create an instance of the `ZFNet` module, a convolutional neural network 
designed for image classification with enhanced feature learning.

**Total Parameters**: 62,357,608

Function Signature
------------------

.. code-block:: python

    def zfnet(num_classes: int = 1000, **kwargs) -> ZFNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `ZFNet` module.

Returns
-------

- **ZFNet**:
  An instance of the `ZFNet` module configured with the specified number 
  of classes and any additional arguments.

Examples
--------

**Creating a Default ZFNet Model**

.. code-block:: python

    import lucid.models as models

    # Create a ZFNet model with 1000 output classes
    model = models.zfnet()

    print(model)  # Displays the ZFNet architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a ZFNet model with 10 output classes
    model = models.zfnet(num_classes=10)

    print(model)  # Displays the ZFNet architecture with modified output
