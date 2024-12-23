models.alexnet
==============

.. autofunction:: lucid.models.alexnet

The `alexnet` provides a convenient way to create an instance of the `AlexNet` module, 
a convolutional neural network designed for image classification tasks.

Function Signature
------------------

.. code-block:: python

    def alexnet(num_classes: int = 1000, **kwargs) -> AlexNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments to customize the `AlexNet` module.

Returns
-------

- **AlexNet**:
  An instance of the `AlexNet` module configured with the specified number of 
  classes and any additional arguments.

Examples
--------

**Creating a Default AlexNet Model**

.. code-block:: python

    import lucid.models as models

    # Create an AlexNet model with 1000 output classes
    model = models.alexnet()

    print(model)  # Displays the AlexNet architecture

**Custom Number of Classes**

.. code-block:: python

    # Create an AlexNet model with 10 output classes
    model = models.alexnet(num_classes=10)

    print(model)  # Displays the AlexNet architecture with modified output
