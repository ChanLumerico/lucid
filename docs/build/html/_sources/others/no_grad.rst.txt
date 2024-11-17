lucid.no_grad
=============

.. autofunction:: lucid.no_grad

The `no_grad` context manager temporarily disables gradient calculation. 

This is especially useful during inference, where gradients are not needed, 
helping to save memory and computational resources.

Function Signature
------------------

.. code-block:: python

    @contextmanager
    def no_grad() -> Generator

Parameters
----------

This context manager does not take any parameters.

Returns
-------

- **Generator**: A generator object that, when used in a `with` block, 
  disables gradient tracking temporarily and restores it afterward.

Usage
-----

The `no_grad` context manager should be used to wrap parts of your code where 
gradient tracking is unnecessary. 

This is typically useful during model inference or when performing operations not related to training.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> with lucid.no_grad():
    >>>     # Inside this block, gradients are disabled
    >>>     output = model(input)
    >>>     print(output)

In the example above, any tensor operation within the `with` block does not track gradients. 
Once the block exits, the gradient state is restored to what it was before entering the context manager.

.. note::
    The context manager ensures that the global gradient tracking state 
    is restored after execution, even if an error occurs inside the block.

How It Works
------------

The `no_grad` context manager temporarily modifies the global `_grad_enabled` variable, 
disabling gradient tracking during the execution of the `yield` block. 

Once the block is finished, the state of gradient tracking is reverted to its previous state.

.. attention::
    Since this context manager modifies a global variable, it should be used with care. 
    Ensure that no unintended side effects occur when disabling gradient tracking globally.

.. warning::
    Avoid using the `no_grad` context manager during training. 
    Disabling gradient tracking will prevent the model from updating its parameters.

Benefits
--------

Using `no_grad` during inference or when performing operations not 
requiring gradients provides several advantages:

- **Reduced memory usage**: Tensors won't store gradient information, saving memory.

- **Faster computation**: Operations that don’t require gradients will be faster 
  since gradient calculations are skipped.

- **Cleaner code**: Instead of manually disabling and re-enabling gradients, 
  the `no_grad` context manager handles this for you.

.. tip::
    When running inference on large models or datasets, wrapping the inference code 
    in `no_grad` can result in significant memory and time savings.

Potential Pitfalls
------------------

While `no_grad` is powerful, it must be used appropriately. Here are some things to keep in mind:

.. caution::

    - **Training operations**: Avoid using `no_grad` in training steps, 
      as it will prevent gradients from being calculated, which are necessary for backpropagation.

    - **Global state modification**: Since it changes a global variable (`_grad_enabled`), 
      ensure it doesn’t interfere with other parts of the code that may rely on gradients.

Conclusion
----------

The `no_grad` context manager is an efficient way to handle non-gradient-related operations in your code, 
optimizing memory usage and computational performance, especially when working with large models during inference.

