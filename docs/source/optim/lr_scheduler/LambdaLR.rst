lr_scheduler.LambdaLR
=====================

.. autoclass:: lucid.optim.lr_scheduler.LambdaLR

The `LambdaLR` learning rate scheduler allows the user to define a custom 
learning rate scaling function (`lr_lambda`). This provides flexibility to adjust 
the learning rate dynamically based on epoch progress.

Class Signature
---------------

.. code-block:: python

    class LambdaLR(
        optimizer: Optimizer, 
        lr_lambda: Callable[[int], float], 
        last_epoch: int = -1, 
        verbose: bool = False
    )

Parameters
----------
- **optimizer** (*Optimizer*):
  The optimizer whose learning rate needs to be scheduled.
- **lr_lambda** (*Callable[[int], float]*):
  A function that takes an epoch index and returns a scaling factor for the learning rate.
- **last_epoch** (*int*, optional):
  The index of the last epoch when resuming training. Default: `-1`.
- **verbose** (*bool*, optional):
  If `True`, logs learning rate updates at each step. Default: `False`.

Mathematical Formula
--------------------
The learning rate at epoch :math:`t` is computed as:

.. math::

    \eta_t = \eta_0 \cdot f(t)

Where:
- :math:`\eta_t` is the learning rate at epoch :math:`t`.
- :math:`\eta_0` is the initial learning rate.
- :math:`f(t)` is the user-defined function (`lr_lambda`) applied at epoch :math:`t`.

Methods
-------
- **get_lr() -> list[float]**:
  Computes the updated learning rate(s) using the lambda function.

- **step(epoch: Optional[int] = None) -> None**:
  Updates the learning rate based on the current epoch.

Usage Example
-------------

.. code-block:: python

    import lucid.optim as optim
    from lucid.optim.lr_scheduler import LambdaLR

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    lambda_fn = lambda epoch: 0.95 ** epoch  # Exponential decay function
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    
    for epoch in range(10):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.last_lr}")

.. note::

    `LambdaLR` is highly flexible and allows users to define complex learning rate 
    schedules by providing a custom function.
