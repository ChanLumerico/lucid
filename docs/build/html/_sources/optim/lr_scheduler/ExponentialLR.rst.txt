lr_scheduler.ExponentialLR
==========================

.. autoclass:: lucid.optim.lr_scheduler.ExponentialLR

The `ExponentialLR` learning rate scheduler reduces the learning rate 
exponentially at every epoch using a fixed multiplicative factor (`gamma`).
This allows for smooth and continuous learning rate decay.

Class Signature
---------------

.. code-block:: python

    class ExponentialLR(
        optimizer: Optimizer, 
        gamma: float, 
        last_epoch: int = -1, 
        verbose: bool = False
    )

Parameters
----------
- **optimizer** (*Optimizer*):
  The optimizer whose learning rate needs to be scheduled.
- **gamma** (*float*):
  Multiplicative factor for learning rate decay at each epoch.
- **last_epoch** (*int*, optional):
  The index of the last epoch when resuming training. Default: `-1`.
- **verbose** (*bool*, optional):
  If `True`, logs learning rate updates at each step. Default: `False`.

Mathematical Formula
--------------------
The learning rate at epoch :math:`t` is computed as:

.. math::

    \eta_t = \eta_0 \cdot \gamma^t

Where:
- :math:`\eta_t` is the learning rate at epoch :math:`t`.
- :math:`\eta_0` is the initial learning rate.
- :math:`\gamma` is the decay factor.

.. image:: _img/exponential_lr.png
    :width: 300
    :align: center

Methods
-------
- **get_lr() -> list[float]**:
  Computes the updated learning rate(s) using exponential decay.

- **step(epoch: Optional[int] = None) -> None**:
  Updates the learning rate based on the current epoch.

Usage Example
-------------

.. code-block:: python

    import lucid.optim as optim
    from lucid.optim.lr_scheduler import ExponentialLR

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(10):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.last_lr}")

.. note::

    `ExponentialLR` is useful for continuous and gradual decay of the 
    learning rate, allowing smoother adaptation to optimization progress.
