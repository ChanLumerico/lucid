lr_scheduler.LRScheduler
========================

.. autoclass:: lucid.optim.lr_scheduler.LRScheduler

The `LRScheduler` class provides the base functionality for all learning rate 
scheduling strategies in Lucid. Schedulers dynamically adjust the learning rate 
during training to enhance model convergence and performance.

Class Signature
---------------

.. code-block:: python

    class LRScheduler(optimizer, last_epoch: int = -1, verbose: bool = False)

Parameters
----------
- **optimizer** (*Optimizer*):
  The optimizer whose learning rate needs to be scheduled.
- **last_epoch** (*int*, optional):
  The index of the last epoch when resuming training. Default: `-1`.
- **verbose** (*bool*, optional):
  If `True`, logs learning rate updates at each step. Default: `False`.

Methods
-------
- **get_lr() -> list[float]**:
  Computes the new learning rate(s). Must be implemented by subclasses.

- **step(epoch: Optional[int] = None) -> None**:
  Updates the learning rate. If `epoch` is provided, applies the schedule up to that epoch.

- **state_dict() -> dict[str, Any]**:
  Returns the current state of the scheduler, including learning rates and epoch information.

- **load_state_dict(state_dict: dict[str, Any]) -> None**:
  Loads a previously saved state dictionary to restore scheduler behavior.

Usage Example
-------------

.. code-block:: python

    import lucid.optims as optim
    from lucid.optims.lr_scheduler import StepLR

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(20):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.last_lr}")
