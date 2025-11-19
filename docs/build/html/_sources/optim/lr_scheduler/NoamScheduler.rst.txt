lr_scheduler.NoamScheduler
==========================

.. autoclass:: lucid.optim.lr_scheduler.NoamScheduler

The `NoamScheduler` implements the warmup + inverse square-root decay strategy
popularized by the Transformer. It gradually increases the learning rate during
the warmup window and then decays it proportionally to :math:`1 / \sqrt{t}`.

Class Signature
---------------

.. code-block:: python

    class NoamScheduler(
        optimizer: Optimizer,
        model_size: int,
        warmup_steps: int,
        factor: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
    )

Parameters
----------
- **optimizer** (*Optimizer*):
  The optimizer whose learning rate is controlled.
- **model_size** (*int*):
  Typically the Transformer hidden dimension. Used to scale the schedule.
- **warmup_steps** (*int*):
  Number of steps over which to linearly ramp up the learning rate.
- **factor** (*float*, optional):
  Global scaling factor for the learning rate curve. Default: `1.0`.
- **last_epoch** (*int*, optional):
  Index of the last epoch when resuming training. Default: `-1`.
- **verbose** (*bool*, optional):
  If `True`, logs learning rate changes every step. Default: `False`.

Mathematical Formula
--------------------
The learning rate at step :math:`t` is:

.. math::

    \eta_t = \text{factor} \cdot \text{model\_size}^{-0.5}
    \cdot \min(t^{-0.5}, \ t \cdot \text{warmup\_steps}^{-1.5})

Where:
- :math:`t` is the current step (1-indexed).
- :math:`\eta_t` is the scaled learning rate factor.

Methods
-------
- **get_lr() -> list[float]**:
  Returns the scaled learning rates for each optimizer parameter group.

- **step(epoch: Optional[int] = None) -> None**:
  Advances the scheduler, updating optimizer learning rates.

Usage Example
-------------

.. code-block:: python

    import lucid.optim as optim
    from lucid.optim.lr_scheduler import NoamScheduler

    optimizer = optim.Adam(model.parameters(), lr=1.0)
    scheduler = NoamScheduler(
        optimizer,
        model_size=512,
        warmup_steps=4000,
        factor=2.0,
    )

    for step in range(1, 10001):
        optimizer.step()
        scheduler.step()
        if step % 1000 == 0:
            print(f"Step {step}, Learning Rate: {scheduler.last_lr}")

.. note::

    Noam scheduling is effective for Transformer-style architectures where
    large model dimensions benefit from warmup to stabilize early training.
