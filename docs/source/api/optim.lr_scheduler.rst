lucid.optim.lr_scheduler
========================

.. currentmodule:: lucid.optim.lr_scheduler

Learning rate schedulers adjust the learning rate of an
:class:`~lucid.optim.Optimizer` according to a pre-defined policy.

Base class
----------

.. autoclass:: LRScheduler
   :members:
   :undoc-members:
   :show-inheritance:

Schedulers
----------

.. autoclass:: StepLR
.. autoclass:: MultiStepLR
.. autoclass:: ConstantLR
.. autoclass:: LinearLR
.. autoclass:: ExponentialLR
.. autoclass:: PolynomialLR
.. autoclass:: CosineAnnealingLR
.. autoclass:: CosineAnnealingWarmRestarts
.. autoclass:: CyclicLR
.. autoclass:: OneCycleLR
.. autoclass:: ReduceLROnPlateau
.. autoclass:: LambdaLR
.. autoclass:: MultiplicativeLR
.. autoclass:: SequentialLR
.. autoclass:: ChainedScheduler
