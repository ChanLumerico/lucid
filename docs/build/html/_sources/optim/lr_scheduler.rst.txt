optim.lr_scheduler
===================

The `optim.lr_scheduler` module provides tools for dynamically 
adjusting learning rates during model training. Learning rate scheduling 
is crucial in deep learning as it helps improve convergence, prevent overshooting, 
and optimize model performance efficiently.

Overview
--------
Learning rate scheduling modifies the optimizer's learning rate over time 
based on predefined strategies. This allows models to start with a relatively 
high learning rate for faster convergence and gradually reduce it to refine optimization. 

.. note::

   Learning rate schedulers do not modify the optimizer itself but adjust the 
   `lr` attribute in the optimizer's parameter groups.

Usage
-----
To use a learning rate scheduler, first, define an optimizer and then wrap 
it with the desired scheduler.

Example using `StepLR`:

.. code-block:: python

    import lucid
    import lucid.nn as nn
    import lucid.optim as optim
    from lucid.optim.lr_scheduler import StepLR

    # Define model and optimizer
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Define scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(20):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.last_lr}")

.. important::
    
    Always call `scheduler.step()` after `optimizer.step()` in each training epoch 
    to update the learning rate properly.

Example using `ExponentialLR`:

.. code-block:: python

    from lucid.optim.lr_scheduler import ExponentialLR
    
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(10):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.last_lr}")

.. caution::

   Ensure that the decay factor (`gamma`) is chosen carefully. 
   A too-small value may cause the learning rate to diminish too quickly.

Conclusion
----------
Learning rate scheduling is a powerful technique to improve model convergence and stability. 
The `optim.lr_scheduler` module provides multiple strategies to suit different 
training needs, ensuring effective model optimization.
