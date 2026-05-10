lucid.serialization
===================

.. currentmodule:: lucid

Save and load model checkpoints, tensors, and arbitrary state dictionaries.

.. autofunction:: save
.. autofunction:: load

Model state
-----------

Saving and restoring an :class:`~lucid.nn.Module` follows the standard
``state_dict`` / ``load_state_dict`` pattern:

.. code-block:: python

   import lucid
   import lucid.nn as nn

   model = nn.Linear(128, 64)

   # Save
   lucid.save(model.state_dict(), "checkpoint.lucid")

   # Restore
   state = lucid.load("checkpoint.lucid")
   model.load_state_dict(state)
