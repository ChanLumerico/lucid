Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install lucid          # requires macOS + Apple Silicon

Basic Usage
-----------

**Tensors**

.. code-block:: python

   import lucid

   x = lucid.randn(3, 4)          # random float32, CPU
   y = lucid.zeros(3, 4)
   z = x + y

   x.metal()                      # move to Metal GPU
   x.cpu()                        # move back to CPU

**Autograd**

.. code-block:: python

   x = lucid.randn(3)
   x.requires_grad_(True)
   y = (x * x).sum()
   y.backward()
   print(x.grad)                  # 2x

   with lucid.no_grad():
       z = x * 2                  # no graph recorded

**Neural Networks**

.. code-block:: python

   import lucid.nn as nn
   import lucid.optim as optim

   model = nn.Sequential(
       nn.Linear(784, 256),
       nn.ReLU(),
       nn.Linear(256, 10),
   )

   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   for x_batch, y_batch in dataloader:
       pred = model(x_batch)
       loss = nn.CrossEntropyLoss()(pred, y_batch)
       optimizer.zero_grad()
       loss.backward()
       nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       optimizer.step()

**DataLoader**

.. code-block:: python

   from lucid.utils.data import TensorDataset, DataLoader

   ds = TensorDataset(X, y)
   loader = DataLoader(ds, batch_size=32, shuffle=True)

   for x_batch, y_batch in loader:
       ...

Devices
-------

+------------------+--------------------------------------------+
| ``"cpu"``        | Apple Accelerate (vDSP/BLAS/LAPACK)        |
+------------------+--------------------------------------------+
| ``"metal"``      | Apple Metal GPU via MLX unified memory     |
+------------------+--------------------------------------------+

.. code-block:: python

   lucid.set_default_device("metal")   # all new tensors on GPU
   model.metal()                        # move model weights to GPU
