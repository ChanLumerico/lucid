Installation
============

Requirements
------------

- **macOS** 12.3 or later (Apple Silicon: M1–M4)
- **Python** 3.12–3.14
- **Xcode Command Line Tools** (for the C++ build)

.. code-block:: bash

   xcode-select --install

Install from PyPI
-----------------

.. code-block:: bash

   pip install lucid

This installs the pre-built wheel with the C++ engine for your
platform.  The ``numpy`` bridge is optional:

.. code-block:: bash

   pip install "lucid[numpy]"

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ChanLumerico/lucid.git
   cd lucid
   pip install -e ".[dev]"     # editable install + dev tools

Build the C++ engine:

.. code-block:: bash

   python setup.py build_ext --inplace

Verify
------

.. code-block:: python

   import lucid
   print(lucid.__version__)        # 3.0.0

   x = lucid.randn(3, 3, device="metal")
   lucid.eval(x)
   print("Metal GPU ready:", x.device)
