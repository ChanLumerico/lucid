Inception
=========

.. toctree::
    :maxdepth: 1
    :hidden:

    inception_v1.rst
    inception_v3.rst
    inception_v4.rst

.. raw:: html

   <span
     style="
       display: inline-block; padding: 0.15em 0.6em;
       border-radius: 999px; border: 1px solid #ffa600;
       color: #ffa600; background-color: transparent;
       font-size: 0.72em; font-weight: 500;
     "
   >
     ConvNet
   </span>

   <span
     style="
       display: inline-block; padding: 0.15em 0.6em;
       border-radius: 999px; border: 1px solid #707070;
       color: #707070; background-color: transparent;
       font-size: 0.72em; font-weight: 500;
     "
   >
     Image Classification
   </span>

.. autoclass:: lucid.models.Inception

Overview
--------

The `Inception` base class provides a flexible implementation for defining 
various versions of the Inception architecture, including Inception v1, v3, and v4. 

It facilitates the configuration of the feature extraction and classification components 
through arguments, making it adaptable for different versions of the Inception series.

.. image:: inception.png
    :width: 600
    :alt: Inception-v4 architecture
    :align: center

Class Signature
---------------

.. code-block:: python

   class Inception(nn.Module):
       def __init__(self, num_classes: int, use_aux: bool = True) -> None

Parameters
----------

- **num_classes** (*int*)
  The number of output classes for the final classification layer.

- **use_aux** (*bool*, optional)
  Whether to include auxiliary classifiers. Auxiliary classifiers are 
  additional branches used during training to assist optimization. Default is `True`.
