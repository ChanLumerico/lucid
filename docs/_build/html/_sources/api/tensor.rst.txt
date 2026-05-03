lucid.Tensor
============

.. currentmodule:: lucid

.. autoclass:: Tensor
   :members:
   :undoc-members: False
   :show-inheritance:

   .. rubric:: Shape & dtype

   .. autosummary::

      Tensor.shape
      Tensor.dtype
      Tensor.device
      Tensor.ndim
      Tensor.numel
      Tensor.nbytes
      Tensor.element_size
      Tensor.size

   .. rubric:: Autograd

   .. autosummary::

      Tensor.requires_grad
      Tensor.grad
      Tensor.grad_fn
      Tensor.is_leaf
      Tensor.backward
      Tensor.detach
      Tensor.requires_grad_
      Tensor.retain_grad

   .. rubric:: Conversion

   .. autosummary::

      Tensor.item
      Tensor.numpy
      Tensor.tolist
      Tensor.to
      Tensor.cpu
      Tensor.metal
      Tensor.float
      Tensor.half
      Tensor.double

   .. rubric:: Shape manipulation

   .. autosummary::

      Tensor.T
      Tensor.mT
      Tensor.reshape
      Tensor.view
      Tensor.permute
      Tensor.squeeze
      Tensor.unsqueeze
      Tensor.flatten
      Tensor.expand
      Tensor.repeat
      Tensor.contiguous
      Tensor.clone

   .. rubric:: Convenience methods

   .. autosummary::

      Tensor.fill_
      Tensor.zero_
      Tensor.copy_
      Tensor.flip
      Tensor.bmm
      Tensor.addmm
      Tensor.lerp
      Tensor.diff
      Tensor.index_select
      Tensor.masked_select
      Tensor.expand_as
      Tensor.view_as
      Tensor.type_as
      Tensor.new_zeros
      Tensor.new_ones
      Tensor.new_full
      Tensor.new_empty
      Tensor.new_tensor
