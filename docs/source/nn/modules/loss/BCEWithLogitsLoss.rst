nn.BCEWithLogitsLoss
====================

.. autoclass:: lucid.nn.BCEWithLogitsLoss

The `BCEWithLogitsLoss` module computes the binary cross-entropy loss between 
the input logits and the target labels, applying a sigmoid internally in a numerically 
stable way.  

This loss is particularly recommended for binary classification tasks when 
the model outputs **logits** rather than probabilities.

Compared to `BCELoss`, which expects probability inputs, `BCEWithLogitsLoss` 
is more stable and convenient when dealing with raw, unnormalized scores (logits), 
as it combines the sigmoid activation and BCE loss into a single function, 
avoiding potential floating-point issues.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.BCEWithLogitsLoss(
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean",
    ) -> None

Parameters
----------

- **weight** (*Tensor* or *None*, optional):
  A manual rescaling weight given to the loss of each batch element. 
  Must be of the same shape as the input tensor. Default is `None`.

- **reduction** (*_ReductionType* | *None*, optional):
  Specifies the reduction to apply to the output:

  - `"mean"`: the sum of the output will be divided by the number of elements in the output.
  - `"sum"`: the output will be summed.
  - If set to `None`, no reduction will be applied, and the loss will be returned as is.
  
  Default is `"mean"`.

Attributes
----------

- **weight** (*Tensor* or *None*):
  The manual rescaling weight tensor. Only present if `weight` is provided.

- **reduction** (*_ReductionType* | *None*):
  The reduction method applied to the loss.

Forward Calculation
-------------------

The `BCEWithLogitsLoss` module calculates the binary cross-entropy 
loss between the raw logits and target labels using a numerically stable formula:

.. math::

    \mathcal{L}(\mathbf{x}, \mathbf{y}) = \frac{1}{N} \sum_{i=1}^{N} \left[
        \max(x_i, 0) - x_i \cdot y_i + \log\left(1 + \exp(-|x_i|)\right)
    \right]
    \quad \text{if reduction} = "mean"

Where:

- :math:`\mathbf{x}` is the input tensor containing logits.
- :math:`\mathbf{y}` is the target tensor with binary labels (0 or 1).
- :math:`N` is the number of elements in the tensor.
- :math:`\mathcal{L}` is the computed loss.

Backward Gradient Calculation
-----------------------------

During backpropagation, the gradient of the loss with respect to the input tensor 
is computed as follows:

.. math::

    \frac{\partial \mathcal{L}}{\partial x_i} = \sigma(x_i) - y_i

Where:

- :math:`\sigma(x)` is the sigmoid activation: :math:`\sigma(x) = \frac{1}{1 + \exp(-x)}`
- :math:`x_i` is the :math:`i`-th logit of the input tensor.
- :math:`y_i` is the corresponding binary target value.

This gradient simplifies training by eliminating the need to explicitly apply 
sigmoid in your model.

Examples
--------

**Using `BCEWithLogitsLoss` with logits and targets:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>>
    >>> # Define logits and binary targets
    >>> logits = Tensor([[1.2, -0.7, 0.0]], requires_grad=True)  # Shape: (1, 3)
    >>> targets = Tensor([[1.0, 0.0, 1.0]])  # Shape: (1, 3)
    >>>
    >>> # Initialize loss function
    >>> criterion = nn.BCEWithLogitsLoss()
    >>>
    >>> # Compute loss
    >>> loss = criterion(logits, targets)
    >>> print(loss)
    Tensor([[0.4742]], grad=None)  # Example scalar loss
    >>>
    >>> # Backward pass
    >>> loss.backward()
    >>> print(logits.grad)
    [[-0.2315, 0.3318, -0.2689]]  # grad = sigmoid(logits) - targets

**Using `BCEWithLogitsLoss` with weight and no reduction:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>>
    >>> logits = Tensor([0.5, -0.5, 0.0], requires_grad=True)
    >>> targets = Tensor([1.0, 0.0, 1.0])
    >>> weight = Tensor([1.0, 2.0, 0.5])  # Per-element weighting
    >>>
    >>> criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=None)
    >>> loss = criterion(logits, targets)
    >>> print(loss)
    Tensor([...])  # Element-wise loss with weights applied
    >>> loss.sum().backward()
    >>> print(logits.grad)
    # Gradient is weighted sigmoid(x) - y for each element

**Training a model with `BCEWithLogitsLoss`:**

.. code-block:: python

    >>> class SimpleBinaryClassifier(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear = nn.Linear(in_features=2, out_features=1)
    ...     def forward(self, x):
    ...         return self.linear(x)  # No sigmoid here!
    ...
    >>> model = SimpleBinaryClassifier()
    >>> loss_fn = nn.BCEWithLogitsLoss()
    >>> input_data = Tensor([[0.3, -0.7]], requires_grad=True)
    >>> target = Tensor([[1.0]])
    >>> output = model(input_data)
    >>> loss = loss_fn(output, target)
    >>> loss.backward()
    >>> print(loss)
    Tensor([[0.5731]], grad=None)  # Example scalar loss
    >>> print(input_data.grad)
    # Gradients for input data after backprop
