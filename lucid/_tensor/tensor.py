from typing import Any, Iterator, Optional, Self, SupportsIndex
import numpy as np
import lucid
from lucid._tensor.tensor_ops import _TensorOps
from lucid.types import _ArrayOrScalar, _NumPyArray


class Tensor(_TensorOps):
    """
    The `Tensor` class is a custom implementation resembling PyTorch's
    `Tensor`, with support for automatic differentiation using NumPy
    as a backend.

    Parameters
    ----------
    data : _ArrayOrScalar
        Initial data for the tensor, which is converted to a NumPy array
        if not already.
    requires_grad : bool, optional
        If True, tracks gradients for this tensor. Defaults to `False`.
    keep_grad : bool, optional
        If True, retains gradient information after the backward pass.
        Defaults to `False`.
    dtype : Any, optional
        Data type of the tensor. Defaults to `np.float32`.

    Attributes
    ----------
    data : _NumPyArray
        Underlying data of the tensor, stored as a NumPy array.
    requires_grad : bool
        Indicates if this tensor requires gradients.
    keep_grad : bool
        Determines whether gradients are retained after backward pass.
    dtype : Any
        Data type of the tensor.
    grad : Optional[_NumPyArray]
        Holds the gradient of the tensor, if applicable.
    """

    def __init__(
        self,
        data: _ArrayOrScalar,
        requires_grad: bool = False,
        keep_grad: bool = False,
        dtype: Any = np.float32,
    ) -> None:
        if not isinstance(data, _NumPyArray):
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = data

        self.requires_grad = requires_grad and lucid.grad_enabled()
        self.keep_grad = keep_grad
        self.dtype = self.data.dtype

        self.grad: Optional[_NumPyArray] = None
        self._backward_op: callable = lambda: None
        self._prev: list[Tensor] = []

    @property
    def is_leaf(self) -> bool:
        """
        Indicates if the tensor is a leaf node, meaning it was not
        derived from other tensors.

        Returns
        -------
        bool
            `True` if the tensor is a leaf in the computational graph.
        """
        return self.requires_grad and len(self._prev) == 0

    def backward(self, keep_grad: bool = False) -> None:
        """
        Computes gradients for all tensors involved in producing this tensor.

        Parameters
        ----------
        keep_grad : bool, optional
            If False, clears the gradient after the backward pass unless
            keep_grad is `True` for this tensor. Defaults to `False`.
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        visited = set()
        topo_order: list[Tensor] = []

        def _build_topo(tensor: Tensor) -> None:
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    _build_topo(parent)
                topo_order.append(tensor)

        _build_topo(self)
        topo_order.reverse()

        for tensor in topo_order:
            tensor._backward_op()

            if not tensor.is_leaf and not keep_grad:
                self.zero_grad()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the tensor.

        Returns
        -------
        tuple[int, ...]
            Shape of the tensor data.
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the tensor.

        Returns
        -------
        int
            Number of dimensions.
        """
        return self.data.ndim

    @property
    def size(self) -> int:
        """
        Returns the total number of elements in the tensor.

        Returns
        -------
        int
            Total number of elements in the tensor.
        """
        return self.data.size

    def zero_grad(self) -> None:
        """
        Clears the gradient for this tensor.
        """
        if not self.keep_grad:
            self.grad = None

    def __getitem__(self, idx: SupportsIndex) -> Self:
        sliced_data = self.data[idx]
        new_tensor = Tensor(sliced_data, requires_grad=self.requires_grad)

        def _backward_op() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                np.add.at(self.grad, idx, new_tensor.grad)

        new_tensor._backward_op = _backward_op
        new_tensor._prev = [self]

        return new_tensor

    def __iter__(self) -> Iterator[Self]:
        for i in range(self.shape[0]):
            yield self[i]

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __str__(self) -> str:
        return str(self.data)

    def __hash__(self) -> int:
        return hash(id(self))
