# LUCID LEGACY API INVENTORY

**Complete and Exhaustive API Reference** for the Lucid deep learning framework.
Sufficient for full codebase reconstruction from this document alone.

---

## lucid

### __init__.py

**Functions**:

```python
def __dir__() -> list[str]:
```

```python
def __getattr__(name: str) -> Any:
```

```python
def _build_registry_entry(model: nn.Module, *, is_class: bool) -> dict[str, Any]:
```

```python
def _check_input_dim(tensor: Tensor, dim: int) -> None:
```

```python
def _check_is_tensor(any: Tensor | _ArrayOrScalar, device: _DeviceType='cpu', dtype: _BuiltinNumeric | Numeric | None=None) -> Tensor:
```

```python
def _conv_view_limit_mb() -> int:
```

```python
def _get_overloaded_shape(args: int | _ShapeLike) -> _ShapeLike:
```

```python
def _get_registry_category_path(func: _ModuleReturnFunc) -> list[str]:
```

```python
def _has_model_entry(registry: dict[str, Any], path: list[str], model_name: str) -> bool:
```

```python
def _is_legacy_flat_registry(registry: dict[str, Any]) -> bool:
```

```python
def _load_models_registry(path: Path) -> dict[str, Any]:
```

```python
def _match_grad_shape(data: _NumPyArray | _MLXArray, grad: _NumPyArray | _MLXArray, device: _DeviceType='cpu') -> _NumPyArray | _MLXArray:
```

```python
def _maybe_register_model(target: Any, model: nn.Module, *, is_class: bool) -> None:
```

```python
def _migrate_flat_registry(registry: dict[str, Any]) -> dict[str, Any]:
```

```python
def _set_tensor_grad(tensor: Tensor, grad: _NumPyArray | _MLXArray, at: SupportsIndex=...) -> None:
```

```python
def _upsert_model_entry(registry: dict[str, Any], path: list[str], model_name: str, entry: dict[str, int]) -> None:
```

```python
@contextmanager
@contextmanager
```

```python
def flops_enabled() -> bool:
```

```python
def grad_enabled() -> bool:
```

```python
def register_model(target: _ModuleReturnFunc | _ModuleClass) -> _ModuleReturnFunc | _ModuleClass:
```

```python
def shape(a: Tensor | _NumPyArray | _MLXArray) -> _ShapeLike:
```

```python
def tensor(data: Tensor | _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, dtype: _BuiltinNumeric | Numeric | None=None, device: _DeviceType='cpu') -> Tensor:
```

```python
def to_tensor(a: _ArrayLike, requires_grad: bool=False, keep_grad: bool=False, dtype: _BuiltinNumeric | Numeric | None=None, device: _DeviceType='cpu') -> Tensor:
```

**Classes**:

#### class _LucidModule(ModuleType)

**Methods**:

```python
def __setattr__(self, name: str, value: Any) -> None:
```

#### class _NoGrad(AbstractContextManager)

**Methods**:

```python
def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
```

```python
def __enter__(self) -> Self:
```

```python
def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> bool:
```

---

### error.py

**__all__**: `BackwardError`, `DeviceMismatchError`, `UnknownDeviceError`

**Exceptions**:

- **UnknownDeviceError(Exception)**
  - `__init__`
- **DeviceMismatchError(Exception)**
  - `__init__`
- **BackwardError(Exception)**
  - `__init__`

---

### port.py

**__all__**: `load`, `save`

**Constants**:

- `EXTENSIONS` = Literal['.lct', '.lcd', '.safetensors']

**Functions**:

```python
def load(path: Path | str) -> _LucidPortable:
```

```python
def save(obj: _LucidPortable, path: Path | str, safetensors: bool=False) -> Path:
```

---

### types.py

**Functions**:

```python
def to_numeric_type(data_dtype: type) -> Numeric:
```

**Classes**:

#### class Numeric

**Methods**:

```python
def __eq__(self, other: Self) -> bool:
```

```python
def __hash__(self) -> int:
```

```python
def __init__(self, base_dtype: type[int | float | complex], bits: int | None) -> None:
```

```python
def __repr__(self) -> str:
```

```python
def __str__(self) -> str:
```

```python
def _dtype_bits(self, dtype: type) -> int:
```

```python
def auto_parse(self, data_dtype: type, device: _DeviceType) -> type | None:
```

```python
@property
```

```python
@property
```

```python
@property
```

```python
def parse(self, device: _DeviceType) -> type | None:
```

#### class _ModuleHookable(Protocol)

**Methods**:

```python
def register_backward_hook(self, hook: Callable) -> Callable:
```

```python
def register_forward_hook(self, hook: Callable, *, with_kwargs: bool=False) -> Callable:
```

```python
def register_forward_pre_hook(self, hook: Callable, *, with_kwargs: bool=False) -> Callable:
```

```python
def register_full_backward_hook(self, hook: Callable) -> Callable:
```

```python
def register_full_backward_pre_hook(self, hook: Callable) -> Callable:
```

```python
def register_load_state_dict_post_hook(self, hook: Callable) -> Callable:
```

```python
def register_load_state_dict_pre_hook(self, hook: Callable) -> Callable:
```

```python
def register_state_dict_hook(self, hook: Callable) -> Callable:
```

```python
def register_state_dict_pre_hook(self, hook: Callable) -> Callable:
```

#### class _TensorLike(Protocol)

**Attributes**:

- `_backward_hooks`
- `_backward_op`
- `_op`
- `_prev`
- `_version`
- `data`
- `device`
- `dtype`
- ... (7 more)

**Methods**:

```python
def backward(self, retain_grad: bool=False, retain_graph: bool=False) -> None:
```

```python
def clear_node(self, clear_op: bool=True) -> None:
```

```python
def free(self) -> None:
```

```python
def is_cpu(self) -> bool:
```

```python
def is_gpu(self) -> bool:
```

```python
def new_tensor(self) -> _TensorLike:
```

```python
def to(self, device: _DeviceType) -> None:
```

---

## lucid/_backend

### __init__.py

---

### core.py

**Functions**:

```python
def _py_func_op(n_in: int | None, n_ret: int | None, has_gradient: bool=True, device: _DeviceType='cpu') -> Callable:
```

```python
def binary_func_op(has_gradient: bool=True, device: _DeviceType='cpu') -> Callable:
```

```python
def fallback(cls: type[Operation]) -> type[Operation]:
```

```python
def func_op(n_in: int | None, n_ret: int | None, has_gradient: bool=True, device: _DeviceType='cpu') -> Callable:
```

```python
def poly_func_op(has_gradient: bool=True, device: _DeviceType='cpu') -> Callable:
```

```python
def unary_func_op(has_gradient: bool=True, device: _DeviceType='cpu') -> Callable:
```

**Classes**:

#### class BackwardOperation

**Methods**:

```python
def __call__(self) -> None:
```

```python
def __init__(self, forward_op_ref: weakref.ref[Operation] | None, grad_func: _GradFuncType | None, tensor_refs: tuple[weakref.ref[_TensorLike]], versions: tuple[int, ...]=(), device: _DeviceType | None='cpu', custom_closure: Callable[[], None] | None=None) -> None:
```

```python
def override_grad_func(self, new_grad_func: _GradFuncType) -> None:
```

```python
def override_tensor_refs(self, new_tensor_refs: tuple[weakref.ref[_TensorLike]], new_versions: tuple[int, ...] | None=None) -> None:
```

#### class Operation(ABC)

**Attributes**:

- `__fallback__`

**Methods**:

```python
def __call__(self, *args, **kwargs) -> _TensorLike | tuple[_TensorLike, ...]:
```

```python
def __flops__(self, *args, **kwargs) -> int:
```

```python
def __grad__(self, *args, **kwargs) -> _GradType:
```

```python
def __grad_cpu__(self, *args, **kwargs) -> _GradType:
```

```python
def __grad_gpu__(self, *args, **kwargs) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
def clear(self) -> None:
```

```python
@abstractmethod
```

```python
@property
```

```python
@flops.setter
```

```python
@abstractmethod
```

```python
def inplace(self, target: int=0) -> Self:
```

---

### metal.py

**Functions**:

```python
def check_metal_availability() -> None:
```

```python
def is_cpu_op(*tensor_or_any) -> bool:
```

```python
def is_gpu_op(*tensor_or_any) -> bool:
```

```python
def parse_mlx_indexing(index: Any) -> Any:
```

```python
def post_step_eval(param: _TensorLike | Any, state: Mapping[str, Any] | None=None) -> None:
```

**Classes**:

#### class MetalNotSupportedWarning(UserWarning)

**Attributes**:

- `_has_warned`

**Methods**:

```python
def __init__(self, message=None):
```

---

## lucid/_func

### __init__.py

**__all__**: `abs`, `add`, `arange`, `arccos`, `arcsin`, `arctan`, `ceil`, `clip`, `cos`, `cosh`, `cube`, `cumprod`, `cumsum`, `diag`, `div`, `dot`, `empty`, `empty_like`, `exp`, `eye`, `floor`, `full`, `full_like`, `inner`, `linspace`, `log`, `log2`, `matmul`, `max`, `maximum`, `mean`, `min`, `minimum`, `multiply`, `ones`, `ones_like`, `outer`, `power`, `reciprocal`, `round`, `sign`, `sin`, `sinh`, `sqrt`, `square`, `sub`, `sum`, `swapaxes`, `tan`, `tanh`, `tensordot`, `trace`, `transpose`, `var`, `zeros`, `zeros_like`

**Functions**:

```python
@property
@property
```

```python
def __check_int_bool_dtype(*ts: Tensor) -> None:
```

```python
def _bitwise_and(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def _bitwise_or(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def _equal(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def _greater(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def _greater_or_equal(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def _invert(a: Tensor, /) -> Tensor:
```

```python
def _less(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def _less_or_equal(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
@property
@property
```

```python
def _neg(a: Tensor, /) -> Tensor:
```

```python
def _not_equal(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def _pow(a: Tensor, /, exp: _Scalar) -> Tensor:
```

```python
def _rpow(a: Tensor, /, base: _Scalar) -> Tensor:
```

```python
def abs(a: Tensor, /) -> Tensor:
```

```python
def abs_(a: Tensor, /) -> Tensor:
```

```python
def add(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def add_(a: Tensor, b: Tensor) -> Tensor:
```

```python
@overload
@overload
```

```python
@overload
@overload
```

```python
@overload
@overload
```

```python
def arange(*args, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def arccos(a: Tensor, /) -> Tensor:
```

```python
def arccos_(a: Tensor, /) -> Tensor:
```

*...and 85 more functions*

---

### bfunc.py

**Functions**:

```python
def _broadcast_flops(a: Tensor, b: Tensor) -> int:
```

**Classes**:

#### class _bitwise_and(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class _bitwise_or(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class _equal(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class _greater(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class _greater_or_equal(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class _less(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class _less_or_equal(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class _not_equal(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class add(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor, b: Tensor) -> int:
```

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op()
```

```python
@binary_func_op(device='gpu')
```

#### class dot(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor, b: Tensor) -> int:
```

```python
def __grad_cpu__(self, a: Tensor, b: Tensor) -> _GradType:
```

```python
def __grad_gpu__(self, a: Tensor, b: Tensor) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op()
```

```python
@binary_func_op(device='gpu')
```

#### class floordiv(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor, b: Tensor) -> int:
```

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op(has_gradient=False)
```

```python
@binary_func_op(has_gradient=False, device='gpu')
```

#### class inner(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor, b: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, b: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op()
```

```python
@binary_func_op(device='gpu')
```

*...and 9 more classes*

---

### gfunc.py

**Functions**:

```python
def _get_tensor_specs(a: Tensor | _ArrayLike, dtype: _BuiltinNumeric | Numeric | None, device: _DeviceType | None) -> tuple[_ArrayLike, type, _DeviceType]:
```

```python
def arange(start: _Scalar, stop: _Scalar, step: _Scalar, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def diag(v: Tensor | _ArrayLike, k: int=0, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType | None=None) -> Tensor:
```

```python
def empty(shape: int | _ShapeLike, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def empty_like(a: Tensor | _ArrayLike, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType | None=None) -> Tensor:
```

```python
def eye(N: int, M: int | None=None, k: int=0, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def full(shape: int | _ShapeLike, fill_value: _Scalar, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def full_like(a: Tensor | _ArrayLike, fill_value: _Scalar, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType | None=None) -> Tensor:
```

```python
def linspace(start: _Scalar, stop: _Scalar, num: int=50, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def ones(shape: _ShapeLike, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def ones_like(a: Tensor | _ArrayLike, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType | None=None) -> Tensor:
```

```python
def zeros(shape: _ShapeLike, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def zeros_like(a: Tensor | _ArrayLike, dtype: _BuiltinNumeric | Numeric | None=None, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType | None=None) -> Tensor:
```

---

### ufunc.py

**Functions**:

```python
def _normalize_axis(axis: int | tuple[int] | None, ndim: int) -> tuple[int, ...]:
```

**Classes**:

#### class _T(Operation)

**Methods**:

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class _abs(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class _invert(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op(has_gradient=False)
```

```python
@unary_func_op(has_gradient=False, device='gpu')
```

#### class _mT(Operation)

**Methods**:

```python
def __grad_cpu__(self) -> _GradType:
```

```python
def __grad_gpu__(self) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class _min_or_max(Operation)

**Methods**:

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, mode: Literal['min', 'max'], axis: int | tuple[int] | None, keepdims: bool) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class _neg(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class _pow(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor) -> _GradType:
```

```python
def __init__(self, exp: _Scalar) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class _rpow(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor) -> _GradType:
```

```python
def __init__(self, base: _Scalar) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class arccos(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class arcsin(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class arctan(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class ceil(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> __init__:
```

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op(has_gradient=False)
```

```python
@unary_func_op(has_gradient=False, device='gpu')
```

*...and 25 more classes*

---

## lucid/_fusion

### __init__.py

---

### base.py

**__all__**: `FusedBackwardOp`, `match_fusion_table`

**Functions**:

```python
def match_fusion_table(op1: Operation, op2: Operation) -> type[FusedBackwardOp] | None:
```

**Classes**:

#### class FusedBackwardOp(ABC)

**Attributes**:

- `heuristic_thresh`
- `op1`
- `op2`

**Methods**:

```python
@classmethod
```

```python
@classmethod
```

```python
@classmethod
```

```python
@classmethod
```

```python
@classmethod
```

```python
def __init_subclass__(cls, **kwargs) -> None:
```

```python
def __new__(cls, *args, **kwargs) -> Never:
```

```python
@classmethod
```

---

### func.py

**__all__**: `DoubleMT`, `DoubleNeg`, `DoubleReciprocal`, `DoubleReshape`, `DoubleT`, `LogExp`, `SqueezeUnsqueeze`, `UnsqueezeSqueeze`

**Classes**:

#### class DoubleMT(_IdentityFusion)

#### class DoubleNeg(_IdentityFusion)

#### class DoubleReciprocal(_IdentityFusion)

#### class DoubleReshape(_IdentityViewFusion)

#### class DoubleT(_IdentityFusion)

#### class LogExp(_IdentityFusion)

#### class SqueezeUnsqueeze(_IdentityViewFusion)

#### class UnsqueezeSqueeze(_IdentityViewFusion)

#### class _DoubleReshapeImmediate(_IdentityViewFusion)

#### class _IdentityFusion(FusedBackwardOp)

**Methods**:

```python
@classmethod
```

#### class _IdentityViewFusion(FusedBackwardOp)

**Methods**:

```python
@classmethod
```

---

## lucid/_jit

### __init__.py

**__all__**: `JITFunction`, `JITModule`, `compile`

---

### api.py

**Functions**:

```python
def _attach_compiled_backward(outputs: tuple, fwd_result: ForwardResult, graph: IRGraph) -> None:
```

```python
def _collect_tensor_inputs(args: tuple, kwargs: dict) -> tuple:
```

```python
def _trace_and_compile(fn: Callable, args: tuple, kwargs: dict, key: CacheKey, param_tensors: tuple=()) -> CompiledPlan:
```

```python
@overload
@overload
```

```python
@overload
@overload
```

```python
def compile(target: Any, *, max_cache_entries: int=8) -> JITFunction | JITModule:
```

**Classes**:

#### class JITFunction

**Methods**:

```python
def __call__(self, *args: Any, **kwargs: Any) -> Any:
```

```python
def __init__(self, fn: Callable, *, max_cache_entries: int=8) -> None:
```

```python
def __repr__(self) -> str:
```

```python
def _run_plan(self, plan: CompiledPlan, inputs: tuple, param_map: dict, grad_enabled: bool) -> Any:
```

```python
def invalidate_cache(self) -> None:
```

#### class JITModule

**Methods**:

```python
def __call__(self, *args: Any, **kwargs: Any) -> Any:
```

```python
def __getattr__(self, name: str) -> Any:
```

```python
def __init__(self, module: Any, *, max_cache_entries: int=8) -> None:
```

```python
def __repr__(self) -> str:
```

```python
def _build_param_map(self, graph) -> dict:
```

```python
def invalidate_cache(self) -> None:
```

---

### cache.py

**Classes**:

#### class CacheKey

**Attributes**:

- `grad_enabled`
- `shape_dtypes`
- `training_mode`

**Methods**:

```python
@classmethod
```

#### class PlanCache

**Methods**:

```python
def __init__(self, max_entries: int=8) -> None:
```

```python
def __len__(self) -> int:
```

```python
def get(self, key: CacheKey) -> CompiledPlan | None:
```

```python
def invalidate(self) -> None:
```

```python
def put(self, key: CacheKey, plan: CompiledPlan) -> None:
```

---

### executor.py

**Classes**:

#### class BackwardExecutor

**Methods**:

```python
def execute(self, forward_result: ForwardResult, output_vids: list, output_upstream_grads: dict, retain_grad: bool=False) -> None:
```

#### class CompiledPlan

**Methods**:

```python
def __init__(self, graph: IRGraph, training: bool, output_treespec: object=None) -> None:
```

```python
def __repr__(self) -> str:
```

#### class ForwardExecutor

**Methods**:

```python
def execute(self, plan: CompiledPlan, inputs: tuple, param_map: dict, grad_enabled: bool) -> tuple:
```

#### class ForwardResult

**Attributes**:

- `exec_order`
- `grad_func_map`
- `leaf_vids`
- `node_device_map`
- `value_map`

---

### ir.py

**Classes**:

#### class IRGraph

**Attributes**:

- `input_ids`
- `nodes`
- `output_ids`
- `param_ids`
- `values`

**Methods**:

```python
@property
```

#### class IRNode

**Attributes**:

- `device`
- `fused_forward_kernel`
- `has_gradient`
- `input_ids`
- `n_in`
- `n_ret`
- `node_id`
- `non_tensor_args`
- ... (5 more)

#### class IRValue

**Attributes**:

- `live_tensor`
- `shape_spec`
- `value_id`

#### class ShapeSpec

**Attributes**:

- `device`
- `dtype`
- `shape`

---

### passes.py

**Functions**:

```python
def run_passes(graph: IRGraph, passes: list[Pass]) -> IRGraph:
```

**Classes**:

#### class DeadNodeElimPass(Pass)

**Methods**:

```python
def run(self, graph: IRGraph) -> IRGraph:
```

#### class NoGradStripPass(Pass)

**Methods**:

```python
def run(self, graph: IRGraph) -> IRGraph:
```

#### class Pass(ABC)

**Attributes**:

- `name`

**Methods**:

```python
@abstractmethod
```

---

### pytree.py

**Functions**:

```python
def _is_namedtuple(value: Any) -> bool:
```

```python
def _unflatten(flat: list[_Tensor], treespec: Any, offset: int) -> tuple[int, Any]:
```

```python
def flatten_output(value: Any) -> tuple[list[_Tensor], Any]:
```

```python
def unflatten_output(flat: list[_Tensor], treespec: Any) -> Any:
```

---

### tracer.py

**Functions**:

```python
def _capture_op_init_state(op_self: Any) -> tuple[tuple, dict]:
```

```python
def _get_active_tracer() -> TracingContext | None:
```

```python
def is_tracing() -> bool:
```

**Classes**:

#### class TracingContext

**Methods**:

```python
def __init__(self) -> None:
```

```python
def finalize(self, output_tensors: tuple) -> IRGraph:
```

```python
def record_op(self, op_self: Any, input_tensors: tuple, output_tensors: tuple, non_tensor_args: tuple, non_tensor_kwargs: dict, device: str, has_gradient: bool, n_in: int | None, n_ret: int | None) -> None:
```

```python
def register_tensor(self, tensor: Any, *, is_input: bool=False, is_param: bool=False) -> int:
```

---

## lucid/_tensor

### __init__.py

---

### base.py

**Classes**:

#### class _TensorBase

**Methods**:

```python
@property
```

```python
def __add__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __and__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __eq__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __floordiv__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __ge__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __gt__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __invert__(self) -> Self:
```

```python
def __le__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __lt__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __matmul__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __mul__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __ne__(self, other: Self | _ArrayOrScalar) -> Self:
```

```python
def __neg__(self) -> Self:
```

```python
def __or__(self, other: Self | _ArrayOrScalar) -> Self:
```

*...and 45 more methods*

#### class _TensorInplace

**Methods**:

```python
def abs_(self) -> Self:
```

```python
def add_(self, other: Self) -> Self:
```

```python
def arccos_(self) -> Self:
```

```python
def arcsin_(self) -> Self:
```

```python
def arctan_(self) -> Self:
```

```python
def ceil_(self) -> Self:
```

```python
def clip_(self, min_value: _Scalar | None=None, max_value: _Scalar | None=None) -> Self:
```

```python
def cos_(self) -> Self:
```

```python
def cosh_(self) -> Self:
```

```python
def cube_(self) -> Self:
```

```python
def div_(self, other: Self, /, floor: bool=False) -> Self:
```

```python
def exp_(self) -> Self:
```

```python
def floor_(self) -> Self:
```

```python
def log2_(self) -> Self:
```

```python
def log_(self) -> Self:
```

*...and 15 more methods*

---

### tensor.py

**__all__**: `BoolTensor`, `CharTensor`, `DoubleTensor`, `FloatTensor`, `HalfTensor`, `IntTensor`, `LongTensor`, `ShortTensor`, `Tensor`

**Classes**:

#### class BoolTensor(Tensor[bool])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class CharTensor(Tensor[types.Int8])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class DoubleTensor(Tensor[types.Float64])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class FloatTensor(Tensor[types.Float32])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class HalfTensor(Tensor[types.Float16])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class IntTensor(Tensor[types.Int32])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class LongTensor(Tensor[types.Int64])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class ShortTensor(Tensor[types.Int16])

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> None:
```

#### class Tensor(Generic[DType], _TensorBase, _TensorInplace)

**Attributes**:

- `_fixed_dtype`

**Methods**:

```python
def __bool__(self) -> bool:
```

```python
def __deepcopy__(self, *args: Any) -> Self:
```

```python
def __getitem__(self, idx: SupportsIndex | Self) -> Self:
```

```python
def __hash__(self) -> int:
```

```python
def __init__(self, data: _ArrayOrScalar, requires_grad: bool=False, keep_grad: bool=False, dtype: _BuiltinNumeric | Numeric | None=None, device: _DeviceType='cpu') -> None:
```

```python
def __iter__(self) -> Iterator[Self]:
```

```python
def __len__(self) -> int:
```

```python
def __repr__(self) -> str:
```

```python
def __setitem__(self, idx: SupportsIndex | Self, value: Self | _ArrayOrScalar) -> None:
```

```python
def __str__(self) -> str:
```

```python
def all(self, axis=None, keepdims=False) -> bool | Self:
```

```python
def any(self, axis: int | None=None, keepdims: bool=False) -> bool | Self:
```

```python
def astype(self, dtype: type | Numeric) -> Self:
```

```python
def backward(self, retain_grad: bool=False, retain_graph: bool=False) -> None:
```

```python
def bool(self) -> Self:
```

*...and 35 more methods*

---

## lucid/_utils

### __init__.py

**__all__**: `argmax`, `argmin`, `argsort`, `broadcast_to`, `chunk`, `concatenate`, `diagonal`, `expand`, `expand_dims`, `flatten`, `gather`, `histogram`, `histogram2d`, `histogramdd`, `hstack`, `masked_fill`, `meshgrid`, `nonzero`, `nonzero`, `pad`, `ravel`, `repeat`, `reshape`, `roll`, `sort`, `split`, `squeeze`, `stack`, `tile`, `topk`, `tril`, `triu`, `unbind`, `unique`, `unsqueeze`, `vstack`, `where`

**Functions**:

```python
def _reshape_immediate(a: Tensor, /, *shape: int | _ShapeLike) -> Tensor:
```

```python
def argmax(a: Tensor, axis: int | None=None, keepdims: bool=False) -> Tensor:
```

```python
def argmin(a: Tensor, axis: int | None=None, keepdims: bool=False) -> Tensor:
```

```python
def argsort(a: Tensor, /, axis: int=-1, descending: bool=False, kind: _SortKind | None=None, stable: bool=False) -> Tensor:
```

```python
def broadcast_to(a: Tensor, /, shape: _ShapeLike) -> Tensor:
```

```python
def chunk(a: Tensor, /, chunks: int, axis: int=0) -> tuple[Tensor, ...]:
```

```python
def concatenate(arr: tuple[Tensor, ...], /, axis: int=0) -> Tensor:
```

```python
def diagonal(a: Tensor, /, offset: int=0, axis1: int=0, axis2: int=1) -> Tensor:
```

```python
def expand(a: Tensor, /, *sizes: int | _ShapeLike) -> Tensor:
```

```python
def expand_dims(a: Tensor, /, axis: _ShapeLike) -> Tensor:
```

```python
def flatten(a: Tensor, /, start_axis: int=0, end_axis: int=-1) -> Tensor:
```

```python
def gather(a: Tensor, /, axis: int, index: Tensor) -> Tensor:
```

```python
def histogram(a: Tensor, /, bins: int=10, range: tuple[float, float] | None=None, density: bool=False) -> tuple[Tensor, Tensor]:
```

```python
def histogram2d(a: Tensor, b: Tensor, /, bins: list[int, int]=[10, 10], range: list[tuple[float, float]] | None=None, density: bool=False) -> tuple[Tensor, Tensor]:
```

```python
def histogramdd(a: Tensor, /, bins: int | list[int], range: list[tuple[float, float]], density: bool=False) -> tuple[Tensor, Tensor]:
```

```python
def hstack(arr: tuple[Tensor, ...], /) -> Tensor:
```

```python
def masked_fill(a: Tensor, /, mask: Tensor, value: _Scalar) -> Tensor:
```

```python
def meshgrid(a: Tensor, b: Tensor, /, indexing: Literal['xy', 'ij']='ij') -> tuple[Tensor, Tensor]:
```

```python
def nonzero(a: Tensor, /) -> Tensor:
```

```python
def pad(a: Tensor, /, pad_width: _ArrayLikeInt) -> Tensor:
```

```python
def ravel(a: Tensor, /) -> Tensor:
```

```python
def repeat(a: Tensor, /, repeats: int | Sequence[int], axis: int | None=None) -> Tensor:
```

```python
def reshape(a: Tensor, /, shape: _ShapeLike) -> Tensor:
```

```python
@overload
@overload
```

```python
def roll(a: Tensor, /, shifts: int | tuple[int, ...], axis: int | tuple[int, ...] | None=None) -> Tensor:
```

*...and 13 more functions*

---

### func.py

**Classes**:

#### class _reshape_immediate(Operation)

**Methods**:

```python
def __grad__(self, a: Tensor) -> _GradType:
```

```python
def __init__(self, shape: _ShapeLike) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class argmax(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, axis: int | None=None, keepdims: bool=False) -> None:
```

```python
@unary_func_op(has_gradient=False)
```

```python
@unary_func_op(has_gradient=False, device='gpu')
```

#### class argmin(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, axis: int=None, keepdims: bool=False) -> None:
```

```python
@unary_func_op(has_gradient=False)
```

```python
@unary_func_op(has_gradient=False, device='gpu')
```

#### class argsort(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, axis: int=-1, descending: bool=False, kind: _SortKind | None=None, stable: bool=False) -> None:
```

```python
@unary_func_op(has_gradient=False)
```

```python
@unary_func_op(has_gradient=False, device='gpu')
```

#### class broadcast_to(Operation)

**Methods**:

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self, shape: _ShapeLike) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class chunk(Operation)

**Methods**:

```python
def __init__(self, chunks: int, axis: int) -> None:
```

```python
def _unified(self, a: Tensor, lib_: ModuleType) -> _FuncOpReturnType:
```

```python
@func_op(n_in=1, n_ret=None)
```

```python
@func_op(n_in=1, n_ret=None, device='gpu')
```

#### class concatenate(Operation)

**Methods**:

```python
def __grad__(self, arr: tuple[Tensor, ...]) -> tuple:
```

```python
def __init__(self, axis: int) -> None:
```

```python
@poly_func_op()
```

```python
@poly_func_op(device='gpu')
```

#### class diagonal(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, lib_) -> _GradType:
```

```python
def __init__(self, offset: int=0, axis1: int=0, axis2: int=1) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class expand(Operation)

**Methods**:

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self, shape: _ShapeLike) -> None:
```

```python
def _resolve_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class expand_dims(unsqueeze)

#### class flatten(Operation)

**Methods**:

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self, start_axis: int=0, end_axis: int=-1) -> None:
```

```python
def _unified(self, a: Tensor) -> _FuncOpReturnType:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class gather(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor, index: Tensor) -> int:
```

```python
def __grad_cpu__(self, a: Tensor, index: Tensor) -> _GradType:
```

```python
def __grad_gpu__(self, a: Tensor, index: Tensor) -> _GradType:
```

```python
def __init__(self, axis: int) -> None:
```

```python
def _normalize_dim(self, ndim: int) -> int:
```

```python
def _scatter_grad_np(self, a_shape: tuple[int, ...], index_np: _NumPyArray, grad_np: _NumPyArray, dim: int) -> _NumPyArray:
```

```python
def _slice_to_index_extents(self, x: _TensorData, index_shape: tuple[int, ...], dim: int) -> _TensorData:
```

```python
def _validate_inputs(self, a: Tensor, index: Tensor) -> int:
```

```python
@binary_func_op()
```

```python
@binary_func_op(device='gpu')
```

*...and 23 more classes*

---

## lucid/autograd

### __init__.py

**__all__**: `backward`, `grad`

**Functions**:

```python
def _as_tuple(value: _TensorLike | Sequence[_TensorLike]) -> tuple[_TensorLike, ...]:
```

```python
def _coerce_grad_output(output: _TensorLike, grad_output: _TensorLike | _Gradient | _Scalar) -> _Gradient:
```

```python
def _try_backward_fusion(topo_order: list[_TensorLike]) -> None:
```

```python
def backward(tensor: _TensorLike, retain_grad: bool=False, retain_graph: bool=False) -> None:
```

```python
def grad(outputs: _TensorLike | Iterable[_TensorLike], inputs: _TensorLike | Iterable[_TensorLike], grad_outputs: _TensorLike | Iterable[_TensorLike] | Iterable[_Scalar] | None=None, retain_graph: bool=False, allow_unused: bool=False) -> tuple[_Gradient, ...] | _Gradient:
```

---

## lucid/einops

### __init__.py

**__all__**: `einsum`, `rearrange`, `reduce`, `repeat`

**Functions**:

```python
def einsum(pattern: str, *tensors: Tensor) -> Tensor:
```

```python
def rearrange(a: Tensor, /, pattern: _EinopsPattern, **shapes: int) -> Tensor:
```

```python
def reduce(a: Tensor, /, pattern: _EinopsPattern, reduction: _ReduceStr='sum', **shapes: int) -> Tensor:
```

```python
def repeat(a: Tensor, /, pattern: _EinopsPattern, **shapes: int) -> Tensor:
```

---

### _func.py

**Functions**:

```python
def _build_intermediate(input_tokens: list[str | tuple[str, ...]], shape: tuple[int, ...], shapes: dict) -> tuple[list[str], list[int]]:
```

```python
def _parse_pattern(pattern_side: str) -> list[str | tuple[str, ...]]:
```

**Classes**:

#### class einsum(Operation)

**Methods**:

```python
def __grad__(self, arr: tuple[Tensor, ...], lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, pattern: str) -> None:
```

```python
def _parse_equation(self, n_in: int) -> tuple[list[str], str]:
```

```python
@poly_func_op()
```

```python
@poly_func_op(device='gpu')
```

#### class rearrange(Operation)

**Methods**:

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, pattern: _EinopsPattern, t_shape: _ShapeLike, **shapes: int) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class reduce(Operation)

**Methods**:

```python
def __flops__(self, _) -> int:
```

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, pattern: _EinopsPattern, reduction: _ReduceStr, t_shape: _ShapeLike, **shapes: int) -> None:
```

```python
def _unified(self, arr: _NumPyArray | _MLXArray) -> _NumPyArray | _MLXArray:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class repeat(Operation)

**Methods**:

```python
def __flops__(self, _) -> int:
```

```python
def __grad__(self, a: Tensor) -> _GradType:
```

```python
def __init__(self, pattern: _EinopsPattern, t_shape: _ShapeLike, **shapes: int) -> None:
```

```python
def _unified(self, arr: _NumPyArray | _MLXArray, lib_: ModuleType) -> _NumPyArray | _MLXArray:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

---

## lucid/linalg

### __init__.py

**__all__**: `cholesky`, `det`, `inv`, `matrix_power`, `norm`, `pinv`, `qr`, `solve`, `svd`

**Functions**:

```python
def cholesky(a: Tensor, /) -> Tensor:
```

```python
def det(a: Tensor, /) -> Tensor:
```

```python
def eig(a: Tensor, /, eps: float=1e-12) -> tuple[Tensor, Tensor]:
```

```python
def inv(a: Tensor, /) -> Tensor:
```

```python
def matrix_power(a: Tensor, /, n: int) -> Tensor:
```

```python
def norm(a: Tensor, /, ord: int=2, axis: tuple[int, ...] | int | None=None, keepdims: bool=False) -> Tensor:
```

```python
def pinv(a: Tensor, /, rcond: float=1e-12) -> Tensor:
```

```python
def qr(a: Tensor, /) -> tuple[Tensor, Tensor]:
```

```python
def solve(a: Tensor, b: Tensor, /) -> Tensor:
```

```python
def svd(a: Tensor, /, full_matrices: bool=True) -> tuple[Tensor, Tensor, Tensor]:
```

---

### _func.py

**Classes**:

#### class cholesky(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class det(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad_cpu__(self, a: Tensor) -> _GradType:
```

```python
def __grad_gpu__(self, a: Tensor) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class eig(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad_eigvals__(self, _fallback: bool=False) -> _GradType:
```

```python
def __grad_eigvecs__(self, _fallback: bool=False) -> _GradType:
```

```python
def __init__(self, eps: float) -> None:
```

```python
def _unified(self, a: Tensor) -> tuple[_NumPyArray, _NumPyArray]:
```

```python
@func_op(n_in=1, n_ret=2)
```

```python
@func_op(n_in=1, n_ret=2, device='gpu')
```

#### class inv(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad_cpu__(self) -> _GradType:
```

```python
def __grad_gpu__(self) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class matrix_power(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, n: int) -> None:
```

```python
def _gpu_matrix_pow(self, arr: _MLXArray, n: int) -> _MLXArray:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class norm(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, ord: int=2, axis: int | tuple[int, ...] | None=None, keepdims: bool=False) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class pinv(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, rcond: float) -> None:
```

```python
@unary_func_op()
```

```python
@unary_func_op(device='gpu')
```

#### class qr(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad_q__(self, lib_: ModuleType) -> _GradType:
```

```python
def __grad_r__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@func_op(n_in=1, n_ret=2)
```

```python
@func_op(n_in=1, n_ret=2, device='gpu')
```

#### class solve(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor, b: Tensor) -> int:
```

```python
def __grad_cpu__(self, a: Tensor) -> _GradType:
```

```python
def __grad_gpu__(self, a: Tensor) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
@binary_func_op()
```

```python
@binary_func_op(device='gpu')
```

#### class svd(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad_s__(self, lib_: ModuleType) -> _GradType:
```

```python
def __grad_u__(self, lib_: ModuleType) -> _GradType:
```

```python
def __grad_vt__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, full_matrices: bool) -> None:
```

```python
@func_op(n_in=1, n_ret=3)
```

```python
@func_op(n_in=1, n_ret=3, device='gpu')
```

---

## lucid/nn

### __init__.py

---

### cache.py

**__all__**: `Cache`, `DynamicKVCache`, `EncoderDecoderCache`, `KVCache`, `StaticKVCache`

**Classes**:

#### class Cache(ABC)

**Methods**:

```python
@abstractmethod
```

```python
@abstractmethod
```

```python
@abstractmethod
```

```python
def get_max_cache_shape(self) -> int | None:
```

```python
def reorder_cache(self, beam_idx: Tensor) -> None:
```

```python
@abstractmethod
```

```python
def update(self, *args, **kwargs) -> Any:
```

#### class DynamicKVCache(KVCache)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def _crop_layer(self, layer_idx: int, max_length: int) -> None:
```

```python
def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
```

```python
def get_seq_length(self, layer_idx: int=0) -> int:
```

```python
def reset(self) -> None:
```

```python
def update(self, key: Tensor, value: Tensor, layer_idx: int, cache_position: Tensor | None=None) -> tuple[Tensor, Tensor]:
```

#### class EncoderDecoderCache(Cache)

**Methods**:

```python
def __init__(self, self_attention_cache: KVCache | None=None, cross_attention_cache: KVCache | None=None) -> None:
```

```python
@staticmethod
```

```python
def batch_repeat_interleave(self, repeats: int) -> None:
```

```python
def batch_select_indices(self, indices: Tensor) -> None:
```

```python
def crop(self, max_length: int) -> None:
```

```python
def get(self, layer_idx: int, is_cross_attention: bool=False) -> tuple[Tensor, Tensor] | None:
```

```python
@override
```

```python
def get_seq_length(self, layer_idx: int=0, is_cross_attention: bool=False) -> int:
```

```python
def reset(self) -> None:
```

```python
def update(self, key: Tensor, value: Tensor, layer_idx: int, cache_position: Tensor | None=None, is_cross_attention: bool=False) -> tuple[Tensor, Tensor]:
```

#### class KVCache(Cache)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def _check_key_value_shape(self, key: Tensor, value: Tensor) -> None:
```

```python
def _check_valid_layer_idx(self, layer_idx: int, dynamic: bool=False) -> None:
```

```python
def _coerce_index_device(self, index: Tensor, device: str) -> Tensor:
```

```python
@abstractmethod
```

```python
def batch_repeat_interleave(self, repeats: int) -> None:
```

```python
def batch_select_indices(self, indices: Tensor) -> None:
```

```python
def crop(self, max_length: int) -> None:
```

```python
@abstractmethod
```

```python
@abstractmethod
```

```python
@override
```

#### class StaticKVCache(KVCache)

**Methods**:

```python
def __init__(self, max_cache_len: int, num_layers: int) -> None:
```

```python
def _crop_layer(self, layer_idx: int, max_length: int) -> None:
```

```python
def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
```

```python
@override
```

```python
def get_seq_length(self, layer_idx: int=0) -> int:
```

```python
def reset(self) -> None:
```

```python
def update(self, key: Tensor, value: Tensor, layer_idx: int, cache_position: Tensor | None=None) -> tuple[Tensor, Tensor]:
```

---

### fused.py

**__all__**: `ConvBNReLU1d`, `ConvBNReLU2d`, `ConvBNReLU3d`, `DepthSeparableConv1d`, `DepthSeparableConv2d`, `DepthSeparableConv3d`, `SEModule`, `SelectiveKernel`

**Constants**:

- `_BN` = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

**Classes**:

#### class ConvBNReLU1d(_ConvBNReLU)

**Attributes**:

- `D`

#### class ConvBNReLU2d(_ConvBNReLU)

**Attributes**:

- `D`

#### class ConvBNReLU3d(_ConvBNReLU)

**Attributes**:

- `D`

#### class DepthSeparableConv1d(_DepthSeparableConv)

**Attributes**:

- `D`

#### class DepthSeparableConv2d(_DepthSeparableConv)

**Attributes**:

- `D`

#### class DepthSeparableConv3d(_DepthSeparableConv)

**Attributes**:

- `D`

#### class SEModule(nn.Module)

**Methods**:

```python
def __init__(self, in_channels: int, reduction: int=16) -> None:
```

```python
def forward(self, x: Tensor) -> Tensor:
```

#### class SelectiveKernel(nn.Module)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int], stride: int=1, padding: _PaddingStr | None=None, groups: int=1, reduction: int=16) -> None:
```

```python
def forward(self, x: Tensor) -> Tensor:
```

#### class _ConvBNReLU(nn.Module)

**Attributes**:

- `D`

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, conv_bias: bool=True, eps: float=1e-05, momentum: float | None=0.1, bn_affine: bool=True, track_running_stats: bool=True) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class _DepthSeparableConv(nn.Module)

**Attributes**:

- `D`

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, bias: bool=True) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

---

### module.py

**__all__**: `Module`, `ModuleDict`, `ModuleList`, `ParameterDict`, `ParameterList`, `Sequential`, `auto_repr`, `set_state_dict_pass_attr`

**Constants**:

- `T` = TypeVar('T', bound=Type[Module])

**Functions**:

```python
def _add_indent(s: str, num_spaces: int) -> str:
```

```python
def auto_repr(*attr_names: str) -> Callable[[T], T]:
```

```python
def set_state_dict_pass_attr(*attr_names: str) -> Callable[[T], T]:
```

**Classes**:

#### class Module

**Attributes**:

- `_alt_name`
- `_registry_map`

**Methods**:

```python
def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
```

```python
def __init__(self) -> None:
```

```python
def __repr__(self) -> str:
```

```python
def __setattr__(self, name: str, value: Any) -> None:
```

```python
def _get_name(self) -> str:
```

```python
def add_module(self, name: str, module: Self) -> None:
```

```python
def apply(self, fn: Callable[[Self, Any], None]) -> Self:
```

```python
def buffers(self, recurse: bool=True) -> Iterator[nn.Buffer]:
```

```python
def children(self: Self) -> Iterator[Self]:
```

```python
def compile(self, **kwargs) -> JITModule:
```

```python
def count_parameters(self, recurse: bool=True) -> int:
```

```python
def eval(self) -> Self:
```

```python
def extra_repr(self) -> str:
```

```python
def forward(self) -> Tensor | tuple[Tensor, ...]:
```

```python
def load_state_dict(self, state_dict: OrderedDict, strict: bool=True, *, verbose: bool=False, progress_desc: str | None=None) -> None:
```

*...and 19 more methods*

#### class ModuleDict(Module)

**Methods**:

```python
def __delitem__(self, key: str) -> None:
```

```python
def __getitem__(self, key: str) -> Module:
```

```python
def __init__(self, modules: dict[str, Module] | None=None) -> None:
```

```python
def __iter__(self) -> Iterator[str]:
```

```python
def __len__(self) -> int:
```

```python
def __setitem__(self, key: str, module: Module) -> None:
```

```python
def clear(self) -> None:
```

```python
def items(self) -> ItemsView[str, Module]:
```

```python
def keys(self) -> KeysView[str]:
```

```python
def pop(self, key: str) -> Module:
```

```python
def update(self, modules: dict[str, Module]) -> None:
```

```python
def values(self) -> ValuesView[Module]:
```

#### class ModuleList(Module)

**Methods**:

```python
def __delitem__(self, idx: int) -> None:
```

```python
def __getitem__(self, idx: int | slice) -> Module | Self:
```

```python
def __init__(self, modules: list[Module] | None=None) -> None:
```

```python
def __iter__(self) -> Iterator[Module]:
```

```python
def __len__(self) -> int:
```

```python
def __setitem__(self, idx: int, module: Module) -> None:
```

```python
def append(self, module: Module) -> None:
```

```python
def extend(self, modules: list[Module]) -> None:
```

```python
def insert(self, index: int, module: Module) -> None:
```

#### class ParameterDict(Module)

**Methods**:

```python
def __delitem__(self, key: str) -> None:
```

```python
def __getitem__(self, key: str) -> nn.Parameter:
```

```python
def __init__(self, parameters: dict[str, nn.Parameter] | None=None) -> None:
```

```python
def __iter__(self) -> Iterator[str]:
```

```python
def __len__(self) -> int:
```

```python
def __repr__(self) -> str:
```

```python
def __setitem__(self, key: str, param: nn.Parameter) -> None:
```

```python
def clear(self) -> None:
```

```python
def items(self) -> ItemsView[str, nn.Parameter]:
```

```python
def keys(self) -> KeysView[str]:
```

```python
def pop(self, key: str) -> nn.Parameter:
```

```python
def update(self, parameters: dict[str, nn.Parameter]) -> None:
```

```python
def values(self) -> ValuesView[nn.Parameter]:
```

#### class ParameterList(Module)

**Methods**:

```python
def __delitem__(self, idx: int) -> None:
```

```python
def __getitem__(self, idx: int | slice) -> nn.Parameter | Self:
```

```python
def __init__(self, parameters: list[nn.Parameter] | None=None) -> None:
```

```python
def __iter__(self) -> Iterator[nn.Parameter]:
```

```python
def __len__(self) -> int:
```

```python
def __repr__(self) -> str:
```

```python
def __setitem__(self, idx: int, param: nn.Parameter) -> None:
```

```python
def append(self, param: nn.Parameter) -> None:
```

```python
def extend(self, parameters) -> None:
```

```python
def insert(self, index: int, param: nn.Parameter) -> None:
```

#### class Sequential(Module)

**Methods**:

```python
def __delitem__(self, idx: int) -> None:
```

```python
def __getitem__(self, idx: int | slice) -> Module | Self:
```

```python
@overload
```

```python
@overload
```

```python
def __init__(self, *args: Module | OrderedDict[str, Module]) -> None:
```

```python
def __len__(self) -> int:
```

```python
def __setitem__(self, idx: int, module: Module) -> None:
```

```python
def append(self, module: Module) -> None:
```

```python
def extend(self, modules: Iterator[Module]) -> None:
```

```python
def forward(self, input: Tensor) -> Tensor:
```

```python
@classmethod
```

```python
@classmethod
```

---

### parameter.py

**__all__**: `Buffer`, `Parameter`

**Classes**:

#### class Buffer(Tensor)

**Methods**:

```python
def __init__(self, data: Tensor | _ArrayOrScalar, dtype: type | None=None, device: _DeviceType='cpu') -> None:
```

#### class Parameter(Tensor)

**Methods**:

```python
def __init__(self, data: Tensor | _ArrayOrScalar, dtype: type | Numeric | None=None, device: _DeviceType='cpu') -> None:
```

---

## lucid/nn/_kernel

### __init__.py

---

### activation.py

**Functions**:

```python
def _norm_axis(axis: int, ndim: int) -> int:
```

**Classes**:

#### class gelu_kernel(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
def _forward(self, a: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=1, n_ret=1, device='cpu')
```

```python
@func_op(n_in=1, n_ret=1, device='gpu')
```

#### class sigmoid_kernel(Operation)

**Methods**:

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
def _forward(self, a: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=1, n_ret=1, device='cpu')
```

```python
@func_op(n_in=1, n_ret=1, device='gpu')
```

#### class silu_kernel(Operation)

**Methods**:

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self) -> None:
```

```python
def _forward(self, a: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=1, n_ret=1, device='cpu')
```

```python
@func_op(n_in=1, n_ret=1, device='gpu')
```

#### class softmax_kernel(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, axis: int=-1) -> None:
```

```python
def _forward(self, a: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=1, n_ret=1, device='cpu')
```

```python
@func_op(n_in=1, n_ret=1, device='gpu')
```

---

### attention.py

**Functions**:

```python
def _make_causal_mask(lib_: ModuleType, L: int, S: int, dtype: object) -> _TensorData:
```

**Classes**:

#### class scaled_dot_product_attention_kernel(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, attn_mask: Tensor | None=None, is_causal: bool=False, scale: float | None=None, dropout_p: float=0.0) -> None:
```

```python
def _forward(self, q: Tensor, k: Tensor, v: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=3, n_ret=1)
```

```python
def get_attention_weight(self, device: _DeviceType) -> Tensor:
```

```python
@func_op(n_in=3, n_ret=1, device='gpu')
```

---

### conv.py

**Constants**:

- `_CONV_VIEW_LIMIT_BYTES` = _load_view_limit_bytes()

**Functions**:

```python
def _as_strided(lib_: ModuleType, data: _Array, shape: _Shape, strides: _Shape) -> _Array | None:
```

```python
def _conv_backward_input(lib_: ModuleType, grad_out: _Array, weight: _Array, x_pad: _Array, stride: _Stride, padding: _Padding, dilation: _Dilation, groups: int) -> _Array:
```

```python
def _conv_backward_weight(lib_: ModuleType, grad_out: _Array, x_pad: _Array, weight: _Array, stride: _Stride, dilation: _Dilation, groups: int) -> _Array:
```

```python
def _conv_fallback(lib_: ModuleType, input_: _Array, weight: _Array, stride: _Stride, padding: _Padding, dilation: _Dilation, groups: int, out_dims: _Shape) -> _Array:
```

```python
def _conv_forward(lib_: ModuleType, input_: _Array, weight: _Array, stride: _Stride, padding: _Padding, dilation: _Dilation, groups: int) -> _Array:
```

```python
def _conv_from_view(lib_: ModuleType, x_view: _Array, weight: _Array, out_dims: _Shape, groups: int) -> _Array:
```

```python
def _conv_out_dims(input_spatial: _Shape, kernel_size: _Shape, stride: _Stride, padding: _Padding, dilation: _Dilation) -> list[int]:
```

```python
def _default_view_limit_bytes() -> int:
```

```python
def _dtype_itemsize(data: _Array) -> int:
```

```python
def _get_total_memory_bytes() -> int | None:
```

```python
def _load_view_limit_bytes() -> int:
```

```python
def _make_input_view(lib_: ModuleType, data: _Array, out_dims: _Shape, kernel_size: _Shape, stride: _Stride, dilation: _Dilation) -> _Array | None:
```

```python
def _pad_input(lib_: ModuleType, data: _Array, padding: _Padding) -> _Array:
```

```python
def _prod(shape: _Shape) -> int:
```

```python
def _round_to_step(value: int, step: int) -> int:
```

```python
def _sysconf_value(name: str) -> int | None:
```

```python
def _to_tuple(value: int | tuple[int, ...] | list[int], dim: int, name: str) -> _Shape:
```

```python
def _validate_conv_shapes(input_: Tensor, weight: Tensor, groups: int) -> None:
```

```python
def _view_exceeds_limit(data: _Array, out_dims: _Shape, kernel_size: _Shape) -> bool:
```

```python
def get_conv_view_limit_mb() -> int:
```

**Classes**:

#### class conv_nd_kernel(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor, b: Tensor) -> int:
```

```python
def __grad__(self, a: Tensor, b: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, stride: int | tuple[int, ...] | list[int], padding: int | tuple[int, ...] | list[int], dilation: int | tuple[int, ...] | list[int], groups: int) -> None:
```

```python
def _normalize(self, weight: Tensor) -> tuple[_Stride, _Padding, _Dilation]:
```

```python
@func_op(n_in=2, n_ret=1)
```

```python
@func_op(n_in=2, n_ret=1, device='gpu')
```

---

### embedding.py

**Classes**:

#### class embedding_kernel(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, padding_idx: int=-1, max_norm: float | None=None, norm_type: float=2.0) -> None:
```

```python
def _forward(self, indices: Tensor, weight: Tensor, lib_: ModuleType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=2, n_ret=1)
```

```python
@func_op(n_in=2, n_ret=1, device='gpu')
```

---

### loss.py

**Functions**:

```python
def _to_int(arr: _TensorData, lib_: ModuleType) -> _TensorData:
```

**Classes**:

#### class binary_cross_entropy_kernel(Operation)

**Methods**:

```python
def __grad__(self) -> _GradType:
```

```python
def __init__(self, reduction: str | None='mean', eps: float=1e-07, has_weight: bool=True) -> None:
```

```python
def _forward(self, input_: Tensor, target: Tensor, weight: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=3, n_ret=1)
```

```python
@func_op(n_in=3, n_ret=1, device='gpu')
```

#### class binary_cross_entropy_with_logits_kernel(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, reduction: str | None='mean', has_weight: bool=True, has_pos_weight: bool=True) -> None:
```

```python
def _forward(self, logits: Tensor, target: Tensor, weight: Tensor, pos_weight: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=4, n_ret=1)
```

```python
@func_op(n_in=4, n_ret=1, device='gpu')
```

#### class cross_entropy_kernel(Operation)

**Methods**:

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, reduction: str | None='mean', eps: float=1e-07, ignore_index: int | None=None, has_weight: bool=True) -> None:
```

```python
def _forward(self, logits: Tensor, target: Tensor, weight: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=3, n_ret=1)
```

```python
@func_op(n_in=3, n_ret=1, device='gpu')
```

---

### norm.py

**Functions**:

```python
def _broadcast_shape(ndim: int, normalized_shape: Sequence[int]) -> tuple[int, ...]:
```

```python
def _clone_array(arr):
```

```python
def _norm_axes(ndim: int, normalized_shape: Sequence[int]) -> tuple[int, ...]:
```

**Classes**:

#### class batch_norm_kernel(Operation)

**Methods**:

```python
def __grad__(self, a: Tensor, w: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, eps: float=1e-05, momentum: float=0.1, training: bool=True, has_running: bool=True, has_weight: bool=True, has_bias: bool=True) -> None:
```

```python
def _forward(self, a: Tensor, running_mean: Tensor, running_var: Tensor, w: Tensor, b: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=5, n_ret=1, device='cpu')
```

```python
@func_op(n_in=5, n_ret=1, device='gpu')
```

#### class group_norm_kernel(Operation)

**Methods**:

```python
def __grad__(self, a: Tensor, w: Tensor, b: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, num_groups: int, eps: float=1e-05, has_weight: bool=True, has_bias: bool=True) -> None:
```

```python
def _forward(self, a: Tensor, w: Tensor, b: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=3, n_ret=1, device='cpu')
```

```python
@func_op(n_in=3, n_ret=1, device='gpu')
```

#### class layer_norm_kernel(Operation)

**Methods**:

```python
def __grad__(self, a: Tensor, w: Tensor, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, normalized_shape: Sequence[int], eps: float=1e-05, has_weight: bool=True, has_bias: bool=True) -> None:
```

```python
def _forward(self, a: Tensor, w: Tensor, b: Tensor, lib_: ModuleType, device: _DeviceType) -> _FuncOpReturnType:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=3, n_ret=1, device='cpu')
```

```python
@func_op(n_in=3, n_ret=1, device='gpu')
```

---

### pool.py

**Functions**:

```python
def _crop_padding(data: _Array, padding: _Shape) -> _Array:
```

```python
def _full_like_int(lib_: ModuleType, ref: _Array, value: int) -> _Array:
```

```python
def _pad_input(lib_: ModuleType, data: _Array, padding: _Shape) -> _Array:
```

```python
def _pool_backward_avg(lib_: ModuleType, grad_out: _Array, input_shape: _Shape, out_dims: _Shape, kernel_size: _Shape, stride: _Shape, padding: _Shape) -> _Array:
```

```python
def _pool_backward_max(lib_: ModuleType, grad_out: _Array, input_shape: _Shape, out_dims: _Shape, kernel_size: _Shape, stride: _Shape, padding: _Shape, max_idx: _Array) -> _Array:
```

```python
def _pool_forward_max(lib_: ModuleType, x_pad: _Array, out_dims: _Shape, kernel_size: _Shape, stride: _Shape) -> tuple[_Array, _Array]:
```

```python
def _pool_forward_sum(x_pad: _Array, out_dims: _Shape, kernel_size: _Shape, stride: _Shape) -> _Array:
```

```python
def _pool_out_dims(input_spatial: _Shape, kernel_size: _Shape, stride: _Shape, padding: _Shape) -> _Shape:
```

```python
def _prod(shape: _Shape) -> int:
```

```python
def _to_tuple(value: int | tuple[int, ...] | list[int], dim: int, name: str) -> _Shape:
```

```python
def _where(lib_: ModuleType, cond: _Array, x: _Array, y: _Array) -> _Array:
```

```python
def _zeros(lib_: ModuleType, shape: _Shape, dtype: object) -> _Array:
```

**Classes**:

#### class pool_nd_kernel(Operation)

**Methods**:

```python
def __flops__(self, a: Tensor) -> int:
```

```python
def __grad__(self, lib_: ModuleType) -> _GradType:
```

```python
def __init__(self, kernel_size: int | tuple[int, ...] | list[int], stride: int | tuple[int, ...] | list[int], padding: int | tuple[int, ...] | list[int], mode: _Mode) -> None:
```

```python
def _normalize(self, input_: Tensor) -> tuple[_Shape, _Shape, _Shape]:
```

```python
def clear(self) -> None:
```

```python
@func_op(n_in=1, n_ret=1)
```

```python
@func_op(n_in=1, n_ret=1, device='gpu')
```

---

## lucid/nn/functional

### __init__.py

**Functions**:

```python
def adaptive_avg_pool1d(input_: Tensor, output_size: int) -> Tensor:
```

```python
def adaptive_avg_pool2d(input_: Tensor, output_size: int | tuple[int, int]) -> Tensor:
```

```python
def adaptive_avg_pool3d(input_: Tensor, output_size: int | tuple[int, int, int]) -> Tensor:
```

```python
def adaptive_max_pool1d(input_: Tensor, output_size: int) -> Tensor:
```

```python
def adaptive_max_pool2d(input_: Tensor, output_size: int | tuple[int, int]) -> Tensor:
```

```python
def adaptive_max_pool3d(input_: Tensor, output_size: int | tuple[int, int, int]) -> Tensor:
```

```python
def affine_grid(theta: Tensor, size: tuple[int, ...], align_corners: bool=True) -> Tensor:
```

```python
def alpha_dropout(input_: Tensor, p: float=0.5, training: bool=True) -> Tensor:
```

```python
def avg_pool1d(input_: Tensor, kernel_size: int | tuple[int]=1, stride: int | tuple[int]=1, padding: int | tuple[int]=0) -> Tensor:
```

```python
def avg_pool2d(input_: Tensor, kernel_size: int | tuple[int, int]=1, stride: int | tuple[int, int]=1, padding: int | tuple[int, int]=0) -> Tensor:
```

```python
def avg_pool3d(input_: Tensor, kernel_size: int | tuple[int, int, int]=1, stride: int | tuple[int, int, int]=1, padding: int | tuple[int, int, int]=0) -> Tensor:
```

```python
def batch_norm(input_: Tensor, running_mean: Tensor | None=None, running_var: Tensor | None=None, weight: Tensor | None=None, bias: Tensor | None=None, training: bool=True, momentum: float=0.1, eps: float=1e-05) -> Tensor:
```

```python
def bilinear(input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None=None) -> Tensor:
```

```python
def binary_cross_entropy(input_: Tensor, target: Tensor, weight: Tensor | None=None, reduction: _ReductionType | None='mean', eps: float=1e-07) -> Tensor:
```

```python
def binary_cross_entropy_with_logits(input_: Tensor, target: Tensor, weight: Tensor | None=None, pos_weight: Tensor | None=None, reduction: _ReductionType | None='mean') -> Tensor:
```

```python
def conv1d(input_: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1) -> Tensor:
```

```python
def conv2d(input_: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1) -> Tensor:
```

```python
def conv3d(input_: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1) -> Tensor:
```

```python
def conv_transpose1d(input_: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0, output_padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1) -> Tensor:
```

```python
def conv_transpose2d(input_: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0, output_padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1) -> Tensor:
```

```python
def conv_transpose3d(input_: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0, output_padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1) -> Tensor:
```

```python
def cross_entropy(input_: Tensor, target: Tensor, weight: Tensor | None=None, reduction: _ReductionType | None='mean', eps: float=1e-07, ignore_index: int | None=None) -> Tensor:
```

```python
def drop_block(input_: Tensor, block_size: int, p: float=0.1, eps: float=1e-07) -> Tensor:
```

```python
def drop_path(input_: Tensor, p: float=0.1, scale_by_keep: bool=True) -> Tensor:
```

```python
def dropout(input_: Tensor, p: float=0.5, training: bool=True) -> Tensor:
```

*...and 33 more functions*

---

### _activation.py

**Functions**:

```python
def elu(input_: Tensor, alpha: float=1.0) -> Tensor:
```

```python
def gelu(input_: Tensor) -> Tensor:
```

```python
def leaky_relu(input_: Tensor, negative_slope: float=0.01) -> Tensor:
```

```python
def relu(input_: Tensor) -> Tensor:
```

```python
def selu(input_: Tensor) -> Tensor:
```

```python
def sigmoid(input_: Tensor) -> Tensor:
```

```python
def silu(input_: Tensor) -> Tensor:
```

```python
def softmax(input_: Tensor, axis: int=-1) -> Tensor:
```

```python
def tanh(input_: Tensor) -> Tensor:
```

---

### _attention.py

**Functions**:

```python
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None=None, dropout_p: float=0.0, is_causal: bool=False, scale: float | None=None, output_weight: bool=False) -> Tensor | tuple[Tensor, Tensor]:
```

---

### _conv.py

**Functions**:

```python
def _upsample_nd(input_: Tensor, stride: Tuple[int, ...]) -> Tensor:
```

```python
def conv(input_: Tensor, weight: Tensor, bias: Optional[Tensor], stride: Tuple[int, ...], padding: Tuple[int, ...], dilation: Tuple[int, ...], groups: int) -> Tensor:
```

```python
def conv_transpose(input_: Tensor, weight: Tensor, bias: Optional[Tensor], stride: Tuple[int, ...], padding: Tuple[int, ...], output_padding: Tuple[int, ...], dilation: Tuple[int, ...], groups: int=1) -> Tensor:
```

```python
def unfold(input_: Tensor, filter_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], dilation: Tuple[int, ...]) -> Tensor:
```

---

### _drop.py

**Functions**:

```python
def _prob_check(p: float) -> None:
```

```python
def alpha_dropout(input_: Tensor, p: float=0.5, training: bool=True) -> Tensor:
```

```python
def drop_block(input_: Tensor, block_size: int, p: float=0.1, eps: float=1e-07) -> Tensor:
```

```python
def drop_path(input_: Tensor, p: float=0.0, scale_by_keep: bool=True) -> Tensor:
```

```python
def dropout(input_: Tensor, p: float=0.5, training: bool=True) -> Tensor:
```

```python
def dropoutnd(input_: Tensor, p: float=0.5, training: bool=True) -> Tensor:
```

---

### _embedding.py

**Functions**:

```python
def embedding(input_: Tensor, weight: Tensor, padding_idx: int | None=None, max_norm: float | None=None, norm_type: float=2.0) -> Tensor:
```

```python
def rotary_pos_embedding(input_: Tensor, position_ids: Tensor | None=None, interleaved: bool=True) -> Tensor:
```

```python
def sinusoidal_pos_embedding(seq_len: int, embed_dim: int, device: _DeviceType='cpu', dtype: Numeric | None=None) -> Tensor:
```

---

### _linear.py

**Functions**:

```python
def bilinear(input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None=None) -> Tensor:
```

```python
def linear(input_: Tensor, weight: Tensor, bias: Tensor | None=None) -> Tensor:
```

---

### _loss.py

**Functions**:

```python
def _ignore_index_loss(loss: Tensor, target_int: Tensor, ignore_index: int, reduction: _ReductionType | None) -> Tensor:
```

```python
def _loss_reduction(loss: Tensor, reduction: _ReductionType | None) -> Tensor:
```

```python
def binary_cross_entropy(input_: Tensor, target: Tensor, weight: Tensor | None=None, reduction: _ReductionType | None='mean', eps: float=1e-07) -> Tensor:
```

```python
def binary_cross_entropy_with_logits(input_: Tensor, target: Tensor, weight: Tensor | None=None, pos_weight: Tensor | None=None, reduction: _ReductionType | None='mean') -> Tensor:
```

```python
def cross_entropy(input_: Tensor, target: Tensor, weight: Tensor | None=None, reduction: _ReductionType | None='mean', eps: float=1e-07, ignore_index: int | None=None) -> Tensor:
```

```python
def huber_loss(input_: Tensor, target: Tensor, delta: float=1.0, reduction: _ReductionType | None='mean') -> Tensor:
```

```python
def mse_loss(input_: Tensor, target: Tensor, reduction: _ReductionType | None='mean') -> Tensor:
```

```python
def nll_loss(input_: Tensor, target: Tensor, weight: Tensor | None=None, reduction: _ReductionType | None='mean', ignore_index: int | None=None) -> Tensor:
```

---

### _norm.py

**Functions**:

```python
def batch_norm(input_: Tensor, running_mean: Tensor | None, running_var: Tensor | None, weight: Tensor | None=None, bias: Tensor | None=None, training: bool=True, momentum: float=0.1, eps: float=1e-05) -> Tensor:
```

```python
def global_response_norm(input_: Tensor, gamma: Tensor, beta: Tensor, eps: float=1e-06) -> Tensor:
```

```python
def group_norm(input_: Tensor, num_groups: int, weight: Tensor | None, bias: Tensor | None, eps: float=1e-05) -> Tensor:
```

```python
def instance_norm(input_: Tensor, running_mean: Tensor | None, running_var: Tensor | None, weight: Tensor | None=None, bias: Tensor | None=None, training: bool=True, momentum: float=0.1, eps: float=1e-05) -> Tensor:
```

```python
def layer_norm(input_: Tensor, normalized_shape: _ShapeLike, weight: Tensor | None=None, bias: Tensor | None=None, eps: float=1e-05) -> Tensor:
```

```python
def normalize(input_: Tensor, ord: int=2, axis: int=1, eps: float=1e-12) -> Tensor:
```

---

### _pool.py

**Functions**:

```python
def adaptive_pool1d(input_: Tensor, output_size: int, avg_or_max: Literal['avg', 'max']) -> Tensor:
```

```python
def adaptive_pool2d(input_: Tensor, output_size: tuple[int, int] | int, avg_or_max: Literal['avg', 'max']) -> Tensor:
```

```python
def adaptive_pool3d(input_: Tensor, output_size: tuple[int, int, int] | int, avg_or_max: Literal['avg', 'max']) -> Tensor:
```

```python
def avg_pool1d(input_: Tensor, kernel_size: int | tuple[int]=1, stride: int | tuple[int]=1, padding: int | tuple[int]=0) -> Tensor:
```

```python
def avg_pool2d(input_: Tensor, kernel_size: int | tuple[int, int]=1, stride: int | tuple[int, int]=1, padding: int | tuple[int, int]=0) -> Tensor:
```

```python
def avg_pool3d(input_: Tensor, kernel_size: int | tuple[int, int, int]=1, stride: int | tuple[int, int, int]=1, padding: int | tuple[int, int, int]=0) -> Tensor:
```

```python
def max_pool1d(input_: Tensor, kernel_size: int | tuple[int]=1, stride: int | tuple[int]=1, padding: int | tuple[int]=0) -> Tensor:
```

```python
def max_pool2d(input_: Tensor, kernel_size: int | tuple[int, int]=1, stride: int | tuple[int, int]=1, padding: int | tuple[int, int]=0) -> Tensor:
```

```python
def max_pool3d(input_: Tensor, kernel_size: int | tuple[int, int, int]=1, stride: int | tuple[int, int, int]=1, padding: int | tuple[int, int, int]=0) -> Tensor:
```

---

### _spatial.py

**Functions**:

```python
def affine_grid(theta: Tensor, size: tuple[int, ...], align_corners: bool=True) -> Tensor:
```

```python
def grid_sample(input_: Tensor, grid: Tensor, mode: _InterpolateType='bilinear', padding_mode: _PaddingType='zeros', align_corners: bool=True) -> Tensor:
```

---

### _utils.py

**Functions**:

```python
def _interpolate_area(input_: Tensor, size: tuple[int, int], align_corners: bool=False) -> Tensor:
```

```python
def _interpolate_bilinear(input_: Tensor, size: tuple[int, int], align_corners: bool=False) -> Tensor:
```

```python
def _interpolate_nearest(input_: Tensor, size: tuple[int, int], align_corners: bool=False) -> Tensor:
```

```python
def _interpolate_nearest_3d(input_: Tensor, size: tuple[int, int, int], align_corners: bool=False) -> Tensor:
```

```python
def _interpolate_trilinear(input_: Tensor, size: tuple[int, int, int], align_corners: bool=False) -> Tensor:
```

```python
def one_hot(input_: Tensor, num_classes: int=-1, dtype: Numeric | bool | None=None) -> Tensor:
```

```python
def rotate(input_: Tensor, angle: float, center: tuple[_Scalar, _Scalar] | None=None) -> Tensor:
```

---

## lucid/nn/init

### __init__.py

**Functions**:

```python
def _tensor_check(value: Any) -> None:
```

```python
def constant(tensor: Tensor, val: _Scalar) -> None:
```

```python
def kaiming_normal(tensor: Tensor, mode: _FanMode='fan_in') -> None:
```

```python
def kaiming_uniform(tensor: Tensor, mode: _FanMode='fan_in') -> None:
```

```python
def normal(tensor: Tensor, mean: _Scalar=0.0, std: _Scalar=1.0) -> None:
```

```python
def uniform(tensor: Tensor, a: _Scalar=0, b: _Scalar=1) -> None:
```

```python
def xavier_normal(tensor: Tensor, gain: _Scalar=1.0) -> None:
```

```python
def xavier_uniform(tensor: Tensor, gain: _Scalar=1.0) -> None:
```

---

### _dist.py

**Functions**:

```python
def _assign_like(tensor: Tensor, data: np.ndarray) -> None:
```

```python
def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
```

```python
def constant(tensor: Tensor, val: _Scalar) -> None:
```

```python
def kaiming_normal(tensor: Tensor, mode: str) -> None:
```

```python
def kaiming_uniform(tensor: Tensor, mode: str) -> None:
```

```python
def normal(tensor: Tensor, mean: _Scalar, std: _Scalar) -> None:
```

```python
def uniform(tensor: Tensor, a: _Scalar, b: _Scalar) -> None:
```

```python
def xavier_normal(tensor: Tensor, gain: _Scalar) -> None:
```

```python
def xavier_uniform(tensor: Tensor, gain: _Scalar) -> None:
```

---

## lucid/nn/modules

### __init__.py

---

### activation.py

**__all__**: `ELU`, `GELU`, `HardSigmoid`, `HardSwish`, `LeakyReLU`, `Mish`, `ReLU`, `ReLU6`, `SELU`, `Sigmoid`, `Softmax`, `Swish`, `Tanh`

**Classes**:

#### class ELU(nn.Module)

**Methods**:

```python
def __init__(self, alpha: float=1.0) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class GELU(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class HardSigmoid(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class HardSwish(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class LeakyReLU(nn.Module)

**Methods**:

```python
def __init__(self, negative_slope: float=0.01) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Mish(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class ReLU(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class ReLU6(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class SELU(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Sigmoid(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Softmax(nn.Module)

**Methods**:

```python
def __init__(self, axis: int=-1) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Swish(nn.Module)

**Methods**:

```python
def __init__(self) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

*...and 1 more classes*

---

### attention.py

**__all__**: `MultiHeadAttention`, `ScaledDotProductAttention`

**Classes**:

#### class MultiHeadAttention(nn.Module)

**Methods**:

```python
def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True, use_separate_proj_weight: bool=True, add_bias_kv: bool=False, add_zero_attn: bool=False, kdim: int | None=None, vdim: int | None=None) -> None:
```

```python
def _reset_parameters(self) -> None:
```

```python
def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None=None, attn_mask: Tensor | None=None, is_causal: bool=False, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_layer_idx: int | None=None) -> Tensor:
```

#### class ScaledDotProductAttention(nn.Module)

**Methods**:

```python
def __init__(self, attn_mask: Tensor | None=None, dropout_p: float=0.0, is_causal: bool=False, scale: _Scalar | None=None) -> None:
```

```python
def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
```

---

### conv.py

**__all__**: `ConstrainedConv1d`, `ConstrainedConv2d`, `ConstrainedConv3d`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`, `Unfold`

**Functions**:

```python
def _single_to_tuple(value: Any, times: int) -> tuple[Any, ...]:
```

**Classes**:

#### class ConstrainedConv1d(_ConstrainedConvNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True, *, constraint: _ConstrainedMode='none', enforce: _ConstraintEnforce='forward', eps: float=1e-12, max_l2: float | None=None, center_value: float=-1.0, neighbor_sum: float=1.0) -> None:
```

```python
def _conv_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class ConstrainedConv2d(_ConstrainedConvNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True, *, constraint: _ConstrainedMode='none', enforce: _ConstraintEnforce='forward', eps: float=1e-12, max_l2: float | None=None, center_value: float=-1.0, neighbor_sum: float=1.0) -> None:
```

```python
def _conv_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class ConstrainedConv3d(_ConstrainedConvNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True, *, constraint: _ConstrainedMode='none', enforce: _ConstraintEnforce='forward', eps: float=1e-12, max_l2: float | None=None, center_value: float=-1.0, neighbor_sum: float=1.0) -> None:
```

```python
def _conv_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Conv1d(_ConvNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True) -> None:
```

```python
def _conv_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Conv2d(_ConvNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True) -> None:
```

```python
def _conv_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Conv3d(_ConvNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True) -> None:
```

```python
def _conv_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class ConvTranspose1d(_ConvTransposeNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, output_padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True) -> None:
```

```python
def _conv_transpose_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class ConvTranspose2d(_ConvTransposeNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, output_padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True) -> None:
```

```python
def _conv_transpose_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class ConvTranspose3d(_ConvTransposeNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...]=1, padding: _PaddingStr | int | tuple[int, ...]=0, output_padding: int | tuple[int, ...]=0, dilation: int | tuple[int, ...]=1, groups: int=1, bias: bool=True) -> None:
```

```python
def _conv_transpose_forward(self, input_: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Unfold(nn.Module)

**Methods**:

```python
def __init__(self, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...], padding: int | tuple[int, ...], dilation: int | tuple[int, ...]=1) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class _ConstrainedConvNd(_ConvNd)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...], padding: _PaddingStr | int | tuple[int, ...], dilation: int | tuple[int, ...], groups: int, bias: bool, *, constraint: _ConstrainedMode='none', enforce: _ConstraintEnforce='forward', eps: float=1e-12, max_l2: float | None=None, center_value: float=-1.0, neighbor_sum: float=1.0, D: int) -> None:
```

```python
def _apply_constraint(self, w: Tensor) -> Tensor:
```

```python
def _constrained_weight(self) -> Tensor:
```

```python
def _l2_spatial(self, w: Tensor) -> Tensor:
```

```python
def _normalize_sum(self, w: Tensor, target_sum: float) -> Tensor:
```

```python
def _sum_spatial(self, w: Tensor) -> Tensor:
```

```python
def extra_repr(self) -> str:
```

```python
def project_(self) -> '_ConstrainedConvNd':
```

#### class _ConvNd(nn.Module)

**Methods**:

```python
def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...], padding: _PaddingStr | int | tuple[int, ...], dilation: int | tuple[int, ...], groups: int, bias: bool, *, D: int) -> None:
```

```python
def extra_repr(self) -> str:
```

```python
def reset_parameters(self) -> None:
```

*...and 1 more classes*

---

### drop.py

**__all__**: `AlphaDropout`, `DropBlock`, `DropPath`, `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`

**Classes**:

#### class AlphaDropout(_DropoutNd)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class DropBlock(nn.Module)

**Methods**:

```python
def __init__(self, block_size: int, p: float=0.1, eps: float=1e-07) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class DropPath(nn.Module)

**Methods**:

```python
def __init__(self, drop_prob: float=0.1, scale_by_keep: bool=True) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Dropout(_DropoutNd)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Dropout1d(_DropoutNd)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Dropout2d(_DropoutNd)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Dropout3d(_DropoutNd)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class _DropoutNd(nn.Module)

**Methods**:

```python
def __init__(self, p: float=0.5) -> None:
```

---

### einops.py

**__all__**: `Rearrange`

**Classes**:

#### class Rearrange(nn.Module)

**Methods**:

```python
def __init__(self, pattern: _EinopsPattern, **shapes: int) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

---

### linear.py

**__all__**: `Bilinear`, `Flatten`, `Identity`, `Linear`

**Classes**:

#### class Bilinear(nn.Module)

**Methods**:

```python
def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool=True) -> None:
```

```python
def extra_repr(self) -> str:
```

```python
def forward(self, input_1: Tensor, input_2: Tensor) -> Tensor:
```

```python
def reset_parameters(self) -> None:
```

#### class Flatten(nn.Module)

**Methods**:

```python
def __init__(self, start_axis: int=1, end_axis: int=-1) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Identity(nn.Module)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class Linear(nn.Module)

**Methods**:

```python
def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
```

```python
def extra_repr(self) -> str:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

```python
def reset_parameters(self) -> None:
```

---

### loss.py

**__all__**: `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `HuberLoss`, `MSELoss`, `NLLLoss`

**Classes**:

#### class BCELoss(_WeightedLoss)

**Methods**:

```python
def __init__(self, weight: Tensor | None=None, reduction: _ReductionType | None='mean', eps: float=1e-07) -> None:
```

```python
def forward(self, input_: Tensor, target: Tensor) -> Tensor:
```

#### class BCEWithLogitsLoss(_WeightedLoss)

**Methods**:

```python
def __init__(self, weight: Tensor | None=None, reduction: _ReductionType | None='mean', eps: float=1e-07) -> None:
```

```python
def forward(self, input_: Tensor, target: Tensor) -> Tensor:
```

#### class CrossEntropyLoss(_WeightedLoss)

**Methods**:

```python
def __init__(self, weight: Tensor | None=None, reduction: _ReductionType | None='mean', eps: float=1e-07) -> None:
```

```python
def forward(self, input_: Tensor, target: Tensor) -> Tensor:
```

#### class HuberLoss(_Loss)

**Methods**:

```python
def __init__(self, reduction: _ReductionType | None='mean', delta: float=1.0) -> None:
```

```python
def forward(self, input_: Tensor, target: Tensor) -> Tensor:
```

#### class MSELoss(_Loss)

**Methods**:

```python
def forward(self, input_: Tensor, target: Tensor) -> Tensor:
```

#### class NLLLoss(_WeightedLoss)

**Methods**:

```python
def forward(self, input_: Tensor, target: Tensor) -> Tensor:
```

#### class _Loss(nn.Module)

**Methods**:

```python
def __init__(self, reduction: _ReductionType | None='mean') -> None:
```

```python
@override
```

#### class _WeightedLoss(nn.Module)

**Methods**:

```python
def __init__(self, weight: Tensor | None=None, reduction: _ReductionType | None='mean') -> None:
```

```python
@override
```

---

### norm.py

**__all__**: `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `GlobalResponseNorm`, `GroupNorm`, `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d`, `LayerNorm`

**Classes**:

#### class BatchNorm1d(_BatchNorm)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class BatchNorm2d(_BatchNorm)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class BatchNorm3d(_BatchNorm)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class GlobalResponseNorm(nn.Module)

**Methods**:

```python
def __init__(self, channels: int, eps: float=1e-06) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class GroupNorm(nn.Module)

**Methods**:

```python
def __init__(self, num_groups: int, num_channels: int, eps: float=1e-05, affine: bool=True) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class InstanceNorm1d(_InstanceNorm)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class InstanceNorm2d(_InstanceNorm)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class InstanceNorm3d(_InstanceNorm)

**Methods**:

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class LayerNorm(nn.Module)

**Methods**:

```python
def __init__(self, normalized_shape: _ShapeLike | int, eps: float=1e-05, elementwise_affine: bool=True, bias: bool=True) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class _BatchNorm(_NormBase)

**Methods**:

```python
def __init__(self, num_features: int, eps: float=1e-05, momentum: float | None=0.1, affine: bool=True, track_running_stats: bool=True) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class _InstanceNorm(_NormBase)

**Methods**:

```python
def __init__(self, num_features: int, eps: float=1e-05, momentum: float | None=0.1, affine: bool=True, track_running_stats: bool=True) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class _NormBase(nn.Module)

**Methods**:

```python
def __init__(self, num_features: int, eps: float=1e-05, momentum: float | None=0.1, affine: bool=True, track_running_stats: bool=True) -> None:
```

```python
def extra_repr(self) -> str:
```

```python
def reset_parameters(self) -> None:
```

```python
def reset_running_stats(self) -> None:
```

---

### pool.py

**__all__**: `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d`, `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `MaxPool1d`, `MaxPool2d`, `MaxPool3d`

**Functions**:

```python
def _single_to_tuple(value: Any, times: int) -> tuple[Any, ...]:
```

**Classes**:

#### class AdaptiveAvgPool1d(nn.Module)

**Methods**:

```python
def __init__(self, output_size: int) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AdaptiveAvgPool2d(nn.Module)

**Methods**:

```python
def __init__(self, output_size: int | tuple[int, int]) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AdaptiveAvgPool3d(nn.Module)

**Methods**:

```python
def __init__(self, output_size: int | tuple[int, int, int]) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AdaptiveMaxPool1d(nn.Module)

**Methods**:

```python
def __init__(self, output_size: int) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AdaptiveMaxPool2d(nn.Module)

**Methods**:

```python
def __init__(self, output_size: int | tuple[int, int]) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AdaptiveMaxPool3d(nn.Module)

**Methods**:

```python
def __init__(self, output_size: int | tuple[int, int, int]) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AvgPool1d(_PoolNd)

**Methods**:

```python
def __init__(self, kernel_size: int | tuple[int, ...]=1, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AvgPool2d(_PoolNd)

**Methods**:

```python
def __init__(self, kernel_size: int | tuple[int, ...]=1, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class AvgPool3d(_PoolNd)

**Methods**:

```python
def __init__(self, kernel_size: int | tuple[int, ...]=1, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class MaxPool1d(_PoolNd)

**Methods**:

```python
def __init__(self, kernel_size: int | tuple[int, ...]=1, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class MaxPool2d(_PoolNd)

**Methods**:

```python
def __init__(self, kernel_size: int | tuple[int, ...]=1, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

#### class MaxPool3d(_PoolNd)

**Methods**:

```python
def __init__(self, kernel_size: int | tuple[int, ...]=1, stride: int | tuple[int, ...]=1, padding: int | tuple[int, ...]=0) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

*...and 1 more classes*

---

### rnn.py

**__all__**: `GRU`, `GRUCell`, `LSTM`, `LSTMCell`, `RNN`, `RNNBase`, `RNNCell`

**Functions**:

```python
def _get_activation(nonlinearity: str) -> type[nn.Module]:
```

**Classes**:

#### class GRU(RNNBase)

**Methods**:

```python
def __init__(self, input_size: int, hidden_size: int, num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0) -> None:
```

#### class GRUCell(nn.Module)

**Methods**:

```python
def __init__(self, input_size: int, hidden_size: int, bias: bool=True) -> None:
```

```python
def forward(self, input_: Tensor, hx: Tensor | None=None) -> Tensor:
```

#### class LSTM(RNNBase)

**Methods**:

```python
def __init__(self, input_size: int, hidden_size: int, num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0) -> None:
```

#### class LSTMCell(nn.Module)

**Methods**:

```python
def __init__(self, input_size: int, hidden_size: int, bias: bool=True, **kwargs) -> None:
```

```python
def forward(self, input_: Tensor, hx: tuple[Tensor, Tensor] | None=None) -> tuple[Tensor, Tensor]:
```

#### class RNN(RNNBase)

**Methods**:

```python
def __init__(self, input_size: int, hidden_size: int, num_layers: int=1, nonlinearity: Literal['tanh', 'relu']='tanh', bias: bool=True, batch_first: bool=False, dropout: float=0.0) -> None:
```

#### class RNNBase(nn.Module)

**Methods**:

```python
def __init__(self, mode: Literal['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU'], input_size: int, hidden_size: int, num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0) -> None:
```

```python
def _gru_cell(self, input_: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor | None, b_hh: Tensor | None) -> Tensor:
```

```python
def _init_hidden(self, batch_size: int, dtype: Numeric, device: _DeviceType) -> Tensor | tuple[Tensor, Tensor]:
```

```python
def _lstm_cell(self, input_: Tensor, hx: Tensor, cx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor | None, b_hh: Tensor | None) -> tuple[Tensor, Tensor]:
```

```python
def _rnn_cell(self, input_: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor | None, b_hh: Tensor | None) -> Tensor:
```

```python
def forward(self, input_: Tensor | PackedSequence, hx: Tensor | tuple[Tensor, Tensor] | None=None) -> tuple[Tensor | PackedSequence, Tensor] | tuple[Tensor | PackedSequence, tuple[Tensor, Tensor]]:
```

#### class RNNCell(nn.Module)

**Methods**:

```python
def __init__(self, input_size: int, hidden_size: int, bias: bool=True, nonlinearity: Literal['tanh', 'relu']='tanh') -> None:
```

```python
def forward(self, input_: Tensor, hx: Tensor | None=None) -> Tensor:
```

---

### sparse.py

**__all__**: `Embedding`

**Classes**:

#### class Embedding(nn.Module)

**Methods**:

```python
def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None=None, max_norm: float | None=None, norm_type: float=2.0, _weight: Tensor | None=None) -> None:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

```python
def reset_parameters(self) -> None:
```

---

### transformer.py

**__all__**: `LearnedPosEmbedding`, `RotaryPosEmbedding`, `SinusoidalPosEmbedding`, `Transformer`, `TransformerDecoder`, `TransformerDecoderLayer`, `TransformerEncoder`, `TransformerEncoderLayer`

**Classes**:

#### class LearnedPosEmbedding(nn.Module)

**Methods**:

```python
def __init__(self, max_len: int, embed_dim: int) -> None:
```

```python
def forward(self, input_: lucid.FloatTensor, offset: int=0) -> lucid.FloatTensor:
```

#### class RotaryPosEmbedding(nn.Module)

**Methods**:

```python
def __init__(self, embed_dim: int | None=None, max_seq_len: int | None=None, interleaved: bool=True) -> None:
```

```python
def _build_cache(self, seq_len: int, embed_dim: int, device: str) -> None:
```

```python
def forward(self, input_: lucid.FloatTensor, position_ids: lucid.LongTensor | None=None) -> lucid.FloatTensor:
```

#### class SinusoidalPosEmbedding(nn.Module)

**Methods**:

```python
def __init__(self, seq_len: int | None=None, embed_dim: int | None=None) -> None:
```

```python
def forward(self, input_: lucid.FloatTensor) -> lucid.FloatTensor:
```

#### class Transformer(nn.Module)

**Methods**:

```python
def __init__(self, d_model: int=512, num_heads: int=8, num_encoder_layers: int=6, num_decoder_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1, activation: Callable[[Tensor], Tensor]=F.relu, layer_norm_eps: float=1e-05, norm_first: bool=False, bias: bool=True, custom_encoder: nn.Module | None=None, custom_decoder: nn.Module | None=None) -> None:
```

```python
def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor | None=None, tgt_mask: Tensor | None=None, mem_mask: Tensor | None=None, src_key_padding_mask: Tensor | None=None, tgt_key_padding_mask: Tensor | None=None, mem_key_padding_mask: Tensor | None=None, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_start_layer_idx: int=0, encoder_kv_cache: nn.KVCache | None=None, use_encoder_cache: bool=False, encoder_cache_position: Tensor | None=None, encoder_cache_start_layer_idx: int=0) -> Tensor:
```

#### class TransformerDecoder(nn.Module)

**Methods**:

```python
def __init__(self, decoder_layer: TransformerDecoderLayer | nn.Module, num_layers: int, norm: nn.Module | None=None) -> None:
```

```python
def extra_repr(self) -> str:
```

```python
def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor | None=None, mem_mask: Tensor | None=None, tgt_key_padding_mask: Tensor | None=None, mem_key_padding_mask: Tensor | None=None, tgt_is_causal: bool=False, mem_is_causal: bool=False, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_start_layer_idx: int=0) -> Tensor:
```

#### class TransformerDecoderLayer(nn.Module)

**Methods**:

```python
def __init__(self, d_model: int, num_heads: int, dim_feedforward: int=2048, dropout: float=0.1, activation: Callable[[Tensor], Tensor]=F.relu, layer_norm_eps: float=1e-05, norm_first: bool=False, bias: bool=True) -> None:
```

```python
def _ff_block(self, x: Tensor) -> Tensor:
```

```python
def _mha_block(self, x: Tensor, memory: Tensor, mem_mask: Tensor | None, mem_key_padding_mask: Tensor | None, is_causal: bool) -> Tensor:
```

```python
def _sa_block(self, x: Tensor, tgt_mask: Tensor | None, tgt_key_padding_mask: Tensor | None, is_causal: bool, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_layer_idx: int | None=None) -> Tensor:
```

```python
def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor | None=None, mem_mask: Tensor | None=None, tgt_key_padding_mask: Tensor | None=None, mem_key_padding_mask: Tensor | None=None, tgt_is_causal: bool=False, mem_is_causal: bool=False, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_layer_idx: int | None=None) -> Tensor:
```

#### class TransformerEncoder(nn.Module)

**Methods**:

```python
def __init__(self, encoder_layer: TransformerEncoderLayer | nn.Module, num_layers: int, norm: nn.Module | None=None) -> None:
```

```python
def extra_repr(self) -> str:
```

```python
def forward(self, src: Tensor, src_mask: Tensor | None=None, src_key_padding_mask: Tensor | None=None, is_causal: bool=False, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_start_layer_idx: int=0) -> Tensor:
```

#### class TransformerEncoderLayer(nn.Module)

**Methods**:

```python
def __init__(self, d_model: int, num_heads: int, dim_feedforward: int=2048, dropout: float=0.1, activation: Callable[[Tensor], Tensor]=F.relu, layer_norm_eps: float=1e-05, norm_first: bool=False, bias: bool=True) -> None:
```

```python
def _ff_block(self, x: Tensor) -> Tensor:
```

```python
def _sa_block(self, x: Tensor, src_mask: Tensor | None, src_key_padding_mask: Tensor | None, is_causal: bool, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_layer_idx: int | None=None) -> Tensor:
```

```python
def forward(self, src: Tensor, src_mask: Tensor | None=None, src_key_padding_mask: Tensor | None=None, is_causal: bool=False, kv_cache: nn.KVCache | None=None, use_cache: bool=False, cache_position: Tensor | None=None, cache_layer_idx: int | None=None) -> Tensor:
```

---

### vision.py

**__all__**: `Upsample`

**Classes**:

#### class Upsample(nn.Module)

**Methods**:

```python
def __init__(self, size: int | Tuple[int, ...] | None=None, scale_factor: float | Tuple[float, ...] | None=None, mode: _InterpolateType='nearest', align_corners: bool=False) -> None:
```

```python
def _calculate_size(self, input_: Tensor) -> Tuple[int, ...] | int:
```

```python
def forward(self, input_: Tensor) -> Tensor:
```

---

## lucid/nn/utils

### __init__.py

---

### _base.py

**__all__**: `apply_chunking_to_forward`, `clip_grad_norm`, `clip_grad_value`, `get_activation_from_name`, `get_activation_module_from_name`, `get_total_norm`, `grad_norm`

**Functions**:

```python
def _as_iter(parameters: Iterable[Tensor] | Tensor) -> list[Tensor]:
```

```python
def apply_chunking_to_forward(forward_fn: Callable[..., Tensor], chunk_size: int, chunk_dim: int, *input_tensors: Tensor) -> Tensor:
```

```python
def clip_grad_norm(parameters: Iterable[Tensor] | Tensor, max_norm: _Scalar, norm_type: int=2, eps: float=1e-07) -> float:
```

```python
def clip_grad_value(parameters: Iterable[Tensor] | Tensor, clip_value: _Scalar) -> None:
```

```python
def get_activation_from_name(act_name: str) -> Callable[[Tensor], Tensor] | None:
```

```python
def get_activation_module_from_name(act_name: str) -> type[Module] | None:
```

```python
def get_total_norm(parameters: Iterable[Tensor] | Tensor, norm_type: int=2) -> Tensor:
```

```python
def grad_norm(parameters: Iterable[Tensor] | Tensor, norm_type: int=2) -> Tensor:
```

---

### rnn.py

**__all__**: `PackedSequence`, `pack_padded_sequence`, `pack_sequence`, `pad_packed_sequence`, `pad_sequence`, `unpack_sequence`

**Functions**:

```python
def _as_lengths(lengths: Sequence[int] | Tensor, *, device: str) -> Tensor:
```

```python
def _invert_permutation(indices: Tensor) -> Tensor:
```

```python
def pack_padded_sequence(input_: Tensor, lengths: Sequence[int] | Tensor, batch_first: bool=False, enforce_sorted: bool=True) -> PackedSequence:
```

```python
def pack_sequence(sequences: Iterable[Tensor], enforce_sorted: bool=True) -> PackedSequence:
```

```python
def pad_packed_sequence(sequence: PackedSequence, batch_first: bool=False, padding_value: _Scalar=0) -> tuple[Tensor, Tensor]:
```

```python
def pad_sequence(sequences: Iterable[Tensor], batch_first: bool=False, padding_value: _Scalar=0) -> Tensor:
```

```python
def unpack_sequence(sequence: PackedSequence, batch_first: bool=False) -> list[Tensor]:
```

**Classes**:

#### class PackedSequence

**Attributes**:

- `batch_sizes`
- `data`
- `sorted_indices`
- `unsorted_indices`

---

## lucid/optim

### __init__.py

---

### _base.py

**Classes**:

#### class Optimizer(ABC)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], defaults: dict[str, Any]) -> None:
```

```python
def __repr__(self) -> str:
```

```python
def _flat_params(self) -> list[nn.Parameter]:
```

```python
def add_param_group(self, param_group: dict[str, Any]) -> None:
```

```python
def load_state_dict(self, state_dict: dict) -> None:
```

```python
def param_groups_setup(self, params: list[nn.Parameter], defaults: dict[str, Any]) -> list[dict[str, Any]]:
```

```python
def state_dict(self) -> dict:
```

```python
@abstractmethod
```

```python
def zero_grad(self) -> None:
```

---

### ada.py

**__all__**: `Adadelta`, `Adagrad`, `Adamax`

**Classes**:

#### class Adadelta(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=1.0, rho: float=0.9, eps: float=1e-06, weight_decay: float=0.0) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

#### class Adagrad(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.01, eps: float=1e-10, weight_decay: float=0.0, initial_accumulator_value: float=0.0) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

#### class Adamax(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.002, betas: tuple[_Scalar, _Scalar]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

---

### adam.py

**__all__**: `Adam`, `AdamW`, `NAdam`, `RAdam`

**Classes**:

#### class Adam(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.001, betas: tuple[_Scalar, _Scalar]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0, amsgrad: bool=False) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

#### class AdamW(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.001, betas: tuple[_Scalar, _Scalar]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.01, amsgrad: bool=False) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

#### class NAdam(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.002, betas: tuple[_Scalar, _Scalar]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0, momentum_decay: float=0.004) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

#### class RAdam(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.001, betas: tuple[_Scalar, _Scalar]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

---

### prop.py

**__all__**: `RMSprop`, `Rprop`

**Classes**:

#### class RMSprop(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.01, alpha: float=0.99, eps: float=1e-08, weight_decay: float=0.0, momentum: float=0.0, centered: bool=False) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

#### class Rprop(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.01, etas: tuple[_Scalar, _Scalar]=(0.5, 1.2), step_sizes: tuple[_Scalar, _Scalar]=(1e-06, 50.0)) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> None:
```

---

### sgd.py

**__all__**: `ASGD`, `SGD`

**Classes**:

#### class ASGD(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.001, momentum: float=0.0, weight_decay: float=0.0, alpha: float=0.75, t0: float=1000000.0, lambd: float=0.0001) -> None:
```

```python
def get_averages(self) -> Iterable[nn.Parameter]:
```

```python
def step(self, closure: _OptimClosure | None=None) -> Any | None:
```

#### class SGD(optim.Optimizer)

**Methods**:

```python
def __init__(self, params: Iterable[nn.Parameter], lr: float=0.001, momentum: float=0.0, weight_decay: float=0.0) -> None:
```

```python
def step(self, closure: _OptimClosure | None=None) -> Any | None:
```

---

## lucid/optim/lr_scheduler

### __init__.py

---

### _base.py

**Classes**:

#### class LRScheduler(ABC)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
@abstractmethod
```

```python
@property
```

```python
def load_state_dict(self, state_dict: dict[str, Any]) -> None:
```

```python
def state_dict(self) -> dict[str, Any]:
```

```python
def step(self, epoch: int | None=None) -> None:
```

---

### _schedulers.py

**__all__**: `CosineAnnealingLR`, `CyclicLR`, `ExponentialLR`, `LambdaLR`, `MultiStepLR`, `NoamScheduler`, `ReduceLROnPlateau`, `StepLR`

**Classes**:

#### class CosineAnnealingLR(LRScheduler)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float=0.0, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
def get_lr(self) -> list[float]:
```

#### class CyclicLR(LRScheduler)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, base_lr: float, max_lr: float, step_size_up: int, step_size_down: int | None=None, mode: Literal['triangular', 'triangular2', 'exp_range']='triangular', gamma: float=1.0, scale_fn: Callable[[int], float] | None=None, cycle_momentum: bool=True, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
def get_lr(self) -> list[float]:
```

#### class ExponentialLR(LRScheduler)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
def get_lr(self) -> list[float]:
```

#### class LambdaLR(LRScheduler)

**Methods**:

```python
def __init__(self, lr_lambda: Callable[[int], float], optimizer: Optimizer, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
def get_lr(self) -> list[float]:
```

#### class MultiStepLR(LRScheduler)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, milestones: list[int], gamma: float=0.1, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
def get_lr(self) -> list[float]:
```

#### class NoamScheduler(LRScheduler)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, model_size: int, warmup_steps: int, factor: float=1.0, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
def get_lr(self) -> list[float]:
```

#### class ReduceLROnPlateau(LRScheduler)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, mode: Literal['min', 'max'], factor: float=0.1, patience: int=10, threshold: float=0.0001, threshold_mode: Literal['rel', 'abs']='rel', cooldown: int=0, min_lr: float=0.0, eps: float=1e-08, verbose: bool=False) -> None:
```

```python
def _reduce_lr(self) -> None:
```

```python
def get_lr(self) -> list[float]:
```

```python
def is_better(self, metrics: float) -> bool:
```

```python
def step(self, metrics: float, epoch: int | None=None) -> None:
```

#### class StepLR(LRScheduler)

**Methods**:

```python
def __init__(self, optimizer: Optimizer, step_size: int, gamma: float=0.1, last_epoch: int=-1, verbose: bool=False) -> None:
```

```python
def get_lr(self) -> list[float]:
```

---

## lucid/random

### __init__.py

**__all__**: `bernoulli`, `get_seed`, `permutation`, `rand`, `randint`, `randn`, `seed`, `uniform`

**Functions**:

```python
def bernoulli(probs: _ArrayOrScalar | Tensor, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def get_seed() -> None:
```

```python
def permutation(n: int, dtype: _BuiltinNumeric | Numeric=int, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
@overload
@overload
```

```python
@overload
@overload
```

```python
def rand(*args: int, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def randint(low: int, high: int | None, size: int | _ShapeLike=1, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
@overload
@overload
```

```python
@overload
@overload
```

```python
def randn(*args: int | _ShapeLike, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def seed(seed: int) -> None:
```

```python
def uniform(low: _Scalar=0, high: _Scalar=1, size: int | _ShapeLike=1, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

---

### _func.py

**Functions**:

```python
def bernoulli(probs: _ArrayOrScalar | Tensor, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def permutation(n: int, dtype: _BuiltinNumeric | Numeric, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def rand(shape: _ShapeLike, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def randint(low: int, high: int | None, size: int | _ShapeLike, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def randn(shape: _ShapeLike, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

```python
def seed(seed: int) -> None:
```

```python
def uniform(low: _Scalar, high: _Scalar, size: int | _ShapeLike, requires_grad: bool=False, keep_grad: bool=False, device: _DeviceType='cpu') -> Tensor:
```

---

## lucid/visual

### __init__.py

---

### mermaid.py

**__all__**: `build_module_mermaid_chart`, `build_tensor_mermaid_chart`

**Constants**:

- `_NN_MODULES_PREFIX` = 'lucid.nn.modules.'

**Functions**:

```python
def _build_tree(module: nn.Module, depth: int, max_depth: int, name: str='', *, collapse_repeats: bool=False, repeat_min: int=3, hide_subpackages: set[str] | None=None, hide_module_names: set[str] | None=None) -> _ModuleNode:
```

```python
def _builtin_subpackage_key(module: nn.Module) -> str | None:
```

```python
def _collapse_repeated_children(children: list[_ModuleNode], *, repeat_min: int) -> list[_ModuleNode]:
```

```python
def _container_attr_label(node: _ModuleNode) -> str | None:
```

```python
def _copy_to_clipboard(text: str) -> None:
```

```python
def _escape_label(text: str) -> str:
```

```python
def _flatten_shapes(obj: object) -> list[tuple[int, ...]]:
```

```python
def _flatten_tensors(obj: object) -> list[Tensor]:
```

```python
def _module_label(module: nn.Module, show_params: bool) -> str:
```

```python
def _node_signature(node: _ModuleNode) -> tuple:
```

```python
def _parse_rgba(value: str) -> tuple[str, float] | None:
```

```python
def _shape_str(shape: object) -> str:
```

```python
def _shape_text_color(module: nn.Module) -> str | None:
```

```python
def _shapes_brief(shapes: list[tuple[int, ...]]) -> str:
```

```python
def build_mermaid_chart(module: nn.Module, input_shape: _ShapeLike | list[_ShapeLike] | None=None, inputs: Iterable[Tensor] | Tensor | None=None, depth: int=2, direction: str='LR', include_io: bool=True, show_params: bool=False, return_lines: bool=False, copy_to_clipboard: bool=False, compact: bool=False, use_class_defs: bool=False, end_semicolons: bool=True, edge_mode: Literal['dataflow', 'execution']='execution', collapse_repeats: bool=True, repeat_min: int=2, color_by_subpackage: bool=True, container_name_from_attr: bool=True, edge_stroke_width: float=2.0, emphasize_model_title: bool=True, model_title_font_px: int=20, show_shapes: bool=False, hide_subpackages: Iterable[str]=(), hide_module_names: Iterable[str]=(), dash_multi_input_edges: bool=True, subgraph_fill: str='#000000', subgraph_fill_opacity: float=0.05, subgraph_stroke: str='#000000', subgraph_stroke_opacity: float=0.75, force_text_color: str | None=None, edge_curve: str='natural', node_spacing: int=50, rank_spacing: int=50, **forward_kwargs) -> str | list[str]:
```

```python
def build_module_mermaid_chart(module: nn.Module, input_shape: _ShapeLike | list[_ShapeLike] | None=None, inputs: Iterable[Tensor] | Tensor | None=None, depth: int=2, direction: str='LR', include_io: bool=True, show_params: bool=False, return_lines: bool=False, copy_to_clipboard: bool=False, compact: bool=False, use_class_defs: bool=False, end_semicolons: bool=True, edge_mode: Literal['dataflow', 'execution']='execution', collapse_repeats: bool=True, repeat_min: int=2, color_by_subpackage: bool=True, container_name_from_attr: bool=True, edge_stroke_width: float=2.0, emphasize_model_title: bool=True, model_title_font_px: int=20, show_shapes: bool=False, hide_subpackages: Iterable[str]=(), hide_module_names: Iterable[str]=(), dash_multi_input_edges: bool=True, subgraph_fill: str='#000000', subgraph_fill_opacity: float=0.05, subgraph_stroke: str='#000000', subgraph_stroke_opacity: float=0.75, force_text_color: str | None=None, edge_curve: str='natural', node_spacing: int=50, rank_spacing: int=50, input_dtype: type | None=None, **forward_kwargs) -> str | list[str]:
```

```python
def build_tensor_mermaid_chart(tensor: Tensor, horizontal: bool=False, title: str | None=None, start_id: int | None=None, end_semicolons: bool=True, copy_to_clipboard: bool=False, use_class_defs: bool=True, op_fill: str='lightgreen', param_fill: str='plum', result_fill: str='lightcoral', leaf_fill: str='lightgray', grad_fill: str='lightblue', start_fill: str='gold', stroke_color: str='#666', stroke_width_px: int=1) -> str:
```

**Classes**:

#### class _ModuleNode

**Attributes**:

- `children`
- `depth`
- `group`
- `module`
- `name`

**Methods**:

```python
@property
```

```python
def iter_modules(self) -> Iterable[nn.Module]:
```

---

## SUMMARY

- **Files processed**: 89
- **Classes documented**: 305
- **Functions documented**: 463
- **Methods documented**: 962

