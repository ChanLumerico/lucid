Module Hooks
============

This page documents `nn.Module` hook APIs. Hooks let you observe or modify inputs,
outputs, and gradients during forward and backward passes, as well as customize
`state_dict` save/load flows.

Overview
--------

Hooks are registered on a module and return a remover callable. Call the remover
to detach the hook.

.. tip::

    Use hooks for debugging, logging, and lightweight instrumentation. For core
    model behavior, prefer explicit code in `forward`.

.. warning::

    Hooks run inside the forward/backward path. Heavy work inside hooks can slow
    training or change determinism.

Forward Hooks
-------------

Forward pre-hook
^^^^^^^^^^^^^^^^

.. code-block:: python

    def register_forward_pre_hook(self, hook: Callable, *, with_kwargs: bool = False)

**Signature**:

- `hook(module, args) -> args | None`
- If `with_kwargs=True`: `hook(module, args, kwargs) -> (args, kwargs) | None`

.. caution::

    If you return new inputs from a pre-hook, ensure shapes/dtypes are compatible
    with the module's `forward`.

**Example**:

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class Scale(nn.Module):
        def forward(self, x):
            return x * 2

    def pre_hook(module, args):
        (x,) = args
        return (x + 1,)

    m = Scale()
    remove = m.register_forward_pre_hook(pre_hook)

    x = lucid.ones((2, 2))
    y = m(x)  # effectively (x + 1) * 2
    remove()

Forward hook
^^^^^^^^^^^^

.. code-block:: python

    def register_forward_hook(self, hook: Callable, *, with_kwargs: bool = False)

**Signature**:

- `hook(module, args, output) -> output | None`
- If `with_kwargs=True`: `hook(module, args, kwargs, output) -> output | None`

.. note::

    Returning `None` keeps the original output. Returning a value replaces the
    output seen by downstream modules.

**Example**:

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class LinearBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.Parameter(lucid.ones((1,)))

        def forward(self, x):
            return x + self.b

    def post_hook(module, args, output):
        return output * 3

    m = LinearBias()
    remove = m.register_forward_hook(post_hook)

    x = lucid.ones((1,))
    y = m(x)  # output is multiplied by 3
    remove()

Backward Hooks
--------------

Backward hook (output tensor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def register_backward_hook(self, hook: Callable)

**Signature**:

- `hook(tensor, grad) -> None`

.. important::

    This hook attaches to the module's output tensor and only runs when the
    module returns a single `Tensor`.

**Example**:

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class Square(nn.Module):
        def forward(self, x):
            return x * x

    def grad_hook(tensor, grad):
        print("grad:", grad)

    m = Square()
    m.register_backward_hook(grad_hook)

    x = lucid.ones((1,), requires_grad=True)
    y = m(x)
    y.backward()

Full backward pre-hook
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def register_full_backward_pre_hook(self, hook: Callable)

**Signature**:

- `hook(module, grad_output_tuple) -> grad_output_tuple | None`

.. note::

    `grad_output_tuple` contains gradients for each `Tensor` output. Non-`Tensor`
    outputs are omitted.

**Example**:

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class Add(nn.Module):
        def forward(self, x, y):
            return x + y

    def pre_full_backward(module, grad_out):
        print("grad_out:", grad_out)
        return grad_out

    m = Add()
    m.register_full_backward_pre_hook(pre_full_backward)

    x = lucid.ones((1,), requires_grad=True)
    y = lucid.ones((1,), requires_grad=True)
    out = m(x, y)
    out.backward()

Full backward hook
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def register_full_backward_hook(self, hook: Callable)

**Signature**:

- `hook(module, grad_input_tuple, grad_output_tuple) -> None`

.. note::

    `grad_input_tuple` is aligned with positional inputs only. Keyword-only
    inputs are not included and non-`Tensor` inputs appear as `None`.

**Example**:

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class Mul(nn.Module):
        def forward(self, x, y):
            return x * y

    def full_backward(module, grad_in, grad_out):
        print("grad_in:", grad_in)
        print("grad_out:", grad_out)

    m = Mul()
    m.register_full_backward_hook(full_backward)

    x = lucid.ones((1,), requires_grad=True)
    y = lucid.ones((1,), requires_grad=True)
    out = m(x, y)
    out.backward()

State Dict Hooks
----------------

State dict pre-hook
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def register_state_dict_pre_hook(self, hook: Callable)

**Signature**:

- `hook(module, prefix, keep_vars) -> None`

.. tip::

    Use this to set up temporary metadata or logging before a save.

**Example**:

.. code-block:: python

    import lucid.nn as nn

    def pre_state(module, prefix, keep_vars):
        print("saving with prefix:", prefix)

    m = nn.Module()
    m.register_state_dict_pre_hook(pre_state)
    _ = m.state_dict()

State dict hook
^^^^^^^^^^^^^^^

.. code-block:: python

    def register_state_dict_hook(self, hook: Callable)

**Signature**:

- `hook(module, state_dict, prefix, keep_vars) -> None`

.. warning::

    Mutating the `state_dict` changes what gets saved. Keep changes minimal and
    well-documented.

**Example**:

.. code-block:: python

    import lucid.nn as nn

    def post_state(module, state_dict, prefix, keep_vars):
        state_dict[prefix + "note"] = "custom"

    m = nn.Module()
    m.register_state_dict_hook(post_state)
    sd = m.state_dict()

Load state dict pre-hook
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def register_load_state_dict_pre_hook(self, hook: Callable)

**Signature**:

- `hook(module, state_dict, strict) -> None`

.. caution::

    If you modify keys here, ensure they still match the module's current
    structure when `strict=True`.

**Example**:

.. code-block:: python

    import lucid.nn as nn

    def pre_load(module, state_dict, strict):
        state_dict.pop("legacy_key", None)

    m = nn.Module()
    m.register_load_state_dict_pre_hook(pre_load)
    m.load_state_dict({})

Load state dict post-hook
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def register_load_state_dict_post_hook(self, hook: Callable)

**Signature**:

- `hook(module, missing_keys, unexpected_keys, strict) -> None`

.. tip::

    This is a good place to emit warnings or metrics when keys are missing.

**Example**:

.. code-block:: python

    import lucid.nn as nn

    def post_load(module, missing, unexpected, strict):
        if missing:
            print("missing:", missing)

    m = nn.Module()
    m.register_load_state_dict_post_hook(post_load)
    m.load_state_dict({}, strict=False)

Notes
-----

- `register_backward_hook` attaches to the output tensor. It only runs when the
  module returns a single `Tensor`.
- `grad_input_tuple` in full backward hooks is aligned with positional `args`.
  Non-`Tensor` inputs appear as `None`. Keyword-only inputs are not included.
- `grad_output_tuple` is built from `Tensor` outputs only.
- Hook registration returns a callable to remove the hook.
