"""User-facing compiled-execution wrappers for :mod:`lucid.compile`.

Each module here lowers a different scope of a training step into a cached
MPSGraph executable:

* :mod:`~lucid.compile._entry.module` — forward-only (:class:`CompiledModule`).
* :mod:`~lucid.compile._entry.step` — forward + backward (``make_step``).
* :mod:`~lucid.compile._entry.fused_step` — forward + backward + optimizer
  update fused into one executable (``fused_step``).
* :mod:`~lucid.compile._entry.function` — the ``compiled_step`` convenience
  combinator.

The public names are re-exported from :mod:`lucid.compile`; import them from
there rather than reaching into this internal package.
"""
