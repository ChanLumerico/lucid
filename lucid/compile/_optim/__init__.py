"""Optimizer-step compilation for :mod:`lucid.compile`.

* :mod:`~lucid.compile._optim.compiler` — ``compile_optimizer`` and the
  per-optimizer compiled update wrappers (in-place output path).  Each
  wrapper's hooks describe its structural flags, state buffers and
  per-step scalars; the hyper-parameters themselves are read from the
  live ``param_groups`` every step.
* :mod:`~lucid.compile._optim.spec` — the shared hyper-parameter reader.
"""
