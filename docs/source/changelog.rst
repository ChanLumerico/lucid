Changelog
=========

3.0.0 (2025)
------------

Initial public release of Lucid 3.0.

**Core engine**

- C++ compute engine with pybind11 bindings
- Accelerate (vDSP / vForce / BLAS / LAPACK) CPU backend
- MLX Metal GPU backend with lazy evaluation

**Autograd**

- Reverse-mode automatic differentiation
- Custom ``Function`` node API
- Gradient checkpointing, anomaly detection

**Neural networks** (``lucid.nn``)

- Linear, Conv 1–3D, Transposed convolutions
- Pooling: Max, Avg, Adaptive variants
- Normalisation: BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm
- Recurrent: LSTM, GRU, RNN
- Attention: MultiheadAttention, ScaledDotProduct
- Transformer: Encoder, Decoder, full Transformer
- Activations: ReLU, GELU, SiLU, Mish, and more
- Dropout, Embedding, Padding, Upsample

**Optimisers** (``lucid.optim``)

- SGD, Adam, AdamW, AdaGrad, RMSprop, LBFGS, NAdam, RAdam, Adamax
- LR schedulers: CosineAnnealing, OneCycle, ReduceLROnPlateau, and more

**Specialised modules**

- ``lucid.linalg`` — SVD, QR, Cholesky, solvers via Accelerate LAPACK
- ``lucid.fft`` — FFT / rFFT / N-D transforms via vDSP
- ``lucid.einops`` — rearrange, reduce, repeat, einsum
- ``lucid.amp`` — autocast + GradScaler
- ``lucid.profiler`` — Metal timeline + CPU wall-clock profiling
- ``lucid.serialization`` — ``save`` / ``load`` checkpoint API
- ``lucid.utils.data`` — Dataset, DataLoader, Samplers
