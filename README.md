# Lucid³

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&pause=1000&color=FFFFFF&center=true&vCenter=true&width=435&height=30&lines=A+Deep+Learning+Framework+Built+From+Scratch" alt="Typing SVG"/>

<br>

![PyPI Version](https://img.shields.io/pypi/v/lucid-dl?color=red)
![PyPI Downloads](https://img.shields.io/pypi/dm/lucid-dl.svg)
[![PyPI Total Downloads](https://static.pepy.tech/personalized-badge/lucid-dl?period=total&units=NONE&left_color=GRAY&right_color=yellow&left_text=total%20downloads)](https://pepy.tech/projects/lucid-dl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ChanLumerico/lucid.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-113133-purple)

</div>

**Lucid** is a production-grade machine learning framework built for Apple Silicon. It exposes a PyTorch-compatible Python API backed by a custom C++ engine that runs natively on Apple's hardware stack — MLX on GPU and Apple Accelerate on CPU — with no NumPy dependency in its compute paths.

Version 3.0 is a complete rewrite. The Python-layer minimalism of earlier releases is preserved at the API surface, but underneath sits a fully engineered C++ engine with a typed exception hierarchy, a memory pool, determinism and thread-safety contracts, op fusion, and a direct Metal shader escape hatch. The result is a framework that is simultaneously a clean learning resource and a platform capable of running real training workloads on Apple Silicon hardware.

[📑 Documentation](https://chanlumerico.github.io/lucid/) | [🤗 Hugging Face](https://huggingface.co/ChanLumerico/lucid) | [📋 Changelog](CHANGELOG.md)

---

## 🚀 What's New in 3.0

### ⚙️ Complete C++ engine rewrite

The entire numerical backend has been rewritten in C++ under a new layered architecture with clean interfaces and enforced dependency rules between each layer. A CI check validates the layer graph on every commit; a violation fails the build.

The engine now contains **260+ ops** across unary, binary, reduction, shape, indexing, convolution, pooling, attention, and BLAS families. Every op carries a name, version, AMP policy, and determinism flag, enabling checkpoint compatibility across releases.

### 📦 New Python sub-packages

| Package | Functions | Notes |
|---------|-----------|-------|
| `lucid.fft` | 22 | Full DFT surface: `fft`/`ifft`/`rfft`/`irfft`, 2D + N-D variants, Hermitian forms, `fftshift`/`fftfreq` |
| `lucid.signal.windows` | 12 | Bartlett, Blackman, Gaussian, Kaiser, Nuttall, Hann, Hamming, and more |
| `lucid.special` | 38 | Error functions, Bessel, gamma, digamma, polygamma, Hurwitz ζ, orthogonal polynomials |
| `lucid.distributions` | 33 distributions + 16 transforms | Full `Distribution` base, constraints, KL registry, 20 analytical KL pairs, MC fallback |
| `lucid.linalg` | 38 | Decompositions (QR, SVD, Cholesky, Eigh, LU), norms, solvers, `matrix_exp` |
| `lucid.metal` | — | `lucid.metal.run_kernel` — Metal shader escape hatch for custom GPU kernels |

### 🔌 NumPy independence

Lucid's core is now a standalone binary. `import lucid` and the full forward + backward + optimizer + save/load lifecycle work without NumPy installed. NumPy is an optional extra (`pip install lucid[numpy]`) used only at six explicit bridge boundaries: tensor conversion, `.numpy()`, `_repr.py`, `_types.py`, serialization state_dict, and the DataLoader ingest path.

### 🏛️ Production pillars

Nine production-grade capabilities that ship as first-class features rather than afterthoughts:

| # | Pillar | Entry point |
|---|--------|-------------|
| P1 | Typed exception hierarchy | `LucidError` + 8 subclasses |
| P2 | Memory accounting | `lucid.memory_stats(device)` |
| P3 | Determinism contract | `lucid.set_deterministic(True)` |
| P4 | Thread-safety | forward is thread-safe; backward is single-threaded per root |
| P5 | Sanitizer-clean builds | `LUCID_BUILD_MODE=debug-asan` / `debug-ubsan` |
| P6 | OpSchema versioning | Checkpoint forward-compatibility |
| P7 | Mixed precision | `lucid.amp.autocast` + `GradScaler` |
| P8 | Op-level profiler | `with lucid.profiler() as p: …` |
| P9 | Inference-only C ABI | `liblucid_infer.dylib` |

### ⚡ Op fusion

`lucid.nn.functional.fused_linear_relu`, `fused_linear_gelu`, and `nn.FusedLinear` dispatch to a fused kernel during inference and fall back to standard autograd during training, with no API-level branching required.

### 🔄 Zero-copy CPU↔GPU transfers

CPU↔GPU transfers for tensors above 64 KB avoid an intermediate copy using a shared memory abstraction. Transfers under the threshold use the fast private upload path.

---

## 🏗️ System Architecture

### 🔢 Layer stack

```
┌────────────────────────────────────────────────────┐
│  Python public API                                  │
│    lucid.*  /  lucid.nn.*  /  lucid.optim.*         │
├────────────────────────────────────────────────────┤
│  Python composite & dispatch layer                  │
│    Pure-Python ops + op registry + type boundary    │
├────────────────────────────────────────────────────┤
│  pybind11 boundary                                  │
├────────────────────────────────────────────────────┤
│  C++ engine                                         │
│    Tensor  —  storage, views, dtype, device         │
│    Autograd  —  dynamic graph, backward engine      │
│    Ops  —  260+ kernels across all op families      │
│    CPU backend  —  Accelerate (BLAS/LAPACK/BNNS)    │
│    GPU backend  —  MLX + Metal                      │
└────────────────────────────────────────────────────┘
```

### 🖥️ Backend design

Lucid enforces a strict backend bifurcation:

| Stream | Backend | Rationale |
|--------|---------|-----------|
| CPU | Apple Accelerate (vDSP / vForce / BLAS / LAPACK / BNNS) | Native arm64 SIMD; no framework overhead |
| GPU | MLX | Unified memory; lazy evaluation; Metal under the hood |
| `lucid.linalg` on CPU | MLX (exception) | MLX is itself CPU-backed here; avoids duplicating LAPACK wrappers |
| Data-dependent output shapes | CPU round-trip | Unavoidable when output size is unknown at graph-build time |

The two backends never mix within a single op.

### 🔁 Autograd engine

Lucid implements **reverse-mode automatic differentiation** with a dynamic computation graph. Each op records what it needs to compute gradients, applies the chain rule on the backward pass, and propagates results to parent tensors.

The backward pass is single-threaded per root call; the forward pass is thread-safe. Higher-order differentiation (Jacobians, VJPs, JVPs, `gradcheck`, `gradgradcheck`) is supported in `lucid.autograd`.

### 👁️ View semantics

View ops — `reshape`, `permute`, `transpose`, `slice` — are metadata-only. They share the underlying storage with the source tensor and allocate zero bytes. Zero-copy CPU↔GPU transfers extend this model across devices for tensors above 64 KB.

---

## 📦 Installation

### 📥 Stable release

```bash
pip install lucid-dl
```

### 🛠️ Development install (from source)

```bash
git clone https://github.com/ChanLumerico/lucid.git
cd lucid
pip install -e ".[dev]"
```

The C++ engine is compiled automatically via `scikit-build-core` + CMake + Ninja. Requires Xcode Command Line Tools.

### 🔧 Optional extras

```bash
pip install lucid-dl[numpy]   # numpy bridge (tensor(), .numpy(), from_numpy())
pip install lucid-dl[test]    # pytest + pytest-benchmark + numpy + safetensors
pip install lucid-dl[dev]     # ruff + mypy + numpy (contributor tooling)
```

The reference framework for parity tests is **not** declared as a dependency — install it separately if you need `pytest -m parity`.

### ⚡ GPU (Metal / MLX)

GPU support is built-in on Apple Silicon. No separate install is needed — MLX is linked into the engine at build time. Verify your setup:

```python
import lucid

x = lucid.ones((4, 4), device="metal")
print(x.device)   # metal
print(lucid.backends.metal.is_available())  # True
```

---

## ⚡ Quick Start

### 🔢 Tensors and autograd

```python
import lucid

# Create a tensor on GPU with gradient tracking
x = lucid.randn(3, 4, device="metal", requires_grad=True)

# Forward pass — builds computation graph automatically
y = (x ** 2).sum()

# Backward pass — populates x.grad
y.backward()
print(x.grad)   # shape (3, 4) — d(sum(x^2))/dx = 2x
```

### 🏋️ Training a neural network

```python
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

model = MLP(784, 256, 10).to("metal")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X = lucid.randn(64, 784, device="metal")
Y = lucid.randn(64, 10, device="metal")

for step in range(200):
    loss = F.mse_loss(model(X), Y)
    loss.eval()            # flush MLX lazy graph before backward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"final loss: {loss.item():.4f}")
```

> **MLX lazy evaluation.** MLX defers execution until a value is needed. Call `.eval()` on the loss after the forward pass to flush the accumulated GPU graph before `backward()`. Skipping this can cause unbounded graph growth and degraded performance.

### 🎚️ Mixed precision training

```python
import lucid
import lucid.nn as nn
from lucid.amp import autocast, GradScaler

model = nn.Linear(512, 512).to("metal")
scaler = GradScaler()

with autocast():
    output = model(lucid.randn(32, 512, device="metal"))
    loss = output.sum()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 💾 Saving and loading checkpoints

```python
lucid.save(model.state_dict(), "checkpoint.lucid")

new_model = nn.Linear(512, 512)
new_model.load_state_dict(lucid.load("checkpoint.lucid"))
```

State dicts follow the PyTorch v2 format: `OrderedDict` with a `_metadata` attribute carrying version information for checkpoint forward-compatibility.

### 🔩 Custom Metal kernel

For operations not covered by the built-in op set, Lucid exposes the Metal shader runtime directly:

```python
from lucid.metal import run_kernel

result = run_kernel(
    source="""
    kernel void scale(device const float* x,
                      device float*       y,
                      uint   gid [[thread_position_in_grid]]) {
        y[gid] = x[gid] * 2.0f;
    }
    """,
    kernel_name="scale",
    inputs=[x],
    output_shape=x.shape,
    output_dtype=lucid.float32,
)
```

---

## 🗂️ Module Coverage

### 🔷 Top-level (`lucid.*`)

314 free functions across creation, math, reduction, shape, indexing, and type-casting. Dtype objects (`lucid.float32`, `lucid.int64`, …) and grad-control utilities (`no_grad`, `enable_grad`, `set_grad_enabled`) are exposed at Tier 1. Sub-packages (`lucid.nn`, `lucid.optim`, `lucid.linalg`, …) are loaded lazily.

### 🧱 Neural networks (`lucid.nn`)

| Category | Modules |
|----------|---------|
| Linear | `Linear`, `Bilinear`, `LazyLinear` |
| Convolution | `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d/2d/3d`, `LazyConv*` |
| Recurrent | `RNN`, `LSTM`, `GRU` (with `proj_size`, bidirectional, `PackedSequence`) |
| Normalization | `BatchNorm1d/2d/3d`, `LayerNorm`, `GroupNorm`, `InstanceNorm*`, `RMSNorm` |
| Attention | `MultiheadAttention`, `Transformer`, `TransformerEncoder/Decoder` |
| Pooling | `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool*`, `AdaptiveMaxPool*` |
| Dropout | `Dropout`, `Dropout1d/2d/3d`, `AlphaDropout`, `FeatureAlphaDropout` |
| Sparse | `Embedding`, `EmbeddingBag` |
| Upsampling | `Upsample`, `PixelShuffle` |
| Padding | `ConstantPad1d/2d/3d`, `ZeroPad1d/2d/3d`, `ReflectionPad*`, `ReplicationPad*` |
| Flatten | `Flatten`, `Unflatten` |
| Container | `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict` |
| Activation | 25+ functions incl. `ReLU`, `GELU`, `SiLU`, `Mish`, `Threshold`, `Hardswish`, `LogSigmoid` |
| Loss | `MSELoss`, `CrossEntropyLoss`, `BCELoss`, `NLLLoss`, `CTCLoss`, `HuberLoss`, and more |
| Fusion (new) | `FusedLinear` — fused kernel at inference, autograd at training |

`lucid.nn.functional` provides 70+ stateless functions mirroring the module API. `lucid.nn.init` provides 26 initializer functions. `lucid.nn.utils` includes gradient clipping, weight norm, spectral norm, parametrize, prune, and RNN pack/unpack utilities.

### 🎯 Optimizers (`lucid.optim`)

13 optimizers: `SGD`, `Adam`, `AdamW`, `Adamax`, `NAdam`, `RAdam`, `RMSprop`, `Adadelta`, `Adagrad`, `ASGD`, `LBFGS`, `SparseAdam`, `Rprop`.

16 LR schedulers: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `CyclicLR`, `OneCycleLR`, `ReduceLROnPlateau`, `CosineAnnealingWarmRestarts`, and more.

### 🔬 Math sub-packages

| Package | Highlights |
|---------|-----------|
| `lucid.linalg` | `cholesky`, `eig`, `eigh`, `svd`, `qr`, `lu`, `solve`, `lstsq`, `norm`, `matrix_exp`, `matrix_power` (38 total) |
| `lucid.fft` | Full DFT surface including Hermitian-symmetric (`rfft`, `hfft`) and N-D variants (22 total) |
| `lucid.special` | `erf`/`erfc`/`erfinv`, Bessel `i0`/`i1`, `ndtr`/`ndtri`, `digamma`, `polygamma`, `multigammaln`, Gumbel, Hurwitz ζ (38 total) |
| `lucid.einops` | `rearrange`, `reduce`, `repeat`, `einsum` |

### 🎲 Probability (`lucid.distributions`)

33 distributions including `Normal`, `Bernoulli`, `Categorical`, `Dirichlet`, `Beta`, `Gamma`, `StudentT`, `Cauchy`, `Laplace`, `Poisson`, `Multinomial`, `MultivariateNormal`, `LowRankMultivariateNormal`, `Wishart`, and more.

16 transforms: `AffineTransform`, `ExpTransform`, `SigmoidTransform`, `TanhTransform`, `CorrCholeskyTransform`, `StackTransform`, `CatTransform`, and others.

20 analytical KL divergence pairs are registered; for unregistered pairs the engine falls back to Monte Carlo estimation automatically.

---

## 📊 Performance

All measurements taken on Apple M-series hardware. Numbers represent wall-clock time relative to the reference framework on an equivalent workload.

### 🚦 Backend overhead

After the 3.0 kernel pipeline refactor (redundant intermediate nodes removed from all GPU templates):

| Op | Before 3.0 | After 3.0 |
|----|-----------|----------|
| `relu` | +78% vs reference | +1–3% vs reference |
| `exp` | +28% vs reference | +11% vs reference |

The dominant remaining overhead on GPU is the Python-to-C++ dispatch boundary and MLX lazy-graph commit, not the numerical kernel itself.

### 🔄 Zero-copy transfers

CPU↔GPU transfers above 64 KB avoid an intermediate copy. Below that threshold the fast private upload path is used instead. The threshold is configurable at build time.

### 🧠 BNNS fast paths

`Conv2d` with batch size 1 and `BatchNorm` dispatch to Apple's BNNS (Basic Neural Network Subroutines) for single-sample inference, delivering lower latency than the general BLAS path.

### ⚡ Op fusion

`FusedLinear` (Linear + ReLU or GELU in one kernel) eliminates an intermediate allocation and the activation's separate launch overhead. During training, the op falls back to standard autograd without any user-visible branch.

---

## 🧠 Design Decisions

A selection of non-obvious decisions in the 3.0 architecture:

**No NumPy in the compute path.** Keeping NumPy out of all op implementations means the import graph is clean, cold-start import time is lower, and the framework can be embedded in environments where NumPy is unavailable or undesirable. The explicit bridge boundaries where NumPy is allowed are documented in `CONTRIBUTING.md`.

**CPU = Accelerate, GPU = MLX, no mixing.** Each backend is a fully independent implementation. Crossing backends inside an op is a hard rule violation. `lucid.linalg` is the only permitted exception because MLX is itself CPU-backed via LAPACK on Apple Silicon.

**A single, auditable Python↔C++ boundary.** All transitions between the Python `Tensor` type and the C++ tensor representation happen at one well-defined crossing point. This keeps the boundary auditable and prevents implementation details from leaking into composite ops.

**Op versioning for checkpoint compatibility.** Every op registration includes a version number. When a checkpoint is loaded, the engine can detect version mismatches and apply migration logic rather than silently producing wrong results.

**DLPack via NumPy.** Rather than implementing DLPack export directly in C++, Lucid delegates to NumPy's existing DLPack implementation at the tensor conversion boundary. The cost of a bespoke DLPack layer is not justified when NumPy is already present there for `.numpy()` support.

---

## 🧪 Testing

The test suite has 1,500+ passing tests organized into seven tiers:

| Tier | Location | What it covers |
|------|----------|----------------|
| Unit | `lucid/test/unit/` | Pure Lucid — no reference framework dependency |
| Neural networks | `lucid/test/nn/` | `nn.Module`, `nn.functional`, all layers |
| Autograd | `lucid/test/autograd/` | backward correctness, `gradcheck`, higher-order |
| Linear algebra | `lucid/test/linalg/` | decomposition accuracy |
| Parity | `lucid/test/parity/` | Numerical parity vs reference framework |
| Integration | `lucid/test/integration/` | End-to-end training loops |
| C++ (Google Test) | `lucid/_C/test/` | Kernel-level correctness, concurrency, memory |

Run the Python suite:

```bash
pytest lucid/test/ -q                        # full suite
pytest lucid/test/ --ignore=lucid/test/parity # without parity (no reference framework needed)
pytest lucid/test/ -m smoke                  # fast sanity only
```

Run the C++ suite:

```bash
cmake --build build/temp.macosx-*/lucid__C_engine/ -j$(sysctl -n hw.ncpu)
ctest --test-dir build/temp.macosx-*/lucid__C_engine/ --output-on-failure
```

---

## 💻 System Requirements

| Requirement | Minimum |
|-------------|---------|
| Hardware | Apple Silicon (M1 or later) |
| OS | macOS 26 Tahoe or later |
| Python | 3.14 only (PEP 649 lazy annotations) |
| MLX | ≥ 0.31 (bundled — provides `macosx_26_0_arm64` wheel + `mlx-metal` split) |
| Build tools | CMake ≥ 3.24, Ninja ≥ 1.11, Xcode CLT |
| Runtime deps | MLX (Python `mlx` package; engine links against `libmlx.dylib`) |

Linux, Windows, x86-64, and macOS ≤ 15 are not supported.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including hard rules, coding conventions, the op addition workflow, and the PR checklist.

---

## 📜 License

See [LICENSE](LICENSE).

---

<div align="center">

**Inspired by**

![](https://skillicons.dev/icons?i=pytorch)
![](https://skillicons.dev/icons?i=tensorflow)
![](https://skillicons.dev/icons?i=stackoverflow)

</div>
