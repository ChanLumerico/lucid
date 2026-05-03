# Lucid vs PyTorch — 기능 격차 분석 (Apple Silicon 전용 관점 재정렬)

Lucid는 **Apple Silicon 전용 프레임워크**라는 포지셔닝을 전제로, PyTorch 기능 호환성 측면에서 부족한 부분을 재정렬한 문서.

**스코프 원칙**:
- ✅ **In-scope**: PyTorch 사용자가 이식할 때 즉시 막히는 API surface
- ✅ **In-scope**: Apple Silicon 차별화 포인트 (CoreML/MLX compile/통합 메모리 활용)
- ⏬ **Deprioritized**: NVIDIA/CUDA 전용 (NCCL, TF32, fused CUDA graph, pin_memory, channels_last 메모리 포맷, FP8)
- ⏬ **Deprioritized**: 분산 학습 (DDP/FSDP/TP/PP/RPC) — 단일 노드 Apple Silicon에서 의미 적음
- ⏬ **Deprioritized**: 양자화 인프라 (qint8/QAT) — Apple Silicon은 fp16/bf16 위주
- ⏬ **Deprioritized**: 그래프 컴파일러 (TorchInductor/Dynamo/TorchScript) — MLX compile에 위임 가능
- ⏬ **Deprioritized**: 저수준 IO (mmap-backed tensor, fd 기반 IPC, file-backed storage)

---

## Tier 1 — Critical (PyTorch 사용자가 즉시 부딪히는 격차)

PyTorch 코드를 Lucid로 옮길 때 첫 1시간 안에 막히는 항목들.

### 1.1 Tensor In-place op 가족

| 항목 | PyTorch | Lucid 현황 | 부족한 디테일 |
|---|---|---|---|
| 산술 in-place | add_, sub_, mul_, div_, pow_, fmod_, remainder_ | fill_, copy_, zero_ | **거의 모든 산술 in-place 변형** 부재 |
| 활성/클램프 in-place | abs_, neg_, clamp_, clamp_min_, clamp_max_, sigmoid_, tanh_, relu_ | 없음 | **클램프/활성 in-place** 부재 |
| 합성 in-place | addcmul_, addcdiv_, addmm_, baddbmm_, lerp_ | 없음 | **fused 합성 op in-place** 부재 |
| 인덱싱 in-place | index_add_, index_copy_, index_fill_, index_put_, masked_fill_, masked_scatter_, scatter_, scatter_add_, scatter_reduce_ | masked_fill 일부 | **scatter/index 가족 in-place** 부재 |

### 1.2 Loss 함수

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Sequence | CTCLoss | 없음 | **CTC** — ASR/OCR 필수 |
| Metric learning | TripletMarginLoss, TripletMarginWithDistanceLoss, CosineEmbeddingLoss, MarginRankingLoss | 없음 | **임베딩 학습 4종** |
| Multi-label | MultiMarginLoss, MultiLabelMarginLoss, MultiLabelSoftMarginLoss, HingeEmbeddingLoss | 없음 | **멀티라벨/힌지 4종** |
| Distribution | PoissonNLLLoss, GaussianNLLLoss, KLDivLoss | KLDiv 가능 | **PoissonNLL/GaussianNLL** 부재 |

### 1.3 RNN 생태계

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Packed sequence | PackedSequence + pack_padded_sequence + pad_packed_sequence + pack_sequence + pad_sequence | nn.utils 일부 | **가변 길이 시퀀스 처리 풀세트** |
| RNN 옵션 | bidirectional, num_layers, dropout (between layers), proj_size (LSTM projection) | 클래스 존재 — 동작 검증 필요 | **bidirectional/projections/dropout 동작** 검증 필요 |

### 1.4 Embedding

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| EmbeddingBag | sum/mean/max 풀링 임베딩 | 없음 | **EmbeddingBag** 부재 — 추천/NLP 필수 |
| Embedding 옵션 | max_norm, norm_type, scale_grad_by_freq, sparse, padding_idx 그래디언트 무시 | padding_idx 정도 | **max_norm, scale_grad_by_freq** 부재 |
| one_hot | F.one_hot | 있음 (추정) | OK |

### 1.5 분포 / 수치 패키지

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| `torch.distributions` | Normal, Bernoulli, Categorical, Dirichlet, Beta, Gamma, ... + KL + transforms + constraints | 없음 | **distributions 패키지 전무** — VAE/RL/베이지안 차단 |
| `torch.fft` | fft/ifft/rfft/irfft/fft2/fftn/fftshift/ifftshift | 없음 | **FFT 패키지 부재** — signal/spectral 차단 |
| `torch.special` | gamma, lgamma, digamma, polygamma, erf, erfc, erfinv, expit, logit, i0, ... | 활성화 일부에만 | **special 패키지 부재** |

### 1.6 DataLoader (단일 노드 multiprocessing)

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Worker 풀 | num_workers, persistent_workers, prefetch_factor, worker_init_fn, multiprocessing_context | 동기 단일 worker | **multiprocessing 워커** 부재 — 학습 처리량 직접적 영향 |
| 병렬 데이터 변환 | per-worker 시드, fork/spawn | 없음 | 시드 + context 부재 |

> CUDA 전용 `pin_memory`는 Apple unified memory 환경에서는 의미 없으므로 제외.

### 1.7 Functorch (현대 PyTorch의 표준)

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| `torch.func.vmap` | 자동 배치화 | 없음 | **vmap 부재** |
| `torch.func.grad / jacrev / jacfwd` | functional grad/Jacobian | 없음 | **함수형 미분 변환 부재** |
| `torch.func.hessian` | Hessian 자동 | 없음 | 부재 |
| `torch.func.functional_call` | stateless module 호출 | 없음 | 부재 |

### 1.8 Module 메서드 표준

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| `Module.apply(fn)` | 재귀 함수 적용 | 없음/미흡 | **초기화 순회 표준** |
| `Module.to_empty(device)` | 메타 디바이스 → 실제 디바이스 | 없음 | **거대 모델 초기화** 차단 |
| 일괄 dtype 변환 | `.half()`, `.float()`, `.double()`, `.bfloat16()` | `.to(dtype)` | **PyTorch 패턴과 1:1 매칭** 부재 |
| Full backward hook | `register_full_backward_hook`, `register_full_backward_pre_hook` | 일부 | **Module 단위 backward hook** 미흡 |

---

## Tier 2 — High (생태계 완성도 / Apple Silicon 차별화)

### 2.1 Apple Silicon 차별화 포인트 ⭐

이 섹션이 Lucid가 PyTorch 대비 **고유 가치**를 가질 수 있는 영역.

| 항목 | 현황 | 우선순위 이유 |
|---|---|---|
| **CoreML export** (`lucid.export.to_coreml(model)`) | 없음 | **Apple Silicon 프레임워크의 가장 큰 missed opportunity** — iOS/macOS 배포 직결 |
| **MLX compile 통합** (`lucid.compile(model)`) | 없음 | MLX의 `mx.compile` 위임으로 PyTorch `torch.compile` 호환 인터페이스 제공 가능 |
| **ANE (Apple Neural Engine) 디스패치** | 없음 | CoreML 경유 ANE 활용은 Apple 전용 프레임워크의 핵심 차별점 |
| **Unified memory 활용** (zero-copy CPU↔GPU view) | SharedStorage variant 존재 | API 노출 강화 — PyTorch에 없는 장점 |
| **MPS Profiler 통합** (Instruments / Metal System Trace) | Chrome trace만 | Xcode Instruments 연계 시 디버깅 경험 우위 |

### 2.2 nn 레이어 빈틈

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Pool 3D / 특수 | MaxPool3d, AvgPool3d, FractionalMaxPool2/3d, LPPool1/2/3d, MaxUnpool1/2/3d | MaxPool1/2d, AvgPool1/2d만 | **3D pool, FractionalMaxPool, MaxUnpool** 부재 |
| Conv 보조 | F.fold (im2col 역연산), F.unfold | unfold만 | **fold** 부재 |
| Activation 빈틈 | Hardshrink, Softshrink, Tanhshrink, RReLU, Softmax2d, AdaptiveLogSoftmaxWithLoss, Gumbel softmax | 19종 보유 | 약 10종 부재 |
| Distance | F.pdist, F.cdist | pairwise/cosine | **pdist/cdist** 부재 |

### 2.3 nn.utils 표준 도구

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Parametrization | `nn.utils.parametrize`, `weight_norm`, `spectral_norm`, `orthogonal` | 없음 | **GAN/Transformer/RL** 표준 도구 부재 |
| Pruning | `nn.utils.prune` (l1/random/structured/global, 영구 적용) | 없음 | **모델 압축** 인프라 부재 |
| Stateless | `torch.nn.utils.stateless` | 없음 | 메타러닝/MAML 차단 |
| Lazy modules | LazyLinear, LazyConv*, LazyBatchNorm* | 없음 | **shape inference** 자동화 부재 |

### 2.4 Optimizer / Scheduler 빈칸

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Optimizer | LBFGS, SparseAdam | 11종 보유 | **LBFGS** 부재 — 2차 방법 차단 |
| LR Scheduler | OneCycleLR, CosineAnnealingWarmRestarts, PolynomialLR, LinearLR, ConstantLR, ChainedScheduler, SequentialLR, MultiplicativeLR | 8종 + Noam | **8종 부재** — 특히 OneCycleLR/Warm-restart는 SOTA 학습 패턴 |
| SWA/EMA | `torch.optim.swa_utils` (AveragedModel, SWALR, update_bn) | 없음 | **SWA/EMA 부재** — modern training recipe |
| param_groups | per-group 하이퍼파라미터, add_param_group | 추정 지원 | **검증 필요** |
| Hook | optimizer step pre/post hook | 없음 | gradient clipping 패턴 영향 |

### 2.5 Autograd 디버깅 / 검증

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| `torch.autograd.gradcheck` | finite-diff 자동 검증 | 없음 (parity harness 내부에만) | **사용자 custom op 검증** 차단 |
| `torch.autograd.gradgradcheck` | 2차 미분 검증 | 없음 | 부재 |
| `torch.testing.assert_close` | atol/rtol 비교 | 없음 (내부 사용) | **공개 API 노출** 부재 |
| `torch.autograd.set_detect_anomaly` | NaN/Inf 추적 | 없음 | **디버깅 차단** |
| Saved tensor hooks | `saved_tensors_hooks`, `disable_saved_tensors_hooks` | 없음 | activation checkpointing/offloading 차단 |
| Forward-mode AD (jvp) | `torch.autograd.forward_ad` | 없음 | **JVP 부재** |
| Custom Function `setup_context` / `vmap` | 새 API | forward/backward만 | **현대 Function API** 미흡 |

### 2.6 Models / Hub / Pretrained

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| `lucid.models` | torchvision.models (resnet/vit/efficientnet/...) 100+ | 없음 (review/는 학습 코드만) | **Models 모듈 전무** |
| `torch.hub` | hub.load() | 없음 | **공식 hub** 부재 |
| Pretrained weights | weights=...로 사전 학습 가중치 | 없음 | **가중치 배포 채널** 부재 |
| HuggingFace 호환 | from_pretrained 인터페이스 | 없음 | **HF 호환 어댑터** 부재 |

### 2.7 Tensor view / 인덱싱 빈틈

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| view 연산 | as_strided, narrow, view_as_complex/real, movedim/moveaxis, swapaxes/swapdims | 기본 reshape/view/permute/transpose | **as_strided, narrow, view_as_complex/real** 부재 |
| 고급 인덱싱 | take_along_dim, scatter_reduce_ (amin/amax/prod/mean), put_, take | 일부 | **take_along_dim, scatter_reduce_** 부재 |
| Ellipsis 케이스 | `x[..., 0]` 전수 지원 | 일부 | **검증 필요** |

### 2.8 Linalg 확장

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| 분해 | linalg.lu, lu_factor, lu_solve, slogdet, householder_product | inv/det/solve/cholesky/qr/svd/eig/eigh/pinv/norm/matrix_power | **lu, slogdet** 부재 |
| 응용 | linalg.lstsq, matrix_exp, multi_dot, tensorsolve, tensorinv, vector_norm, matrix_norm | norm 통합 | **lstsq, matrix_exp, multi_dot** 부재 |

---

## Tier 3 — Medium (Polish)

### 3.1 직렬화 / 로깅

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| safetensors | 표준 외부 라이브러리 | 없음 | **safetensors 미지원** — HF 모델 로딩 차단 |
| TensorBoard | `torch.utils.tensorboard.SummaryWriter` | 없음 | **로깅 인프라** 부재 |
| `torch.save` 옵션 | weights_only=True 보안, map_location | save/load 존재 | **map_location의 GPU↔CPU 변환** 검증 필요 |
| Profiler export | TensorBoard plugin, FlameGraph | Chrome trace만 | TensorBoard plugin 부재 |
| Checkpoint | `torch.utils.checkpoint` (use_reentrant, sequential, CPU offload) | 기본 존재 추정 | **고급 옵션** 미흡 |

### 3.2 Random / Generator

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Generator 객체 | `torch.Generator(device)` per-device | manual_seed | **Generator 객체** 부재 |
| RNG state 직렬화 | get_rng_state / set_rng_state / fork_rng | 없음/추정 | **재현성 도구** 부재 |

### 3.3 Meta / FakeTensor

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| `device='meta'` | 메타 텐서 (메모리 0, shape만) | 없음 | **거대 모델 초기화 후 to_empty** 차단 |
| FakeTensor | shape inference 시뮬레이션 | 없음 | **그래프 사전 검증** 부재 |

### 3.4 Nested / Sparse Tensor

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| Nested tensor | 가변 길이 batched tensor | 없음 | **Transformer 가변 시퀀스** 차단 |
| Sparse tensor | COO/CSR/CSC | 없음 | **sparse 연산 부재** |

### 3.5 Vision functional

| 항목 | PyTorch | Lucid 현황 | 부족 |
|---|---|---|---|
| F.interpolate | 모든 mode + antialias | 일부 | **mode 전수 / antialias** 검증 필요 |
| F.grid_sample | bicubic, padding_mode 전수, align_corners | 존재 | **mode/padding 검증** |
| torchvision ops | nms, roi_align, deform_conv2d, ps_roi_align | 없음 | **detection 도메인** 차단 (별도 패키지로 분리 가능) |

---

## Tier 4 — Low / N/A (Apple Silicon 환경에서 의미 적음)

명시적으로 **후순위로 빼두는** 항목 (정책적 결정).

| 항목 | 후순위 사유 |
|---|---|
| DDP / FSDP / TP / PP / RPC | 단일 노드 Apple Silicon 환경. multi-machine 분산은 별도 로드맵 |
| NCCL / Gloo / MPI backend | NVIDIA/Linux 클러스터 전용 |
| TF32 | NVIDIA Ampere+ 전용 |
| FP8 (e4m3/e5m2) | Apple Silicon 미지원 |
| Quantization (qint8/quint8/QAT/observer) | Apple은 fp16/bf16 위주, CoreML 양자화에 위임 |
| Channels_last memory_format | CUDA Tensor Core 친화 — Apple Silicon에서 효익 미미 |
| `pin_memory` / `pin_memory_device` | Apple unified memory에서 무의미 |
| TorchScript (`jit.trace`/`jit.script`) | Deprecated 방향 — `torch.export` + CoreML로 대체 |
| `torch.compile` (Inductor/Dynamo) | MLX `mx.compile`에 위임. 자체 컴파일러는 비용 대비 효익 낮음 |
| SyncBatchNorm | 분산 학습 부재로 의미 없음 (다중 노드 도입 시점에 함께) |
| DataParallel | Apple 단일 GPU |
| `mmap` 텐서 / fd 기반 IPC / file-backed shared storage | 저수준 IO — 우선순위 낮음 |
| Optimizer fused/foreach/capturable | CUDA graph 의존 — Apple은 MLX lazy graph로 자연 fusion |
| Named tensor | PyTorch에서도 실험적 |
| FX graph 양자화 / GraphModule | 그래프 컴파일러 도입 후 검토 |

---

## 재정렬된 우선순위 로드맵

### Phase Q (Critical) — PyTorch 이식성 1차

PyTorch 사용자가 첫날 부딪히는 격차 해소.

1. **In-place op 가족 전수** — `add_/mul_/sub_/div_/pow_/clamp_/abs_/neg_/sigmoid_/tanh_/relu_` + scatter/index in-place
2. **Loss 보강** — CTCLoss, TripletMargin*, MarginRanking, CosineEmbedding, MultiMargin*, Hinge*, PoissonNLL, GaussianNLL
3. **PackedSequence + 4 helper** — pack_padded/pad_packed/pack_sequence/pad_sequence
4. **EmbeddingBag + 옵션** — max_norm, scale_grad_by_freq
5. **DataLoader workers** — num_workers, persistent_workers, prefetch_factor, worker_init_fn
6. **`lucid.distributions`** — 최소 Normal/Bernoulli/Categorical/Dirichlet/Beta/Gamma + KL + sample/log_prob/rsample
7. **`lucid.fft`** — fft/ifft/rfft/irfft/fft2/fftn (MLX `mx.fft` 위임 가능)
8. **`lucid.special`** — gamma/lgamma/erf/erfc/digamma 등 (MLX/Accelerate 위임)
9. **Module 표준 메서드** — `apply()`, `.half()/.float()/.double()/.bfloat16()`, `to_empty()`, full backward hook

### Phase R (High — Apple 차별화) ⭐

Lucid가 PyTorch 대비 **고유 가치**를 갖는 핵심.

10. **`lucid.export.to_coreml(model)`** — ONNX-tier 우선순위, iOS/macOS 배포 직결
11. **`lucid.compile(model)`** — MLX `mx.compile` 통합 PyTorch-호환 API
12. **ANE 디스패치 정책** — CoreML 경유 ANE 활용 옵션
13. **Unified memory API 노출** — zero-copy CPU↔GPU view 명시화
14. **Xcode Instruments 통합** — Metal System Trace, MPS Profiler 연계

### Phase S (High — Modern API)

현대 PyTorch 사용자가 기대하는 표준.

15. **Functorch** — `lucid.func.vmap/grad/jacrev/jacfwd/hessian/functional_call`
16. **Forward-mode AD** — `lucid.autograd.forward_ad` + Function `jvp`
17. **`lucid.autograd.gradcheck` + `lucid.testing.assert_close` + `detect_anomaly`** — 디버깅/검증 표준
18. **`saved_tensors_hooks`** — activation checkpointing 기반
19. **Lazy modules** — LazyLinear, LazyConv*, LazyBatchNorm*
20. **Meta device + `to_empty`** — 거대 모델 초기화

### Phase T (Polish — 생태계)

21. **`lucid.models` zoo + `lucid.hub`** — ResNet/ViT/EfficientNet/BERT 사전 학습 가중치
22. **HuggingFace `from_pretrained` 어댑터** — safetensors 로더 포함
23. **`lucid.utils.tensorboard.SummaryWriter`** — 로깅 표준
24. **`nn.utils.parametrize/weight_norm/spectral_norm/prune`** — modern training 도구
25. **`torch.optim.swa_utils` 호환** — AveragedModel, SWALR, update_bn
26. **LBFGS, OneCycleLR, CosineAnnealingWarmRestarts, ChainedScheduler, SequentialLR** — 빈칸 메우기
27. **Tensor view 보강** — as_strided, narrow, view_as_complex/real, movedim
28. **Linalg 확장** — lstsq, lu, slogdet, matrix_exp, multi_dot
29. **F.fold, F.pdist/cdist, MaxPool3d, FractionalMaxPool, MaxUnpool**
30. **Nested tensor (가변 시퀀스)** — Transformer 효율 학습용

### Phase U (Future / 별도 로드맵)

- Sparse tensor (특수 도메인)
- 분산 학습 (multi-machine — Apple Silicon 클러스터 도입 시점)
- 그래프 컴파일러 자체 구현 (MLX compile로 충분하면 보류)
- Quantization (CoreML quantization에 위임 검토)

---

## 종합 평가

**Apple Silicon 전용 + PyTorch 호환** 두 축으로 재정렬했을 때:

- **Tier 1 (Critical)** 9개 항목을 채우면 PyTorch 코드 이식성이 80% → 95%
- **Tier 2 Apple 차별화** 5개 항목 (CoreML export / MLX compile / ANE / unified memory / Instruments)이 **Lucid의 진짜 경쟁력** — 이게 없으면 그냥 "PyTorch 클론"
- **Tier 4** 분산/양자화/CUDA-specific은 명시적 후순위 → 로드맵에서 정리 부담 제거

**핵심 메시지**:
> "Lucid는 PyTorch의 작은 부분집합이 아니라, **Apple Silicon 환경에서 PyTorch보다 자연스러운 프레임워크**"

이 포지션을 살리려면 **Phase Q (이식성) + Phase R (Apple 차별화)** 두 묶음을 함께 진행하는 게 가장 효과적.
