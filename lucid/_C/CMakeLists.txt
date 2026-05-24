cmake_minimum_required(VERSION 3.24)

if(NOT APPLE)
    message(FATAL_ERROR "Lucid C++ engine supports macOS arm64 Apple Silicon only")
endif()

if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "Target Apple Silicon only" FORCE)
endif()

project(lucid_engine LANGUAGES CXX OBJCXX)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(FATAL_ERROR "Lucid C++ engine supports macOS arm64 Apple Silicon only")
endif()

set(LUCID_BUILD_MODE "release" CACHE STRING "Lucid build mode: release, debug, debug-ubsan, debug-asan, debug-tsan")
option(LUCID_COVERAGE "Enable LLVM coverage instrumentation" OFF)
set(LUCID_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

foreach(required_var
        LUCID_PYTHON_INCLUDE_DIR
        LUCID_PYBIND11_INCLUDE_DIR
        LUCID_MLX_INCLUDE_DIR
        LUCID_MLX_LIBRARY_DIR
        LUCID_PYTHON_EXTENSION_SUFFIX
        LUCID_EXTENSION_OUTPUT_DIR)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "${required_var} must be provided by setup.py")
    endif()
endforeach()

find_library(LUCID_MLX_LIBRARY NAMES mlx PATHS "${LUCID_MLX_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)

add_library(lucid_compile_options INTERFACE)
target_compile_features(lucid_compile_options INTERFACE cxx_std_20)
target_include_directories(lucid_compile_options INTERFACE "${LUCID_REPO_ROOT}")
target_include_directories(
    lucid_compile_options
    SYSTEM
    INTERFACE
        "${LUCID_PYTHON_INCLUDE_DIR}"
        "${LUCID_PYBIND11_INCLUDE_DIR}"
        "${LUCID_MLX_INCLUDE_DIR}")
target_compile_definitions(lucid_compile_options INTERFACE LUCID_BUILDING_ENGINE=1)
# Apple deprecated the legacy CBLAS / CLAPACK Fortran-name interfaces in
# macOS 13.3 in favour of a new header layout (cblas_sgemm, sgetrf_, …).
# ACCELERATE_NEW_LAPACK opts into the new symbol routing without changing
# index width (LP64); add ACCELERATE_LAPACK_ILP64 alongside if 64-bit
# indices are ever required.
target_compile_definitions(lucid_compile_options INTERFACE ACCELERATE_NEW_LAPACK)
target_compile_options(
    lucid_compile_options
    INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wno-unused-parameter
        -fvisibility=hidden)

if(LUCID_BUILD_MODE STREQUAL "release")
    target_compile_options(lucid_compile_options INTERFACE -O3)
    target_compile_definitions(lucid_compile_options INTERFACE NDEBUG)
elseif(LUCID_BUILD_MODE STREQUAL "debug")
    target_compile_options(lucid_compile_options INTERFACE -O0 -g -fno-omit-frame-pointer)
    target_link_options(lucid_compile_options INTERFACE -g)
elseif(LUCID_BUILD_MODE STREQUAL "debug-ubsan")
    target_compile_options(
        lucid_compile_options INTERFACE -O1 -g -fno-omit-frame-pointer -fsanitize=undefined)
    target_link_options(lucid_compile_options INTERFACE -fsanitize=undefined)
elseif(LUCID_BUILD_MODE STREQUAL "debug-asan")
    target_compile_options(
        lucid_compile_options
        INTERFACE
            -O1
            -g
            -fno-omit-frame-pointer
            -fsanitize=address,undefined)
    target_link_options(lucid_compile_options INTERFACE -fsanitize=address,undefined)
elseif(LUCID_BUILD_MODE STREQUAL "debug-tsan")
    target_compile_options(
        lucid_compile_options INTERFACE -O1 -g -fno-omit-frame-pointer -fsanitize=thread)
    target_link_options(lucid_compile_options INTERFACE -fsanitize=thread)
else()
    message(FATAL_ERROR "Unknown LUCID_BUILD_MODE=${LUCID_BUILD_MODE}")
endif()

if(LUCID_COVERAGE)
    target_compile_options(lucid_compile_options INTERFACE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(lucid_compile_options INTERFACE -fprofile-instr-generate -fcoverage-mapping)
endif()

function(lucid_object_library target)
    add_library(${target} OBJECT ${ARGN})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${target} PRIVATE lucid_compile_options)
endfunction()

lucid_object_library(
    lucid_core
    version.cpp
    core/Allocator.cpp
    core/AmpPolicy.cpp
    core/Determinism.cpp
    core/ErrorBuilder.cpp
    core/Error.cpp
    core/Generator.cpp
    core/GradMode.cpp
    core/MemoryStats.cpp
    core/OpRegistry.cpp
    core/Profiler.cpp
    core/SchemaGuard.cpp
    core/TensorImpl.cpp
    core/Validate.cpp)

lucid_object_library(
    lucid_backend_cpu
    backend/cpu/Blas.cpp
    backend/cpu/Lapack.cpp
    backend/cpu/Im2Col.cpp
    backend/cpu/Norm.cpp
    backend/cpu/Pool.cpp
    backend/cpu/Reduce.cpp
    backend/cpu/Shape.cpp
    backend/cpu/Vdsp.cpp
    backend/cpu/Vforce.cpp)

# Phase 9.1 + 18: Objective-C++ files require ARC and the Metal framework.
# 3.4 Phase 1: gpu/mps/ subdirectory adds MPSGraph dispatch sources.
lucid_object_library(
    lucid_backend_gpu
    backend/gpu/MlxBridge.cpp
    backend/gpu/MetalAllocator.mm
    backend/gpu/MetalKernelRunner.mm
    backend/gpu/mps/MpsBridge.mm
    backend/gpu/mps/MpsDispatch.cpp
    backend/gpu/mps/MpsKernels.mm)
set_source_files_properties(
    backend/gpu/MetalAllocator.mm
    backend/gpu/MetalKernelRunner.mm
    backend/gpu/mps/MpsBridge.mm
    backend/gpu/mps/MpsKernels.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

lucid_object_library(lucid_backend_init backend/BackendInit.cpp)

lucid_object_library(
    lucid_autograd
    autograd/Node.cpp
    autograd/Helpers.cpp
    autograd/AccumulateGrad.cpp
    autograd/Engine.cpp
    autograd/ModuleHookNode.cpp
    autograd/CustomFunction.cpp    # Phase 12: PythonBackwardNode
    autograd/FusionPass.cpp)       # Phase 19: Op Fusion pass

lucid_object_library(
    lucid_ops
    ops/bfunc/Add.cpp
    ops/bfunc/Sub.cpp
    ops/bfunc/Mul.cpp
    ops/bfunc/Div.cpp
    ops/bfunc/Pow.cpp
    ops/bfunc/Maximum.cpp
    ops/bfunc/Minimum.cpp
    ops/bfunc/Matmul.cpp
    ops/bfunc/Compare.cpp
    ops/bfunc/Bitwise.cpp
    ops/bfunc/Dot.cpp
    ops/bfunc/Inner.cpp
    ops/bfunc/Outer.cpp
    ops/bfunc/Tensordot.cpp
    ops/bfunc/Floordiv.cpp
    ops/bfunc/Inplace.cpp
    ops/ufunc/Arith.cpp
    ops/ufunc/Exponential.cpp
    ops/ufunc/Trig.cpp
    ops/ufunc/Hyperbolic.cpp
    ops/ufunc/Activation.cpp
    ops/ufunc/ScalarParam.cpp
    ops/ufunc/Discrete.cpp
    ops/ufunc/Softmax.cpp
    ops/ufunc/UnaryGpu.cpp
    ops/ufunc/Reductions.cpp
    ops/ufunc/Transpose.cpp
    ops/ufunc/Var.cpp
    ops/ufunc/Trace.cpp
    ops/ufunc/Scan.cpp
    ops/ufunc/Inplace.cpp
    ops/ufunc/CubeRoot.cpp
    ops/ufunc/Predicate.cpp
    ops/gfunc/Gfunc.cpp
    ops/utils/View.cpp
    ops/utils/Contiguous.cpp
    ops/utils/Concat.cpp
    ops/utils/Repeat.cpp
    ops/utils/Pad.cpp
    ops/utils/Layout.cpp
    ops/utils/Tri.cpp
    ops/utils/Select.cpp
    ops/utils/Sort.cpp
    ops/utils/Meshgrid.cpp
    ops/utils/Histogram.cpp
    ops/utils/Nextafter.cpp
    ops/composite/Math.cpp
    ops/composite/Reductions.cpp
    ops/composite/Matrix.cpp
    ops/composite/Logical.cpp
    ops/composite/Indexing.cpp
    ops/composite/Layout.cpp
    ops/composite/Stats.cpp
    ops/composite/Search.cpp
    ops/einops/Rearrange.cpp
    ops/einops/Reduce.cpp
    ops/einops/Repeat.cpp
    ops/einops/Einsum.cpp
    ops/linalg/Inv.cpp
    ops/linalg/Det.cpp
    ops/linalg/Solve.cpp
    ops/linalg/Cholesky.cpp
    ops/linalg/Norm.cpp
    ops/linalg/QR.cpp
    ops/linalg/SVD.cpp
    ops/linalg/MatrixPower.cpp
    ops/linalg/Pinv.cpp
    ops/linalg/Eig.cpp
    ops/linalg/Eigh.cpp
    ops/ufunc/Astype.cpp
    ops/linalg/LUFactor.cpp
    ops/linalg/SolveTriangular.cpp
    ops/linalg/Lstsq.cpp
    ops/linalg/LUSolve.cpp
    ops/linalg/HouseholderProduct.cpp
    ops/linalg/LDLFactor.cpp
    ops/fft/Fftn.cpp
    ops/fft/Ifftn.cpp
    ops/fft/Rfftn.cpp
    ops/fft/Irfftn.cpp
    ops/complex/Real.cpp
    ops/complex/Imag.cpp
    ops/complex/Complex.cpp
    ops/complex/Conj.cpp
    nn/Linear.cpp
    nn/Dropout.cpp
    nn/LayerNorm.cpp
    nn/RMSNorm.cpp
    nn/ConvNd.cpp
    nn/ConvTransposeNd.cpp
    nn/PoolNd.cpp
    nn/AdaptivePool.cpp
    nn/BatchNorm.cpp
    nn/GroupNorm.cpp
    nn/NormExt.cpp
    nn/Loss.cpp
    nn/Attention.cpp
    nn/LSTM.cpp
    nn/Embedding.cpp
    nn/Spatial.cpp
    nn/Interpolate.cpp
    nn/Vision.cpp)

lucid_object_library(lucid_random random/Random.cpp)

lucid_object_library(
    lucid_optim
    optim/Optimizer.cpp
    optim/SGD.cpp
    optim/Adam.cpp
    optim/Prop.cpp
    optim/Ada.cpp
    optim/LRScheduler.cpp)

# 3.5 Phase 1.1+1.2: lucid.compile() graph-capture path.
# Phase 1.1 = Tracer + TraceIR + I/O wiring.
# Phase 1.2 step 1 = MpsBuilder + ExecutableCache + first OpEmitter (Linear).
# Emitters are organised into six sub-packages: core/ linalg/ shape/
# reduce/ nn/ misc/.  Each subdir owns one bucket of MPSGraph emitters
# plus its own ``Registrar`` static initialiser that pushes them into
# the process-global registry on load.
set(LUCID_OP_EMITTERS
    # framework
    compile/OpEmitters/OpEmitter.cpp
    # elementwise/ — one-output ops where each output element only
    # depends on the same-position input element(s)
    compile/OpEmitters/elementwise/Activation.mm
    compile/OpEmitters/elementwise/Arith.mm
    compile/OpEmitters/elementwise/Compare.mm
    compile/OpEmitters/elementwise/DtypeCast.mm
    compile/OpEmitters/elementwise/Math.mm
    compile/OpEmitters/elementwise/Softmax.mm
    compile/OpEmitters/elementwise/Trig.mm
    # reduce/ — collapse one or more axes
    compile/OpEmitters/reduce/Reduction.mm
    compile/OpEmitters/reduce/Cumulative.mm
    compile/OpEmitters/reduce/ArgReduce.mm
    # linalg/ — matrix algebra primitives
    compile/OpEmitters/linalg/Linear.mm
    compile/OpEmitters/linalg/Matmul.mm
    compile/OpEmitters/linalg/MatrixOps.mm
    # shape/ — pure view + layout rearrangement (no data-dep indexing)
    compile/OpEmitters/shape/Concat.mm
    compile/OpEmitters/shape/Layout.mm
    compile/OpEmitters/shape/Permute.mm
    compile/OpEmitters/shape/Reshape.mm
    compile/OpEmitters/shape/Split.mm
    # index/ — data-dependent reads / writes
    compile/OpEmitters/index/Gather.mm
    compile/OpEmitters/index/Indexing.mm
    compile/OpEmitters/index/Scatter.mm
    # nn/ — neural network op families
    compile/OpEmitters/nn/Conv.mm
    compile/OpEmitters/nn/Dropout.mm
    compile/OpEmitters/nn/Pool.mm
    compile/OpEmitters/nn/Norm.mm
    compile/OpEmitters/nn/Loss.mm
    compile/OpEmitters/nn/Attention.mm
    compile/OpEmitters/nn/Embedding.mm
    compile/OpEmitters/nn/Spatial.mm
    compile/OpEmitters/nn/Rnn.mm
    # special/ — compile-pipeline plumbing (factories, complex split,
    # nan_to_num, eager-fallback stubs)
    compile/OpEmitters/special/Complex.mm
    compile/OpEmitters/special/NanToNum.mm
    compile/OpEmitters/special/Factory.mm
    compile/OpEmitters/special/Random.mm
    compile/OpEmitters/special/Stubs.mm)
lucid_object_library(
    lucid_compile
    compile/Tracer.cpp
    compile/ExecutableCache.cpp
    compile/CompiledExecutable.mm
    compile/MpsBuilder.mm
    ${LUCID_OP_EMITTERS})
# ObjC++ sources in the compile path need ARC, same as the other GPU
# bridge ``.mm`` files.  Strip the framework ``.cpp`` from the ARC
# list (it's pure C++) and apply to every emitter ``.mm``.
set(LUCID_OP_EMITTER_MM ${LUCID_OP_EMITTERS})
list(REMOVE_ITEM LUCID_OP_EMITTER_MM compile/OpEmitters/OpEmitter.cpp)
set_source_files_properties(
    compile/CompiledExecutable.mm
    compile/MpsBuilder.mm
    ${LUCID_OP_EMITTER_MM}
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

lucid_object_library(
    lucid_bindings
    bindings/bind_errors.cpp
    bindings/bind_tensor.cpp
    bindings/bind_amp.cpp
    bindings/bind_profiler.cpp
    bindings/bind_op_registry.cpp
    bindings/bind_autograd.cpp
    bindings/bind_nn.cpp
    bindings/bind_random.cpp
    bindings/bind_optim.cpp
    bindings/bind_gfunc.cpp
    bindings/bind_bfunc.cpp
    bindings/bind_ufunc.cpp
    bindings/bind_utils.cpp
    bindings/bind_composite.cpp
    bindings/bind_linalg.cpp
    bindings/bind_einops.cpp
    bindings/bind_fft.cpp
    bindings/bind_complex.cpp
    bindings/bind_compile.cpp        # 3.5 Phase 1.1: lucid.compile() bindings
    bindings/bind.cpp)

add_library(
    lucid_engine MODULE
    $<TARGET_OBJECTS:lucid_core>
    $<TARGET_OBJECTS:lucid_backend_cpu>
    $<TARGET_OBJECTS:lucid_backend_gpu>
    $<TARGET_OBJECTS:lucid_backend_init>
    $<TARGET_OBJECTS:lucid_autograd>
    $<TARGET_OBJECTS:lucid_ops>
    $<TARGET_OBJECTS:lucid_random>
    $<TARGET_OBJECTS:lucid_optim>
    $<TARGET_OBJECTS:lucid_compile>     # 3.5 Phase 1.1: graph-capture tracer
    $<TARGET_OBJECTS:lucid_bindings>)

target_link_libraries(
    lucid_engine
    PRIVATE
        lucid_compile_options
        "${LUCID_MLX_LIBRARY}"
        "-framework Accelerate"
        "-framework Metal"                          # Phase 9.1: MetalAllocator
        "-framework Foundation"                     # required by Metal on macOS
        "-framework MetalPerformanceShaders"        # 3.4 Phase 1: MPSGraph deps
        "-framework MetalPerformanceShadersGraph") # 3.4 Phase 1: graph kernels
if(APPLE)
    target_link_options(lucid_engine PRIVATE -undefined dynamic_lookup)
endif()
set_target_properties(
    lucid_engine
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "engine"
        SUFFIX "${LUCID_PYTHON_EXTENSION_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${LUCID_EXTENSION_OUTPUT_DIR}"
        # BUILD_RPATH = absolute path to the build-env's MLX install so
        # `import lucid` works directly out of the build tree during
        # `pip install -e .` (no install step copies the .so anywhere).
        BUILD_RPATH "${LUCID_MLX_LIBRARY_DIR}"
        # INSTALL_RPATH = the path list dyld searches for ``@rpath/``
        # references at runtime.  Two entries, tried in order:
        #
        #   1. ``@loader_path/../../mlx/lib`` — the canonical wheel layout:
        #         site-packages/lucid/_C/engine.cpython-*.so   ← @loader_path
        #         site-packages/mlx/lib/libmlx.dylib           ← target
        #      Two ``..`` levels reach site-packages, then ``mlx/lib/``.
        #      Resolves correctly inside any venv or system Python that
        #      has MLX installed via pip.
        #
        #   2. ``${LUCID_MLX_LIBRARY_DIR}`` — the build-env's absolute MLX
        #      path.  Used by editable installs (``pip install -e .``),
        #      where the .so lives in the source tree and entry 1 falls
        #      through (no site-packages mlx alongside the source tree).
        #
        # 3.0.0 had only entry 2, so dyld followed the absolute build-
        # machine path verbatim on every user machine — ignoring the
        # user's venv mlx and hitting ABI mismatches when the build path
        # happened to coincide with a different MLX version.  Listing
        # entry 1 FIRST means wheels resolve correctly on any machine,
        # while entry 2 remains a safety net for in-tree development.
        INSTALL_RPATH "@loader_path/../../mlx/lib;${LUCID_MLX_LIBRARY_DIR}"
        # Use INSTALL_RPATH at link time too, so the build artifact has
        # both entries (no separate install relink step).
        BUILD_WITH_INSTALL_RPATH ON
        # Don't auto-add the linker's search path to RPATH — that would
        # add an unintended third entry pointing into the build directory.
        INSTALL_RPATH_USE_LINK_PATH OFF)

# ── Optional C++ unit tests (Google Test) ────────────────────────────────────
option(BUILD_TESTING "Build C++ unit tests using Google Test" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
