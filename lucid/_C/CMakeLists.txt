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
target_compile_options(lucid_backend_cpu PRIVATE -Wno-deprecated-declarations)

# Phase 9.1 + 18: Objective-C++ files require ARC and the Metal framework.
lucid_object_library(
    lucid_backend_gpu
    backend/gpu/MlxBridge.cpp
    backend/gpu/MetalAllocator.mm
    backend/gpu/MetalKernelRunner.mm)
set_source_files_properties(
    backend/gpu/MetalAllocator.mm
    backend/gpu/MetalKernelRunner.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

lucid_object_library(lucid_backend_init backend/BackendInit.cpp)

lucid_object_library(
    lucid_autograd
    autograd/Node.cpp
    autograd/Helpers.cpp
    autograd/AccumulateGrad.cpp
    autograd/Engine.cpp
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
    bindings/bind_linalg.cpp
    bindings/bind_einops.cpp
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
    $<TARGET_OBJECTS:lucid_bindings>)

target_link_libraries(
    lucid_engine
    PRIVATE
        lucid_compile_options
        "${LUCID_MLX_LIBRARY}"
        "-framework Accelerate"
        "-framework Metal"       # Phase 9.1: MetalAllocator
        "-framework Foundation") # required by Metal on macOS
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
        BUILD_RPATH "${LUCID_MLX_LIBRARY_DIR}"
        INSTALL_RPATH "${LUCID_MLX_LIBRARY_DIR}")
