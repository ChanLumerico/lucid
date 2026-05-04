# lucid/_C/test/CMakeLists.txt
#
# Google Test unit tests for the Lucid C++ engine.
#
# Build with:
#   BUILD_TESTING=ON pip install -e . --no-build-isolation
# Or directly:
#   cmake -DBUILD_TESTING=ON ... && cmake --build .
# Run with:
#   ctest --test-dir <build_dir> --output-on-failure

cmake_minimum_required(VERSION 3.24)

# ── Fetch Google Test ─────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
# Prevent overriding parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST          OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK            ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── Test sources ──────────────────────────────────────────────────────────────
file(GLOB_RECURSE LUCID_TEST_SRCS
    core/*.cpp
    ops/*.cpp
    nn/*.cpp
    backend/*.cpp
    autograd/*.cpp
    integration/*.cpp
)

if(NOT LUCID_TEST_SRCS)
    message(WARNING "[lucid_test] No test source files found — skipping test target.")
    return()
endif()

# ── Test executable ───────────────────────────────────────────────────────────
add_executable(lucid_test ${LUCID_TEST_SRCS})

target_link_libraries(lucid_test PRIVATE
    GTest::gtest_main
    GTest::gmock
    lucid_core
    lucid_backend_cpu
    lucid_backend_gpu
    lucid_ops
    lucid_autograd
)

# Inherit the engine compile options (C++20, warnings, etc.)
target_compile_options(lucid_test PRIVATE
    $<TARGET_PROPERTY:lucid_compile_options,INTERFACE_COMPILE_OPTIONS>
)

# Include paths: helpers/, engine root, and project root
target_include_directories(lucid_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    ${CMAKE_CURRENT_SOURCE_DIR}/..        # lucid/_C — for #include "core/..."
)

# ── Register with CTest ───────────────────────────────────────────────────────
include(GoogleTest)
gtest_discover_tests(lucid_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES        LABELS "lucid_unit"
)
