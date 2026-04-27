#pragma once

// =====================================================================
// Lucid C++ engine — typed exception hierarchy.
// =====================================================================
//
// Every `throw` in the engine throws a `LucidError` subclass — never a bare
// `std::runtime_error`. The pybind11 translator (`bind_errors.cpp`) turns each
// subclass into a same-named Python class so user code can do
// `except lucid.ShapeMismatch:` precisely.
//
// Adding a new exception:
//   1. Subclass `LucidError` here, with structured payload as ctor params.
//   2. Implement the message format in `Exceptions.cpp`.
//   3. Register it in `bind_errors.cpp` (one new `make_subclass` line + one
//      new `catch` arm in the translator).
//   4. Document in `CHANGELOG.md`.
//
// Layer: core/.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "../api.h"

namespace lucid {

// Base for every Lucid-thrown C++ exception. The pybind11 translator maps
// concrete subclasses to matching Python classes (lucid.LucidError,
// lucid.OutOfMemory, ...), so user code can `except lucid.ShapeMismatch:`
// instead of catching opaque RuntimeError.
class LUCID_API LucidError : public std::exception {
public:
    explicit LucidError(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }

protected:
    std::string msg_;
};

// Allocation failed. Carries the requested size and a snapshot of current
// memory state so user / debugger can act on it.
class OutOfMemory : public LucidError {
public:
    OutOfMemory(std::size_t requested_bytes, std::size_t current_bytes,
                std::size_t peak_bytes, std::string device);
    std::size_t requested_bytes() const { return requested_bytes_; }
    std::size_t current_bytes() const { return current_bytes_; }
    std::size_t peak_bytes() const { return peak_bytes_; }
    const std::string& device() const { return device_; }

private:
    std::size_t requested_bytes_;
    std::size_t current_bytes_;
    std::size_t peak_bytes_;
    std::string device_;
};

// Shape doesn't match what the op expected.
class ShapeMismatch : public LucidError {
public:
    ShapeMismatch(std::vector<std::int64_t> expected,
                  std::vector<std::int64_t> got, std::string context);
    const std::vector<std::int64_t>& expected() const { return expected_; }
    const std::vector<std::int64_t>& got() const { return got_; }

private:
    std::vector<std::int64_t> expected_;
    std::vector<std::int64_t> got_;
};

class DtypeMismatch : public LucidError {
public:
    DtypeMismatch(std::string expected, std::string got, std::string context);
    const std::string& expected() const { return expected_; }
    const std::string& got() const { return got_; }

private:
    std::string expected_;
    std::string got_;
};

class DeviceMismatch : public LucidError {
public:
    DeviceMismatch(std::string expected, std::string got, std::string context);
    const std::string& expected() const { return expected_; }
    const std::string& got() const { return got_; }

private:
    std::string expected_;
    std::string got_;
};

// Tensor mutated in-place between forward and backward — the saved version
// no longer matches the live one. (P3 determinism / autograd correctness.)
class VersionMismatch : public LucidError {
public:
    VersionMismatch(std::int64_t expected, std::int64_t got, std::string context);
    std::int64_t expected_version() const { return expected_; }
    std::int64_t got_version() const { return got_; }

private:
    std::int64_t expected_;
    std::int64_t got_;
};

// User asked for the GPU code path but the runtime hasn't enabled it yet
// (Phase 3 wires MLX) or Metal isn't available on the host.
class GpuNotAvailable : public LucidError {
public:
    explicit GpuNotAvailable(std::string reason);
};

// Out-of-bounds index, mirroring Python's IndexError. We keep the C++ name
// suffixed with `Error` to avoid shadowing builtin macros on some toolchains.
class IndexError : public LucidError {
public:
    explicit IndexError(std::string msg) : LucidError(std::move(msg)) {}
};

// A code path that's intentionally not implemented in this phase. Different
// from `LucidError` callers swallowing things — this is "feature explicitly
// not shipped yet."
class NotImplementedError : public LucidError {
public:
    explicit NotImplementedError(std::string msg) : LucidError(std::move(msg)) {}
};

}  // namespace lucid
