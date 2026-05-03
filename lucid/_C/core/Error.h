// lucid/_C/core/Error.h
//
// Lucid exception hierarchy.  All engine exceptions derive from LucidError,
// which itself derives from std::exception so that callers using a generic
// catch(std::exception&) still see a meaningful message.
//
// Exception types are purposely granular so that Python bindings can map them
// to distinct Python exception types (e.g. ShapeMismatch → ValueError) and
// so that C++ callers can selectively catch specific error conditions without
// relying on string-matching what().
//
// Error messages are formatted at construction time and stored in msg_; what()
// is therefore allocation-free at catch sites.

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "../api.h"

namespace lucid {

// Base class for all Lucid engine exceptions.
//
// Concrete subclasses should populate msg_ by constructing the formatted
// message string in their constructor body and assigning it after calling the
// base constructor with an empty string.  This keeps the constructor
// arguments available for inspection through typed accessors.
class LUCID_API LucidError : public std::exception {
public:
    explicit LucidError(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }

protected:
    std::string msg_;
};

// Thrown when an allocation request cannot be satisfied.
// Carries the failed request size plus the current and peak usage counters
// from MemoryTracker so the caller or the user can diagnose fragmentation.
class OutOfMemory : public LucidError {
public:
    OutOfMemory(std::size_t requested_bytes,
                std::size_t current_bytes,
                std::size_t peak_bytes,
                std::string device);
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

// Thrown when two tensors have incompatible shapes for an operation.
// Retains the expected and actual shape vectors for programmatic inspection.
class ShapeMismatch : public LucidError {
public:
    ShapeMismatch(std::vector<std::int64_t> expected,
                  std::vector<std::int64_t> got,
                  std::string context);
    const std::vector<std::int64_t>& expected() const { return expected_; }
    const std::vector<std::int64_t>& got() const { return got_; }

private:
    std::vector<std::int64_t> expected_;
    std::vector<std::int64_t> got_;
};

// Thrown when a dtype mismatch prevents an operation from executing.
// expected and got are human-readable names (e.g. "float32", "int8").
class DtypeMismatch : public LucidError {
public:
    DtypeMismatch(std::string expected, std::string got, std::string context);
    const std::string& expected() const { return expected_; }
    const std::string& got() const { return got_; }

private:
    std::string expected_;
    std::string got_;
};

// Thrown when tensors from different devices are combined incorrectly
// (e.g. a CPU tensor passed where a GPU tensor is required).
class DeviceMismatch : public LucidError {
public:
    DeviceMismatch(std::string expected, std::string got, std::string context);
    const std::string& expected() const { return expected_; }
    const std::string& got() const { return got_; }

private:
    std::string expected_;
    std::string got_;
};

// Thrown by the autograd engine when a saved tensor is mutated between
// the forward and backward passes.  expected_version is the version captured
// at forward time; got_version is the version observed during backward.
class VersionMismatch : public LucidError {
public:
    VersionMismatch(std::int64_t expected, std::int64_t got, std::string context);
    std::int64_t expected_version() const { return expected_; }
    std::int64_t got_version() const { return got_; }

private:
    std::int64_t expected_;
    std::int64_t got_;
};

// Thrown when an operation requires a GPU but no Metal-capable device is
// available, or when the GPU backend cannot be initialised.
class GpuNotAvailable : public LucidError {
public:
    explicit GpuNotAvailable(std::string reason);
};

// Thrown by index or slice operations when an index is out of bounds.
class IndexError : public LucidError {
public:
    explicit IndexError(std::string msg) : LucidError(std::move(msg)) {}
};

// Thrown when an op is not yet implemented for a given dtype, device, or
// configuration (as opposed to an error in the caller's usage).
class NotImplementedError : public LucidError {
public:
    explicit NotImplementedError(std::string msg) : LucidError(std::move(msg)) {}
};

}  // namespace lucid
