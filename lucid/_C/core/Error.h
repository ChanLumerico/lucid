#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "../api.h"

namespace lucid {

class LUCID_API LucidError : public std::exception {
public:
    explicit LucidError(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }

protected:
    std::string msg_;
};

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

class VersionMismatch : public LucidError {
public:
    VersionMismatch(std::int64_t expected, std::int64_t got, std::string context);
    std::int64_t expected_version() const { return expected_; }
    std::int64_t got_version() const { return got_; }

private:
    std::int64_t expected_;
    std::int64_t got_;
};

class GpuNotAvailable : public LucidError {
public:
    explicit GpuNotAvailable(std::string reason);
};

class IndexError : public LucidError {
public:
    explicit IndexError(std::string msg) : LucidError(std::move(msg)) {}
};

class NotImplementedError : public LucidError {
public:
    explicit NotImplementedError(std::string msg) : LucidError(std::move(msg)) {}
};

}  // namespace lucid
