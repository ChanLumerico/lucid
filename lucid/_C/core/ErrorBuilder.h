#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "Error.h"
#include "Shape.h"

namespace lucid {

class LUCID_API ErrorContext {
public:
    static void push(std::string op_name);

    static void pop();

    static std::string trace();

    static void reset();
};

class LUCID_API ErrorContextGuard {
public:
    explicit ErrorContextGuard(std::string op_name);
    ~ErrorContextGuard();
    ErrorContextGuard(const ErrorContextGuard&) = delete;
    ErrorContextGuard& operator=(const ErrorContextGuard&) = delete;
};

class LUCID_API ErrorBuilder {
public:
    explicit ErrorBuilder(std::string op_name) : op_(std::move(op_name)) {}
    explicit ErrorBuilder(std::string_view op_name) : op_(op_name) {}
    explicit ErrorBuilder(const char* op_name) : op_(op_name) {}

    [[noreturn]] void fail(const std::string& msg) const;

    [[noreturn]] void not_implemented(const std::string& msg) const;

    [[noreturn]] void index_error(const std::string& msg) const;

    [[noreturn]] void
    shape_mismatch(const Shape& expected, const Shape& got, const std::string& detail = "") const;

    [[noreturn]] void
    dtype_mismatch(Dtype expected, Dtype got, const std::string& detail = "") const;

    [[noreturn]] void
    device_mismatch(Device expected, Device got, const std::string& detail = "") const;

private:
    std::string op_;
    std::string format_with_context(const std::string& msg) const;
};

}  // namespace lucid
