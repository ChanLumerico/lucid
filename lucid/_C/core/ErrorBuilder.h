// lucid/_C/core/ErrorBuilder.h
//
// Fluent error-construction helpers and a thread-local operation call-stack
// for enriching error messages with context about which ops were active at
// the point of failure.
//
// ErrorContext maintains a thread-local stack of operation names.  Each op
// that wants to annotate errors with its name pushes onto the stack via
// ErrorContextGuard (RAII) or the manual push/pop API.  When an error is
// thrown, ErrorBuilder::format_with_context() appends the current trace
// string (e.g. "linear > matmul > gemm") to the message.
//
// Usage:
//   ErrorContextGuard guard("my_op");
//   ErrorBuilder("my_op").shape_mismatch(expected, got);
//
// ErrorBuilder does NOT hold a reference to any tensor; its only job is to
// call the right throw expression with a well-formatted message.

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

// Thread-local call-stack of operation names.
//
// Each entry is typically the human-readable name of an op (e.g. "conv2d").
// The stack is manipulated directly by push/pop for low-overhead code paths,
// or through ErrorContextGuard for RAII safety.
//
// Thread safety: each thread has its own private stack, so no locking is
// required.  The stack is not preserved across thread boundaries.
class LUCID_API ErrorContext {
public:
    // Pushes op_name onto the calling thread's error-context stack.
    static void push(std::string op_name);

    // Pops the most recently pushed name.  No-op if the stack is empty.
    static void pop();

    // Returns the current call stack as a " > " delimited string, or an empty
    // string when the stack is empty.
    static std::string trace();

    // Clears the entire stack.  Intended for test teardown or error recovery.
    static void reset();
};

// RAII wrapper that pushes an op name on construction and pops it on
// destruction.  Preferred over manual push/pop in any scope that may throw.
class LUCID_API ErrorContextGuard {
public:
    explicit ErrorContextGuard(std::string op_name);
    ~ErrorContextGuard();
    ErrorContextGuard(const ErrorContextGuard&) = delete;
    ErrorContextGuard& operator=(const ErrorContextGuard&) = delete;
};

// Constructs and throws typed exceptions, annotating messages with the
// current ErrorContext trace.
//
// Typical usage:
//   ErrorBuilder("my_op").shape_mismatch(expected, got, "weight tensor");
//
// All methods are [[noreturn]]; the compiler will warn if a call site's
// control flow analysis incorrectly assumes these can return.
class LUCID_API ErrorBuilder {
public:
    explicit ErrorBuilder(std::string op_name) : op_(std::move(op_name)) {}
    explicit ErrorBuilder(std::string_view op_name) : op_(op_name) {}
    explicit ErrorBuilder(const char* op_name) : op_(op_name) {}

    // Throws LucidError with msg prepended by the op name and trace.
    [[noreturn]] void fail(const std::string& msg) const;

    // Throws NotImplementedError — use for unimplemented dtype/device combos.
    [[noreturn]] void not_implemented(const std::string& msg) const;

    // Throws IndexError — use for out-of-bounds index access.
    [[noreturn]] void index_error(const std::string& msg) const;

    // Throws ShapeMismatch with the op name and current trace as context.
    // detail is appended after the op name when non-empty.
    [[noreturn]] void
    shape_mismatch(const Shape& expected, const Shape& got, const std::string& detail = "") const;

    // Throws DtypeMismatch, converting Dtype enumerators to name strings.
    [[noreturn]] void
    dtype_mismatch(Dtype expected, Dtype got, const std::string& detail = "") const;

    // Throws DeviceMismatch, converting Device enumerators to name strings.
    [[noreturn]] void
    device_mismatch(Device expected, Device got, const std::string& detail = "") const;

private:
    std::string op_;
    // Prepends op_ and appends the current trace to msg, producing the
    // "op: msg [trace=...]" format consumed by the exception constructors.
    std::string format_with_context(const std::string& msg) const;
};

}  // namespace lucid
