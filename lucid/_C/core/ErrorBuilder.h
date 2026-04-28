#pragma once

// =====================================================================
// Lucid C++ engine — error construction helpers + thread-local context.
// =====================================================================
//
// Two complementary tools that sit on top of the typed exceptions in
// `Exceptions.h`:
//
// 1. `ErrorBuilder`     — chainable convenience for the most common
//                          throw patterns. Equivalent to constructing
//                          the typed exception directly, but reads
//                          better at call sites.
//
// 2. `ErrorContext`     — a thread-local stack of op names. When op
//                          forwards push their name on entry (via
//                          `ErrorContextGuard`), any thrown
//                          `LucidError` automatically gains a stack
//                          trace in `what()` showing the op nesting.
//
// Both are migrated incrementally — Phase 1 introduces them; the bulk
// of `throw LucidError("...")` call sites move over phase by phase.
//
// Layer: core/. Depends on Exceptions.h, Shape.h, Dtype.h.

#include <string>
#include <string_view>
#include <vector>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "Exceptions.h"
#include "Shape.h"

namespace lucid {

// --------------------------------------------------------------------------- //
// ErrorContext — thread-local op-name stack threaded through every throw.
// --------------------------------------------------------------------------- //

class LUCID_API ErrorContext {
public:
    /// Push an op name onto the calling thread's context stack.
    static void push(std::string op_name);
    /// Pop the most recently pushed op name.
    static void pop();
    /// Snapshot of the current trace (top-of-stack first), formatted as
    /// "outer > middle > inner". Returns empty string if stack is empty.
    static std::string trace();
    /// Clear the stack (test fixtures, panic recovery).
    static void reset();
};

/// RAII guard that pushes/pops an op name. Use on every op forward entry:
///   ErrorContextGuard ctx_guard{"matmul"};
class LUCID_API ErrorContextGuard {
public:
    explicit ErrorContextGuard(std::string op_name);
    ~ErrorContextGuard();
    ErrorContextGuard(const ErrorContextGuard&) = delete;
    ErrorContextGuard& operator=(const ErrorContextGuard&) = delete;
};

// --------------------------------------------------------------------------- //
// ErrorBuilder — chainable convenience for the common throw patterns.
// --------------------------------------------------------------------------- //

class LUCID_API ErrorBuilder {
public:
    explicit ErrorBuilder(std::string op_name) : op_(std::move(op_name)) {}
    explicit ErrorBuilder(std::string_view op_name) : op_(op_name) {}
    explicit ErrorBuilder(const char* op_name) : op_(op_name) {}

    /// Throw a generic `LucidError` with `op: msg` formatting.
    [[noreturn]] void fail(const std::string& msg) const;

    /// Throw `NotImplementedError` with `op: msg`.
    [[noreturn]] void not_implemented(const std::string& msg) const;

    /// Throw `IndexError` with `op: msg`.
    [[noreturn]] void index_error(const std::string& msg) const;

    /// Throw `ShapeMismatch(expected, got, op_name)`.
    [[noreturn]] void shape_mismatch(const Shape& expected,
                                     const Shape& got,
                                     const std::string& detail = "") const;

    /// Throw `DtypeMismatch(expected, got, op_name)`.
    [[noreturn]] void dtype_mismatch(Dtype expected,
                                     Dtype got,
                                     const std::string& detail = "") const;

    /// Throw `DeviceMismatch(expected, got, op_name)`.
    [[noreturn]] void device_mismatch(Device expected,
                                      Device got,
                                      const std::string& detail = "") const;

private:
    std::string op_;
    std::string format_with_context(const std::string& msg) const;
};

}  // namespace lucid
