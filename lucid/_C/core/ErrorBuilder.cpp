// lucid/_C/core/ErrorBuilder.cpp
//
// Implementation of ErrorContext and ErrorBuilder.
//
// The call-stack state is stored in a thread_local std::vector<std::string>.
// Using thread_local avoids any locking; each thread's stack is independent.
// The stack is intentionally simple (no capacity limit) because deeply nested
// op call chains are uncommon and each name is typically a short string.

#include "ErrorBuilder.h"

#include <vector>

namespace lucid {

namespace {

// Thread-local call stack of operation names.
thread_local std::vector<std::string> tl_stack_;

}  // namespace

void ErrorContext::push(std::string op_name) {
    tl_stack_.emplace_back(std::move(op_name));
}

void ErrorContext::pop() {
    if (!tl_stack_.empty())
        tl_stack_.pop_back();
}

// Joins all entries on the stack with " > " as the separator.
// Returns an empty string when the stack is empty so that callers can
// omit the trace annotation entirely in the common no-context case.
std::string ErrorContext::trace() {
    if (tl_stack_.empty())
        return {};
    std::string out;
    for (std::size_t i = 0; i < tl_stack_.size(); ++i) {
        if (i)
            out += " > ";
        out += tl_stack_[i];
    }
    return out;
}

void ErrorContext::reset() {
    tl_stack_.clear();
}

ErrorContextGuard::ErrorContextGuard(std::string op_name) {
    ErrorContext::push(std::move(op_name));
}

ErrorContextGuard::~ErrorContextGuard() {
    ErrorContext::pop();
}

// Produces a string of the form "op_name: msg [trace=A > B > C]".
// The trace annotation is omitted when the stack is empty to keep messages
// concise in the common single-op case.
std::string ErrorBuilder::format_with_context(const std::string& msg) const {
    std::string out = op_;
    out += ": ";
    out += msg;
    const std::string trace = ErrorContext::trace();
    if (!trace.empty()) {
        out += " [trace=";
        out += trace;
        out += "]";
    }
    return out;
}

void ErrorBuilder::fail(const std::string& msg) const {
    throw LucidError(format_with_context(msg));
}

void ErrorBuilder::not_implemented(const std::string& msg) const {
    throw NotImplementedError(format_with_context(msg));
}

void ErrorBuilder::index_error(const std::string& msg) const {
    throw IndexError(format_with_context(msg));
}

// For typed mismatch throws, the context string is built separately so the
// ShapeMismatch / DtypeMismatch / DeviceMismatch constructors receive a fully
// qualified context (op + detail + trace) as their context argument.
void ErrorBuilder::shape_mismatch(const Shape& expected,
                                  const Shape& got,
                                  const std::string& detail) const {
    std::string ctx = op_;
    if (!detail.empty()) {
        ctx += ": ";
        ctx += detail;
    }
    const std::string trace = ErrorContext::trace();
    if (!trace.empty()) {
        ctx += " [trace=";
        ctx += trace;
        ctx += "]";
    }
    throw ShapeMismatch(expected, got, std::move(ctx));
}

void ErrorBuilder::dtype_mismatch(Dtype expected, Dtype got, const std::string& detail) const {
    std::string ctx = op_;
    if (!detail.empty()) {
        ctx += ": ";
        ctx += detail;
    }
    const std::string trace = ErrorContext::trace();
    if (!trace.empty()) {
        ctx += " [trace=";
        ctx += trace;
        ctx += "]";
    }
    throw DtypeMismatch(std::string(dtype_name(expected)), std::string(dtype_name(got)),
                        std::move(ctx));
}

void ErrorBuilder::device_mismatch(Device expected, Device got, const std::string& detail) const {
    std::string ctx = op_;
    if (!detail.empty()) {
        ctx += ": ";
        ctx += detail;
    }
    const std::string trace = ErrorContext::trace();
    if (!trace.empty()) {
        ctx += " [trace=";
        ctx += trace;
        ctx += "]";
    }
    throw DeviceMismatch(std::string(device_name(expected)), std::string(device_name(got)),
                         std::move(ctx));
}

}  // namespace lucid
