#include "ErrorBuilder.h"

#include <vector>

namespace lucid {

// --------------------------------------------------------------------------- //
// ErrorContext — thread-local op-name stack.
// --------------------------------------------------------------------------- //

namespace {

thread_local std::vector<std::string> tl_stack_;

}  // namespace

void ErrorContext::push(std::string op_name) {
    tl_stack_.emplace_back(std::move(op_name));
}

void ErrorContext::pop() {
    if (!tl_stack_.empty())
        tl_stack_.pop_back();
}

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

// --------------------------------------------------------------------------- //
// ErrorContextGuard — RAII push/pop helper.
// --------------------------------------------------------------------------- //

ErrorContextGuard::ErrorContextGuard(std::string op_name) {
    ErrorContext::push(std::move(op_name));
}

ErrorContextGuard::~ErrorContextGuard() {
    ErrorContext::pop();
}

// --------------------------------------------------------------------------- //
// ErrorBuilder — chainable typed-throw helpers.
// --------------------------------------------------------------------------- //

std::string ErrorBuilder::format_with_context(const std::string& msg) const {
    // "op: msg [trace=outer > inner]" when a trace exists, otherwise "op: msg".
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
