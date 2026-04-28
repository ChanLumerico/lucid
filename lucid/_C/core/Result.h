#pragma once

// =====================================================================
// Lucid C++ engine — Result<T> primitive for non-throwing fast paths.
// =====================================================================
//
// Most engine errors are thrown as `LucidError` subclasses (see
// `Error.h`) and translated to Python exceptions at the binding
// layer. That's correct for the slow path: a malformed dtype, a
// shape mismatch, an OOM — these are rare and the throw overhead is
// invisible.
//
// `Result<T>` exists for the *hot* paths that may legitimately fail
// at high frequency: kernel dispatch fall-through, optional
// fast-path attempts, validation of internal contracts where the
// caller has a meaningful fallback. There the cost of constructing
// + unwinding an exception is observable, and a plain
// std::variant<T, ErrorPayload> is preferable.
//
// Design:
//   - `Result<T>` is a thin wrapper over `std::variant<T, ErrorPayload>`.
//   - `ErrorPayload` carries an error code + message (no shape /
//     dtype payload — those go through typed exceptions on the
//     slow path).
//   - Helpers `Ok(...)` and `Err(...)` keep call sites readable.
//   - `expect()` / `value_or_throw()` bridge back into the
//     exception world when the caller wants the typed throw.
//
// Layer: core/. Header-only. No deps beyond std.

#include <string>
#include <utility>
#include <variant>

#include "Error.h"
#include "ErrorBuilder.h"

namespace lucid {

enum class ErrorCode : int {
    Ok = 0,
    InvalidArgument,
    ShapeMismatch,
    DtypeMismatch,
    DeviceMismatch,
    NotImplemented,
    IndexOutOfRange,
    Internal,
};

struct ErrorPayload {
    ErrorCode code = ErrorCode::Internal;
    std::string msg;
};

template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(ErrorPayload err) : data_(std::move(err)) {}

    bool is_ok() const noexcept { return data_.index() == 0; }
    bool is_err() const noexcept { return data_.index() == 1; }
    explicit operator bool() const noexcept { return is_ok(); }

    T& value() & { return std::get<0>(data_); }
    const T& value() const& { return std::get<0>(data_); }
    T&& value() && { return std::get<0>(std::move(data_)); }

    const ErrorPayload& error() const& { return std::get<1>(data_); }
    ErrorPayload&& error() && { return std::get<1>(std::move(data_)); }

    /// Bridge to the typed-exception world: returns the value or
    /// throws `LucidError(error.msg)`. Callers that want a more
    /// specific typed exception should branch on `error().code`.
    T value_or_throw() && {
        if (is_ok())
            return std::get<0>(std::move(data_));
        ErrorBuilder("Result").fail(std::move(std::get<1>(data_).msg));
    }

private:
    std::variant<T, ErrorPayload> data_;
};

template <typename T>
inline Result<T> Ok(T value) {
    return Result<T>(std::move(value));
}

inline ErrorPayload Err(ErrorCode code, std::string msg) {
    return ErrorPayload{code, std::move(msg)};
}

}  // namespace lucid
