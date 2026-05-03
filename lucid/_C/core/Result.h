// lucid/_C/core/Result.h
//
// Lightweight expected-value type — analogous to std::expected (C++23) or
// Rust's Result<T, E>.  Used by subsystems that need to propagate errors
// without unwinding the stack (e.g. FFI boundaries, async callbacks).
//
// A Result<T> holds either a value of type T (success) or an ErrorPayload
// (failure).  The two cases are stored in a std::variant so there is no extra
// heap allocation.  Callers can inspect is_ok() / is_err(), access the value
// directly (value()), or call value_or_throw() to convert the error case into
// a thrown LucidError.
//
// The Ok() and Err() factory functions produce the appropriate type without
// requiring the caller to spell out the template parameter.

#pragma once

#include <string>
#include <utility>
#include <variant>

#include "Error.h"
#include "ErrorBuilder.h"

namespace lucid {

// Identifies the category of a propagated error without requiring a live
// exception object.  Mirrors the exception hierarchy in Error.h.
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

// POD error descriptor carried by the error arm of Result<T>.
struct ErrorPayload {
    ErrorCode code = ErrorCode::Internal;
    std::string msg;
};

// Value-or-error container.
//
// Invariants:
//   - is_ok()  ⟺  data_ holds a T  (variant index 0)
//   - is_err() ⟺  data_ holds an ErrorPayload (variant index 1)
//
// Thread safety: Result<T> instances are not shared across threads; there is
// no internal synchronisation.
template <typename T>
class Result {
public:
    // Constructs a successful result wrapping value.
    Result(T value) : data_(std::move(value)) {}
    // Constructs a failed result wrapping err.
    Result(ErrorPayload err) : data_(std::move(err)) {}

    bool is_ok() const noexcept { return data_.index() == 0; }
    bool is_err() const noexcept { return data_.index() == 1; }
    // Allows if (result) { ... } as a convenient success test.
    explicit operator bool() const noexcept { return is_ok(); }

    // Access the stored value.  Undefined behaviour (std::bad_variant_access
    // in debug builds) if is_err().
    T& value() & { return std::get<0>(data_); }
    const T& value() const& { return std::get<0>(data_); }
    T&& value() && { return std::get<0>(std::move(data_)); }

    const ErrorPayload& error() const& { return std::get<1>(data_); }
    ErrorPayload&& error() && { return std::get<1>(std::move(data_)); }

    // Returns the stored value, or throws a LucidError with the payload's
    // message if is_err().  Consuming (rvalue overload) to avoid a copy of T.
    T value_or_throw() && {
        if (is_ok())
            return std::get<0>(std::move(data_));
        ErrorBuilder("Result").fail(std::move(std::get<1>(data_).msg));
    }

private:
    std::variant<T, ErrorPayload> data_;
};

// Convenience factory — constructs a successful Result<T> from value.
template <typename T>
inline Result<T> Ok(T value) {
    return Result<T>(std::move(value));
}

// Convenience factory — constructs an ErrorPayload.  The returned payload can
// be implicitly converted to Result<T> for any T.
inline ErrorPayload Err(ErrorCode code, std::string msg) {
    return ErrorPayload{code, std::move(msg)};
}

}  // namespace lucid
