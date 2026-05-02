#pragma once

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
