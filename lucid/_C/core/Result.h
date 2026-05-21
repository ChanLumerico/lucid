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
// exception object.
//
// Each enumerator mirrors a class in the :class:`LucidError` hierarchy and
// is carried inside :cpp:struct:`ErrorPayload` so a :cpp:class:`Result` can
// be inspected (and acted on) without having to rethrow.  Use this when
// you want to handle failures across an FFI / async boundary where C++
// exceptions cannot propagate.
//
// Attributes
// ----------
// Ok : ErrorCode
//     Sentinel for the success arm.  Not normally set on an
//     :cpp:struct:`ErrorPayload` (which only exists for the error arm) but
//     reserved as the zero value so future ``Result<void>``-style
//     specialisations can use it.
// InvalidArgument : ErrorCode
//     Caller passed an argument that violates a documented precondition
//     not covered by the other categories.
// ShapeMismatch : ErrorCode
//     Mirrors :class:`ShapeMismatch`.
// DtypeMismatch : ErrorCode
//     Mirrors :class:`DtypeMismatch`.
// DeviceMismatch : ErrorCode
//     Mirrors :class:`DeviceMismatch`.
// NotImplemented : ErrorCode
//     Mirrors :class:`NotImplementedError`.
// IndexOutOfRange : ErrorCode
//     Mirrors :class:`IndexError`.
// Internal : ErrorCode
//     Catch-all for engine-internal failures that do not fit a more
//     specific category.  Default value of :cpp:struct:`ErrorPayload`.
//
// See Also
// --------
// :cpp:struct:`ErrorPayload` : POD that bundles an ``ErrorCode`` with a
//     human-readable message.
// :class:`LucidError` : Live-exception counterpart.
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

// Plain-old-data error descriptor carried by the error arm of
// :cpp:class:`Result`.
//
// Designed to be trivially copyable so a ``Result`` can be moved across
// async / FFI boundaries without pulling in the C++ exception machinery.
// The message is a fully-formatted human-readable string — for typed
// inspection (shapes, dtypes, ...) the caller should re-throw via
// :cpp:func:`Result::value_or_throw` and catch the resulting typed
// :class:`LucidError`.
//
// Attributes
// ----------
// code : ErrorCode
//     Category of the failure.  Defaults to ``ErrorCode::Internal`` so a
//     default-constructed payload is always a recognisable error rather
//     than a misleading ``Ok``.
// msg : std::string
//     Final ``what()``-style text.  May be empty when the category alone
//     conveys the information.
//
// Examples
// --------
// ```
// return Err(ErrorCode::ShapeMismatch, "rhs has incompatible inner dim");
// ```
struct ErrorPayload {
    ErrorCode code = ErrorCode::Internal;
    std::string msg;
};

// Value-or-error container that holds either a ``T`` (success) or an
// :cpp:struct:`ErrorPayload` (failure) in a single ``std::variant``.
//
// Use ``Result<T>`` instead of an exception when a failure must be
// propagated across a boundary that cannot safely unwind C++ exceptions:
// the pybind11 FFI surface (when ``noexcept`` is required), async callbacks
// dispatched onto a worker queue, or any C ABI shim.  Inside the engine
// itself, ops typically throw a :class:`LucidError` directly because the
// stack-unwinding cost is acceptable and the typed exception carries
// richer state.
//
// Invariants
// ----------
// - ``is_ok()``  ⟺  ``data_`` holds a ``T`` (variant index 0).
// - ``is_err()`` ⟺  ``data_`` holds an :cpp:struct:`ErrorPayload`
//   (variant index 1).
//
// Thread safety
// -------------
// ``Result<T>`` instances are not internally synchronised.  Sharing a
// single instance across threads requires external coordination.
//
// Parameters
// ----------
// T : typename
//     Type carried by the success arm.  Must be movable; otherwise no
//     constraints.
//
// See Also
// --------
// :cpp:func:`Ok` : Factory that wraps a value as a successful ``Result``.
// :cpp:func:`Err` : Factory that builds an :cpp:struct:`ErrorPayload`,
//     implicitly convertible to ``Result<T>`` for any ``T``.
template <typename T>
class Result {
public:
    // Constructs a successful result wrapping the given value.
    //
    // Parameters
    // ----------
    // value : T
    //     Payload moved into the success arm.
    Result(T value) : data_(std::move(value)) {}

    // Constructs a failed result wrapping the given error payload.
    //
    // Parameters
    // ----------
    // err : ErrorPayload
    //     Error descriptor moved into the error arm.
    Result(ErrorPayload err) : data_(std::move(err)) {}

    // Returns whether this result holds a value.
    //
    // Returns
    // -------
    // bool
    //     ``true`` if the success arm is active, ``false`` otherwise.
    bool is_ok() const noexcept { return data_.index() == 0; }

    // Returns whether this result holds an error.
    //
    // Returns
    // -------
    // bool
    //     ``true`` if the error arm is active, ``false`` otherwise.
    bool is_err() const noexcept { return data_.index() == 1; }

    // Allows the result to be used as a boolean test for success.
    //
    // Returns
    // -------
    // bool
    //     Same as :cpp:func:`is_ok`.
    //
    // Examples
    // --------
    // ```
    // if (auto r = compute(); r) { use(r.value()); }
    // ```
    explicit operator bool() const noexcept { return is_ok(); }

    // Accesses the stored value (lvalue overload).
    //
    // Returns
    // -------
    // T&
    //     Reference to the success payload.  Lifetime is tied to ``*this``.
    //
    // Raises
    // ------
    // std::bad_variant_access
    //     If :cpp:func:`is_err` is true.  In release builds this is
    //     equivalent to undefined behaviour — call :cpp:func:`is_ok` first.
    T& value() & { return std::get<0>(data_); }

    // Accesses the stored value (const lvalue overload).
    //
    // Returns
    // -------
    // const T&
    //     Const reference to the success payload.
    //
    // Raises
    // ------
    // std::bad_variant_access
    //     If :cpp:func:`is_err` is true.
    const T& value() const& { return std::get<0>(data_); }

    // Accesses the stored value (rvalue overload, consuming).
    //
    // Returns
    // -------
    // T&&
    //     Rvalue reference to the success payload, suitable for moving the
    //     value out of an expiring ``Result``.
    //
    // Raises
    // ------
    // std::bad_variant_access
    //     If :cpp:func:`is_err` is true.
    T&& value() && { return std::get<0>(std::move(data_)); }

    // Accesses the stored error payload (const lvalue overload).
    //
    // Returns
    // -------
    // const ErrorPayload&
    //     Const reference to the error descriptor.  Lifetime is tied to
    //     ``*this``.
    //
    // Raises
    // ------
    // std::bad_variant_access
    //     If :cpp:func:`is_ok` is true.
    const ErrorPayload& error() const& { return std::get<1>(data_); }

    // Accesses the stored error payload (rvalue overload, consuming).
    //
    // Returns
    // -------
    // ErrorPayload&&
    //     Rvalue reference suitable for moving the payload out of an
    //     expiring ``Result``.
    //
    // Raises
    // ------
    // std::bad_variant_access
    //     If :cpp:func:`is_ok` is true.
    ErrorPayload&& error() && { return std::get<1>(std::move(data_)); }

    // Returns the stored value or throws the error as a :class:`LucidError`.
    //
    // The rvalue-ref qualifier means this method consumes the ``Result`` —
    // callers should write ``std::move(r).value_or_throw()`` (or call it
    // on a prvalue) to avoid an unnecessary copy of ``T``.
    //
    // Returns
    // -------
    // T
    //     The success payload, moved out of the variant.
    //
    // Raises
    // ------
    // LucidError
    //     If the error arm is active.  The payload's ``msg`` is forwarded
    //     to :cpp:func:`ErrorBuilder::fail`, which prepends ``"Result: "``
    //     and the current :class:`ErrorContext` trace.
    //
    // Examples
    // --------
    // ```
    // T x = std::move(compute()).value_or_throw();
    // ```
    T value_or_throw() && {
        if (is_ok())
            return std::get<0>(std::move(data_));
        ErrorBuilder("Result").fail(std::move(std::get<1>(data_).msg));
    }

private:
    std::variant<T, ErrorPayload> data_;
};

// Builds a successful :cpp:class:`Result` from the given value.
//
// Parameters
// ----------
// value : T
//     Payload to wrap.  Moved into the success arm.
//
// Returns
// -------
// Result<T>
//     A result for which :cpp:func:`Result::is_ok` is ``true``.
//
// Examples
// --------
// ```
// Result<int> r = Ok(42);
// ```
template <typename T>
inline Result<T> Ok(T value) {
    return Result<T>(std::move(value));
}

// Builds an :cpp:struct:`ErrorPayload` from a code and message.
//
// The returned payload is implicitly convertible to ``Result<T>`` for any
// ``T``, so a ``return Err(code, msg);`` inside a function returning
// ``Result<Something>`` Just Works.
//
// Parameters
// ----------
// code : ErrorCode
//     Category of the failure.
// msg : std::string
//     Human-readable message moved into ``ErrorPayload::msg``.
//
// Returns
// -------
// ErrorPayload
//     Error descriptor ready to be returned as the failure arm of a
//     ``Result``.
//
// Examples
// --------
// ```
// return Err(ErrorCode::IndexOutOfRange, "i must be in [0, n)");
// ```
inline ErrorPayload Err(ErrorCode code, std::string msg) {
    return ErrorPayload{code, std::move(msg)};
}

}  // namespace lucid
