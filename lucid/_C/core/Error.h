// lucid/_C/core/Error.h
//
// Lucid exception hierarchy.  All engine exceptions derive from LucidError,
// which itself derives from std::exception so that callers using a generic
// catch(std::exception&) still see a meaningful message.
//
// Exception types are purposely granular so that Python bindings can map them
// to distinct Python exception types (e.g. ShapeMismatch → ValueError) and
// so that C++ callers can selectively catch specific error conditions without
// relying on string-matching what().
//
// Error messages are formatted at construction time and stored in msg_; what()
// is therefore allocation-free at catch sites.

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "../api.h"

namespace lucid {

// Base class for every typed exception thrown by the Lucid engine.
//
// Derives from ``std::exception`` so generic handlers
// (``catch (const std::exception&)``) still observe the formatted message via
// :cpp:func:`what`.  Concrete subclasses are expected to populate the
// inherited ``msg_`` field after their member fields are initialised, allowing
// the formatted text to reference moved-from constructor arguments through
// member accessors.
//
// Pybind11 maps this class to ``lucid._C.engine.LucidError``, which is a
// Python subclass of ``RuntimeError``.  All derived exception classes are
// further re-exported as Python subclasses of ``LucidError`` by
// ``bind_errors.cpp``.
//
// Parameters
// ----------
// msg : std::string
//     Fully formatted human-readable message.  Stored verbatim; no further
//     prefix is applied.
//
// Attributes
// ----------
// msg_ : std::string
//     Owning storage for the formatted ``what()`` payload.  Protected so
//     subclasses can overwrite it after their members are initialised.
//
// Examples
// --------
// ```
// throw LucidError("unexpected null tensor in saved-graph slot");
// ```
//
// See Also
// --------
// :class:`ErrorBuilder` : Fluent helper that throws typed subclasses with a
//     consistent ``"op: msg [trace=...]"`` format.
class LUCID_API LucidError : public std::exception {
public:
    // Constructs the base exception with a pre-formatted message.
    //
    // Parameters
    // ----------
    // msg : std::string
    //     Final text returned by :cpp:func:`what`.  Move-stored into
    //     ``msg_``.
    explicit LucidError(std::string msg) : msg_(std::move(msg)) {}

    // Returns a pointer to the stored message.
    //
    // Returns
    // -------
    // const char*
    //     Null-terminated string owned by this exception.  Lifetime is tied
    //     to ``*this``; do not retain beyond the catch site.
    //
    // Notes
    // -----
    // Allocation-free: the string was built at construction time.
    const char* what() const noexcept override { return msg_.c_str(); }

protected:
    std::string msg_;
};

// Thrown when an allocation request cannot be satisfied by the device
// allocator.
//
// The exception captures the failed request size together with the current
// and peak usage counters tracked by ``MemoryTracker`` for the offending
// device, so callers (or end-users reading the Python traceback) can
// distinguish OOM caused by absolute size from OOM caused by fragmentation.
// Mapped to ``lucid._C.engine.OutOfMemory`` on the Python side.
//
// Parameters
// ----------
// requested_bytes : std::size_t
//     Number of bytes the allocator was asked to provide.
// current_bytes : std::size_t
//     Live (in-use) bytes on the device immediately before the failed
//     request.
// peak_bytes : std::size_t
//     High-water-mark of bytes ever observed in-use on the device since the
//     tracker was last reset.
// device : std::string
//     Human-readable device label (e.g. ``"cpu"``, ``"metal:0"``).
//
// Attributes
// ----------
// requested_bytes_ : std::size_t
//     Stored copy of the failed request size.
// current_bytes_ : std::size_t
//     Stored copy of the live byte count.
// peak_bytes_ : std::size_t
//     Stored copy of the peak byte count.
// device_ : std::string
//     Stored copy of the device label.
//
// Examples
// --------
// ```
// throw OutOfMemory(1ull << 30, current, peak, "metal:0");
// ```
class OutOfMemory : public LucidError {
public:
    // Construct an ``OutOfMemory`` error from the failed allocation size, the
    // tracker's current/peak usage at the moment of failure, and the offending
    // device label.  All four arguments are formatted into the human-readable
    // ``what()`` message so end-users can distinguish absolute-size OOM from
    // fragmentation-induced OOM.
    OutOfMemory(std::size_t requested_bytes,
                std::size_t current_bytes,
                std::size_t peak_bytes,
                std::string device);

    // Returns the number of bytes that the failed allocation requested.
    //
    // Returns
    // -------
    // std::size_t
    //     The original ``requested_bytes`` argument.
    std::size_t requested_bytes() const { return requested_bytes_; }

    // Returns the live byte count at the time of failure.
    //
    // Returns
    // -------
    // std::size_t
    //     Bytes already in-use on the device when the allocator failed.
    std::size_t current_bytes() const { return current_bytes_; }

    // Returns the peak byte count observed on the device.
    //
    // Returns
    // -------
    // std::size_t
    //     The high-water mark recorded by the memory tracker.
    std::size_t peak_bytes() const { return peak_bytes_; }

    // Returns the device label where the allocation was attempted.
    //
    // Returns
    // -------
    // const std::string&
    //     Reference to the stored device string.  Lifetime is tied to
    //     ``*this``.
    const std::string& device() const { return device_; }

private:
    std::size_t requested_bytes_;
    std::size_t current_bytes_;
    std::size_t peak_bytes_;
    std::string device_;
};

// Thrown when two tensors have shapes that are incompatible with an
// operation's broadcasting or contraction rules.
//
// The expected and actual shapes are retained as integer vectors so callers
// can inspect them programmatically rather than parsing ``what()``.  The
// message string formatted by the constructor includes both shapes plus the
// ``context`` annotation supplied by :class:`ErrorBuilder`.  Mapped to
// ``lucid._C.engine.ShapeMismatch`` on the Python side.
//
// Parameters
// ----------
// expected : std::vector<std::int64_t>
//     Shape the operation required, in row-major dimension order.
// got : std::vector<std::int64_t>
//     Shape that was actually supplied.
// context : std::string
//     Free-form annotation produced by :class:`ErrorBuilder` (typically the
//     op name plus the optional ``detail`` argument and any thread-local
//     trace).
//
// Attributes
// ----------
// expected_ : std::vector<std::int64_t>
//     Stored copy of the required shape.
// got_ : std::vector<std::int64_t>
//     Stored copy of the offending shape.
//
// Examples
// --------
// ```
// ErrorBuilder("matmul").shape_mismatch(expected, got, "rhs");
// ```
//
// See Also
// --------
// :class:`Validator` : Wraps the common precondition checks that raise this
//     exception via :class:`ErrorBuilder`.
class ShapeMismatch : public LucidError {
public:
    // Construct a ``ShapeMismatch`` with the expected vs actual shape and the
    // op-level context annotation the call site captured.  All three become
    // part of the formatted ``what()`` message.
    ShapeMismatch(std::vector<std::int64_t> expected,
                  std::vector<std::int64_t> got,
                  std::string context);

    // Returns the shape the operation required.
    //
    // Returns
    // -------
    // const std::vector<std::int64_t>&
    //     Stored ``expected`` argument; lifetime tied to ``*this``.
    const std::vector<std::int64_t>& expected() const { return expected_; }

    // Returns the shape that was actually supplied.
    //
    // Returns
    // -------
    // const std::vector<std::int64_t>&
    //     Stored ``got`` argument; lifetime tied to ``*this``.
    const std::vector<std::int64_t>& got() const { return got_; }

private:
    std::vector<std::int64_t> expected_;
    std::vector<std::int64_t> got_;
};

// Thrown when a dtype mismatch prevents an operation from executing.
//
// Both ``expected`` and ``got`` are stored as human-readable dtype names
// (e.g. ``"float32"``, ``"int8"``) rather than as raw enumerators so the
// exception can be formatted directly into ``what()`` without depending on
// the :cpp:enum:`Dtype` header.  Mapped to ``lucid._C.engine.DtypeMismatch``
// on the Python side.
//
// Parameters
// ----------
// expected : std::string
//     Required dtype name (or a ``"|"``-separated set when multiple dtypes
//     are accepted, as emitted by :cpp:func:`Validator::dtype_in`).
// got : std::string
//     Dtype name that was actually supplied.
// context : std::string
//     Annotation produced by :class:`ErrorBuilder` (op name and trace).
//
// Attributes
// ----------
// expected_ : std::string
//     Stored copy of the required dtype name.
// got_ : std::string
//     Stored copy of the offending dtype name.
//
// Examples
// --------
// ```
// ErrorBuilder("linear").dtype_mismatch(Dtype::F32, t.dtype(), "weight");
// ```
class DtypeMismatch : public LucidError {
public:
    // Construct a ``DtypeMismatch`` with the expected vs actual dtype name and
    // the op-level context annotation the call site captured.  Stored verbatim
    // in ``what()`` so callers see ``expected float32, got int8`` style text.
    DtypeMismatch(std::string expected, std::string got, std::string context);

    // Returns the dtype name that the operation required.
    //
    // Returns
    // -------
    // const std::string&
    //     Stored ``expected`` argument; lifetime tied to ``*this``.
    const std::string& expected() const { return expected_; }

    // Returns the dtype name that was actually supplied.
    //
    // Returns
    // -------
    // const std::string&
    //     Stored ``got`` argument; lifetime tied to ``*this``.
    const std::string& got() const { return got_; }

private:
    std::string expected_;
    std::string got_;
};

// Thrown when tensors from different devices are combined incorrectly.
//
// Raised, for example, when a CPU tensor is passed to an op that requires a
// GPU tensor (or vice-versa), or when two operands disagree on their device
// placement.  Mapped to ``lucid._C.engine.DeviceMismatch`` on the Python
// side.
//
// Parameters
// ----------
// expected : std::string
//     Device name the operation required (e.g. ``"cpu"`` or ``"metal:0"``).
// got : std::string
//     Device name of the offending tensor.
// context : std::string
//     Annotation produced by :class:`ErrorBuilder`.
//
// Attributes
// ----------
// expected_ : std::string
//     Stored copy of the required device label.
// got_ : std::string
//     Stored copy of the offending device label.
//
// Examples
// --------
// ```
// ErrorBuilder("matmul").device_mismatch(a.device(), b.device());
// ```
class DeviceMismatch : public LucidError {
public:
    // Construct a ``DeviceMismatch`` with the expected vs actual device label
    // and the op-level context annotation the call site captured.  All three
    // are formatted into the ``what()`` message.
    DeviceMismatch(std::string expected, std::string got, std::string context);

    // Returns the device name the operation required.
    //
    // Returns
    // -------
    // const std::string&
    //     Stored ``expected`` argument; lifetime tied to ``*this``.
    const std::string& expected() const { return expected_; }

    // Returns the device name that was actually supplied.
    //
    // Returns
    // -------
    // const std::string&
    //     Stored ``got`` argument; lifetime tied to ``*this``.
    const std::string& got() const { return got_; }

private:
    std::string expected_;
    std::string got_;
};

// Thrown by the autograd engine when a saved tensor is mutated between the
// forward and backward pass.
//
// Each tensor carries a monotonically increasing version counter; the saved
// snapshot records the version observed at save time.  When the snapshot is
// restored during backward, the engine compares the saved version against
// the tensor's current version and raises :class:`VersionMismatch` if they
// differ — preventing silently incorrect gradients caused by in-place
// mutation.  Mapped to ``lucid._C.engine.VersionMismatch`` on the Python
// side.
//
// Parameters
// ----------
// expected : std::int64_t
//     Version that was captured during the forward pass.
// got : std::int64_t
//     Version observed during the backward pass.
// context : std::string
//     Annotation produced by :class:`ErrorBuilder` (typically the saving
//     op's name).
//
// Attributes
// ----------
// expected_ : std::int64_t
//     Stored snapshot version.
// got_ : std::int64_t
//     Stored current version.
//
// Notes
// -----
// In-place ops on tensors that participate in autograd should either bump
// the version counter on the underlying storage or be re-implemented as a
// non-aliasing copy.  Catching :class:`VersionMismatch` and continuing is
// strongly discouraged — the gradient is no longer well-defined.
class VersionMismatch : public LucidError {
public:
    // Construct a ``VersionMismatch`` with the version captured at forward
    // time vs the version observed at backward time, plus the op-level context
    // annotation the call site captured.  Raised when a saved tensor was
    // mutated in-place between forward and backward.
    VersionMismatch(std::int64_t expected, std::int64_t got, std::string context);

    // Returns the version observed when the tensor was saved.
    //
    // Returns
    // -------
    // std::int64_t
    //     Forward-pass version captured at save time.
    std::int64_t expected_version() const { return expected_; }

    // Returns the version observed during the backward pass.
    //
    // Returns
    // -------
    // std::int64_t
    //     Current version of the tensor when the saved snapshot was
    //     restored.
    std::int64_t got_version() const { return got_; }

private:
    std::int64_t expected_;
    std::int64_t got_;
};

// Thrown when an operation requires a GPU but the Metal backend is
// unavailable.
//
// Sources include: a build with the GPU backend compiled out, a host with no
// Metal-capable device, or a Metal command-queue initialisation failure at
// runtime.  Mapped to ``lucid._C.engine.GpuNotAvailable`` on the Python
// side.
//
// Parameters
// ----------
// reason : std::string
//     Short human-readable explanation appended to the ``"GpuNotAvailable: "``
//     prefix when building the final message.
//
// Examples
// --------
// ```
// throw GpuNotAvailable("MLX failed to acquire default Metal device");
// ```
class GpuNotAvailable : public LucidError {
public:
    // Construct a ``GpuNotAvailable`` error from a short human-readable reason
    // the call site captured (e.g. ``"MLX failed to acquire default Metal
    // device"``).  Prefixed with ``"GpuNotAvailable: "`` when formatted into
    // ``what()``.
    explicit GpuNotAvailable(std::string reason);
};

// Thrown by indexing and slicing operations when an index falls outside the
// valid range of its dimension.
//
// Mapped to ``lucid._C.engine.IndexError`` on the Python side, which itself
// inherits from ``LucidError`` (not from the built-in Python
// ``IndexError``).  Callers wanting to catch both Lucid and native index
// errors should match on ``LookupError`` or both classes explicitly.
//
// Parameters
// ----------
// msg : std::string
//     Fully formatted message; passed through unchanged to the
//     :class:`LucidError` base.
class IndexError : public LucidError {
public:
    // Construct an ``IndexError`` from the pre-formatted message the call
    // site built describing the offending index vs the valid range.  Passed
    // through unchanged to the :class:`LucidError` base.
    explicit IndexError(std::string msg) : LucidError(std::move(msg)) {}
};

// Thrown when an op recognises its inputs but has no implementation for the
// requested dtype, device, or configuration.
//
// Distinct from a user-error such as :class:`ShapeMismatch` — the caller did
// nothing wrong, the engine simply has no kernel for this combination
// (e.g. ``conv3d`` on integer dtypes, or an autograd op that has not yet
// shipped a backward formula).  Mapped to
// ``lucid._C.engine.NotImplementedError`` on the Python side.
//
// Parameters
// ----------
// msg : std::string
//     Fully formatted message; passed through unchanged to the
//     :class:`LucidError` base.
//
// Examples
// --------
// ```
// ErrorBuilder("conv3d").not_implemented("int8 not supported on GPU");
// ```
class NotImplementedError : public LucidError {
public:
    // Construct a ``NotImplementedError`` from the pre-formatted message the
    // call site built describing the unsupported dtype / device / config
    // combination.  Passed through unchanged to the :class:`LucidError` base.
    explicit NotImplementedError(std::string msg) : LucidError(std::move(msg)) {}
};

}  // namespace lucid
