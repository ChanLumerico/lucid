// lucid/_C/core/Validate.h
//
// Fluent validation API for TensorImpl inputs.  Op implementations call
// Validator::input() or Validator::pair() to build a chain of precondition
// checks; each check throws the appropriate typed exception on failure and
// returns *this on success so checks can be chained.
//
// Example (single tensor):
//   Validator::input(t, "linear.weight")
//       .non_null()
//       .float_only()
//       .ndim(2);
//
// Example (two tensors):
//   Validator::pair(a, b, "matmul")
//       .both_non_null()
//       .same_dtype()
//       .same_device();
//
// The label / op_name argument is passed to ErrorBuilder so that exception
// messages identify the problematic tensor by name.

#pragma once

#include <initializer_list>
#include <string>

#include "../api.h"
#include "Dtype.h"
#include "Error.h"
#include "Shape.h"
#include "TensorImpl.h"
#include "fwd.h"

namespace lucid {

// Fluent precondition checker for a single tensor input.
//
// Each method either evaluates one precondition and returns ``*this``
// (success), or throws the appropriate typed :class:`LucidError` subclass
// via :class:`ErrorBuilder` (failure).  Chaining lets op implementations
// state all of their input requirements in one compact expression at the
// top of their body.
//
// Instances cannot be constructed directly — use the static
// :cpp:func:`input` factory.  The validator holds a *reference* to the
// caller's ``TensorImplPtr`` rather than a copy, so the chain must live no
// longer than the tensor it inspects (this is the normal case when the
// chain appears on a single statement at the op's entry point).
//
// Attributes
// ----------
// t_ : const TensorImplPtr&
//     Reference to the tensor being checked.  Lifetime is the caller's
//     responsibility.
// label_ : std::string
//     Human-readable identifier used as the op-name argument to
//     :class:`ErrorBuilder` so error messages name the offending tensor
//     (e.g. ``"linear.weight"``).
//
// Examples
// --------
// ```
// Validator::input(t, "linear.weight")
//     .non_null()
//     .float_only()
//     .ndim(2);
// ```
//
// See Also
// --------
// :class:`Validator::Pair` : Two-tensor sibling for cross-tensor checks.
// :class:`ErrorBuilder` : Underlying throw machinery.
class LUCID_API Validator {
public:
    // Builds a single-tensor validator.
    //
    // Parameters
    // ----------
    // t : const TensorImplPtr&
    //     Tensor to be validated.  The validator retains a reference, not a
    //     copy — ``t`` must remain alive for the duration of the chain.
    // label : std::string
    //     Identifier used in error messages (typically the parameter name,
    //     e.g. ``"weight"``).
    //
    // Returns
    // -------
    // Validator
    //     A new validator bound to ``t``.
    static Validator input(const TensorImplPtr& t, std::string label);

    // Asserts that the tensor pointer is not null.
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``, to allow chaining.
    //
    // Raises
    // ------
    // LucidError
    //     If ``t_`` is null.
    Validator& non_null();

    // Asserts that the dtype is either :cpp:enumerator:`Dtype::F32` or
    // :cpp:enumerator:`Dtype::F64`.
    //
    // Implicitly calls :cpp:func:`non_null` first.
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``.
    //
    // Raises
    // ------
    // LucidError
    //     If the tensor is null.
    // NotImplementedError
    //     If the dtype is anything other than ``F32`` or ``F64``.
    Validator& float_only();

    // Asserts that the dtype equals ``expected``.
    //
    // Implicitly calls :cpp:func:`non_null` first.
    //
    // Parameters
    // ----------
    // expected : Dtype
    //     Required dtype.
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``.
    //
    // Raises
    // ------
    // LucidError
    //     If the tensor is null.
    // DtypeMismatch
    //     If ``t_->dtype() != expected``.
    Validator& dtype_eq(Dtype expected);

    // Asserts that the dtype is one of ``allowed``.
    //
    // Implicitly calls :cpp:func:`non_null` first.  The error message lists
    // the allowed dtypes joined by ``"|"`` (e.g. ``"float32|float64"``).
    //
    // Parameters
    // ----------
    // allowed : std::initializer_list<Dtype>
    //     Set of acceptable dtypes.  Iterated linearly, so callers should
    //     keep the list short (the common case is 2–4 entries).
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``.
    //
    // Raises
    // ------
    // LucidError
    //     If the tensor is null.
    // DtypeMismatch
    //     If no entry in ``allowed`` matches ``t_->dtype()``.
    Validator& dtype_in(std::initializer_list<Dtype> allowed);

    // Asserts that the tensor has exactly ``expected`` dimensions.
    //
    // Implicitly calls :cpp:func:`non_null` first.
    //
    // Parameters
    // ----------
    // expected : int
    //     Required number of dimensions.
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``.
    //
    // Raises
    // ------
    // LucidError
    //     If the tensor is null.
    // ShapeMismatch
    //     If its rank differs from ``expected`` (see :cpp:func:`require_rank`).
    Validator& ndim(int expected);

    // Asserts that the tensor has at least ``min_n`` dimensions.
    //
    // Implicitly calls :cpp:func:`non_null` first.
    //
    // Parameters
    // ----------
    // min_n : int
    //     Lower bound (inclusive) on the tensor's rank.
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``.
    //
    // Raises
    // ------
    // LucidError
    //     If the tensor is null.
    // ShapeMismatch
    //     If its rank is below ``min_n`` (see :cpp:func:`require_min_rank`).
    Validator& ndim_at_least(int min_n);

    // Asserts that the tensor's shape equals ``expected``.
    //
    // Implicitly calls :cpp:func:`non_null` first.
    //
    // Parameters
    // ----------
    // expected : const Shape&
    //     Required shape.
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``.
    //
    // Raises
    // ------
    // LucidError
    //     If the tensor is null.
    // ShapeMismatch
    //     If ``t_->shape() != expected``.
    Validator& shape_eq(const Shape& expected);

    // Asserts that the tensor is at least 2-D and its trailing two
    // dimensions are equal.
    //
    // Used by linalg ops that require square matrices or batches thereof —
    // e.g. a shape ``[B, N, N]`` passes while ``[B, N, M]`` with
    // ``N != M`` fails.  Implicitly calls :cpp:func:`non_null` first.
    //
    // Shape
    // -----
    // Accepted: any ``[*, N, N]``.
    //
    // Returns
    // -------
    // Validator&
    //     ``*this``.
    //
    // Raises
    // ------
    // InvalidArgument
    //     If the tensor has rank < 2.
    // ShapeMismatch
    //     If its last two dims disagree (see :cpp:func:`require_square_matrix`).
    Validator& square_2d();

    // Fluent precondition checker for a pair of tensors.
    //
    // Mirrors :class:`Validator` but for checks that compare two operands
    // (same dtype, same device, same shape, etc.).  Constructed via the
    // static :cpp:func:`Validator::pair` factory — direct construction
    // exists only so the factory can return by value.
    //
    // Attributes
    // ----------
    // a_ : const TensorImplPtr&
    //     Reference to the first operand.
    // b_ : const TensorImplPtr&
    //     Reference to the second operand.
    // op_ : std::string
    //     Op name used in error messages (e.g. ``"matmul"``).
    //
    // Examples
    // --------
    // ```
    // Validator::pair(a, b, "matmul")
    //     .both_non_null()
    //     .same_dtype()
    //     .same_device();
    // ```
    class LUCID_API Pair {
    public:
        // Constructs a Pair validator.
        //
        // Parameters
        // ----------
        // a : const TensorImplPtr&
        //     First operand.  Reference retained; must outlive the chain.
        // b : const TensorImplPtr&
        //     Second operand.  Reference retained; must outlive the chain.
        // op : std::string
        //     Op name used as the prefix in error messages.
        Pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op);

        // Asserts that both operand pointers are non-null.
        //
        // Returns
        // -------
        // Pair&
        //     ``*this``.
        //
        // Raises
        // ------
        // LucidError
        //     If either ``a_`` or ``b_`` is null.
        Pair& both_non_null();

        // Asserts that both operands share the same dtype.
        //
        // Implicitly calls :cpp:func:`both_non_null` first.
        //
        // Returns
        // -------
        // Pair&
        //     ``*this``.
        //
        // Raises
        // ------
        // LucidError
        //     If either operand is null.
        // DtypeMismatch
        //     If ``a_->dtype() != b_->dtype()``.
        Pair& same_dtype();

        // Asserts that both operands reside on the same device.
        //
        // Implicitly calls :cpp:func:`both_non_null` first.
        //
        // Returns
        // -------
        // Pair&
        //     ``*this``.
        //
        // Raises
        // ------
        // LucidError
        //     If either operand is null.
        // DeviceMismatch
        //     If ``a_->device() != b_->device()``.
        Pair& same_device();

        // Asserts that both operands share the same shape.
        //
        // Implicitly calls :cpp:func:`both_non_null` first.
        //
        // Returns
        // -------
        // Pair&
        //     ``*this``.
        //
        // Raises
        // ------
        // LucidError
        //     If either operand is null.
        // ShapeMismatch
        //     If ``a_->shape() != b_->shape()``.
        Pair& same_shape();

    private:
        const TensorImplPtr& a_;
        const TensorImplPtr& b_;
        std::string op_;
    };

    // Builds a two-tensor validator.
    //
    // Parameters
    // ----------
    // a : const TensorImplPtr&
    //     First operand.  Reference retained; must outlive the chain.
    // b : const TensorImplPtr&
    //     Second operand.  Reference retained; must outlive the chain.
    // op_name : std::string
    //     Op name used in error messages.
    //
    // Returns
    // -------
    // Pair
    //     A new validator bound to ``a`` and ``b``.
    static Pair pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op_name);

private:
    // Construct a ``Validator`` bound to ``t`` with the given ``label``.
    //
    // Private because callers must go through the :cpp:func:`input`
    // factory.  Stores ``t`` by reference (not copy), so the resulting
    // validator must live no longer than the caller's ``TensorImplPtr``.
    // ``label`` is woven into every error message the chain emits, so
    // callers should pass the parameter's public name (e.g. ``"weight"``).
    Validator(const TensorImplPtr& t, std::string label) : t_(t), label_(std::move(label)) {}
    const TensorImplPtr& t_;
    std::string label_;
};

// Refuse a shape that is not a square matrix or a batch of them: rank < 2
// is ``InvalidArgument``, unequal trailing dims ``ShapeMismatch`` (both
// ``ValueError``s on the Python side).  ``label`` names the op and operand.
LUCID_API void require_square_matrix(const Shape& sh, const std::string& label);

// The door checks every kernel that indexes fixed axes relies on.  A kernel
// handed a rank, a per-channel length or a geometry value it does not handle
// reads past its operands — a segfault on the CPU, a foreign MLX exception
// on Metal — so each op refuses such an input here, on both devices alike,
// before anything is dispatched.  ``what`` names the op and the operand
// (``"batch_norm: x"``); ``layout`` spells the axes the kernel indexes
// (``"(N, C, H, W)"``).

// Refuse a shape whose rank is not ``rank``: ``ShapeMismatch`` (a
// ``ValueError``) with both ranks.  The expected shape is printed with a
// ``-1`` wildcard per axis.
LUCID_INTERNAL void require_rank(const Shape& sh,
                                 std::size_t rank,
                                 const std::string& what,
                                 const std::string& layout = "");

// Refuse a shape of rank below ``min_rank``, as :cpp:func:`require_rank`.
LUCID_INTERNAL void require_min_rank(const Shape& sh,
                                     std::size_t min_rank,
                                     const std::string& what,
                                     const std::string& layout = "");

// The ``layout`` of a channels-first input over ``n_spatial`` spatial axes:
// ``(N, C, L)``, ``(N, C, H, W)`` or ``(N, C, D, H, W)``, with ``lead`` and
// ``channel`` naming the first two axes.
LUCID_INTERNAL std::string channels_first_layout(int n_spatial,
                                                 const std::string& lead = "N",
                                                 const std::string& channel = "C");

// Refuse a per-channel operand (scale, shift, bias, running statistic)
// that is not 1-D of length ``n``: ``ShapeMismatch`` naming ``(n,)``.
LUCID_INTERNAL void require_length(const Shape& sh, std::int64_t n, const std::string& what);

// Refuse an integer hyper-parameter below ``lo`` — a kernel extent, stride,
// dilation or group count below 1, a padding below 0, an output extent the
// kernel would allocate negatively: ``InvalidArgument`` (a ``ValueError``).
LUCID_INTERNAL void require_at_least(std::int64_t v, std::int64_t lo, const std::string& what);

// :cpp:func:`require_at_least` over every entry of a per-axis parameter.
template <class Range>
void require_each_at_least(const Range& values, std::int64_t lo, const std::string& what) {
    for (const auto v : values)
        require_at_least(static_cast<std::int64_t>(v), lo, what);
}

}  // namespace lucid
