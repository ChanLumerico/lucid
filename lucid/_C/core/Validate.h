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
    //     If the tensor is null or its rank differs from ``expected``.
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
    //     If the tensor is null or its rank is below ``min_n``.
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
    // LucidError
    //     If the tensor is null, has rank < 2, or its last two dims
    //     disagree.
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

}  // namespace lucid
