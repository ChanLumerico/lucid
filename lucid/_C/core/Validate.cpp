// lucid/_C/core/Validate.cpp
//
// Implementation of the Validator and Validator::Pair fluent validation
// chains.  Each method is intentionally short — its only job is to evaluate
// a single precondition and delegate to ErrorBuilder for the throw.

#include "Validate.h"

#include <set>

#include "Dtype.h"
#include "Error.h"
#include "ErrorBuilder.h"
#include "TensorImpl.h"

namespace lucid {

Validator Validator::input(const TensorImplPtr& t, std::string label) {
    return Validator(t, std::move(label));
}

Validator& Validator::non_null() {
    if (!t_)
        ErrorBuilder(label_).fail("null input");
    return *this;
}

Validator& Validator::float_only() {
    non_null();
    if (t_->dtype() != Dtype::F32 && t_->dtype() != Dtype::F64) {
        ErrorBuilder(label_).not_implemented("only F32/F64 supported (got " +
                                             std::string(dtype_name(t_->dtype())) + ")");
    }
    return *this;
}

Validator& Validator::dtype_eq(Dtype expected) {
    non_null();
    if (t_->dtype() != expected) {
        throw DtypeMismatch(std::string(dtype_name(expected)), std::string(dtype_name(t_->dtype())),
                            label_);
    }
    return *this;
}

// Iterates the allowed set linearly — the list is almost always very short
// (2–4 entries) so a std::set or bitset would be overkill.  The two-pass
// approach (check first, then build the error string) keeps the fast path
// (success) free of any string construction.
Validator& Validator::dtype_in(std::initializer_list<Dtype> allowed) {
    non_null();
    for (Dtype d : allowed)
        if (t_->dtype() == d)
            return *this;
    // Build a "|"-separated string of allowed dtype names for the error message.
    std::string allowed_str;
    bool first = true;
    for (Dtype d : allowed) {
        if (!first)
            allowed_str += "|";
        allowed_str += dtype_name(d);
        first = false;
    }
    throw DtypeMismatch(allowed_str, std::string(dtype_name(t_->dtype())), label_);
}

Validator& Validator::ndim(int expected) {
    non_null();
    require_rank(t_->shape(), static_cast<std::size_t>(expected), label_);
    return *this;
}

Validator& Validator::ndim_at_least(int min_n) {
    non_null();
    require_min_rank(t_->shape(), static_cast<std::size_t>(min_n), label_);
    return *this;
}

Validator& Validator::shape_eq(const Shape& expected) {
    non_null();
    if (t_->shape() != expected) {
        throw ShapeMismatch(expected, t_->shape(), label_);
    }
    return *this;
}

// Checks both that the tensor is at least 2-D and that the trailing two
// dimensions are equal (square matrix / batch of square matrices).  A 3-D
// tensor [B, N, N] passes; [B, N, M] with N != M fails.
Validator& Validator::square_2d() {
    non_null();
    require_square_matrix(t_->shape(), label_);
    return *this;
}

// The one refusal of a non-square matrix, for every linalg op: a shape
// error (``ShapeMismatch``, so a ``ValueError`` as well as a ``LucidError``)
// naming the square shape it needed — never the bare ``LucidError`` a
// failed factorisation used to share, which the ``*_ex`` variants then had
// to tell apart by checking the shape themselves first.
void require_square_matrix(const Shape& sh, const std::string& label) {
    if (sh.size() < 2)
        ErrorBuilder(label).invalid_argument("expected a square matrix or a batch of them "
                                             "(>=2-D), got ndim=" +
                                             std::to_string(sh.size()));
    const auto n = sh.size();
    if (sh[n - 1] != sh[n - 2]) {
        Shape square = sh;
        square[n - 1] = sh[n - 2];
        throw ShapeMismatch(square, sh, label + ": a square matrix or a batch of them");
    }
}

namespace {

std::string rank_refusal(const std::string& what,
                         const char* relation,
                         std::size_t rank,
                         const std::string& layout,
                         std::size_t got) {
    std::string msg = what + " must be " + relation + std::to_string(rank) + "-D";
    if (!layout.empty())
        msg += " " + layout;
    return msg + ", got " + std::to_string(got) + "-D";
}

}  // namespace

void require_rank(const Shape& sh,
                  std::size_t rank,
                  const std::string& what,
                  const std::string& layout) {
    if (sh.size() != rank)
        throw ShapeMismatch(Shape(rank, -1), sh, rank_refusal(what, "", rank, layout, sh.size()));
}

void require_min_rank(const Shape& sh,
                      std::size_t min_rank,
                      const std::string& what,
                      const std::string& layout) {
    if (sh.size() < min_rank)
        throw ShapeMismatch(Shape(min_rank, -1), sh,
                            rank_refusal(what, "at least ", min_rank, layout, sh.size()));
}

std::string
channels_first_layout(int n_spatial, const std::string& lead, const std::string& channel) {
    static const char* const kSpatial[] = {"", ", L", ", H, W", ", D, H, W"};
    const char* spatial = (n_spatial >= 0 && n_spatial <= 3) ? kSpatial[n_spatial] : ", *";
    return "(" + lead + ", " + channel + spatial + ")";
}

void require_length(const Shape& sh, std::int64_t n, const std::string& what) {
    if (sh.size() != 1 || sh[0] != n)
        throw ShapeMismatch(Shape{n}, sh,
                            what + " must be 1-D of length " + std::to_string(n) +
                                " (one entry per channel)");
}

void require_at_least(std::int64_t v, std::int64_t lo, const std::string& what) {
    if (v < lo)
        throw InvalidArgument(what + " must be >= " + std::to_string(lo) + ", got " +
                              std::to_string(v));
}

Validator::Pair::Pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op)
    : a_(a), b_(b), op_(std::move(op)) {}

Validator::Pair
Validator::pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op_name) {
    return Pair(a, b, std::move(op_name));
}

Validator::Pair& Validator::Pair::both_non_null() {
    if (!a_ || !b_)
        ErrorBuilder(op_).fail("null input");
    return *this;
}

Validator::Pair& Validator::Pair::same_dtype() {
    both_non_null();
    if (a_->dtype() != b_->dtype()) {
        throw DtypeMismatch(std::string(dtype_name(a_->dtype())),
                            std::string(dtype_name(b_->dtype())), op_);
    }
    return *this;
}

Validator::Pair& Validator::Pair::same_device() {
    both_non_null();
    if (a_->device() != b_->device()) {
        throw DeviceMismatch(std::string(device_name(a_->device())),
                             std::string(device_name(b_->device())), op_);
    }
    return *this;
}

Validator::Pair& Validator::Pair::same_shape() {
    both_non_null();
    if (a_->shape() != b_->shape()) {
        throw ShapeMismatch(a_->shape(), b_->shape(), op_);
    }
    return *this;
}

}  // namespace lucid
