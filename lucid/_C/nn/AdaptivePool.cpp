#include "AdaptivePool.h"

#include <vector>

#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "PoolNd.h"

namespace lucid {

namespace {

inline void check_uniform(int S, int O, int axis, const char* op) {
    if (O <= 0)
        ErrorBuilder(op).fail("output_size must be > 0");
    if (S % O != 0) {
        ErrorBuilder(op).not_implemented("non-uniform adaptive pooling not supported (axis" +
                                         std::to_string(axis) + ": input " + std::to_string(S) +
                                         " not divisible by output " + std::to_string(O) +
                                         "). Pad input to a divisible size or use a regular pool.");
    }
}

inline void check_rank(const TensorImplPtr& x, int expected_rank, const char* op) {
    Validator::input(x, std::string(op) + ".x").non_null();
    if (static_cast<int>(x->shape().size()) != expected_rank)
        throw ShapeMismatch(x->shape(), Shape{}, std::string(op) + ": x rank mismatch");
}

}  // namespace

TensorImplPtr adaptive_max_pool1d_op(const TensorImplPtr& x, int OL) {
    check_rank(x, 3, "adaptive_max_pool1d");
    const int L = static_cast<int>(x->shape()[2]);
    check_uniform(L, OL, 0, "adaptive_max_pool1d");
    const int K = L / OL;
    return max_pool1d_op(x, K, K, 0);
}

TensorImplPtr adaptive_max_pool2d_op(const TensorImplPtr& x, int OH, int OW) {
    check_rank(x, 4, "adaptive_max_pool2d");
    const int H = static_cast<int>(x->shape()[2]);
    const int W = static_cast<int>(x->shape()[3]);
    check_uniform(H, OH, 0, "adaptive_max_pool2d");
    check_uniform(W, OW, 1, "adaptive_max_pool2d");
    const int KH = H / OH;
    const int KW = W / OW;
    return max_pool2d_op(x, KH, KW, KH, KW, 0, 0);
}

TensorImplPtr adaptive_max_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW) {
    check_rank(x, 5, "adaptive_max_pool3d");
    const int D = static_cast<int>(x->shape()[2]);
    const int H = static_cast<int>(x->shape()[3]);
    const int W = static_cast<int>(x->shape()[4]);
    check_uniform(D, OD, 0, "adaptive_max_pool3d");
    check_uniform(H, OH, 1, "adaptive_max_pool3d");
    check_uniform(W, OW, 2, "adaptive_max_pool3d");
    const int KD = D / OD, KH = H / OH, KW = W / OW;
    return max_pool3d_op(x, KD, KH, KW, KD, KH, KW, 0, 0, 0);
}

TensorImplPtr adaptive_avg_pool1d_op(const TensorImplPtr& x, int OL) {
    check_rank(x, 3, "adaptive_avg_pool1d");
    const int L = static_cast<int>(x->shape()[2]);
    check_uniform(L, OL, 0, "adaptive_avg_pool1d");
    const int K = L / OL;
    return avg_pool1d_op(x, K, K, 0);
}

TensorImplPtr adaptive_avg_pool2d_op(const TensorImplPtr& x, int OH, int OW) {
    check_rank(x, 4, "adaptive_avg_pool2d");
    const int H = static_cast<int>(x->shape()[2]);
    const int W = static_cast<int>(x->shape()[3]);
    check_uniform(H, OH, 0, "adaptive_avg_pool2d");
    check_uniform(W, OW, 1, "adaptive_avg_pool2d");
    const int KH = H / OH;
    const int KW = W / OW;
    return avg_pool2d_op(x, KH, KW, KH, KW, 0, 0);
}

TensorImplPtr adaptive_avg_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW) {
    check_rank(x, 5, "adaptive_avg_pool3d");
    const int D = static_cast<int>(x->shape()[2]);
    const int H = static_cast<int>(x->shape()[3]);
    const int W = static_cast<int>(x->shape()[4]);
    check_uniform(D, OD, 0, "adaptive_avg_pool3d");
    check_uniform(H, OH, 1, "adaptive_avg_pool3d");
    check_uniform(W, OW, 2, "adaptive_avg_pool3d");
    const int KD = D / OD, KH = H / OH, KW = W / OW;
    return avg_pool3d_op(x, KD, KH, KW, KD, KH, KW, 0, 0, 0);
}

}  // namespace lucid
