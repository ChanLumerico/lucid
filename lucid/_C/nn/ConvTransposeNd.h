#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

template <int N>

class LUCID_API ConvTransposeNdBackward : public FuncOp<ConvTransposeNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];
    int pad_[N];
    int opad_[N];

    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& W,
                                 const TensorImplPtr& b,
                                 const int (&stride)[N],
                                 const int (&pad)[N],
                                 const int (&opad)[N]);
    std::vector<Storage> apply(Storage grad_out) override;
};

using ConvTranspose1dBackward = ConvTransposeNdBackward<1>;
using ConvTranspose2dBackward = ConvTransposeNdBackward<2>;
using ConvTranspose3dBackward = ConvTransposeNdBackward<3>;

LUCID_API TensorImplPtr conv_transpose1d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_l = 1,
                                            int pad_l = 0,
                                            int opad_l = 0);

LUCID_API TensorImplPtr conv_transpose2d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_h = 1,
                                            int stride_w = 1,
                                            int pad_h = 0,
                                            int pad_w = 0,
                                            int opad_h = 0,
                                            int opad_w = 0);

LUCID_API TensorImplPtr conv_transpose3d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_d = 1,
                                            int stride_h = 1,
                                            int stride_w = 1,
                                            int pad_d = 0,
                                            int pad_h = 0,
                                            int pad_w = 0,
                                            int opad_d = 0,
                                            int opad_h = 0,
                                            int opad_w = 0);

}  // namespace lucid
