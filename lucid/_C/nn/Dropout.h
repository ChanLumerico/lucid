#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class Generator;

class LUCID_API DropoutBackward : public FuncOp<DropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;

    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API DropoutNdBackward : public FuncOp<DropoutNdBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;

    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API AlphaDropoutBackward : public FuncOp<AlphaDropoutBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    double a_coef_ = 1.0;
    Storage mask_;

    static TensorImplPtr forward(const TensorImplPtr& a, double p, bool training, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API DropBlockBackward : public FuncOp<DropBlockBackward, 1> {
public:
    static const OpSchema schema_v1;
    double p_ = 0.0;
    Storage mask_;

    static TensorImplPtr
    forward(const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

class LUCID_API DropPathBackward : public FuncOp<DropPathBackward, 1> {
public:
    static const OpSchema schema_v1;
    Storage mask_;

    static TensorImplPtr
    forward(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen);

LUCID_API TensorImplPtr dropoutnd_op(const TensorImplPtr& a,
                                     double p,
                                     bool training,
                                     Generator* gen);

LUCID_API TensorImplPtr alpha_dropout_op(const TensorImplPtr& a,
                                         double p,
                                         bool training,
                                         Generator* gen);

LUCID_API TensorImplPtr drop_block_op(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen);

LUCID_API TensorImplPtr drop_path_op(const TensorImplPtr& a,
                                     double p,
                                     bool scale_by_keep,
                                     Generator* gen);

}  // namespace lucid
