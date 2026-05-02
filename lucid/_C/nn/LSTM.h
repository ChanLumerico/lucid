#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/Node.h"
#include "../backend/IBackend.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class LUCID_API LstmBackward : public Node {
public:
    Storage saved_input;
    Storage saved_h0;
    std::vector<Storage> saved_weights;

    Storage gates_all;
    Storage cells_all;

    backend::IBackend::LstmOpts opts;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    std::string_view name() const noexcept { return "LstmBackward"; }

    std::vector<Storage> apply(Storage grad_out) override;

    static std::vector<TensorImplPtr> forward(const TensorImplPtr& input,
                                              const TensorImplPtr& h0,
                                              const TensorImplPtr& c0,
                                              const std::vector<TensorImplPtr>& weights,
                                              const backend::IBackend::LstmOpts& opts);
};

LUCID_API std::vector<TensorImplPtr> lstm_op(const TensorImplPtr& input,
                                             const TensorImplPtr& h0,
                                             const TensorImplPtr& c0,
                                             const std::vector<TensorImplPtr>& weights,
                                             const backend::IBackend::LstmOpts& opts);

}  // namespace lucid
