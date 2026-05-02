#include "LSTM.h"

#include <cstring>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../backend/Dispatcher.h"
#include "../core/Allocator.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/TensorImpl.h"

namespace lucid {

std::vector<Storage> LstmBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device);

    const Shape zero_shape{static_cast<std::int64_t>(opts.batch_size),
                           static_cast<std::int64_t>(opts.hidden_size)};
    Storage zero_hn = be.zeros(zero_shape, dtype);
    Storage zero_cn = be.zeros(zero_shape, dtype);

    return be.lstm_backward(grad_out, zero_hn, zero_cn, saved_input, saved_h0, saved_weights,
                            gates_all, cells_all, opts, dtype);
}

std::vector<TensorImplPtr> LstmBackward::forward(const TensorImplPtr& input,
                                                 const TensorImplPtr& h0,
                                                 const TensorImplPtr& c0,
                                                 const std::vector<TensorImplPtr>& weights,
                                                 const backend::IBackend::LstmOpts& opts) {
    if (!input || !h0 || !c0)
        ErrorBuilder("lstm").fail("null input");

    auto& be = backend::Dispatcher::for_device(input->device());
    const Dtype dt = input->dtype();
    const Device dev = input->device();
    const int T = opts.seq_len, B = opts.batch_size, H = opts.hidden_size;

    std::vector<Storage> w_storages;
    w_storages.reserve(weights.size());
    for (const auto& w : weights) {
        if (!w)
            ErrorBuilder("lstm").fail("null weight");
        w_storages.push_back(w->storage());
    }

    const bool needs_grad =
        GradMode::is_enabled() &&
        (input->requires_grad() || h0->requires_grad() || c0->requires_grad() ||
         std::any_of(weights.begin(), weights.end(),
                     [](const TensorImplPtr& w) { return w && w->requires_grad(); }));

    Shape out_shape{static_cast<std::int64_t>(T), static_cast<std::int64_t>(B),
                    static_cast<std::int64_t>(H)};
    Shape hn_shape{static_cast<std::int64_t>(opts.num_layers), static_cast<std::int64_t>(B),
                   static_cast<std::int64_t>(H)};

    if (!needs_grad) {
        auto res = be.lstm_forward(input->storage(), h0->storage(), c0->storage(), w_storages, opts,
                                   out_shape, dt);
        return {std::make_shared<TensorImpl>(std::move(res[0]), out_shape, dt, dev, false),
                std::make_shared<TensorImpl>(std::move(res[1]), hn_shape, dt, dev, false),
                std::make_shared<TensorImpl>(std::move(res[2]), hn_shape, dt, dev, false)};
    }

    auto res =
        be.lstm_forward_train(input->storage(), h0->storage(), c0->storage(), w_storages, opts, dt);
    if (res.size() < 5)
        ErrorBuilder("lstm").fail("lstm_forward_train returned < 5 outputs");

    auto out_t = std::make_shared<TensorImpl>(std::move(res[0]), out_shape, dt, dev, true);
    auto hn_t = std::make_shared<TensorImpl>(std::move(res[1]), hn_shape, dt, dev, false);
    auto cn_t = std::make_shared<TensorImpl>(std::move(res[2]), hn_shape, dt, dev, false);

    auto bwd = std::make_shared<LstmBackward>();
    bwd->saved_input = input->storage();
    bwd->saved_h0 = h0->storage();
    bwd->saved_weights = w_storages;
    bwd->gates_all = std::move(res[3]);
    bwd->cells_all = std::move(res[4]);
    bwd->opts = opts;
    bwd->dtype = dt;
    bwd->device = dev;

    std::vector<TensorImplPtr> edge_tensors{input, h0, c0};
    for (const auto& w : weights)
        edge_tensors.push_back(w);

    std::vector<Edge> edges;
    std::vector<std::int64_t> versions;
    for (const auto& t : edge_tensors) {
        if (!t || !t->requires_grad()) {
            edges.emplace_back(nullptr, 0);
            versions.push_back(0);
        } else {
            if (t->is_leaf() && !t->grad_fn())
                t->set_grad_fn(std::make_shared<AccumulateGrad>(t));
            edges.emplace_back(t->grad_fn(), 0);
            versions.push_back(t->version());
        }
    }
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions(std::move(versions));
    out_t->set_grad_fn(std::move(bwd));
    out_t->set_leaf(false);

    return {out_t, hn_t, cn_t};
}

std::vector<TensorImplPtr> lstm_op(const TensorImplPtr& input,
                                   const TensorImplPtr& h0,
                                   const TensorImplPtr& c0,
                                   const std::vector<TensorImplPtr>& weights,
                                   const backend::IBackend::LstmOpts& opts) {
    return LstmBackward::forward(input, h0, c0, weights, opts);
}

}  // namespace lucid
