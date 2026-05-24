// lucid/_C/nn/LSTM.cpp
//
// LSTM forward and BPTT backward implementation.
//
// forward() decides between two backend paths based on whether any input
// requires a gradient:
//   Inference (no grad): IBackend::lstm_forward (returns just output/hn/cn).
//   Training   (grad):   IBackend::lstm_forward_train (saves gates/cells for BPTT).
//
// The training path returns 5 Storage objects:
//   res[0] – output sequence (T, B, H).
//   res[1] – final hidden state hn (1, B, H).
//   res[2] – final cell state cn (1, B, H).
//   res[3] – gates_all (T, B, 4H).
//   res[4] – cells_all (T+1, B, H).
//
// Edges are wired manually in forward(): each weight tensor that requires_grad
// gets an AccumulateGrad leaf node; non-differentiable tensors get a null edge.
// Only out_t carries the grad_fn; hn_t and cn_t are detached (requires_grad=false).
//
// In apply(), gradients for hn and cn at the sequence end are zero because they
// are not used further in the computation graph in the standard single-layer case.

#include "LSTM.h"

#include <cstring>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Allocator.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"

namespace lucid {

std::vector<Storage> LstmBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device);

    // Gradients for hn and cn at the sequence end are zero here because only
    // the output sequence is connected in the graph for the single-layer case.
    // ``hn`` lives in the recurrent dim (proj_size when projection is on),
    // ``cn`` always lives in the cell-state dim (hidden_size).
    const std::int64_t Hrec = (opts.proj_size > 0) ? opts.proj_size : opts.hidden_size;
    const Shape zero_hn_shape{static_cast<std::int64_t>(opts.batch_size), Hrec};
    const Shape zero_cn_shape{static_cast<std::int64_t>(opts.batch_size),
                              static_cast<std::int64_t>(opts.hidden_size)};
    Storage zero_hn = be.zeros(zero_hn_shape, dtype);
    Storage zero_cn = be.zeros(zero_cn_shape, dtype);

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

    // With proj_size > 0, the output / hn dimensions shrink to proj_size,
    // while c_n keeps the cell-state dim H.
    const int Hout = (opts.proj_size > 0) ? opts.proj_size : H;
    Shape out_shape{static_cast<std::int64_t>(T), static_cast<std::int64_t>(B),
                    static_cast<std::int64_t>(Hout)};
    Shape hn_shape{static_cast<std::int64_t>(opts.num_layers), static_cast<std::int64_t>(B),
                   static_cast<std::int64_t>(Hout)};
    Shape cn_shape{static_cast<std::int64_t>(opts.num_layers), static_cast<std::int64_t>(B),
                   static_cast<std::int64_t>(H)};

    // Open an OpScope so the trace sees ``lstm`` as a single 3-output
    // op.  Attrs carry the shape parameters the compile-path emitter
    // needs to set up the MPSGraph LSTMDescriptor.  Without this the
    // C++ fused call would run invisible to the trace and downstream
    // ops would treat its outputs as fresh external feeds.
    OpScopeFull scope{"lstm", dev, dt, out_shape};
    scope.set_attr("hidden_size", static_cast<std::int64_t>(H));
    scope.set_attr("num_layers", static_cast<std::int64_t>(opts.num_layers));
    scope.set_attr("batch_first", opts.batch_first);
    scope.set_attr("bidirectional", opts.bidirectional);
    scope.set_attr("proj_size", static_cast<std::int64_t>(opts.proj_size));
    scope.set_attr("has_bias", weights.size() >= 4);

    auto register_trace_outputs = [&](const TensorImplPtr& out_t,
                                       const TensorImplPtr& hn_t,
                                       const TensorImplPtr& cn_t) {
        auto* trc = ::lucid::compile::current_tracer();
        if (trc == nullptr) return;
        // Pass all inputs on the first on_op_io call; the subsequent
        // calls pass empty input lists so the trace's input record
        // doesn't duplicate (see Tracer's first-vs-subsequent
        // detection logic).
        std::vector<TensorImplPtr> all_inputs{input, h0, c0};
        for (const auto& w : weights) all_inputs.push_back(w);
        trc->on_op_io(all_inputs, out_t);
        trc->on_op_io({}, hn_t);
        trc->on_op_io({}, cn_t);
    };

    // Backend storage convention: ``lstm_forward`` / ``lstm_forward_train``
    // emit hn / cn as 2-D ``(B, Hrec/H)`` arrays, while Lucid's Python
    // API contract is 3-D ``(num_layers, B, Hrec/H)``.  Without an
    // explicit reshape the TensorImpl claims the 3-D shape but the
    // underlying MLX storage still reports 2 dims — any downstream op
    // that walks the storage rank (``sum`` / ``flatten`` / …) then
    // fails with ``Invalid axis 2 for array with 2 dimensions``.  Add
    // the unsqueeze here so the storage rank matches what callers see.
    const Shape hn_storage_2d{static_cast<std::int64_t>(B),
                              static_cast<std::int64_t>(Hout)};
    const Shape cn_storage_2d{static_cast<std::int64_t>(B), static_cast<std::int64_t>(H)};

    if (!needs_grad) {
        // Backends that don't implement projection in lstm_forward route
        // through the training kernel for proj_size > 0 and discard the
        // saved gates/cells outputs.
        if (opts.proj_size > 0) {
            auto res_p = be.lstm_forward_train(input->storage(), h0->storage(), c0->storage(),
                                               w_storages, opts, dt);
            if (res_p.size() < 3)
                ErrorBuilder("lstm").fail("lstm_forward_train returned < 3 outputs");
            auto hn_3d = be.reshape(res_p[1], hn_storage_2d, hn_shape, dt);
            auto cn_3d = be.reshape(res_p[2], cn_storage_2d, cn_shape, dt);
            auto out_t = std::make_shared<TensorImpl>(std::move(res_p[0]), out_shape, dt, dev, false);
            auto hn_t = std::make_shared<TensorImpl>(std::move(hn_3d), hn_shape, dt, dev, false);
            auto cn_t = std::make_shared<TensorImpl>(std::move(cn_3d), cn_shape, dt, dev, false);
            register_trace_outputs(out_t, hn_t, cn_t);
            return {out_t, hn_t, cn_t};
        }
        auto res = be.lstm_forward(input->storage(), h0->storage(), c0->storage(), w_storages, opts,
                                   out_shape, dt);
        auto hn_3d = be.reshape(res[1], hn_storage_2d, hn_shape, dt);
        auto cn_3d = be.reshape(res[2], cn_storage_2d, cn_shape, dt);
        auto out_t = std::make_shared<TensorImpl>(std::move(res[0]), out_shape, dt, dev, false);
        auto hn_t = std::make_shared<TensorImpl>(std::move(hn_3d), hn_shape, dt, dev, false);
        auto cn_t = std::make_shared<TensorImpl>(std::move(cn_3d), cn_shape, dt, dev, false);
        register_trace_outputs(out_t, hn_t, cn_t);
        return {out_t, hn_t, cn_t};
    }

    auto res =
        be.lstm_forward_train(input->storage(), h0->storage(), c0->storage(), w_storages, opts, dt);
    if (res.size() < 5)
        ErrorBuilder("lstm").fail("lstm_forward_train returned < 5 outputs");

    auto hn_3d = be.reshape(res[1], hn_storage_2d, hn_shape, dt);
    auto cn_3d = be.reshape(res[2], cn_storage_2d, cn_shape, dt);
    auto out_t = std::make_shared<TensorImpl>(std::move(res[0]), out_shape, dt, dev, true);
    auto hn_t = std::make_shared<TensorImpl>(std::move(hn_3d), hn_shape, dt, dev, false);
    auto cn_t = std::make_shared<TensorImpl>(std::move(cn_3d), cn_shape, dt, dev, false);

    auto bwd = std::make_shared<LstmBackward>();
    bwd->saved_input = input->storage();
    bwd->saved_h0 = h0->storage();
    bwd->saved_weights = w_storages;
    bwd->gates_all = std::move(res[3]);
    bwd->cells_all = std::move(res[4]);
    bwd->opts = opts;
    bwd->dtype = dt;
    bwd->device = dev;

    // Build the edge list manually: {input, h0, c0, wih, whh, bih, bhh}.
    // NaryKernel is not used here because the number of edges is dynamic
    // (it depends on the number of weight tensors supplied).
    std::vector<TensorImplPtr> edge_tensors{input, h0, c0};
    for (const auto& w : weights)
        edge_tensors.push_back(w);

    std::vector<Edge> edges;
    std::vector<std::int64_t> versions;
    for (const auto& t : edge_tensors) {
        if (!t || !t->requires_grad()) {
            // A null edge signals that this input does not participate in
            // gradient accumulation; the backward skips it.
            edges.emplace_back(nullptr, 0);
            versions.push_back(0);
        } else {
            if (t->is_leaf() && !t->grad_fn())
                t->set_grad_fn(std::make_shared<AccumulateGrad>(t));
            edges.emplace_back(t->grad_fn(), t->grad_output_nr());
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
