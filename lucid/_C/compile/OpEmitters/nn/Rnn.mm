// lucid/_C/compile/OpEmitters/nn/Rnn.mm
//
// Recurrent-layer emitters for the trace IR.  Currently covers the
// single-layer, unidirectional, F32, no-projection LSTM — the same
// surface the MLX fused backend supports.  Configurations outside that
// envelope (multi-layer, bidirectional, projected LSTMP, non-F32) make
// the emitter return ``nullptr`` so the builder aborts and Lucid falls
// back to the (still-fast) MLX eager path.
//
// Trace contract (set by :file:`lucid/_C/nn/LSTM.cpp`):
//
//   - Op name           : ``"lstm"``
//   - Inputs            : ``input``, ``h0``, ``c0``, ``W_ih``, ``W_hh``,
//                         ``b_ih``, ``b_hh`` (bias pair optional —
//                         engine always supplies for now)
//   - Outputs           : ``out``   shape (T, B, H)
//                         ``h_n``   shape (1, B, H)   — single-layer
//                         ``c_n``   shape (1, B, H)
//   - Attributes        : ``hidden_size`` (i64), ``num_layers`` (i64),
//                         ``batch_first`` (bool), ``bidirectional``
//                         (bool), ``proj_size`` (i64), ``has_bias``
//                         (bool)
//
// Weight layout (matches the reference framework):
//   - W_ih    : (4H, I)   gates concatenated as [i, f, g, o]
//   - W_hh    : (4H, H)   ditto
//   - b_ih    : (4H,)
//   - b_hh    : (4H,)
//
// MPSGraph's LSTM API expects the same gate ordering with
// ``descriptor.forgetGateLast == NO`` (the default) — input gate first,
// forget second, cell (g) third, output last.  That matches the
// engine's slicing in :file:`lucid/_C/backend/gpu/GpuBackend.h`
// (``raw[:, :H]`` = i, ``raw[:, H:2H]`` = f, ``raw[:, 2H:3H]`` = g,
// ``raw[:, 3H:4H]`` = o), so no gate reorder is needed.
//
// Bias handling: PyTorch splits the affine bias into two halves so the
// state-dict round-trips with the reference framework's
// ``LSTMCell`` layout.  MPSGraph wants a single (4H,) bias — we add
// the two halves before feeding the descriptor (matches the engine's
// ``bias = b_ih + b_hh`` pre-compute).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

class LstmEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "lstm"; }

    void* emit(BuilderContext& ctx, const OpNode& node) override {
        // Outside this envelope we bail to eager — matches the MLX
        // backend's ``lstm_metal_supported`` predicate.
        const std::int64_t num_layers = int_attr(node, "num_layers", 1);
        const std::int64_t proj_size = int_attr(node, "proj_size", 0);
        const bool bidirectional = bool_attr(node, "bidirectional", false);
        if (num_layers != 1 || proj_size != 0 || bidirectional)
            return nullptr;

        // batch_first is a Python-side reshape: the engine always
        // receives (T, B, I) and emits (T, B, H).  Reject the
        // hypothetical case where the trace records batch_first=true
        // (currently not possible — the eager wrapper transposes
        // before calling into C++).
        if (bool_attr(node, "batch_first", false))
            return nullptr;

        // Need at least x, h0, c0, W_ih, W_hh, b_ih, b_hh (7 inputs).
        if (node.inputs.size() < 7)
            return nullptr;
        TensorId x_id = node.inputs[0];
        TensorId h0_id = node.inputs[1];
        TensorId c0_id = node.inputs[2];
        TensorId wih_id = node.inputs[3];
        TensorId whh_id = node.inputs[4];
        TensorId bih_id = node.inputs[5];
        TensorId bhh_id = node.inputs[6];
        if (x_id < 0 || h0_id < 0 || c0_id < 0 || wih_id < 0 || whh_id < 0 || bih_id < 0 ||
            bhh_id < 0)
            return nullptr;

        if (node.outputs.size() != 3)
            return nullptr;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        if (g == nil)
            return nullptr;

        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* h0 = (__bridge MPSGraphTensor*)ctx.resolve(h0_id);
        MPSGraphTensor* c0 = (__bridge MPSGraphTensor*)ctx.resolve(c0_id);
        MPSGraphTensor* Wih = (__bridge MPSGraphTensor*)ctx.resolve(wih_id);
        MPSGraphTensor* Whh = (__bridge MPSGraphTensor*)ctx.resolve(whh_id);
        MPSGraphTensor* bih = (__bridge MPSGraphTensor*)ctx.resolve(bih_id);
        MPSGraphTensor* bhh = (__bridge MPSGraphTensor*)ctx.resolve(bhh_id);
        if (x == nil || h0 == nil || c0 == nil || Wih == nil || Whh == nil || bih == nil ||
            bhh == nil)
            return nullptr;

        // h0 / c0 carry the (num_layers=1, B, H) shape the engine
        // returns.  MPSGraph expects (B, H); squeeze the leading dim.
        const TensorMeta& hn_meta = node.outputs[1];  // shape (1, B, Hrec)
        const TensorMeta& cn_meta = node.outputs[2];  // shape (1, B, H)
        if (hn_meta.shape.size() != 3 || cn_meta.shape.size() != 3)
            return nullptr;
        const std::int64_t B = hn_meta.shape[1];
        const std::int64_t Hrec = hn_meta.shape[2];
        const std::int64_t H = cn_meta.shape[2];
        if (B <= 0 || H <= 0 || Hrec != H)
            return nullptr;  // proj_size==0 ⇒ Hrec must equal H

        NSArray<NSNumber*>* twoD_shape = @[
            [NSNumber numberWithLongLong:B],
            [NSNumber numberWithLongLong:H],
        ];
        MPSGraphTensor* h0_2d = [g reshapeTensor:h0 withShape:twoD_shape name:@"lstm_h0"];
        MPSGraphTensor* c0_2d = [g reshapeTensor:c0 withShape:twoD_shape name:@"lstm_c0"];

        // Combined bias for the descriptor.
        MPSGraphTensor* bias =
            [g additionWithPrimaryTensor:bih secondaryTensor:bhh name:@"lstm_bias"];

        // MPSGraph LSTM expects recurrent weight (4H, H) and input
        // weight (4H, I), with the leading axis grouping the four
        // gates.  PyTorch / Lucid match this layout exactly, so feed
        // straight through.
        MPSGraphLSTMDescriptor* d = [MPSGraphLSTMDescriptor descriptor];
        d.reverse = NO;
        d.bidirectional = NO;
        d.training = NO;          // eager path saves gates separately for BPTT
        d.produceCell = YES;
        d.forgetGateLast = NO;    // ⇒ (i, f, g, o) — matches Lucid
        // Activations default to (sigmoid, sigmoid, tanh, sigmoid)
        // with tanh on the cell output — that matches the standard
        // LSTM equations the engine implements.

        NSArray<MPSGraphTensor*>* outs =
            [g LSTMWithSourceTensor:x
                    recurrentWeight:Whh
                        inputWeight:Wih
                               bias:bias
                          initState:h0_2d
                           initCell:c0_2d
                         descriptor:d
                               name:@"lstm"];
        if (outs == nil || outs.count < 3)
            return nullptr;

        MPSGraphTensor* Y_full = outs[0];      // (T, B, H) — every step's hidden
        MPSGraphTensor* C_full = outs[2];      // (T, B, H) — every step's cell

        // Final-step slice for h_n / c_n.  MPSGraph returns the full
        // hidden / cell trajectories; the eager engine exposes only
        // the last step.
        const TensorMeta& out_meta = node.outputs[0];
        if (out_meta.shape.size() != 3)
            return nullptr;
        const std::int64_t T = out_meta.shape[0];
        if (T <= 0)
            return nullptr;

        auto slice_last_step = [&](MPSGraphTensor* full, NSString* nm) -> MPSGraphTensor* {
            MPSGraphTensor* last =
                [g sliceTensor:full
                    dimension:0
                        start:(NSInteger)(T - 1)
                       length:1
                         name:nm];
            // Reshape (1, B, H) — already (1, B, H) from the slice, so
            // the reshape is just a sanity-stable view.  Use the meta
            // shape so MPSGraph sees the same (num_layers=1, B, H)
            // shape every other op expects.
            NSArray<NSNumber*>* three_d = @[
                [NSNumber numberWithLongLong:1],
                [NSNumber numberWithLongLong:B],
                [NSNumber numberWithLongLong:H],
            ];
            return [g reshapeTensor:last withShape:three_d name:nm];
        };

        // Bind the three outputs explicitly (the builder skips
        // auto-bind for ops with multiple outputs).
        const TensorId out_id = node.outputs[0].id;
        const TensorId hn_id = node.outputs[1].id;
        const TensorId cn_id = node.outputs[2].id;

        ctx.bind(out_id, (__bridge void*)Y_full);
        if (ctx.is_consumed(hn_id)) {
            MPSGraphTensor* hn = slice_last_step(Y_full, @"lstm_hn");
            ctx.bind(hn_id, (__bridge void*)hn);
        }
        if (ctx.is_consumed(cn_id)) {
            MPSGraphTensor* cn = slice_last_step(C_full, @"lstm_cn");
            ctx.bind(cn_id, (__bridge void*)cn);
        }

        return (__bridge void*)Y_full;
    }
};

struct RnnEmitterRegistrar {
    RnnEmitterRegistrar() {
        register_emitter(std::make_unique<LstmEmitter>());
    }
};

[[maybe_unused]] static const RnnEmitterRegistrar g_rnn_registrar;

}  // namespace

}  // namespace lucid::compile
