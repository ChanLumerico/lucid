// lucid/_C/compile/VjpEmitters/nn/Loss.mm
//
// VJP for MSE loss (the smoke-test surface for Phase 1-4).
//
// Forward (see :file:`OpEmitters/nn/Loss.mm`):
//   diff = input - target
//   sq   = diff * diff
//   out  = reduce(sq, reduction)
//      where reduction ∈ {None (0), Mean (1), Sum (2)}.
//
// Backward:
//   d(out)/d(input)  =  2 * diff * grad_out / N   (mean)
//                       2 * diff * grad_out       (sum, none)
//   d(out)/d(target) = -d(out)/d(input)
//
// For ``reduction='none'`` and ``reduction='sum'`` the grad_out arrives
// with the same shape as ``diff`` (or the squeezed 0-D for sum, which
// MPSGraph broadcasts on multiply); for ``reduction='mean'`` we divide
// by the total number of elements ``numel(input)``.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

class MseLossVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "mse_loss"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() < 2 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0) return false;

        std::int64_t reduction = 1;  // Mean default
        if (auto it = node.attrs.find("reduction"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second))
                reduction = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(in_id));
        MPSGraphTensor* t = as_tensor(bctx.forward(tg_id));
        if (g == nil || grad == nil || x == nil || t == nil) return false;

        MPSDataType dt = x.dataType;
        MPSGraphTensor* diff =
            [g subtractionWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* two = [g constantWithScalar:2.0 dataType:dt];
        MPSGraphTensor* two_diff =
            [g multiplicationWithPrimaryTensor:two secondaryTensor:diff name:nil];

        // For Mean (1): divide by N = numel(x).
        double scale = 1.0;
        if (reduction == 1) {
            std::vector<std::int64_t> x_shape = shape_of_mps(x);
            double n = 1.0;
            for (std::int64_t d : x_shape) n *= (double)d;
            if (n > 0.0) scale = 1.0 / n;
        }

        MPSGraphTensor* coef = two_diff;
        if (scale != 1.0) {
            MPSGraphTensor* s = [g constantWithScalar:scale dataType:dt];
            coef = [g multiplicationWithPrimaryTensor:two_diff
                                       secondaryTensor:s
                                                  name:nil];
        }
        // dx = coef * grad_out  (grad_out broadcasts up to x's shape
        //                        for the Mean/Sum cases where it's 0-D).
        MPSGraphTensor* dx =
            [g multiplicationWithPrimaryTensor:coef secondaryTensor:grad name:@"mse_vjp_dx"];
        MPSGraphTensor* dt_grad = [g negativeWithTensor:dx name:@"mse_vjp_dt"];

        bctx.accumulate_grad(in_id, from_tensor(dx));
        bctx.accumulate_grad(tg_id, from_tensor(dt_grad));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// nll_loss + cross_entropy_loss VJPs
// ────────────────────────────────────────────────────────────────────
//
// Forward (see :file:`OpEmitters/nn/Loss.mm`):
//   per_sample = -log_p[i, target[i]] * keep[i] * w[target[i]]
//   out = per_sample        if reduction == 0 (None)
//       = sum(per_sample)   if reduction == 2 (Sum)
//       = sum(per_sample) / sum(keep)  if reduction == 1 (Mean)
//
// where ``log_p`` is the input (already log-probs for NLL; the
// CE emitter applies softMax+log internally so the CE VJP can use the
// closed form ``softmax(logits) - one_hot(target)`` instead of going
// through the log_softmax chain rule manually).
//
// d(weight) and d(target) are not produced: weight is a constant
// hyperparameter and target is integer-typed (no gradient).
//
// ────────────────────────────────────────────────────────────────────
// Helpers shared by NLL + CE.

namespace {

// Build the per-sample scale factor that multiplies the sparse one-hot
// gradient.  Inputs:
//   * grad_out  — incoming grad; shape (B,) for reduction='none',
//                 scalar for sum/mean.
//   * w_gather  — w[target[i]] for each i, or nullptr if no weight.
//   * keep_f    — float mask (1 where target != ignore_index).
//   * denom     — sum(keep_f) scalar; needed only for reduction='mean'.
//   * reduction — 0=None, 1=Mean, 2=Sum.
//   * batch_shape — (B,) as int64 vector.
//   * dt        — dtype for any constants.
inline MPSGraphTensor* scale_per_sample(
    MPSGraph* g, MPSGraphTensor* grad_out, MPSGraphTensor* w_gather,
    MPSGraphTensor* keep_f, MPSGraphTensor* denom,
    std::int64_t reduction, MPSDataType dt) {
    // base = -1 * keep * (w_gather or 1)
    MPSGraphTensor* neg_one = [g constantWithScalar:-1.0 dataType:dt];
    MPSGraphTensor* s = [g multiplicationWithPrimaryTensor:neg_one
                                            secondaryTensor:keep_f name:nil];
    if (w_gather != nil) {
        s = [g multiplicationWithPrimaryTensor:s
                                secondaryTensor:w_gather name:nil];
    }
    // Multiply by grad_out (broadcasts when grad_out is scalar).
    s = [g multiplicationWithPrimaryTensor:s
                            secondaryTensor:grad_out name:nil];
    // Divide by denom for reduction='mean'.
    if (reduction == 1) {
        s = [g divisionWithPrimaryTensor:s
                          secondaryTensor:denom name:nil];
    }
    return s;
}

// Common entry point: resolves (input, target, optional weight)
// + ignore_index + reduction.  Returns the pieces NLL/CE both need.
struct LossPieces {
    bool ok = false;
    MPSGraph* g = nil;
    MPSGraphTensor* input = nil;       // log_p (NLL) or logits (CE)
    MPSGraphTensor* target = nil;      // (B,), integer
    MPSGraphTensor* w_gather = nil;    // (B,) or nil
    MPSGraphTensor* keep_f = nil;      // (B,) float mask
    MPSGraphTensor* denom = nil;       // scalar, sum(keep_f)
    MPSGraphTensor* grad_out = nil;
    TensorId input_id = -1;
    std::int64_t reduction = 1;
    std::int64_t B = 0;
    std::int64_t C = 0;
};

inline LossPieces unpack_loss(BackwardContext& bctx, const OpNode& node,
                               const std::vector<void*>& grad_outs) {
    LossPieces p;
    if (node.inputs.size() < 2 || grad_outs.empty() || grad_outs[0] == nullptr)
        return p;
    p.input_id = node.inputs[0];
    TensorId tg_id = node.inputs[1];
    if (p.input_id < 0 || tg_id < 0) return p;

    p.reduction = 1;
    if (auto it = node.attrs.find("reduction"); it != node.attrs.end()) {
        if (const auto* x = std::get_if<std::int64_t>(&it->second))
            p.reduction = *x;
    }
    std::int64_t ignore_index = -100;
    if (auto it = node.attrs.find("ignore_index"); it != node.attrs.end()) {
        if (const auto* x = std::get_if<std::int64_t>(&it->second))
            ignore_index = *x;
    }

    p.g = (__bridge MPSGraph*)bctx.graph();
    p.input = as_tensor(bctx.forward(p.input_id));
    p.target = as_tensor(bctx.forward(tg_id));
    if (p.g == nil || p.input == nil || p.target == nil) return p;

    std::vector<std::int64_t> in_shape = shape_of_mps(p.input);
    if (in_shape.size() < 2) return p;
    p.B = in_shape[0];
    p.C = in_shape[1];

    MPSDataType ft = p.input.dataType;
    // Optional class weight at inputs[2].
    if (node.inputs.size() >= 3 && node.inputs[2] >= 0) {
        MPSGraphTensor* w = as_tensor(bctx.forward(node.inputs[2]));
        if (w != nil) {
            p.w_gather =
                [p.g gatherAlongAxis:0
                   withUpdatesTensor:w
                       indicesTensor:p.target
                                name:nil];
        }
    }
    // keep mask.
    MPSGraphTensor* ig_c =
        [p.g constantWithScalar:(double)ignore_index dataType:p.target.dataType];
    MPSGraphTensor* keep_bool =
        [p.g notEqualWithPrimaryTensor:p.target secondaryTensor:ig_c name:nil];
    p.keep_f = [p.g castTensor:keep_bool toType:ft name:nil];
    // denom = sum(keep_f).
    NSArray<NSNumber*>* axes_all =
        @[ [NSNumber numberWithLongLong:0] ];
    p.denom = [p.g reductionSumWithTensor:p.keep_f axes:axes_all name:nil];
    p.grad_out = as_tensor(grad_outs[0]);
    p.ok = true;
    return p;
}

// Emit the sparse "one-hot gradient" at axis 1: a (B, C) zeros tensor
// with ``scale_per_sample[i]`` written at position [i, target[i]].
// Used by both NLL (final dx) and CE (subtracted from softmax * scale).
inline MPSGraphTensor* sparse_onehot_grad(
    MPSGraph* g, MPSGraphTensor* target, MPSGraphTensor* scale,
    std::int64_t B, std::int64_t C, MPSDataType dt) {
    // base = zeros(B, C)
    NSArray<NSNumber*>* base_shape =
        @[ [NSNumber numberWithLongLong:B], [NSNumber numberWithLongLong:C] ];
    MPSGraphTensor* base =
        [g constantWithScalar:0.0 shape:base_shape dataType:dt];
    // indices = reshape(target, [B, 1])
    NSArray<NSNumber*>* idx_shape =
        @[ [NSNumber numberWithLongLong:B], @1 ];
    MPSGraphTensor* indices =
        [g reshapeTensor:target withShape:idx_shape name:nil];
    // updates = reshape(scale, [B, 1])
    MPSGraphTensor* updates =
        [g reshapeTensor:scale withShape:idx_shape name:nil];
    return [g scatterAlongAxis:1
                withDataTensor:base
                 updatesTensor:updates
                 indicesTensor:indices
                          mode:MPSGraphScatterModeAdd
                          name:@"loss_vjp_scatter"];
}

}  // namespace

// ────────────────────────────────────────────────────────────────────
// NLL Loss: d(log_p)[i, c] = -keep[i] * w[t_i] * δ(c==t_i) * grad_scale_i
// ────────────────────────────────────────────────────────────────────
class NllLossVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "nll_loss"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        LossPieces p = unpack_loss(bctx, node, grad_outs);
        if (!p.ok) return false;
        if (![p.g respondsToSelector:@selector(scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:)])
            return false;

        MPSDataType dt = p.input.dataType;
        MPSGraphTensor* scale =
            scale_per_sample(p.g, p.grad_out, p.w_gather, p.keep_f, p.denom,
                              p.reduction, dt);
        MPSGraphTensor* d_input =
            sparse_onehot_grad(p.g, p.target, scale, p.B, p.C, dt);
        bctx.accumulate_grad(p.input_id, from_tensor(d_input));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// Cross-Entropy Loss: d(logits)[i, c] = (softmax(logits)[i, c]
//                                        - δ(c==t_i)) * |scale_i|
// where the sign baked into ``scale_per_sample`` is -1 (NLL's neg);
// the formula expands to:
//     d(logits)[i, c] = -scale_per_sample[i] * softmax[i, c]
//                       + scale_per_sample[i] * δ(c==t_i)
// which is ``-softmax * scale`` (broadcast) plus the sparse term, both
// emitted directly here.
// ────────────────────────────────────────────────────────────────────
class CrossEntropyLossVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "cross_entropy_loss"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        LossPieces p = unpack_loss(bctx, node, grad_outs);
        if (!p.ok) return false;
        if (![p.g respondsToSelector:@selector(scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:)])
            return false;

        MPSDataType dt = p.input.dataType;
        // scale_per_sample carries the -1 factor (NLL convention).
        MPSGraphTensor* scale =
            scale_per_sample(p.g, p.grad_out, p.w_gather, p.keep_f, p.denom,
                              p.reduction, dt);
        // softmax(logits) — recompute (same cost as a saved tensor on
        // MPSGraph; we'd otherwise need to thread a saved id through).
        MPSGraphTensor* y =
            [p.g softMaxWithTensor:p.input axis:1 name:nil];
        // Reshape scale (B,) to (B, 1) for broadcast onto y.
        NSArray<NSNumber*>* scale_col =
            @[ [NSNumber numberWithLongLong:p.B], @1 ];
        MPSGraphTensor* scale_b =
            [p.g reshapeTensor:scale withShape:scale_col name:nil];
        // term1 = -softmax * scale  (broadcast on axis 1)
        // (scale already carries the -1; multiply by softmax to get
        // the dense part of the gradient.)
        // Note: scale = -|absolute scale|, so softmax * scale = -softmax * |scale|.
        // We need dx = -scale * y - scale * (-onehot) = -scale*y + scale*onehot
        //            = scale * (onehot - y), or equivalently |scale|*(y - onehot).
        // Since scale is already negative, term1 = y * scale handles the
        // ``-softmax * |scale|`` part directly.  Then the sparse +scale at the
        // target index handles the ``+|scale|*onehot`` ... but wait, we want
        // ``-scale*y + scale*onehot``.  scale is negative, so:
        //   first_term = scale * y           (negative dense)
        //   second_term = +(-scale)*onehot   = positive scatter of (-scale).
        // Equivalent: dx = scale*y - sparse_onehot(scale).
        MPSGraphTensor* dense_term =
            [p.g multiplicationWithPrimaryTensor:y
                                  secondaryTensor:scale_b name:nil];
        MPSGraphTensor* sparse_term =
            sparse_onehot_grad(p.g, p.target, scale, p.B, p.C, dt);
        MPSGraphTensor* d_input =
            [p.g subtractionWithPrimaryTensor:dense_term
                              secondaryTensor:sparse_term
                                         name:@"ce_vjp"];
        bctx.accumulate_grad(p.input_id, from_tensor(d_input));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// BCE Loss VJP
//
// Forward (see :file:`OpEmitters/nn/Loss.mm`):
//   xc        = clip(x, eps, 1-eps)
//   per_sample = -[t * log(xc) + (1-t) * log(1-xc)] * w
//   out       = reduce(per_sample, reduction)
//
// Backward (using the unclipped x for grad — clipping is a forward
// numerical-safety detail; the eager backend's bce_loss_backward
// uses the same convention):
//   dx = (x - t) / (x*(1-x) + eps) * w * scale_per_sample
//   where scale_per_sample is 1 for `none`, 1/N for `mean`, 1 for `sum`
//   multiplied by grad_out (broadcasting for scalar grad).
//   target / weight: no grad emitted (typically non-learnable).
// ────────────────────────────────────────────────────────────────────
class BceLossVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "bce_loss"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() < 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        TensorId w_id = node.inputs[2];
        if (in_id < 0 || tg_id < 0 || w_id < 0) return false;
        std::int64_t reduction = 1;
        if (auto it = node.attrs.find("reduction"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second))
                reduction = *p;
        }
        double eps = 1e-12;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(in_id));
        MPSGraphTensor* t = as_tensor(bctx.forward(tg_id));
        MPSGraphTensor* w = as_tensor(bctx.forward(w_id));
        if (g == nil || grad == nil || x == nil || t == nil || w == nil)
            return false;

        MPSDataType dt = x.dataType;
        MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
        MPSGraphTensor* eps_t = [g constantWithScalar:eps dataType:dt];
        // x*(1-x) + eps
        MPSGraphTensor* one_m_x =
            [g subtractionWithPrimaryTensor:one secondaryTensor:x name:nil];
        MPSGraphTensor* x_omx =
            [g multiplicationWithPrimaryTensor:x secondaryTensor:one_m_x name:nil];
        MPSGraphTensor* denom =
            [g additionWithPrimaryTensor:x_omx secondaryTensor:eps_t name:nil];
        // (x - t)
        MPSGraphTensor* x_minus_t =
            [g subtractionWithPrimaryTensor:x secondaryTensor:t name:nil];
        // base = (x - t) / denom * w
        MPSGraphTensor* ratio =
            [g divisionWithPrimaryTensor:x_minus_t secondaryTensor:denom name:nil];
        MPSGraphTensor* base =
            [g multiplicationWithPrimaryTensor:ratio secondaryTensor:w name:nil];

        // Multiply by grad_out (broadcasts for scalar grad of sum/mean).
        MPSGraphTensor* dx_unred =
            [g multiplicationWithPrimaryTensor:base secondaryTensor:grad name:nil];

        // For mean reduction, divide by numel(x).
        if (reduction == 1) {
            std::vector<std::int64_t> x_shape = shape_of_mps(x);
            double N = 1.0;
            for (std::int64_t d : x_shape) N *= (double)d;
            MPSGraphTensor* inv_N =
                [g constantWithScalar:(1.0 / N) dataType:dt];
            dx_unred =
                [g multiplicationWithPrimaryTensor:dx_unred
                                    secondaryTensor:inv_N name:@"bce_vjp_dx"];
        }
        bctx.accumulate_grad(in_id, from_tensor(dx_unred));
        // target / weight: no grad emitted.
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// BCE-with-logits Loss VJP
//
// Forward (in stable form):
//   s = sigmoid(x)
//   per_sample = -[t * log(s) + (1-t) * log(1-s)] * factor * w
//   factor = (pw - 1) * t + 1   (pos_weight factor)
//
// Backward (closed form):
//   dx = (s - t) * factor * w * scale_per_sample
//   where s = sigmoid(x) is recomputed.
// ────────────────────────────────────────────────────────────────────
class BceWithLogitsVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "bce_with_logits"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() < 4 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        TensorId w_id = node.inputs[2];
        TensorId pw_id = node.inputs[3];
        if (in_id < 0 || tg_id < 0 || w_id < 0 || pw_id < 0) return false;
        std::int64_t reduction = 1;
        if (auto it = node.attrs.find("reduction"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second))
                reduction = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(in_id));
        MPSGraphTensor* t = as_tensor(bctx.forward(tg_id));
        MPSGraphTensor* w = as_tensor(bctx.forward(w_id));
        MPSGraphTensor* pw = as_tensor(bctx.forward(pw_id));
        if (g == nil || grad == nil || x == nil || t == nil || w == nil || pw == nil)
            return false;

        MPSDataType dt = x.dataType;
        MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
        MPSGraphTensor* s = [g sigmoidWithTensor:x name:nil];
        // factor = (pw - 1) * t + 1
        MPSGraphTensor* pw_m1 =
            [g subtractionWithPrimaryTensor:pw secondaryTensor:one name:nil];
        MPSGraphTensor* pw_m1_t =
            [g multiplicationWithPrimaryTensor:pw_m1 secondaryTensor:t name:nil];
        MPSGraphTensor* factor =
            [g additionWithPrimaryTensor:pw_m1_t secondaryTensor:one name:nil];
        // base = (s - t) * factor * w
        MPSGraphTensor* s_m_t =
            [g subtractionWithPrimaryTensor:s secondaryTensor:t name:nil];
        MPSGraphTensor* s_m_t_factor =
            [g multiplicationWithPrimaryTensor:s_m_t secondaryTensor:factor name:nil];
        MPSGraphTensor* base =
            [g multiplicationWithPrimaryTensor:s_m_t_factor secondaryTensor:w name:nil];
        // Multiply by grad_out.
        MPSGraphTensor* dx_unred =
            [g multiplicationWithPrimaryTensor:base secondaryTensor:grad name:nil];

        if (reduction == 1) {
            std::vector<std::int64_t> x_shape = shape_of_mps(x);
            double N = 1.0;
            for (std::int64_t d : x_shape) N *= (double)d;
            MPSGraphTensor* inv_N =
                [g constantWithScalar:(1.0 / N) dataType:dt];
            dx_unred =
                [g multiplicationWithPrimaryTensor:dx_unred
                                    secondaryTensor:inv_N
                                               name:@"bce_logits_vjp_dx"];
        }
        bctx.accumulate_grad(in_id, from_tensor(dx_unred));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// Huber Loss VJP
//
// Forward: per_sample = |diff|<delta ? 0.5*diff² : delta*(|diff|-0.5*delta)
// Backward:
//   d(per_sample)/dx = |diff|<delta ? diff : delta * sign(diff)
//   d(per_sample)/dt = -d(per_sample)/dx
// scale by grad_out, then by 1/N for mean reduction.
// ────────────────────────────────────────────────────────────────────
class HuberLossVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "huber_loss"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() < 2 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0) return false;
        double delta = 1.0;
        if (auto it = node.attrs.find("delta"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) delta = *p;
        }
        std::int64_t reduction = 1;
        if (auto it = node.attrs.find("reduction"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second))
                reduction = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(in_id));
        MPSGraphTensor* t = as_tensor(bctx.forward(tg_id));
        if (g == nil || grad == nil || x == nil || t == nil) return false;

        MPSDataType dt = x.dataType;
        MPSGraphTensor* delta_t = [g constantWithScalar:delta dataType:dt];
        MPSGraphTensor* diff =
            [g subtractionWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* abs_d = [g absoluteWithTensor:diff name:nil];
        MPSGraphTensor* sign_d = [g signWithTensor:diff name:nil];
        // delta * sign(diff) — large branch
        MPSGraphTensor* lg =
            [g multiplicationWithPrimaryTensor:delta_t secondaryTensor:sign_d name:nil];
        // |diff| < delta
        MPSGraphTensor* mask =
            [g lessThanWithPrimaryTensor:abs_d secondaryTensor:delta_t name:nil];
        // base = mask ? diff : delta * sign(diff)
        MPSGraphTensor* base =
            [g selectWithPredicateTensor:mask
                     truePredicateTensor:diff
                    falsePredicateTensor:lg
                                    name:nil];
        MPSGraphTensor* dx_unred =
            [g multiplicationWithPrimaryTensor:base secondaryTensor:grad name:nil];
        if (reduction == 1) {
            std::vector<std::int64_t> x_shape = shape_of_mps(x);
            double N = 1.0;
            for (std::int64_t d : x_shape) N *= (double)d;
            MPSGraphTensor* inv_N =
                [g constantWithScalar:(1.0 / N) dataType:dt];
            dx_unred =
                [g multiplicationWithPrimaryTensor:dx_unred
                                    secondaryTensor:inv_N
                                               name:@"huber_vjp_dx"];
        }
        MPSGraphTensor* dt_unred = [g negativeWithTensor:dx_unred name:@"huber_vjp_dt"];
        bctx.accumulate_grad(in_id, from_tensor(dx_unred));
        bctx.accumulate_grad(tg_id, from_tensor(dt_unred));
        return true;
    }
};

struct LossVjpRegistrar {
    LossVjpRegistrar() {
        register_vjp_emitter(std::make_unique<MseLossVjp>());
        register_vjp_emitter(std::make_unique<NllLossVjp>());
        register_vjp_emitter(std::make_unique<CrossEntropyLossVjp>());
        register_vjp_emitter(std::make_unique<BceLossVjp>());
        register_vjp_emitter(std::make_unique<BceWithLogitsVjp>());
        register_vjp_emitter(std::make_unique<HuberLossVjp>());
    }
};

[[maybe_unused]] static const LossVjpRegistrar g_loss_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
