// lucid/_C/autograd/AccumulateGrad.cpp
//
// Implements AccumulateGrad::apply(), which is the final gradient sink for
// every leaf tensor that participates in a backward pass.

#include "AccumulateGrad.h"

#include <utility>

#include "../backend/Dispatcher.h"
#include "../core/Storage.h"
#include "Helpers.h"
#include "TensorHooks.h"

namespace lucid {

AccumulateGrad::AccumulateGrad(std::weak_ptr<TensorImpl> leaf) : leaf_(std::move(leaf)) {}

// Write grad_out into the leaf's gradient storage, or drop it when the leaf
// TensorImpl has been destroyed (weak_ptr expired) — there is nothing to
// accumulate into.  Returns an empty vector because AccumulateGrad has no
// outgoing edges.
std::vector<Storage> AccumulateGrad::apply(Storage grad_out) {
    if (auto t = leaf_.lock())
        accumulate_leaf(*t, std::move(grad_out));
    return {};
}

// Store grad_out (a TensorImplPtr with its own grad_fn) into the leaf's
// grad_impl slot so the gradient tensor itself is differentiable.
// This path is taken when Engine::backward is called with create_graph=true.
std::vector<TensorImplPtr> AccumulateGrad::apply_for_graph(const TensorImplPtr& grad_out) {
    if (auto t = leaf_.lock())
        accumulate_leaf_for_graph(t, grad_out);
    return {};
}

// Three cases:
//   1. The leaf no longer requires a gradient (e.g. the user called
//      requires_grad_(False) after the forward pass) — discard.
//   2. Its hooks run on the gradient, cast to its dtype, and may replace it.
//   3. If the leaf has no gradient yet, move the gradient in as-is;
//      otherwise call accumulate_into() which does an in-place += using the
//      appropriate backend (CPU element-wise loop or MLX add for GPU).  A
//      gradient a hook saw is a copy of its own (run_leaf_hooks), so the +=
//      of a later pass never reaches a tensor the hook kept.
void accumulate_leaf(TensorImpl& leaf, Storage grad_out) {
    if (!leaf.requires_grad()) {
        return;
    }

    // 3.3 AMP fix: under autocast, the same leaf parameter can be reached
    // via two different effective dtypes — e.g. a Conv with eff_dt=F16
    // emits an F16 grad while a sibling path that ran ForceFP32 emits an
    // F32 grad.  ``accumulate_into`` asserts identical dtype on GPU and
    // would throw DtypeMismatch in that case.  Always cast incoming grads
    // to the leaf parameter's own dtype before storing/accumulating —
    // this matches the reference framework's policy of keeping the
    // gradient slot at the parameter's dtype.
    const Dtype target_dt = leaf.dtype();
    const Dtype src_dt = storage_dtype(grad_out);
    if (src_dt != target_dt) {
        auto& be = backend::Dispatcher::for_device(leaf.device());
        if (is_complex(src_dt) && !is_complex(target_dt)) {
            // A complex gradient arriving at a real leaf keeps its real
            // part.  That is a projection, not a cast, and it is the
            // convention the reference uses: ``fft`` of a real input is
            // complex, so the gradient coming back is too, while
            // ``d/dx`` of a real parameter has to be real.
            //
            // Reached only once the complex projections had backwards —
            // before that no complex gradient ever flowed — and it
            // arrived here as ``astype: complex64 -> float32``, an
            // unimplemented cast standing in for a well-defined
            // operation.
            grad_out = be.complex_real(grad_out, leaf.shape());
            const Dtype lane = real_lane_of(src_dt);
            if (lane != target_dt)
                grad_out = be.astype(grad_out, leaf.shape(), lane, target_dt);
        } else {
            grad_out = be.astype(grad_out, leaf.shape(), src_dt, target_dt);
        }
    }

    grad_out = run_leaf_hooks(leaf, std::move(grad_out));

    auto& grad = leaf.mutable_grad_storage();
    if (!grad.has_value()) {
        // First gradient arriving at this leaf — take ownership directly
        // rather than allocating a zero buffer and immediately adding to it.
        grad = std::move(grad_out);
    } else {
        // Subsequent gradient: add in-place into the existing accumulator.
        accumulate_into(*grad, grad_out);
    }
}

void accumulate_leaf_for_graph(const TensorImplPtr& leaf, const TensorImplPtr& grad) {
    if (!leaf || !leaf->requires_grad()) {
        return;
    }
    leaf->accumulate_grad_impl(run_leaf_hooks_for_graph(*leaf, gradient_in_dtype_of(grad, leaf)));
}

TensorImplPtr gradient_in_dtype_of(const TensorImplPtr& grad, const TensorImplPtr& like) {
    if (!grad || !like || grad->dtype() == like->dtype())
        return grad;
    // The ops layer sits above this one; declared here the way
    // TensorImpl.cpp declares add_op, rather than included.
    extern TensorImplPtr real_op(const TensorImplPtr&);
    extern TensorImplPtr astype_op(const TensorImplPtr&, Dtype);
    TensorImplPtr out = grad;
    if (is_complex(out->dtype()) && !is_complex(like->dtype()))
        out = real_op(out);
    if (out->dtype() != like->dtype())
        out = astype_op(out, like->dtype());
    return out;
}

}  // namespace lucid
