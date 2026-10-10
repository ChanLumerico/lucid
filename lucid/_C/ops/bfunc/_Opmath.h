// lucid/_C/ops/bfunc/_Opmath.h
//
// Op-math for a half tensor and a wider 0-d operand.
//
// ``mul_op``, ``div_op`` and ``floordiv_op`` accept one mixed pair that
// BinaryKernel refuses: a float16 / bfloat16 tensor ``T`` of any shape and a
// 0-d float32 operand ``S`` (float64 too on the CPU stream, which is the only
// stream that holds it), in either order.  ``S`` is read at its own width,
// the op runs in float32 and the result is rounded once into ``T``'s dtype.
// Casting ``S`` to the half dtype first — all a same-dtype pair can do — is
// what made ``ones(f16) / 1e5`` zero (1e-5 underflows float16),
// ``full(1e-3, f16) * 1e5`` inf (1e5 overflows it) and ``bf16(3.09375) * 0.1``
// 0.3105 instead of 0.3086.  Only these three ops: the reference applies
// op-math to a scalar operand of mul, div and floor_divide and rounds it first
// for every other binary op, which Lucid already does.
//
// Floor division is ``floor`` of the float32 quotient, rounded once — the
// same rule as every other floating-point ``//`` in Lucid (``floor_op`` over
// ``div_op``).
//
// The 0-d operand must not require grad: its gradient would be a reduction
// over ``T`` in float32, which nothing sends yet.  Under autocast, when the
// policy moves the half dtype elsewhere, both operands are cast to the
// autocast dtype and the ordinary same-dtype op runs.
//
// The Metal stream fuses widen, op and narrow into one kernel with the 0-d
// operand as a kernel input.  The old route through a same-dtype pair spent
// a pass materialising the broadcast 0-d before the multiply.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/HalfAccumulation.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Device.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpSchema.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Scope.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Arith.h"
#include "../ufunc/Astype.h"
#include "Div.h"
#include "Mul.h"

namespace lucid::opmath {

enum class Op : std::uint8_t { Mul, Div, FloorDiv };

constexpr std::string_view op_name(Op op) {
    switch (op) {
    case Op::Mul:
        return "mul";
    case Op::Div:
        return "div";
    case Op::FloorDiv:
        return "floordiv";
    }
    return "mul";
}

// The schema whose AMP policy decides the op's effective dtype.  Floor
// division of floats is ``floor(div)``, so it follows ``div``.
inline const OpSchema& amp_schema(Op op) {
    return op == Op::Mul ? MulBackward::schema_v1 : DivBackward::schema_v1;
}

struct Pair {
    TensorImplPtr t;
    TensorImplPtr s;
    bool scalar_first;
};

inline bool is_wide_scalar(const TensorImplPtr& s) {
    return s->shape().empty() &&
           (s->dtype() == Dtype::F32 || (s->dtype() == Dtype::F64 && s->device() == Device::CPU));
}

// ``(a, b)`` as an op-math pair, or nothing — every other mixed pair is left
// to BinaryKernel, which rejects it as before.
inline std::optional<Pair> match(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b || a->device() != b->device() || a->dtype() == b->dtype())
        return std::nullopt;
    if (is_half_float(a->dtype()) && is_wide_scalar(b))
        return Pair{a, b, false};
    if (is_half_float(b->dtype()) && is_wide_scalar(a))
        return Pair{b, a, true};
    return std::nullopt;
}

namespace gpu_fn {

namespace mx = ::mlx::core;

// ``in = {T, S}``; the result in ``T``'s dtype.  One type per (op, order):
// ``mlx_fused`` keeps one compiled kernel per functor type.
template <Op K, bool ScalarFirst>
struct Forward {
    std::vector<mx::array> operator()(const std::vector<mx::array>& in) const {
        const mx::array t = backend::widen_half(in[0]);
        const mx::array& s = in[1];
        mx::array r = K == Op::Mul  ? mx::multiply(t, s)
                      : ScalarFirst ? mx::divide(s, t)
                                    : mx::divide(t, s);
        if constexpr (K == Op::FloorDiv)
            r = mx::floor(r);
        return {backend::narrow_to(r, in[0].dtype())};
    }
};

// ``in = {g, T, S}``: the gradient of ``S / T`` with respect to ``T``,
// ``-(S * g) / T^2``, in float32 and rounded once into ``g``'s dtype.
struct ReciprocalGrad {
    std::vector<mx::array> operator()(const std::vector<mx::array>& in) const {
        const mx::array t = backend::widen_half(in[1]);
        const mx::array r = mx::negative(
            mx::divide(mx::multiply(in[2], backend::widen_half(in[0])), mx::square(t)));
        return {backend::narrow_to(r, in[0].dtype())};
    }
};

inline mx::array fused(Op op, bool scalar_first, const std::vector<mx::array>& ins) {
    switch (op) {
    case Op::Mul:
        return backend::mlx_fused(ins, Forward<Op::Mul, false>{})[0];
    case Op::Div:
        return scalar_first ? backend::mlx_fused(ins, Forward<Op::Div, true>{})[0]
                            : backend::mlx_fused(ins, Forward<Op::Div, false>{})[0];
    case Op::FloorDiv:
        return scalar_first ? backend::mlx_fused(ins, Forward<Op::FloorDiv, true>{})[0]
                            : backend::mlx_fused(ins, Forward<Op::FloorDiv, false>{})[0];
    }
    return backend::mlx_fused(ins, Forward<Op::Mul, false>{})[0];
}

inline Storage wrap(mx::array r, Dtype dt) {
    // A strided ``T`` gives a strided result; everything downstream of a
    // Storage assumes row-major bytes (engine-mlx-data-ignores-strides).
    return Storage{gpu::wrap_mlx_array(mx::contiguous(r), dt)};
}

inline const mx::array& arr(const Storage& s) {
    return *std::get<GpuStorage>(s).arr;
}

}  // namespace gpu_fn

inline double read_cpu_scalar(const Storage& s, Dtype dt) {
    const auto& c = std::get<CpuStorage>(s);
    return dt == Dtype::F64 ? *reinterpret_cast<const double*>(c.ptr.get())
                            : static_cast<double>(*reinterpret_cast<const float*>(c.ptr.get()));
}

// ``x op s`` (or ``s op x``) for ``x`` of dtype ``h`` and shape ``shape`` and
// a 0-d ``s`` of dtype ``s_dt``: computed in float32, rounded once to ``h``.
inline Storage compute(Op op,
                       bool scalar_first,
                       const Storage& x,
                       const Shape& shape,
                       Dtype h,
                       const Storage& s,
                       Dtype s_dt,
                       Device device) {
    if (device == Device::GPU) {
        return gpu_fn::wrap(gpu_fn::fused(op, scalar_first, {gpu_fn::arr(x), gpu_fn::arr(s)}), h);
    }
    auto& be = backend::Dispatcher::for_device(Device::CPU);
    // The half ``mul_scalar`` already widens, runs vsmul with float(s) and
    // rounds once.
    if (op == Op::Mul)
        return be.mul_scalar(x, shape, h, read_cpu_scalar(s, s_dt));
    // Accelerate has a vector-by-vector division and the backend nothing
    // narrower, so the 0-d is spread to ``shape`` first.  Each lane is still
    // one IEEE float32 division of float(T) by float(S).
    const Storage x32 = be.astype(x, shape, h, Dtype::F32);
    const Storage s32 = s_dt == Dtype::F32 ? s : be.astype(s, Shape{}, s_dt, Dtype::F32);
    const Storage s_full{
        detail::broadcast_cpu(std::get<CpuStorage>(s32), Shape{}, shape, Dtype::F32)};
    Storage q = scalar_first ? be.div(s_full, x32, shape, Dtype::F32)
                             : be.div(x32, s_full, shape, Dtype::F32);
    if (op == Op::FloorDiv)
        q = be.floor(q, shape, Dtype::F32);
    return be.astype(q, shape, Dtype::F32, h);
}

// ``-(S * g) / T^2`` in float32, rounded once to ``h``.
inline Storage reciprocal_grad(const Storage& g,
                               const Storage& t,
                               const Shape& shape,
                               Dtype h,
                               const Storage& s,
                               Dtype s_dt,
                               Device device) {
    if (device == Device::GPU) {
        return gpu_fn::wrap(backend::mlx_fused({gpu_fn::arr(g), gpu_fn::arr(t), gpu_fn::arr(s)},
                                               gpu_fn::ReciprocalGrad{})[0],
                            h);
    }
    auto& be = backend::Dispatcher::for_device(Device::CPU);
    const Storage g32 = be.astype(g, shape, h, Dtype::F32);
    const Storage t32 = be.astype(t, shape, h, Dtype::F32);
    const Storage t_sq = be.mul(t32, t32, shape, Dtype::F32);
    const Storage neg_sg = be.mul_scalar(g32, shape, Dtype::F32, -read_cpu_scalar(s, s_dt));
    return be.astype(be.div(neg_sg, t_sq, shape, Dtype::F32), shape, Dtype::F32, h);
}

// One node per op.  Slot ``t_index_`` is ``T``; the other slot is the 0-d,
// which never requires grad and so gets no gradient.
template <Op K>
class ScalarBackward : public kernel::NaryKernel<ScalarBackward<K>, 2> {
public:
    static inline const OpSchema schema_v1{op_name(K), 1, AmpPolicy::Promote, true};

    std::size_t t_index_ = 0;
    Dtype s_dtype_ = Dtype::F32;

    std::string node_name() const override { return std::string(schema_v1.name); }

    std::vector<Storage> apply(Storage grad_out) override {
        std::vector<Storage> grads(2, Storage{CpuStorage{}});
        grads[t_index_] = grad_t(grad_out);
        return grads;
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const TensorImplPtr& t = this->saved_impl_inputs_[t_index_];
        const TensorImplPtr& s = this->saved_impl_inputs_[1 - t_index_];
        std::vector<TensorImplPtr> grads(2);
        if constexpr (K == Op::Mul) {
            grads[t_index_] = mul_op(grad_out, s);
        } else if constexpr (K == Op::Div) {
            grads[t_index_] = scalar_first() ? neg_op(div_op(mul_op(grad_out, s), mul_op(t, t)))
                                             : div_op(grad_out, s);
        } else {
            grads[t_index_] = zeros_like_op(t);
        }
        return grads;
    }

private:
    bool scalar_first() const { return t_index_ == 1; }

    Storage grad_t(const Storage& g) const {
        const Storage& s = this->saved_inputs_[1 - t_index_];
        if constexpr (K == Op::Mul) {
            return compute(Op::Mul, false, g, this->out_shape_, this->dtype_, s, s_dtype_,
                           this->device_);
        } else if constexpr (K == Op::Div) {
            if (scalar_first())
                return reciprocal_grad(g, this->saved_inputs_[t_index_], this->out_shape_,
                                       this->dtype_, s, s_dtype_, this->device_);
            return compute(Op::Div, false, g, this->out_shape_, this->dtype_, s, s_dtype_,
                           this->device_);
        } else {
            return make_zero_storage(this->out_shape_, this->dtype_, this->device_);
        }
    }
};

template <Op K>
TensorImplPtr forward(const Pair& p, const TensorImplPtr& a, const TensorImplPtr& b) {
    const Dtype h = p.t->dtype();
    const Device device = p.t->device();
    const Shape& shape = p.t->shape();
    OpScopeFull scope{op_name(K), device, h, shape};
    Storage out_storage =
        compute(K, p.scalar_first, p.t->storage(), shape, h, p.s->storage(), p.s->dtype(), device);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), shape, h, device, false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    auto bwd = std::make_shared<ScalarBackward<K>>();
    bwd->t_index_ = p.scalar_first ? 1 : 0;
    bwd->s_dtype_ = p.s->dtype();
    ScalarBackward<K>* node = bwd.get();
    // wire_autograd takes the node's dtype from the first input, which is
    // the 0-d when it leads; the gradient lives in the half dtype.
    if (kernel::NaryKernel<ScalarBackward<K>, 2>::wire_autograd(std::move(bwd), {a, b}, out))
        node->dtype_ = h;
    return out;
}

using SameDtypeOp = TensorImplPtr (*)(const TensorImplPtr&, const TensorImplPtr&);

// The op-math result for an op-math pair, or null for any other pair.
// ``same_dtype`` is the public op, reused when autocast moves the dtype.
template <Op K>
TensorImplPtr try_scalar(const TensorImplPtr& a, const TensorImplPtr& b, SameDtypeOp same_dtype) {
    const std::optional<Pair> p = match(a, b);
    if (!p)
        return nullptr;
    if (p->s->requires_grad())
        ErrorBuilder(std::string(op_name(K))).fail("op-math scalar operand cannot require grad");
    const SchemaGuard sg{amp_schema(K), p->t->dtype(), p->t->device()};
    const Dtype eff = sg.effective_dtype();
    if (eff != p->t->dtype())
        return same_dtype(astype_op(a, eff), astype_op(b, eff));
    return forward<K>(*p, a, b);
}

}  // namespace lucid::opmath
