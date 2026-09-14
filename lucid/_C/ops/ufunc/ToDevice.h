// lucid/_C/ops/ufunc/ToDevice.h
//
// Differentiable device transfer.  Moving a tensor between the CPU and the
// GPU changes where the numbers live, not what they are, so its derivative
// is the identity — carried back to the device the input came from.
// ``Tensor.to`` used to build a fresh leaf instead, and backward stopped at
// the move: the source of anything moved to Metal and back never received a
// gradient, and nothing said so.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/Device.h"
#include "../../core/OpSchema.h"
#include "../../core/fwd.h"
#include "../../kernel/IKernel.h"

namespace lucid {

// Backward node for :func:`to_device_op`.
//
// Returns the incoming gradient on the device the forward input lived on.
//
// Attributes
// ----------
// src_device_ : Device
//     Device of the forward input; the gradient is moved back there.
class LUCID_API ToDeviceBackward : public FuncOp<ToDeviceBackward, 1>, public kernel::IKernel {
public:
    static const OpSchema schema_v1;
    Device src_device_ = Device::CPU;

    std::string_view name() const noexcept override { return schema_v1.name; }
    std::string node_name() const override { return std::string(schema_v1.name); }

    // Eager backward: the gradient's storage, moved to ``src_device_``.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward: :func:`to_device_op` again, so a second
    // derivative crosses the devices through the same differentiable op.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// ``a`` on ``target``, differentiable.
//
// A tensor over a Metal shared buffer is relabelled — an alias of the same
// bytes; anything else is copied by the engine.  When grad mode is on and
// ``a`` requires grad, the result carries a :class:`ToDeviceBackward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor to move.
// target : Device
//     Destination device.
//
// Returns
// -------
// TensorImplPtr
//     ``a`` itself when it is already on ``target``; otherwise ``a`` on
//     ``target``.
LUCID_API TensorImplPtr to_device_op(const TensorImplPtr& a, Device target);

}  // namespace lucid
