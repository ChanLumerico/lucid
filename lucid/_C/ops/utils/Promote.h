// lucid/_C/ops/utils/Promote.h
//
// One line of the kernel templates, for the ops that do not use them, and
// the autocast entry of an engine composite.

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/SchemaGuard.h"
#include "../../core/TensorImpl.h"
#include "../ufunc/Astype.h"

namespace lucid {

// Cast an input to the dtype its schema asks for.
//
// ``UnaryKernel``, ``BinaryKernel`` and ``ReduceKernel`` each open with a
// :class:`SchemaGuard` and cast to ``effective_dtype()``, so every op
// built on them inherits both the autocast policy and the integer
// promotion that goes with ``real_valued`` / ``ForceFP32``.  An op that
// assembles its own forward — softmax, variance, the activations that
// need an extra term — has to ask for the same thing, and the ones that
// did not computed in the input's own dtype.
//
// For an integer input that is not an approximation of the answer, it is
// a different answer: ``softmax`` of an int32 tensor came back
// ``[0, 0, 0]`` on Metal and ``var`` came back ``0``, both integer-typed
// and both silently wrong, while the CPU raised ``NotImplementedError``
// for the same call.  Every one of those schemas already said the result
// was float; nothing was reading it.
//
// ``astype_op`` rather than ``maybe_cast_for_kernel``, for the reason
// ``BatchNorm`` gives: the cast tensor carries an ``AstypeBackward``, so
// a float input cast under an autocast scope keeps its place in the
// graph instead of silently detaching.
//
// Parameters
// ----------
// schema : const OpSchema&
//     The op's schema — supplies the AMP policy and ``real_valued``.
// a : const TensorImplPtr&
//     The input.
//
// Returns
// -------
// TensorImplPtr
//     ``a`` unchanged when it is already at the effective dtype,
//     otherwise a cast copy.
inline TensorImplPtr promote_for_schema(const OpSchema& schema, const TensorImplPtr& a) {
    if (!a)
        return a;
    SchemaGuard sg{schema, a->dtype(), a->device()};
    return astype_op(a, sg.effective_dtype());
}

// The autocast entry of an engine composite — an op with no schema of its
// own, built from other ``*_op`` calls.
//
// The composite's row in the policy table is applied once, here, to its
// inputs; autocast is then off on the device until the scope ends, as
// :class:`SchemaGuard` does for a primitive.  Without it each primitive
// inside applied its own row: ``log1p`` lowered its ``add`` to float16 and
// raised its ``log`` to float32, then refused to combine the two
// (``DtypeMismatch``), and ``log10`` came out in float16 because its last
// step is a ``div``.
//
// Only float16, bfloat16 and float32 inputs are cast; float64, complex and
// integer inputs pass through.  Outside an autocast scope nothing is cast
// and nothing is turned off.
//
// Examples
// --------
// .. code-block:: cpp
//
//     TensorImplPtr log10_op(const TensorImplPtr& a_in) {
//         const AmpScope amp{"log10", a_in->device()};
//         const TensorImplPtr a = amp.cast(a_in);
//         ...
class AmpScope {
public:
    // Resolves ``name``'s row for ``device`` and, inside an autocast scope,
    // turns autocast off on ``device`` until the scope is destroyed.
    AmpScope(std::string_view name, Device device)
        : device_(device),
          policy_(amp::policy_for(name, device)),
          autocast_dt_(amp::active_dtype(device)) {
        if (autocast_dt_)
            autocast_off_.emplace(*autocast_dt_, device, /*enabled=*/false);
    }

    AmpScope(const AmpScope&) = delete;
    AmpScope& operator=(const AmpScope&) = delete;

    // ``t`` at the dtype the row gives one input.  ``PromoteWidest`` leaves
    // a lone input alone — use :meth:`cast_widest` for an op's operands.
    TensorImplPtr cast(const TensorImplPtr& t) const {
        if (!t || !autocast_dt_ || !amp::is_autocast_eligible(t->dtype()))
            return t;
        return astype_op(t, amp::policy_dtype(policy_, t->dtype(), *autocast_dt_, device_));
    }

    // The operands of one op, cast together.  Under ``PromoteWidest`` every
    // eligible operand goes to the widest eligible dtype among them —
    // float16 with bfloat16 is float32, as the reference promotes them;
    // under any other policy each is :meth:`cast` on its own.
    std::vector<TensorImplPtr> cast_widest(std::vector<TensorImplPtr> ts) const {
        if (!autocast_dt_)
            return ts;
        if (policy_ != AmpPolicy::PromoteWidest) {
            for (auto& t : ts)
                t = cast(t);
            return ts;
        }
        std::optional<Dtype> widest;
        for (const auto& t : ts) {
            if (!t || !amp::is_autocast_eligible(t->dtype()))
                continue;
            widest = (!widest || *widest == t->dtype()) ? t->dtype() : Dtype::F32;
        }
        if (!widest)
            return ts;
        for (auto& t : ts) {
            if (t && amp::is_autocast_eligible(t->dtype()))
                t = astype_op(t, *widest);
        }
        return ts;
    }

    // The row's policy on this scope's device (``KeepInput`` for an op
    // with no row), whether or not autocast is active.
    AmpPolicy policy() const noexcept { return policy_; }

private:
    Device device_;
    AmpPolicy policy_;
    // The autocast dtype at construction, before the scope turned it off.
    std::optional<Dtype> autocast_dt_;
    std::optional<amp::AutocastGuard> autocast_off_;
};

}  // namespace lucid
