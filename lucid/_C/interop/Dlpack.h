// lucid/_C/interop/Dlpack.h
//
// DLPack export for tensors that live on Metal.
//
// Why this exists alongside the NumPy-backed DLPack in
// ``lucid/_factories/converters.py``: NumPy's implementation can only
// ever describe host memory, so a Metal tensor handed to any consumer
// through it is downloaded to the CPU first.  Between Lucid and MLX that
// download is pure waste — both sit on the same unified memory, and
// Lucid's GPU storage *is* an ``mlx::core::array``.
//
// Measured (2026-09-01, mlx 0.32): MLX tags its own DLPack capsules
// ``kDLMetal`` (device type 8) and its consumer shares pages rather than
// copying, while NumPy rejects such a capsule outright with
// ``BufferError: Unsupported device``.  So the two frameworks already
// agree on a zero-copy wire format for Metal; Lucid was the one only able
// to speak the CPU dialect.  See ``obsidian/architecture/
// arch-mlx-zero-copy-bridge-spike.md``.
//
// The CPU path deliberately keeps going through NumPy — that decision
// (``arch-dlpack-via-numpy``) is still right for host consumers.  This
// header adds the Metal dialect only.
//
// Wire format
// -----------
// The structs below are the DLPack ABI (the unversioned
// ``DLManagedTensor`` set, capsule name ``"dltensor"``), declared here
// rather than vendored so the engine keeps no third-party headers.  They
// are layout-compatible by definition — the ABI is fixed by the spec, and
// the field order was verified byte-for-byte against a capsule MLX
// produced before this file was written.
//
// For ``kDLMetal``, ``data`` is the ``id<MTLBuffer>`` handle, **not** a
// host pointer — MLX's own export was read to confirm it (its ``data``
// differs from the buffer's ``contents``).  Getting that wrong yields a
// consumer reading an arbitrary address, so it is asserted by test.

#pragma once

#include <cstdint>

#include "../api.h"
#include "../core/fwd.h"

namespace lucid::interop {

// Subset of ``DLDeviceType`` this engine can produce or accept.
enum : std::int32_t {
    kDLCPU = 1,
    kDLMetal = 8,
};

// Subset of ``DLDataTypeCode``.
enum : std::uint8_t {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
};

struct DLDevice {
    std::int32_t device_type;
    std::int32_t device_id;
};

struct DLDataType {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

struct DLTensor {
    void* data;
    DLDevice device;
    std::int32_t ndim;
    DLDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};

struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor* self);
};

// Export a Metal-resident tensor as a ``kDLMetal`` DLPack tensor.
//
// The returned block is heap-allocated and owns a reference to the
// source ``mlx::core::array``, so the buffer stays alive for exactly as
// long as the capsule does regardless of what happens to the Lucid tensor
// meanwhile.  Ownership passes to the caller, who must either invoke
// ``deleter`` or hand the block to a DLPack consumer that will.
//
// Parameters
// ----------
// impl : const TensorImplPtr&
//     Tensor to export.  Must be on ``Device::GPU``.
//
// Returns
// -------
// DLManagedTensor*
//     Never null; failures raise instead.
//
// Raises
// ------
// std::invalid_argument
//     ``impl`` is null, lives on the CPU, or has a dtype with no DLPack
//     spelling on Metal (``float64`` among them — Metal has no such
//     type, so there is nothing to point a consumer at).
LUCID_API DLManagedTensor* metal_to_dlpack(const TensorImplPtr& impl);

// Adopt a ``kDLMetal`` DLPack tensor as a Lucid GPU tensor.
//
// Wraps the incoming ``MTLBuffer`` in a fresh ``mlx::core::array``
// without copying.  Per the DLPack contract the consumer owns ``managed``
// from this point: its ``deleter`` is called when the last reference to
// the resulting tensor's storage dies, not before, so the producer's
// buffer outlives every view Lucid hands out.
//
// Parameters
// ----------
// managed : DLManagedTensor*
//     Tensor to adopt.  Must be ``kDLMetal`` and contiguous.
//
// Returns
// -------
// TensorImplPtr
//     A leaf tensor sharing the producer's pages.
//
// Raises
// ------
// std::invalid_argument
//     Wrong device, unsupported dtype, or non-contiguous strides.  The
//     ``deleter`` is invoked before raising so the producer is not
//     leaked by a rejected import.
LUCID_API TensorImplPtr dlpack_to_metal(DLManagedTensor* managed);

}  // namespace lucid::interop
