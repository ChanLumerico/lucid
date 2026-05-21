// lucid/_C/tensor/TensorMeta.h
//
// Lightweight value type bundling a tensor's shape, stride, dtype, and
// device without owning any :class:`Storage`.
//
// :class:`TensorMeta` is used in shape-inference code paths — broadcast
// rule resolution, view/reshape planning, output-shape inference for op
// dispatch — where the underlying byte buffer is not needed.  Decoupling
// metadata from storage lets the engine reason about shapes without
// triggering allocation or device synchronisation, which keeps hot
// dispatch paths free of GPU round-trips.
//
// Strides follow the BYTE-OFFSET convention: ``stride[i]`` is the number
// of bytes to advance in memory to step one element along dimension
// ``i``.  A tensor is contiguous when its strides match the reference
// :func:`contiguous_stride` for its shape and element size.
//
// Notes
// -----
// This header is the *tensor-layer* twin of :file:`_C/core/TensorMeta.h`
// — both expose a ``TensorMeta`` struct under ``namespace lucid``, but
// the core version is the one embedded inside :class:`TensorImpl`
// while this one is the standalone descriptor used by shape-inference
// helpers and op signature planners.  Keep both in sync when adding
// fields.
//
// See Also
// --------
// :class:`lucid::core::TensorMeta` — the engine-side twin used during
//     runtime dispatch.
// :func:`contiguous_stride` — reference row-major byte-stride producer.
// :func:`shape_numel`       — element-count helper used by :meth:`numel`.

#pragma once

#include <cstddef>
#include <utility>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"

namespace lucid {

// Pure metadata descriptor for a tensor — bundles shape, stride, dtype,
// and device without owning any :class:`Storage`.
//
// Used wherever the engine needs to reason about a tensor's geometry
// without touching its data: broadcast resolution, view / reshape
// planning, output-shape inference for op dispatch, and signature
// hashing for kernel caches.  Cheap to copy (a handful of small
// ``std::vector`` handles plus two enum ints), so callers pass it by
// value freely.
//
// Strides follow the BYTE-OFFSET convention — ``stride[i]`` is the
// byte delta required to step one element along axis ``i``.  A
// :class:`TensorMeta` is considered contiguous when its strides match
// :func:`contiguous_stride(shape, dtype_size(dtype))`.
//
// Attributes
// ----------
// shape : Shape
//     Per-dimension sizes.  An empty shape denotes a 0-dim scalar.
// stride : Stride
//     Per-dimension byte strides.  Length must equal ``shape.size()``
//     for the layout to be considered well-formed; the default
//     constructor leaves both vectors empty.
// dtype : Dtype, optional
//     Element type tag.  Defaults to :attr:`Dtype::F32`.
// device : Device, optional
//     Memory domain the descriptor refers to (CPU = Accelerate, GPU =
//     MLX).  Defaults to :attr:`Device::CPU`.
//
// Notes
// -----
// All constructors are deliberately non-explicit to allow
// aggregate-style initialisation in shape-inference helpers
// (``TensorMeta out{shape, dtype, device};``).
//
// See Also
// --------
// :class:`lucid::core::TensorMeta` — engine-internal twin embedded in
//     :class:`TensorImpl`.
// :func:`contiguous_stride` — canonical stride producer.
struct TensorMeta {
    Shape shape;
    Stride stride;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    // Default-construct an empty descriptor with zero-length shape and
    // stride, dtype :attr:`F32`, and device :attr:`CPU`.
    //
    // Useful as a placeholder when a :class:`TensorMeta` will be filled
    // in by a later assignment (e.g. inside output-shape inference
    // loops that materialise one meta per op output).
    //
    // Notes
    // -----
    // The default-constructed instance describes a 0-dim scalar of
    // dtype ``F32`` on the CPU; do not assume the shape vector has any
    // particular capacity reserved.
    TensorMeta() = default;

    // Construct a contiguous descriptor; stride is synthesised from
    // ``shape_in`` and the element size of ``dtype_in``.
    //
    // This is the common case for output tensors of element-wise ops —
    // the result inherits the input's logical shape and a freshly
    // computed row-major byte stride.
    //
    // Parameters
    // ----------
    // shape_in : Shape
    //     Logical per-dimension sizes; moved into :attr:`shape`.
    // dtype_in : Dtype
    //     Element type tag; recorded in :attr:`dtype` and also fed to
    //     :func:`dtype_size` to seed the stride computation.
    // device_in : Device
    //     Memory domain the resulting descriptor refers to.
    //
    // Notes
    // -----
    // The stride is computed from the *post-move* ``shape`` member, so
    // the result always matches what :func:`contiguous_stride` would
    // produce for the final stored shape.
    //
    // See Also
    // --------
    // :func:`contiguous_stride` — the underlying stride producer.
    TensorMeta(Shape shape_in, Dtype dtype_in, Device device_in)
        : shape(std::move(shape_in)),
          stride(contiguous_stride(shape, dtype_size(dtype_in))),
          dtype(dtype_in),
          device(device_in) {}

    // Construct a descriptor with explicit byte strides — used for
    // non-contiguous views such as slices, transposes, or
    // ``as_strided`` outputs.
    //
    // Callers are responsible for supplying a stride vector whose
    // length matches ``shape_in.size()`` and whose values describe a
    // valid layout into the parent storage.  No validation is
    // performed here; downstream code (e.g.
    // :meth:`is_contiguous`) interprets the values as given.
    //
    // Parameters
    // ----------
    // shape_in : Shape
    //     Logical per-dimension sizes; moved into :attr:`shape`.
    // stride_in : Stride
    //     Per-dimension byte strides; moved into :attr:`stride`.
    // dtype_in : Dtype
    //     Element type tag recorded in :attr:`dtype`.
    // device_in : Device
    //     Memory domain the resulting descriptor refers to.
    //
    // Notes
    // -----
    // Strides may legitimately be zero (broadcast views) or negative
    // (reversed views); the constructor stores them verbatim.
    TensorMeta(Shape shape_in, Stride stride_in, Dtype dtype_in, Device device_in)
        : shape(std::move(shape_in)),
          stride(std::move(stride_in)),
          dtype(dtype_in),
          device(device_in) {}

    // Total number of logical elements described by :attr:`shape`.
    //
    // Computed as the product of all dimension sizes; a 0-dim scalar
    // (empty ``shape``) returns ``1`` by convention.
    //
    // Returns
    // -------
    // std::size_t
    //     ``shape[0] * shape[1] * ... * shape[N-1]``, or ``1`` for a
    //     scalar.
    //
    // Math
    // ----
    // $$ \mathrm{numel} = \prod_{i=0}^{N-1} \mathrm{shape}_i $$
    //
    // Notes
    // -----
    // Delegates to :func:`shape_numel` from :file:`Shape.h`, which
    // handles unresolved negative dimensions (returns ``0``) gracefully.
    //
    // See Also
    // --------
    // :func:`shape_numel` — the underlying element-count helper.
    // :meth:`nbytes` — contiguous byte count for the same shape/dtype.
    std::size_t numel() const noexcept { return shape_numel(shape); }

    // Total bytes occupied by a contiguous layout of :attr:`shape` and
    // :attr:`dtype`.
    //
    // Equal to ``numel() * dtype_size(dtype)``; does NOT inspect
    // :attr:`stride`, so non-contiguous descriptors still report the
    // size of the *equivalent contiguous* tensor, not their actual
    // memory footprint in the parent storage.
    //
    // Returns
    // -------
    // std::size_t
    //     Number of bytes required to hold ``numel()`` elements of
    //     :attr:`dtype`.
    //
    // Math
    // ----
    // $$ \mathrm{nbytes} = \mathrm{numel} \cdot \mathrm{sizeof}(\mathrm{dtype}) $$
    //
    // See Also
    // --------
    // :meth:`numel`     — logical element count.
    // :func:`dtype_size` — width of a single element.
    std::size_t nbytes() const noexcept { return numel() * dtype_size(dtype); }

    // Whether the stored :attr:`stride` matches the reference
    // contiguous (row-major) stride for :attr:`shape` and
    // :attr:`dtype`.
    //
    // A tensor is C-contiguous when, for every axis ``i``,
    // ``stride[i] == dtype_size(dtype) * prod(shape[i+1:])``.  This
    // method delegates the layout check to
    // :func:`contiguous_stride` and compares the two ``Stride``
    // vectors element-wise.
    //
    // Returns
    // -------
    // bool
    //     ``true`` if :attr:`stride` equals
    //     ``contiguous_stride(shape, dtype_size(dtype))``, ``false``
    //     otherwise.
    //
    // Notes
    // -----
    // Used by op kernels that require contiguous input to decide
    // whether to materialise the descriptor via a copy before
    // dispatch.  A default-constructed (empty-shape, empty-stride)
    // descriptor is contiguous by definition because both vectors
    // compare equal.
    //
    // See Also
    // --------
    // :func:`contiguous_stride` — produces the reference stride.
    // :meth:`lucid::core::TensorMeta::is_contiguous_for` — engine-side
    //     equivalent that takes an explicit ``elem_size``.
    bool is_contiguous() const noexcept {
        return stride == contiguous_stride(shape, dtype_size(dtype));
    }
};

}  // namespace lucid
