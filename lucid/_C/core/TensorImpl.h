// lucid/_C/core/TensorImpl.h
//
// Engine-side reference-counted tensor value type.
//
// :class:`TensorImpl` is the C++ object that the Python :class:`lucid.Tensor`
// wraps via a single ``shared_ptr`` (:type:`TensorImplPtr`).  Every attribute
// the Python layer exposes — ``.shape``, ``.dtype``, ``.device``, ``.data``,
// ``.grad``, ``.requires_grad``, ``.is_leaf``, ``.numpy()``, ``.to()``,
// ``.tolist()``, ``.item()`` — ultimately reads or mutates this struct.
//
// A :class:`TensorImpl` combines three independent concerns:
//
//   1. :class:`Storage` — the backing memory (:class:`CpuStorage`,
//      :class:`GpuStorage`, or :class:`SharedStorage`), possibly shared
//      across multiple views via ``shared_ptr``-ownership.
//
//   2. :class:`TensorMeta` — shape, stride, dtype, device.  Held by value
//      (not heap-allocated) so that geometry queries are a single pointer
//      dereference.  Views own an independent :class:`TensorMeta` while
//      sharing the base's :class:`Storage`.
//
//   3. :class:`AutogradMeta` — ``requires_grad`` / ``is_leaf`` flags,
//      version counter, ``grad_fn`` pointer, and accumulated gradient.
//      Stored as :type:`std::optional` so inference-only tensors pay zero
//      allocation cost.
//
// Ownership model
// ---------------
// :class:`TensorImpl` is always managed through :type:`TensorImplPtr`
// (``std::shared_ptr<TensorImpl>``).  Views produced by :func:`make_view`
// share the same :class:`Storage` as the base tensor but own an independent
// :class:`TensorMeta`; the base's lifetime is implicitly extended by the
// shared storage.
//
// Thread safety
// -------------
// Individual field accessors are **not** thread-safe.  Concurrent writes to
// the same :class:`TensorImpl` require external synchronisation.  The
// :type:`VersionCounter` inside :class:`Storage` uses atomic operations for
// cross-thread version reads, but :class:`AutogradMeta::version` is a plain
// :type:`int64` and must only be mutated from a single thread at a time.
//
// See Also
// --------
// :class:`Storage`     — backing memory variants.
// :class:`TensorMeta`  — shape / stride / dtype / device bundle.
// :class:`Node`        — autograd graph node referenced by ``grad_fn``.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "Shape.h"
#include "Storage.h"
#include "TensorMeta.h"

namespace py = pybind11;

namespace lucid {

class TensorImpl;

// The CPU tensors that read one buffer, as :func:`TensorImpl::make_view`
// made them.
//
// Every member holds this through ``shared_ptr``, so ``use_count()`` is how
// many of them are alive.  They share one version counter — a write
// through any member moves every member's ``version()``, so a node that
// saved one of them notices — and an in-place write to a member goes into
// the buffer instead of a new storage slot, so the others see it
// (:func:`TensorImpl::write_through`).  ``members`` is how autograd finds
// the others to re-derive from the one written; entries expire with their
// tensors.
struct ViewFamily {
    VersionCounter version{0};
    // The whole buffer's size.  A member that reads fewer bytes is a slice
    // (make_block_view) rather than a full-buffer reshape.
    std::size_t nbytes = 0;
    std::mutex mu;
    std::vector<std::weak_ptr<TensorImpl>> members;
};

// Reference-counted tensor implementation owned by every Python-level
// :class:`lucid.Tensor`.
//
// The C++ op layer receives and returns :type:`TensorImplPtr`
// (``shared_ptr<TensorImpl>``) directly; the Python wrapper is a thin layer
// that forwards attribute access here.  :class:`TensorImpl` extends
// :class:`std::enable_shared_from_this` so member functions can hand out
// additional ``shared_ptr`` references to ``*this`` safely (e.g. when
// building autograd graph nodes that capture the producing tensor).
//
// Attributes
// ----------
// storage : Storage
//     Backing memory variant — :class:`CpuStorage`, :class:`GpuStorage`,
//     or :class:`SharedStorage`.  Shared with every view.
// offset : std::size_t
//     Byte offset of this tensor's first element from the start of
//     :attr:`storage`.  Non-zero only for views created by :func:`make_view`.
// meta : TensorMeta
//     Shape, stride, dtype, and device.  Views carry their own
//     :class:`TensorMeta` independent of the base.
// autograd : std::optional<AutogradMeta>
//     Autograd bookkeeping (``requires_grad``, ``is_leaf``, version,
//     ``grad_fn``, accumulated gradient).  ``std::nullopt`` for tensors
//     that have never participated in the autograd graph.
//
// See Also
// --------
// :class:`Storage`        — backing memory variant.
// :class:`TensorMeta`     — shape / stride / dtype / device bundle.
// :class:`AutogradMeta`   — autograd bookkeeping.
class LUCID_API TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    // Constructs a :class:`TensorImpl` from an already-allocated
    // :class:`Storage`.
    //
    // The C-contiguous (row-major) stride is computed automatically from
    // ``shape`` and ``dtype``.  When ``requires_grad`` is ``true`` an
    // :class:`AutogradMeta` is constructed immediately; otherwise it is
    // left unallocated until needed.
    //
    // Parameters
    // ----------
    // storage : Storage
    //     Backing memory (any variant).  Ownership is moved in.
    // shape : Shape
    //     Dimension vector.  May be empty for a scalar.
    // dtype : Dtype
    //     Element type.
    // device : Device
    //     CPU or GPU.  Must match the variant of ``storage``.
    // requires_grad : bool
    //     Whether the tensor should track gradients from the start.
    TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad);

    // Constructs a :class:`TensorImpl` by copying data from a NumPy array.
    //
    // The data is always copied into a fresh :class:`CpuStorage` buffer.
    // When ``device == Device::GPU`` the buffer is subsequently uploaded
    // to MLX via the GPU bridge, yielding a :class:`GpuStorage`-backed
    // tensor.
    //
    // Parameters
    // ----------
    // arr : py::array
    //     Source NumPy array.  Must be a signed / floating / complex dtype.
    // device : Device
    //     Target device for the constructed tensor.
    // requires_grad : bool
    //     Whether the resulting tensor tracks gradients.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     Newly constructed tensor.
    //
    // Raises
    // ------
    // DtypeError
    //     The array has an unsigned dtype.  The caller must cast to a
    //     signed integer or floating-point type first.
    static std::shared_ptr<TensorImpl> from_numpy(py::array arr, Device device, bool requires_grad);

    // ---------------------------------------------------------------------------
    // Storage accessors
    // ---------------------------------------------------------------------------

    // The backing :class:`Storage`, as a dense row-major buffer of exactly
    // this tensor's elements.
    //
    // Nearly every op hands ``storage()`` straight to a backend kernel that
    // reads it as dense, from its first byte.  A CPU view at an offset or
    // with strides read the wrong bytes there — about a hundred ops did, and
    // the audit's layout axis found them once it fed them real views.  So a
    // non-dense CPU tensor answers with a packed copy of the elements it
    // reads, rebuilt whenever its version moves (a write through it or
    // through any member of its view family).  Every other tensor answers
    // with its own buffer, as before.
    //
    // Code that walks the geometry itself — ``contiguous``, the NumPy bridge,
    // ``data_ptr``, buffer-identity and overlap tests — reads
    // :func:`raw_storage` instead.
    const Storage& storage() const;

    // The backing :class:`Storage` as it is, before any packing: the whole
    // buffer a view reads into, to be walked with :func:`stride` and
    // :func:`storage_offset`.
    const Storage& raw_storage() const noexcept { return storage_; }

    // Returns a mutable reference to the backing :class:`Storage` variant.
    //
    // Used by in-place ops; callers are responsible for bumping the version
    // counter when they mutate the underlying bytes.
    Storage& mutable_storage() noexcept { return storage_; }

    // Returns the byte offset of this tensor's first element from the start
    // of :func:`storage`.
    //
    // Returns
    // -------
    // std::size_t
    //     ``0`` for owning tensors; non-zero for views created with
    //     :func:`make_view`.
    std::size_t storage_offset() const noexcept { return offset_; }

    // Predicate: is this tensor's CPU storage aliased by another tensor?
    //
    // Returns
    // -------
    // bool
    //     ``true`` when the underlying :class:`CpuStorage` ``ptr`` is
    //     referenced by more than one ``shared_ptr`` — i.e. at least one
    //     other view of this tensor exists.  Always ``false`` for
    //     :class:`GpuStorage` and :class:`SharedStorage` variants (those
    //     variants do not currently report sharing through this hook).
    bool storage_is_shared() const noexcept;

    // Predicate: does another live tensor read this one's buffer as a view
    // of it — a member of the same :class:`ViewFamily`?
    //
    // Such a tensor is written in place rather than handed a new storage
    // slot, so its views see the write (:func:`write_through`).  Unlike
    // :func:`storage_is_shared`, a NumPy array or a storage saved for
    // backward does not count: neither is a view.
    //
    // Returns
    // -------
    // bool
    //     ``true`` while at least one other member of the family is alive.
    bool is_aliased() const noexcept { return family_ && family_.use_count() > 1; }

    // Predicate: is this a detached alias — what ``.data`` hands out — rather
    // than a view autograd follows?  Other clone_with_grad aliases are not
    // detached: a factory's ``requires_grad=True`` tensor is one, and it has
    // to stay an ordinary member once the temporary it aliased is gone.
    //
    // It reads the same buffer as the rest of its family and shares their
    // version counter, but a write through it is untracked by design: the
    // values change and no graph is re-derived, as with the reference's
    // ``.data``.  Reshape-family views are not detached.
    bool is_detached_alias() const noexcept { return detached_alias_; }

    // Marks this tensor as a detached alias; see :func:`is_detached_alias`.
    void set_detached_alias(bool v) noexcept { detached_alias_ = v; }

    // The other live members of this tensor's view family.
    //
    // Returns
    // -------
    // std::vector<std::shared_ptr<TensorImpl>>
    //     Every member but ``this``; empty for a tensor with no views.
    std::vector<std::shared_ptr<TensorImpl>> live_views() const;

    // Writes ``src``'s values into this tensor's buffer, in place.
    //
    // The write an in-place op makes to a tensor with live views: into the
    // bytes the views read, where a new storage slot would leave them
    // reading the old ones.  Bumps the version, which every member shares.
    //
    // Parameters
    // ----------
    // src : TensorImpl
    //     The new values; same shape and dtype.
    // name : const char*
    //     Op name for error messages.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     ``src`` has another shape.
    // LucidError
    //     ``src`` has another dtype, or a view of a leaf that requires grad
    //     would be modified while autograd records.
    // NotImplementedError
    //     The tensor is on the GPU or is not dense, or its buffer is also
    //     held by something that is not one of its views — a storage saved
    //     for backward, a NumPy array — which would read the new values
    //     without being told.
    void write_through(const TensorImpl& src, const char* name);

    // The Metal shared buffer this tensor still is, or ``nullptr``.
    //
    // Non-null only while the storage is the whole buffer, contiguous from
    // its first byte in its own dtype — the one case a relabel to the other
    // device can alias rather than copy.  ``nullptr`` for views that cover
    // less, and once an op has replaced the storage.
    //
    // Returns
    // -------
    // const SharedStorage*
    //     The descriptor, or ``nullptr``.
    const SharedStorage* metal_shared() const;

    // Take ``out``'s contents after an in-place op.
    //
    // Ordinarily a swap of the storage *slot* — see
    // :file:`ops/utils/InplaceGraph.h` for why the slot and not the buffer.
    // A tensor over a Metal shared buffer writes the new values into the
    // buffer instead, so every alias of it sees them — but only when ``out``
    // recorded no graph, since a node that saved the pre-op input shares the
    // buffer.  A CPU tensor with live views always writes into its buffer
    // (:func:`write_through`).  Otherwise the swap happens and the tensor
    // stops being shared.
    //
    // Parameters
    // ----------
    // out : TensorImpl&
    //     The op's result; its storage is moved from.
    // name : const char*
    //     Op name for error messages.
    void take_storage_from(TensorImpl& out, const char* name);

    // Creates a new :class:`TensorImpl` that aliases ``base``'s
    // :class:`Storage` with a different shape, stride, and optional byte
    // offset.
    //
    // No data is copied.  The view starts with no autograd state of its
    // own; a caller that makes a differentiable view wires its grad_fn.  A
    // CPU view joins ``base``'s :class:`ViewFamily` — one version counter,
    // in-place writes into the buffer — unless ``join_family`` is ``false``.
    //
    // Parameters
    // ----------
    // base : std::shared_ptr<TensorImpl>
    //     Tensor whose storage the view aliases.
    // shape : Shape
    //     Shape of the new view.
    // stride : Stride
    //     Byte-stride vector for the new view.  Need not be contiguous.
    // offset_bytes : std::size_t, optional
    //     Byte offset from ``base``'s storage origin.  Defaults to ``0``.
    // join_family : bool, optional
    //     Whether a CPU view joins ``base``'s family.  Defaults to ``true``.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     New view tensor sharing ``base``'s storage.
    static std::shared_ptr<TensorImpl> make_view(const std::shared_ptr<TensorImpl>& base,
                                                 Shape shape,
                                                 Stride stride,
                                                 std::size_t offset_bytes = 0,
                                                 bool join_family = true);

    // A view of a contiguous run of ``base``'s buffer: ``byte_offset`` bytes
    // in, ``shape``'s elements long — what a leading-dimension slice reads.
    //
    // Its storage points at the run itself (an aliasing ``shared_ptr`` into
    // ``base``'s buffer, sharing its ownership), so the view is dense and
    // every op reads it from its first byte, as it reads any tensor.  It
    // joins ``base``'s :class:`ViewFamily`.  ``base`` must be a dense CPU
    // tensor and the run must lie inside its buffer.
    //
    // Parameters
    // ----------
    // base : std::shared_ptr<TensorImpl>
    //     Dense CPU tensor whose buffer the view reads.
    // shape : Shape
    //     Shape of the view; its elements are the run's, in order.
    // byte_offset : std::size_t
    //     Where the run starts in ``base``'s storage.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     The view, with no autograd state of its own.
    static std::shared_ptr<TensorImpl>
    make_block_view(const std::shared_ptr<TensorImpl>& base, Shape shape, std::size_t byte_offset);

    // ---------------------------------------------------------------------------
    // Metadata accessors
    // ---------------------------------------------------------------------------

    // Returns a const reference to the :class:`TensorMeta` bundle.
    const TensorMeta& meta() const noexcept { return meta_; }

    // Returns the shape vector.  Backs the Python ``.shape`` attribute.
    const Shape& shape() const noexcept { return meta_.shape; }

    // Returns the byte-stride vector.  Backs the Python ``.stride()`` method.
    const Stride& stride() const noexcept { return meta_.stride; }

    // Returns the element dtype.  Backs the Python ``.dtype`` attribute.
    Dtype dtype() const noexcept { return meta_.dtype; }

    // Returns the device tag.  Backs the Python ``.device`` attribute.
    Device device() const noexcept { return meta_.device; }

    // ---------------------------------------------------------------------------
    // Autograd accessors
    // ---------------------------------------------------------------------------

    // Returns the current ``requires_grad`` flag.
    //
    // Returns
    // -------
    // bool
    //     ``false`` when there is no :class:`AutogradMeta` at all (the
    //     common inference path — avoids an indirection).
    bool requires_grad() const noexcept { return autograd_ ? autograd_->requires_grad : false; }

    // Returns the leaf-status flag.
    //
    // A *leaf* tensor was constructed directly by the user; a non-leaf
    // tensor is the output of a differentiable op and carries a non-null
    // ``grad_fn``.
    //
    // Returns
    // -------
    // bool
    //     ``true`` for leaves, ``true`` (default) when there is no
    //     :class:`AutogradMeta` yet.
    bool is_leaf() const noexcept { return autograd_ ? autograd_->is_leaf : true; }

    // Returns the autograd version counter.
    //
    // Bumped by :func:`bump_version` on every in-place mutation; autograd
    // snapshots this at forward time and checks it at backward time to
    // catch illegal mutations.  A tensor over a Metal shared buffer adds the
    // buffer's own counter, which every alias of the buffer bumps — so a
    // write through the CPU view invalidates a GPU alias saved for backward,
    // and the other way round.
    //
    // Returns
    // -------
    // std::int64_t
    //     Current version.  ``0`` when no :class:`AutogradMeta` exists and no
    //     shared buffer or view family has been written.
    std::int64_t version() const noexcept {
        std::int64_t v = autograd_ ? autograd_->version : 0;
        if (shared_)
            v += static_cast<std::int64_t>(shared_->get_version());
        // A tensor with live views adds their family's counter the same way.
        if (family_)
            v += static_cast<std::int64_t>(family_->version.load(std::memory_order_relaxed));
        return v;
    }

    // Returns the backward function pointer for this tensor.
    //
    // Returns
    // -------
    // const std::shared_ptr<Node>&
    //     Non-null for non-leaf tensors produced by a differentiable op,
    //     ``nullptr`` for leaves and tensors with no :class:`AutogradMeta`.
    const std::shared_ptr<Node>& grad_fn() const noexcept {
        static const std::shared_ptr<Node> kNull;
        return autograd_ ? autograd_->grad_fn : kNull;
    }

    // Returns the output index that connects this tensor to its
    // :func:`grad_fn`.
    //
    // Multi-output ops produce several tensors that share one
    // :class:`Node`; this index tells the backward pass which of the
    // node's outputs the tensor corresponds to.
    //
    // Returns
    // -------
    // std::uint32_t
    //     Output index, or ``0`` when no :class:`AutogradMeta` exists.
    std::uint32_t grad_output_nr() const noexcept {
        return autograd_ ? autograd_->grad_output_nr : 0;
    }

    // Returns the accumulated gradient :class:`Storage`, if any.
    //
    // Returns
    // -------
    // const std::optional<Storage>&
    //     :type:`std::nullopt` when no gradient has been accumulated, or
    //     when the tensor has no :class:`AutogradMeta` at all.
    const std::optional<Storage>& grad_storage() const noexcept {
        static const std::optional<Storage> kEmpty;
        return autograd_ ? autograd_->grad : kEmpty;
    }

    // Returns a mutable reference to the gradient :class:`Storage`,
    // constructing the :class:`AutogradMeta` on demand.
    //
    // Used by the engine's gradient accumulation paths.
    //
    // Returns
    // -------
    // std::optional<Storage>&
    //     Mutable reference; the optional may still be empty even though
    //     :class:`AutogradMeta` is now allocated.
    std::optional<Storage>& mutable_grad_storage() noexcept {
        ensure_autograd();
        return autograd_->grad;
    }

    // ---------------------------------------------------------------------------
    // Mutation helpers
    // ---------------------------------------------------------------------------

    // Sets the dtype field.  Does not touch the underlying bytes; used by
    // pure metadata reinterpretation paths.
    void set_dtype(Dtype dt) noexcept { meta_.dtype = dt; }

    // Sets the device field.  Does not move data; used after an in-place
    // device transfer has updated :attr:`storage_`.
    void set_device(Device dv) noexcept { meta_.device = dv; }

    // Enables or disables gradient tracking.
    //
    // Constructs :class:`AutogradMeta` lazily when ``v`` is ``true``.
    // Setting to ``false`` does not destroy an already-existing
    // :class:`AutogradMeta` — only the flag is cleared, so historical
    // version counters / saved tensors remain valid.
    //
    // Parameters
    // ----------
    // v : bool
    //     New ``requires_grad`` value.
    void set_requires_grad(bool v) noexcept {
        if (v || autograd_)
            ensure_autograd()->requires_grad = v;
    }

    // Sets the leaf-status flag, constructing :class:`AutogradMeta` if it
    // does not yet exist.
    //
    // Parameters
    // ----------
    // v : bool
    //     New ``is_leaf`` value.
    void set_leaf(bool v) noexcept { ensure_autograd()->is_leaf = v; }

    // Attaches the backward function pointer.
    //
    // Marks the tensor as non-leaf (callers usually pair this with
    // ``set_leaf(false)``).
    //
    // Parameters
    // ----------
    // fn : std::shared_ptr<Node>
    //     Backward node produced by the forward-time op kernel.
    void set_grad_fn(std::shared_ptr<Node> fn) noexcept {
        ensure_autograd()->grad_fn = std::move(fn);
    }

    // Sets the output index associated with :func:`grad_fn`.
    //
    // Parameters
    // ----------
    // nr : std::uint32_t
    //     Zero-based output index within the producing :class:`Node`.
    void set_grad_output_nr(std::uint32_t nr) noexcept { ensure_autograd()->grad_output_nr = nr; }

    // Drops the backward function pointer without removing the
    // :class:`AutogradMeta`.
    //
    // Used when an in-place op detaches a tensor from the autograd graph
    // (``tensor.detach_()``-style semantics).
    void clear_grad_fn() noexcept {
        if (autograd_)
            autograd_->grad_fn.reset();
    }

    // Stores a gradient :class:`Storage` directly.
    //
    // Parameters
    // ----------
    // grad : Storage
    //     Gradient buffer to adopt.  Must have the same dtype / device /
    //     shape as this tensor.
    void set_grad_storage(Storage grad) { ensure_autograd()->grad = std::move(grad); }

    // Returns the graph-mode gradient as a full :class:`TensorImpl`.
    //
    // Populated when ``backward(create_graph=True)`` was called — allowing
    // the gradient itself to participate in further autograd (Hessian-vector
    // products, MAML, second-order ops).
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     The graph-mode gradient tensor, or ``nullptr`` when no graph-mode
    //     gradient has been accumulated.
    std::shared_ptr<TensorImpl> grad_as_impl() const noexcept {
        return autograd_ ? autograd_->grad_impl : nullptr;
    }

    // Sets the graph-mode gradient tensor.
    //
    // Parameters
    // ----------
    // g : std::shared_ptr<TensorImpl>
    //     Gradient tensor to install.
    void set_grad_impl(std::shared_ptr<TensorImpl> g) noexcept {
        ensure_autograd()->grad_impl = std::move(g);
    }

    // Accumulates a graph-mode gradient.
    //
    // First call sets the gradient outright; subsequent calls add ``g``
    // into the existing :func:`grad_as_impl` via the engine's ``add`` op
    // so the accumulation is differentiable when ``create_graph=true``.
    //
    // Parameters
    // ----------
    // g : std::shared_ptr<TensorImpl>
    //     Gradient contribution to accumulate.
    void accumulate_grad_impl(std::shared_ptr<TensorImpl> g);

    // Returns whether this tensor will retain its gradient even when it is
    // a non-leaf.
    //
    // Mirrors the reference framework's ``tensor.retain_grad()`` API.
    //
    // Returns
    // -------
    // bool
    //     ``true`` when the engine should accumulate gradients into
    //     :attr:`grad_storage` for this tensor regardless of leaf status.
    bool retains_grad() const noexcept { return autograd_ ? autograd_->retain_grad : false; }

    // Toggles gradient retention for non-leaf tensors.
    //
    // Parameters
    // ----------
    // v : bool
    //     New ``retain_grad`` value.
    void set_retain_grad(bool v) noexcept { ensure_autograd()->retain_grad = v; }

    // Increments the autograd version counter.
    //
    // Called by every in-place op so autograd can detect mutations of
    // tensors that were saved for backward.  The tensor's own count is left
    // alone when no :class:`AutogradMeta` exists — it has never been in a
    // graph — but a Metal shared buffer's counter always moves, because
    // another alias of the same bytes may have been saved.  So does a view
    // family's, for the same reason.
    void bump_version() noexcept {
        if (autograd_)
            ++autograd_->version;
        if (shared_)
            shared_->bump_version();
        if (family_)
            family_->version.fetch_add(1, std::memory_order_relaxed);
    }

    // ---------------------------------------------------------------------------
    // Derived shape utilities
    // ---------------------------------------------------------------------------

    // Returns the total number of elements (product of shape dimensions).
    //
    // Returns
    // -------
    // std::size_t
    //     ``1`` for a scalar (empty shape).
    std::size_t numel() const;

    // Returns the total byte footprint of the tensor's logical data.
    //
    // Equal to ``numel() * dtype_size(dtype())``; does **not** include any
    // extra storage capacity beyond the tensor's view.
    //
    // Returns
    // -------
    // std::size_t
    //     Logical byte size of the tensor data.
    std::size_t nbytes() const;

    // Predicate: does the current stride equal the row-major contiguous
    // stride for this shape and dtype?
    //
    // Returns
    // -------
    // bool
    //     ``true`` for canonically laid-out tensors; ``false`` for views
    //     that were transposed / sliced / otherwise reordered.
    bool is_contiguous() const;

    // Predicate: contiguous, at the first byte of its storage, and covering
    // all of it — the layout every backend kernel assumes when it reads a
    // Storage from byte 0.  A contiguous view at an offset, or one smaller
    // than its buffer, is contiguous but not dense.
    bool is_dense() const;

    // ---------------------------------------------------------------------------
    // Python / NumPy interop
    // ---------------------------------------------------------------------------

    // Returns the tensor data as a NumPy array.
    //
    // For CPU tensors the returned array is a zero-copy view backed by a
    // :class:`py::capsule` that keeps the :class:`TensorImpl` alive.  For
    // GPU tensors the data is synchronously downloaded to a temporary CPU
    // buffer first.
    //
    // Returns
    // -------
    // py::object
    //     A NumPy array (one of the supported dtypes, except for
    //     :attr:`Dtype::C64` which is represented as a complex64 array).
    //
    // See Also
    // --------
    // :func:`to_bytes`  — NumPy-free alternative.
    // :func:`tolist`    — nested Python list alternative.
    py::object data_as_python() const;

    // Returns the accumulated gradient as a NumPy array.
    //
    // Identical download semantics to :func:`data_as_python`.
    //
    // Returns
    // -------
    // py::object
    //     The gradient array, or :func:`py::none` when no gradient has been
    //     computed yet.
    py::object grad_as_python() const;

    // ---------------------------------------------------------------------------
    // NumPy-free Python interop  (used by serialization + repr; keeps these
    // paths off the numpy bridge so ``import lucid`` stays numpy-free)
    // ---------------------------------------------------------------------------

    // Returns the tensor data as a contiguous bytes blob, in row-major order.
    //
    // GPU tensors are downloaded to a temporary CPU buffer first.  No NumPy
    // dependency, by design — this path is what the serialization layer and
    // ``__repr__`` use to avoid pulling NumPy into ``import lucid``.
    //
    // Returns
    // -------
    // py::bytes
    //     Row-major bytes blob of length ``nbytes()``.
    py::bytes to_bytes() const;

    // Reconstructs a :class:`TensorImpl` from a raw bytes blob + metadata.
    //
    // Inverse of :func:`to_bytes`.  When ``device == Device::GPU`` the
    // buffer is uploaded to MLX before returning.
    //
    // Parameters
    // ----------
    // data : py::bytes
    //     Row-major byte payload.  Must be ``shape_numel(shape) *
    //     dtype_size(dtype)`` bytes long.
    // shape : Shape
    //     Target shape.
    // dtype : Dtype
    //     Element type of the payload.
    // device : Device
    //     Target device.  GPU triggers an MLX upload.
    // requires_grad : bool
    //     Whether the reconstructed tensor tracks gradients.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     Newly constructed tensor.
    static std::shared_ptr<TensorImpl>
    from_bytes(py::bytes data, Shape shape, Dtype dtype, Device device, bool requires_grad);

    // Cross-device copy without round-tripping through Python or NumPy.
    //
    // Dispatch summary:
    //
    //   * CPU → GPU: contiguous-snapshot the source view, then
    //     ``gpu::upload_cpu_to_gpu`` into a fresh GPU-private MLX array.
    //   * GPU → CPU: ``gpu::download_gpu_to_cpu`` into a fresh
    //     :class:`CpuStorage`.
    //   * :class:`SharedStorage` is handled by the separate
    //     ``transfer_storage`` API (zero-copy relabel); calling this method
    //     on a :class:`SharedStorage` tensor falls back to a contiguous CPU
    //     snapshot path.
    //
    // Parameters
    // ----------
    // target : Device
    //     Destination device.
    // requires_grad : bool
    //     Whether the resulting tensor tracks gradients.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     New tensor on ``target`` with the same shape and dtype.
    std::shared_ptr<TensorImpl> transfer_to_device(Device target, bool requires_grad) const;

    // Converts the tensor to a nested Python list (or a Python scalar for
    // 0-d tensors).
    //
    // NumPy-free — mirrors :func:`item`'s dtype dispatch but walks the full
    // shape.  Handles every supported dtype including :attr:`Dtype::C64`
    // (yields Python ``complex``), so callers never need to fall back to
    // the NumPy bridge.  GPU tensors are downloaded to a temporary
    // :class:`CpuStorage` via the :func:`to_bytes` snapshot path; the heavy
    // lifting then runs against a contiguous row-major byte buffer.
    //
    // Returns
    // -------
    // py::object
    //     Nested ``list`` of Python scalars, or a single scalar when the
    //     tensor is 0-d.
    py::object tolist() const;

    // Extracts a single-element tensor's value as a Python scalar.
    //
    // GPU tensors are downloaded to CPU first.  The :attr:`Dtype::F16`
    // IEEE-754 binary16 → ``float`` conversion is performed engine-side so
    // the Python wrapper avoids duplicating the bit-fiddling logic from
    // :func:`to_string`.
    //
    // Returns
    // -------
    // py::object
    //     ``int`` / ``float`` / ``bool`` / ``complex`` depending on dtype.
    //
    // Raises
    // ------
    // ValueError
    //     ``numel() != 1``.
    py::object item() const;

    // Renders the tensor data as a human-readable string suitable for
    // ``__repr__``.
    //
    // The format roughly mirrors NumPy's ``array2string`` defaults but is
    // implemented entirely engine-side so neither Lucid nor its consumers
    // need NumPy.
    //
    // Parameters
    // ----------
    // precision : int, optional
    //     Significant digits for floating-point output.  Defaults to ``4``.
    // threshold : std::size_t, optional
    //     If ``numel > threshold``, an edge-summary is rendered instead of
    //     the full data.  Defaults to ``1000``.
    // edgeitems : std::size_t, optional
    //     How many items to keep at each edge of a truncated axis.
    //     Defaults to ``3``.
    //
    // Returns
    // -------
    // std::string
    //     Human-readable rendering of the tensor.
    std::string
    to_string(int precision = 4, std::size_t threshold = 1000, std::size_t edgeitems = 3) const;

    // Wraps the tensor's accumulated gradient as a fresh :class:`TensorImpl`
    // that shares the underlying :class:`Storage`.
    //
    // Prefers the graph-mode :attr:`grad_impl` when present; otherwise wraps
    // the standard :attr:`grad` :class:`Storage` directly.  Replaces the
    // older NumPy round-trip used by the Python-side ``Tensor.grad``
    // accessor.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     Tensor view of the gradient, or ``nullptr`` when no gradient has
    //     been accumulated.
    std::shared_ptr<TensorImpl> grad_to_tensor() const;

    // ---------------------------------------------------------------------------
    // Data operations
    // ---------------------------------------------------------------------------

    // Copies data from ``other`` into this tensor.
    //
    // Supports :class:`CpuStorage`↔:class:`CpuStorage`,
    // :class:`GpuStorage`↔:class:`GpuStorage`, and all combinations
    // involving :class:`SharedStorage`.  Bumps the version counter on
    // success.
    //
    // Parameters
    // ----------
    // other : TensorImpl
    //     Source tensor.  Must have identical device, dtype, and shape.
    //
    // Raises
    // ------
    // DeviceMismatch
    //     ``other.device() != device()``.
    // DtypeMismatch
    //     ``other.dtype() != dtype()``.
    // ShapeMismatch
    //     ``other.shape() != shape()``.
    void copy_from(const TensorImpl& other);

    // An in-place assignment as a *user* means it, as opposed to
    // ``copy_from``, which is the raw buffer write every internal caller
    // uses — an optimiser step, a ``load_state_dict``, the compiled
    // executable's ghost-grad placeholders.  Putting the graph handling
    // into ``copy_from`` itself broke all of those at once, because the
    // two want opposite things from the same function.
    //
    // Here it is the assignment: the values are overwritten, so the
    // tensor's place in the graph becomes the *source's* where there is
    // one — that is what routes the gradient to ``z`` in
    // ``y.copy_(z)`` — and ends where there is not.  A leaf that
    // requires grad is refused for the same reason every other in-place
    // op refuses one.
    void assign_from(const TensorImpl& other, const char* name);

    // Clears the accumulated gradient.
    //
    // Sets :attr:`grad` to :type:`std::nullopt` but leaves the
    // :class:`AutogradMeta` itself in place so subsequent backward passes
    // continue to accumulate.
    void zero_grad();

    // Forces evaluation of this tensor's lazy MLX computation graph.
    //
    // GPU tensors: calls ``mlx::core::eval()`` on the underlying array,
    // materialising the buffer.
    // CPU tensors: no-op (CPU storage is always materialised).
    //
    // Notes
    // -----
    // Required before any path that wants to dereference the GPU buffer
    // through a CPU pointer (e.g. :func:`to_bytes`, :func:`data_as_python`).
    void eval() const;

private:
    Storage storage_;
    // The packed copy :func:`storage` hands out for a non-dense CPU view, and
    // the version it was packed at.  Guarded by ``packed_mu_``: the reference
    // it returns has to outlive the call, so the copy lives here.
    mutable std::mutex packed_mu_;
    mutable std::optional<Storage> packed_;
    mutable std::int64_t packed_version_ = -1;
    // Byte offset of the first element from the start of storage_.
    std::size_t offset_ = 0;
    TensorMeta meta_;
    // Lazily constructed; nullptr-equivalent (std::nullopt) for inference-only
    // tensors to avoid the overhead of the heap allocation.
    std::optional<AutogradMeta> autograd_;
    // The Metal shared buffer this tensor was built over, kept aside by the
    // constructor while storage_ holds the CPU view or GPU alias; see
    // metal_shared().  Views inherit it.
    std::optional<SharedStorage> shared_;
    // The CPU tensors that read this one's buffer, this one included; set by
    // make_view on the base and every view made from it, and null for a
    // tensor that has never had a view.
    std::shared_ptr<ViewFamily> family_;
    // Set on what ``.data`` hands out (the data_alias binding): a member of
    // the family whose writes autograd does not follow.  See
    // is_detached_alias().
    bool detached_alias_ = false;

    // Writes ``src``'s values into the shared buffer.  False when this
    // tensor no longer is the buffer or ``src`` does not match it.
    bool write_into_shared(const TensorImpl& src);

    // Puts ``view`` into ``base``'s view family, making one if ``base`` has
    // none yet.
    static void add_to_family(const std::shared_ptr<TensorImpl>& base,
                              const std::shared_ptr<TensorImpl>& view);

    // Stops tracking the shared buffer, folding its counter into this
    // tensor's own so version() never moves backwards.
    void drop_shared();

    // Returns a pointer to the AutogradMeta, constructing it in-place if it
    // does not yet exist.  The returned pointer is stable (optional stores
    // in-place) as long as autograd_ is not reset.
    AutogradMeta* ensure_autograd() {
        if (!autograd_)
            autograd_.emplace();
        return &*autograd_;
    }

    // Private mutable references to the meta fields — used only within
    // TensorImpl's own methods; callers should use the public set_* functions.
    Shape& shape_() noexcept { return meta_.shape; }

    // Mutable stride accessor — returns the underlying ``Stride`` by reference
    // so engine-side ops can rewrite strides in place without going through
    // the full ``set_meta`` path.  Pairs with the public const :func:`stride`.
    //
    // Notes
    // -----
    // Modifying the stride WITHOUT also touching ``storage_`` or ``shape_``
    // produces an invalid TensorImpl — only call this from internal kernel
    // code that holds the necessary invariants.  Does NOT bump the version
    // counter; callers that change tensor geometry through this hook are
    // responsible for whatever bookkeeping autograd requires.
    Stride& stride_() noexcept { return meta_.stride; }

    // Mutable dtype accessor — returns the underlying ``Dtype`` by reference.
    // Rarely used outside engine internals; pairs with the public const
    // :func:`dtype`.
    //
    // Notes
    // -----
    // Changing the dtype WITHOUT also reinterpreting / reallocating
    // ``storage_`` is a footgun: the bytes on disk no longer match the
    // declared element type, so subsequent ops will read garbage.  Only call
    // from paths that have already updated storage to match the new dtype.
    Dtype& dtype_field() noexcept { return meta_.dtype; }

    // Mutable device accessor — returns the underlying ``Device`` by
    // reference.  Rarely used outside engine internals; pairs with the
    // public const :func:`device`.
    //
    // Notes
    // -----
    // Same footgun as :func:`dtype_field`: changing the device WITHOUT also
    // migrating ``storage_`` to the new device produces a TensorImpl that
    // claims to live on one stream while its bytes live on another.  Only
    // call from paths that have already moved the storage.
    Device& device_field() noexcept { return meta_.device; }
};

}  // namespace lucid
