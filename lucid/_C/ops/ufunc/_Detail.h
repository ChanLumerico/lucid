// lucid/_C/ops/ufunc/_Detail.h
//
// Private implementation-detail header for ufunc translation units that
// hand-write their forward / backward pass instead of inheriting from the
// :class:`UnaryOp` / :class:`ReduceOp` CRTP bases.
//
// The two main consumers are :file:`Var.cpp` (variance / standard
// deviation — needs both mean and squared-residual passes), :file:`Trace.cpp`
// (matrix trace — requires diag-extraction logic that does not fit the
// reduction CRTP), and :file:`Scan.cpp` (``cumsum`` / ``cumprod`` — output
// shape equals input shape but the kernel is sequential along one axis).
// These files reach into ``lucid::ufunc_detail`` for a small set of
// allocation helpers re-exported from :namespace:`lucid::helpers`.
//
// Notes
// -----
// **Namespace isolation.**  All re-exports live in the
// ``lucid::ufunc_detail`` namespace so that ``using namespace`` lines in
// ``.cpp`` files cannot leak helpers into the top-level ``lucid``
// namespace and collide with public op names.  Translation units are
// expected to write either a local
// ``using namespace lucid::ufunc_detail;`` at the top or fully-qualified
// ``lucid::ufunc_detail::allocate_cpu(...)`` calls.
//
// **Inclusion policy.**  Do **not** include this header from any public
// header (anything reachable from ``lucid/_C/api.h`` or the Python
// bindings).  It exists only to keep the hand-rolled ufunc ``.cpp``
// files concise; pulling it into a public header would silently expose
// the helper namespace to downstream code.
//
// See Also
// --------
// :file:`Var.cpp`, :file:`Trace.cpp`, :file:`Scan.cpp` — sole consumers.
// :namespace:`lucid::helpers` — upstream source of the re-exported
//     helpers (``core/Helpers.h``).

#pragma once

#include <cstring>

#include "../../core/Allocator.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

// Implementation-detail namespace for hand-written ufunc ``.cpp`` files.
//
// Contents are *not* part of the public Lucid C++ API and may change
// without notice between minor versions; only the ufunc translation
// units listed in the file header are permitted consumers.
//
// Notes
// -----
// The namespace is intentionally distinct from :namespace:`lucid` so
// that ``using namespace lucid::ufunc_detail;`` inside a single ``.cpp``
// file cannot bleed helper names into op-level lookup.
namespace lucid::ufunc_detail {

// Re-export of :func:`lucid::helpers::allocate_cpu` into the
// ``ufunc_detail`` namespace.
//
// :func:`lucid::helpers::allocate_cpu` returns a freshly-zeroed
// :class:`CpuStorage` of the requested element count and dtype, using
// the global :class:`Allocator` so allocation metrics flow into the
// profiler.  See ``core/Helpers.h`` for the full signature and
// alignment guarantees.
//
// See Also
// --------
// :func:`lucid::helpers::fresh` — sibling helper re-exported below.
using ::lucid::helpers::allocate_cpu;

// Re-export of :func:`lucid::helpers::fresh` into the ``ufunc_detail``
// namespace.
//
// :func:`lucid::helpers::fresh` constructs a brand-new
// :class:`TensorImplPtr` of the given shape / dtype / device, backed by
// a newly-allocated :class:`Storage` and detached from any autograd
// graph.  Used by hand-written ufunc forwards to materialise a clean
// output tensor before populating its storage.
//
// See Also
// --------
// :func:`lucid::helpers::allocate_cpu` — companion CPU-storage helper.
using ::lucid::helpers::fresh;

}  // namespace lucid::ufunc_detail
