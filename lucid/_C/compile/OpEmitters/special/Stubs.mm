// lucid/_C/compile/OpEmitters/misc/Stubs.mm
//
// Eager-fallback markers — :class:`StubEmitter` + the registration
// list.  Each registered name resolves through :func:`find_emitter`
// and returns nullptr from ``emit()``, which signals the
// :class:`MpsBuilder` to abort the compile (clean fallback to eager
// dispatch).
//
// The categorisation comments group the stubs by *why* they can't be
// real-emit — see ``obsidian/`` notes for the long-form rationale.
//
//   - linalg long tail       : need iterative algorithms (Jacobi /
//                              Householder / Gram-Schmidt) MPSGraph
//                              doesn't expose
//   - multi-output / dynamic : :class:`OpNode` IR currently models
//                              one output per node
//   - FFT                    : no MPSGraph FFT primitive
//   - histogram              : no native histogram primitive
//   - 3D pool / conv-transpose / interpolate : Apple SDK only ships
//     2D variants (or 4D-spatial via rank-6 pooling, not the rank-5
//     form Lucid uses)
//   - rotate / grid_sample   : per-pixel bilinear gather; MPSGraph's
//                              ``gatherAlongAxis`` can't express it
//                              in one step
//   - embedding_bag          : dynamic per-bag offsets need a
//                              segment-reduce
//   - complex                : 2-storage backing path the real-input
//                              pipeline doesn't model
//   - factory header rows    : ``arange`` / ``eye`` / etc. — the
//                              builder already factory-skips empty
//                              inputs, but registering keeps
//                              :func:`find_emitter` non-null for
//                              every traceable op name

#include <memory>
#include <string>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

class StubEmitter final : public OpEmitter {
public:
    explicit StubEmitter(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    void* emit(BuilderContext&, const OpNode&) override { return nullptr; }

private:
    std::string name_;
};

struct StubsRegistrar {
    StubsRegistrar() {
        for (const char* name : {
                 // ── linalg long tail (iterative algorithms) ─────────
                 "cholesky", "eig", "eigh", "pinv",
                 "qr", "solve", "svd", "erfinv",
                 // ── multi-output / dynamic shape ────────────────────
                 // ``split`` / ``split_at`` / ``topk`` moved to real-emit
                 // (OpEmitters/shape/Split.mm) via manual multi-output
                 // bind through BuilderContext::bind.  ``nonzero`` and
                 // ``unique`` remain stubbed (dynamic output shape).
                 "nonzero", "unique",
                 // ── FFT (no MPSGraph primitive) ─────────────────────
                 "fftn", "ifftn", "rfftn", "irfftn",
                 // ── histogram (no MPSGraph primitive) ───────────────
                 "histogram", "histogram2d", "histogramdd",
                 // ── 3D pool / conv-transpose (SDK 2-D only) ─────────
                 "conv_transpose3d", "max_pool3d", "avg_pool3d",
                 // ── 3D interpolate (resize is 2-D only) ─────────────
                 "interpolate_nearest_3d", "interpolate_trilinear",
                 // ── spatial sampling / segment-reduce (dyn. gather) ─
                 "grid_sample", "rotate", "embedding_bag",
                 // ── complex / 2-storage path ────────────────────────
                 "complex",
                 // ── host-precomputed factory headers ────────────────
                 // (already factory-skipped by the builder when inputs
                 // are empty; stubbed for find_emitter completeness).
                 // ``zeros`` / ``ones`` moved to real-emit
                 // (OpEmitters/misc/Factory.mm) because they appear
                 // mid-trace in RNN-style code (default ``hx = zeros``).
                 "arange", "eye", "linspace", "logspace", "meshgrid",
                 "empty",
                 // ``rand`` / ``uniform`` / ``randn`` / ``normal`` /
                 // ``randint`` / ``bernoulli`` moved to real-emit in
                 // OpEmitters/special/Random.mm (deterministic-per-
                 // executable via the stateless MPSGraph RNG path).
             }) {
            register_emitter(std::make_unique<StubEmitter>(std::string(name)));
        }
    }
};

[[maybe_unused]] static const StubsRegistrar g_stubs_registrar;

}  // namespace

}  // namespace lucid::compile
