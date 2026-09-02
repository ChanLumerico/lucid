// lucid/_C/coreml/CoreMLRuntime.h
//
// Loading and running a ``.mlpackage`` through CoreML.framework.
//
// This is the half of ``lucid.coreml`` that reaches the Neural Engine.
// Apple's framework is a system library, the same standing Accelerate and
// Metal already have in this engine — there is no third-party dependency
// here, and none in the writer either.
//
// Core ML executes a *compiled* model (``.mlmodelc``), not the package
// itself, so :func:`load_model` compiles on the way in.  Compilation is
// the expensive step (hundreds of milliseconds for a real network) and
// the compiled artifact is cached for the handle's lifetime, which is why
// loading returns a handle rather than each prediction taking a path.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "../api.h"
#include "../core/fwd.h"

namespace lucid::coreml {

// Opaque handle; the Objective-C objects stay inside the .mm.
class CoreMLModel;

// Which processors Core ML may schedule on.  Mirrors
// ``MLComputeUnits``; ``CpuAndNeuralEngine`` is the one that makes this
// package worth having.
enum class ComputeUnits : int {
    All = 0,
    CpuOnly = 1,
    CpuAndGpu = 2,
    CpuAndNeuralEngine = 3,
};

// Compile and load the package at ``path``.
//
// Raises
// ------
// std::runtime_error
//     Compilation or load failed; the message carries Core ML's own
//     description, which names the offending layer for a malformed
//     program.
LUCID_API CoreMLModel* load_model(const std::string& path, ComputeUnits units);

// Release a handle.  Safe with ``nullptr``.
LUCID_API void destroy_model(CoreMLModel* model);

// Run one prediction.
//
// Parameters
// ----------
// model : CoreMLModel*
//     From :func:`load_model`.
// inputs : const std::vector<std::pair<std::string, TensorImplPtr>>&
//     Each feature name as the exported description declares it, with
//     its tensor. Every tensor must be a contiguous CPU float32 or int32
//     one. A Metal tensor is rejected rather than downloaded behind the
//     caller's back — the Python layer decides whether that copy is
//     acceptable and says so.
// output_names : const std::vector<std::string>&
//     Outputs to read back, in the order the caller wants them.
// images : const std::vector<std::pair<std::string, int>>&
//     Inputs the package declared as images, with the colour space it
//     declared them in. Core ML refuses a multi-array for those, so they
//     are copied into a pixel buffer instead.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     Freshly allocated CPU float32 tensors, one per requested output.
LUCID_API std::vector<TensorImplPtr>
predict(CoreMLModel* model,
        const std::vector<std::pair<std::string, TensorImplPtr>>& inputs,
        const std::vector<std::string>& output_names,
        const std::vector<std::pair<std::string, int>>& images);

// Run a classifier and read back what it declares.
//
// A classifier's outputs are a string and a dictionary, not arrays, so
// ``predict`` cannot read them — the feature types are different, and
// asking for a multi-array gets nothing.
//
// Returns
// -------
// std::pair<std::string, std::vector<std::pair<std::string, double>>>
//     The winning label, and every label with its probability.
LUCID_API std::pair<std::string, std::vector<std::pair<std::string, double>>>
classify(CoreMLModel* model,
         const std::vector<std::pair<std::string, TensorImplPtr>>& inputs,
         const std::vector<std::pair<std::string, int>>& images,
         const std::string& label_name,
         const std::string& probabilities_name);

// One operation's device assignment, as Core ML planned it.
struct OpPlacement {
    std::string op_type;  // MIL operator name, e.g. "conv"
    std::string device;   // "ANE" | "GPU" | "CPU" | "unknown"
};

// Ask Core ML which device each operation will actually run on.
//
// This exists because the failure it detects is silent.  Requesting
// ``CpuAndNeuralEngine`` on a program the Neural Engine cannot take —
// float32 is the common reason — does not warn, does not error, and
// returns correct results at CPU speed.  Without this the only evidence
// that the accelerator is being used is that the model got faster, which
// is not evidence at all.
//
// Backed by ``MLComputePlan`` (macOS 14.4+).  Returns an empty vector
// when the platform is older, which callers must treat as "unknown"
// rather than "not accelerated".
LUCID_API std::vector<OpPlacement> compute_plan(const std::string& path, ComputeUnits units);

// Forget everything a stateful model has accumulated.
//
// A state persists across predictions by design, so a caller that wants
// to start a fresh sequence has to say so; there is no other way back to
// the initial value.
LUCID_API void reset_state(CoreMLModel* model);

// Whether the loaded model declares any state.
LUCID_API bool carries_state(const CoreMLModel* model);

// Feature names the loaded model declares, for diagnostics and for the
// Python layer to check a package it did not write itself.
LUCID_API std::vector<std::string> input_feature_names(const CoreMLModel* model);
LUCID_API std::vector<std::string> output_feature_names(const CoreMLModel* model);

}  // namespace lucid::coreml
