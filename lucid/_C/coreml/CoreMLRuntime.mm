// lucid/_C/coreml/CoreMLRuntime.mm — see CoreMLRuntime.h.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

#include "CoreMLRuntime.h"

#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace lucid::coreml {

namespace {

std::string describe(NSError* error) {
    if (error == nil)
        return "unknown error";
    return std::string([[error localizedDescription] UTF8String]);
}

MLComputeUnits to_mlcompute(ComputeUnits units) {
    switch (units) {
    case ComputeUnits::CpuOnly:
        return MLComputeUnitsCPUOnly;
    case ComputeUnits::CpuAndGpu:
        return MLComputeUnitsCPUAndGPU;
    case ComputeUnits::CpuAndNeuralEngine:
        return MLComputeUnitsCPUAndNeuralEngine;
    case ComputeUnits::All:
    default:
        return MLComputeUnitsAll;
    }
}

// Every feature name a description declares, sorted.
//
// Core ML hands these back in a dictionary, so the order the package
// declared them in is not recoverable here.  Sorted is at least stable;
// a caller that needs the declared order keeps it from the export, which
// is what the Python layer does.
std::vector<std::string> all_names(NSDictionary<NSString*, MLFeatureDescription*>* features) {
    std::vector<std::string> names;
    names.reserve(features.count);
    for (NSString* key in features)
        names.emplace_back([key UTF8String]);
    std::sort(names.begin(), names.end());
    return names;
}

}  // namespace

// Holds the compiled model plus the temporary directory it was compiled
// into.  Core ML writes the ``.mlmodelc`` next to a caller-chosen URL and
// does not clean it up, so the handle owns that too and removes it on
// destruction rather than leaving artifacts in the user's temp space.
class CoreMLModel {
public:
    MLModel* model = nil;  // ARC strong
    NSURL* compiled_url = nil;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    ~CoreMLModel() {
        if (compiled_url != nil) {
            NSError* error = nil;
            [[NSFileManager defaultManager] removeItemAtURL:compiled_url error:&error];
            (void)error;  // best effort; a leftover cache is not worth raising over
        }
    }
};

CoreMLModel* load_model(const std::string& path, ComputeUnits units) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        NSError* error = nil;

        // Core ML runs compiled models; the package is the source form.
        NSURL* compiled = [MLModel compileModelAtURL:url error:&error];
        if (compiled == nil)
            throw std::runtime_error("lucid.coreml: failed to compile " + path + ": " +
                                     describe(error));

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = to_mlcompute(units);

        MLModel* model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:config
                                                   error:&error];
        if (model == nil) {
            NSError* cleanup = nil;
            [[NSFileManager defaultManager] removeItemAtURL:compiled error:&cleanup];
            throw std::runtime_error("lucid.coreml: failed to load " + path + ": " +
                                     describe(error));
        }

        auto* handle = new CoreMLModel();
        handle->model = model;
        handle->compiled_url = compiled;
        handle->input_names = all_names(model.modelDescription.inputDescriptionsByName);
        handle->output_names = all_names(model.modelDescription.outputDescriptionsByName);
        return handle;
    }
}

void destroy_model(CoreMLModel* model) {
    delete model;
}

std::vector<OpPlacement> compute_plan(const std::string& path, ComputeUnits units) {
    std::vector<OpPlacement> placements;
    if (@available(macOS 14.4, *)) {
        @autoreleasepool {
            NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
            NSError* error = nil;
            NSURL* compiled = [MLModel compileModelAtURL:url error:&error];
            if (compiled == nil)
                throw std::runtime_error("lucid.coreml: failed to compile " + path + ": " +
                                         describe(error));

            MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
            config.computeUnits = to_mlcompute(units);

            // The API is asynchronous; this call is a diagnostic that runs
            // once, so waiting is simpler than threading a callback out to
            // Python and cannot deadlock — the handler runs on Core ML's
            // own queue, not this one.
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            __block MLComputePlan* plan = nil;
            __block NSError* plan_error = nil;
            [MLComputePlan loadContentsOfURL:compiled
                               configuration:config
                           completionHandler:^(MLComputePlan* loaded, NSError* failure) {
                             plan = loaded;
                             plan_error = failure;
                             dispatch_semaphore_signal(done);
                           }];
            dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);

            NSError* cleanup = nil;
            [[NSFileManager defaultManager] removeItemAtURL:compiled error:&cleanup];

            if (plan == nil)
                throw std::runtime_error("lucid.coreml: could not plan " + path + ": " +
                                         describe(plan_error));

            MLModelStructureProgram* program = plan.modelStructure.program;
            if (program == nil)
                return placements;  // not an ML Program — nothing to report

            for (NSString* function_name in program.functions) {
                MLModelStructureProgramFunction* function = program.functions[function_name];
                for (MLModelStructureProgramOperation* op in function.block.operations) {
                    MLComputePlanDeviceUsage* usage =
                        [plan computeDeviceUsageForMLProgramOperation:op];
                    std::string device = "unknown";
                    if (usage != nil) {
                        id<MLComputeDeviceProtocol> preferred = usage.preferredComputeDevice;
                        if ([preferred isKindOfClass:[MLNeuralEngineComputeDevice class]])
                            device = "ANE";
                        else if ([preferred isKindOfClass:[MLGPUComputeDevice class]])
                            device = "GPU";
                        else if ([preferred isKindOfClass:[MLCPUComputeDevice class]])
                            device = "CPU";
                    }
                    placements.push_back(
                        {std::string([op.operatorName UTF8String]), std::move(device)});
                }
            }
        }
    }
    return placements;
}

std::vector<std::string> input_feature_names(const CoreMLModel* model) {
    return model == nullptr ? std::vector<std::string>{} : model->input_names;
}

std::vector<std::string> output_feature_names(const CoreMLModel* model) {
    return model == nullptr ? std::vector<std::string>{} : model->output_names;
}

namespace {

// Copy one named output into a fresh CPU tensor.
//
// ``getBytesWithHandler`` hands over the array's backing store, not a
// normalised view of it: Core ML pads the innermost dimension for
// alignment on the paths that permit the Neural Engine, and copying that
// buffer as if it were packed interleaves padding with data — the output
// has the right shape and the wrong values, which is the failure this
// whole package exists to make impossible.  The size check does not catch
// it either, since a padded buffer is larger.
TensorImplPtr read_output(id<MLFeatureProvider> result, const std::string& output_name) {
    NSString* out_key = [NSString stringWithUTF8String:output_name.c_str()];
    MLFeatureValue* value = [result featureValueForName:out_key];
    if (value == nil || value.multiArrayValue == nil)
        throw std::runtime_error("lucid.coreml: the model produced no output named " +
                                 output_name);
    MLMultiArray* out = value.multiArrayValue;

    Shape out_shape;
    out_shape.reserve(out.shape.count);
    std::size_t count = 1;
    for (NSNumber* dim in out.shape) {
        const auto d = static_cast<std::int64_t>([dim longLongValue]);
        out_shape.push_back(d);
        count *= static_cast<std::size_t>(d);
    }

    const std::size_t nbytes = count * sizeof(float);
    auto bytes = std::shared_ptr<std::byte[]>(new std::byte[nbytes]);

    std::vector<std::int64_t> out_strides;
    out_strides.reserve(out.strides.count);
    for (NSNumber* stride in out.strides)
        out_strides.push_back(static_cast<std::int64_t>([stride longLongValue]));

    std::vector<std::int64_t> packed(out_shape.size(), 1);
    for (std::size_t i = out_shape.size(); i-- > 1;)
        packed[i - 1] = packed[i] * out_shape[i];

    const bool contiguous = out_strides == packed;

    // Largest element offset the strides can reach, so the strided path
    // can check the buffer before reading past it.
    std::int64_t span = 1;
    for (std::size_t d = 0; d < out_shape.size(); ++d)
        span += (out_shape[d] - 1) * (d < out_strides.size() ? out_strides[d] : 0);

    __block bool copied = false;
    [out getBytesWithHandler:^(const void* raw, NSInteger size) {
      if (raw == nullptr)
          return;
      const float* src = static_cast<const float*>(raw);
      float* dst = reinterpret_cast<float*>(bytes.get());
      const auto available = static_cast<std::size_t>(size) / sizeof(float);

      if (contiguous) {
          if (available < count)
              return;
          std::memcpy(dst, src, nbytes);
          copied = true;
          return;
      }
      if (available < static_cast<std::size_t>(span))
          return;
      // Walk the logical index space rather than the buffer.
      std::vector<std::int64_t> index(out_shape.size(), 0);
      for (std::size_t linear = 0; linear < count; ++linear) {
          std::int64_t offset = 0;
          for (std::size_t d = 0; d < index.size(); ++d)
              offset += index[d] * out_strides[d];
          dst[linear] = src[offset];
          for (std::size_t d = index.size(); d-- > 0;) {
              ++index[d];
              if (index[d] < out_shape[d])
                  break;
              index[d] = 0;
          }
      }
      copied = true;
    }];
    if (!copied)
        throw std::runtime_error("lucid.coreml: could not read the output buffer for " +
                                 output_name);

    CpuStorage cpu;
    cpu.ptr = std::move(bytes);
    cpu.nbytes = nbytes;
    cpu.dtype = Dtype::F32;
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, out_shape, Dtype::F32,
                                        Device::CPU, false);
}

}  // namespace

std::vector<TensorImplPtr> predict(CoreMLModel* model,
                                   const std::vector<std::pair<std::string, TensorImplPtr>>& inputs,
                                   const std::vector<std::string>& output_names) {
    if (model == nullptr || model->model == nil)
        throw std::invalid_argument("lucid.coreml: null model handle");
    if (inputs.empty())
        throw std::invalid_argument("lucid.coreml: no inputs given");
    if (output_names.empty())
        throw std::invalid_argument("lucid.coreml: no outputs requested");

    for (const auto& [name, tensor] : inputs) {
        if (!tensor)
            throw std::invalid_argument("lucid.coreml: null input tensor for " + name);
        if (tensor->device() != Device::CPU)
            throw std::invalid_argument(
                "lucid.coreml: input " + name +
                " must be a CPU tensor — Core ML reads host memory, and moving it "
                "here would hide a copy the caller did not ask for");
        if (tensor->dtype() != Dtype::F32 && tensor->dtype() != Dtype::I32)
            throw std::invalid_argument(
                "lucid.coreml: input " + name +
                " must be float32 or int32 — Core ML's multi-array has no int64, "
                "so token ids are narrowed by the caller");
        if (!std::get<CpuStorage>(tensor->storage()).ptr)
            throw std::runtime_error("lucid.coreml: input " + name +
                                     " has no host storage");
    }

    @autoreleasepool {
        NSMutableDictionary<NSString*, MLFeatureValue*>* feature_map =
            [NSMutableDictionary dictionaryWithCapacity:inputs.size()];
        NSError* error = nil;

        for (const auto& [name, tensor] : inputs) {
            const auto& storage = std::get<CpuStorage>(tensor->storage());
            const Shape& shape = tensor->shape();
            NSMutableArray<NSNumber*>* ns_shape =
                [NSMutableArray arrayWithCapacity:shape.size()];
            for (std::int64_t dim : shape)
                [ns_shape addObject:@(dim)];

            // Row-major strides, in elements, as MLMultiArray counts them.
            std::vector<std::int64_t> strides(shape.size(), 1);
            for (std::size_t i = shape.size(); i-- > 1;)
                strides[i - 1] = strides[i] * shape[i];
            NSMutableArray<NSNumber*>* ns_strides =
                [NSMutableArray arrayWithCapacity:strides.size()];
            for (std::int64_t stride : strides)
                [ns_strides addObject:@(stride)];

            // The array borrows Lucid's buffer; ``deallocator:nil`` says
            // Core ML must not free it.  The tensor outlives this scope
            // because the caller holds it.
            const MLMultiArrayDataType in_type = tensor->dtype() == Dtype::I32
                                                     ? MLMultiArrayDataTypeInt32
                                                     : MLMultiArrayDataTypeFloat32;
            MLMultiArray* array =
                [[MLMultiArray alloc] initWithDataPointer:static_cast<void*>(storage.ptr.get())
                                                    shape:ns_shape
                                                 dataType:in_type
                                                  strides:ns_strides
                                              deallocator:nil
                                                    error:&error];
            if (array == nil)
                throw std::runtime_error("lucid.coreml: cannot wrap the input " + name +
                                         ": " + describe(error));
            feature_map[[NSString stringWithUTF8String:name.c_str()]] =
                [MLFeatureValue featureValueWithMultiArray:array];
        }

        MLDictionaryFeatureProvider* features =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:feature_map error:&error];
        if (features == nil)
            throw std::runtime_error("lucid.coreml: cannot build the input features: " +
                                     describe(error));

        id<MLFeatureProvider> result = [model->model predictionFromFeatures:features error:&error];
        if (result == nil)
            throw std::runtime_error("lucid.coreml: prediction failed: " + describe(error));

        std::vector<TensorImplPtr> produced;
        produced.reserve(output_names.size());
        for (const std::string& output_name : output_names)
            produced.push_back(read_output(result, output_name));
        return produced;
    }
}

}  // namespace lucid::coreml
