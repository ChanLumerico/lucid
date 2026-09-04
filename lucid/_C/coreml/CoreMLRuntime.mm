// lucid/_C/coreml/CoreMLRuntime.mm — see CoreMLRuntime.h.

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

#include "CoreMLRuntime.h"
#include "MilSchema.h"
#include "../core/Dtype.h"

#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace pb = lucid::coreml::pb;

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

// Wrap a tensor as the pixel buffer an image input wants.
//
// Core ML refuses a multi-array for a feature declared as an image, so a
// package exported with ``image_input`` could not be run — or verified —
// from Lucid without this.  The tensor is ``(1, C, H, W)`` in the colour
// space the package declared, holding pixel values, and the buffer is
// whatever format the model's own constraint asks for.
MLFeatureValue* make_image_feature(MLModel* model,
                                   NSString* key,
                                   const TensorImplPtr& tensor,
                                   int color_space) {
    MLFeatureDescription* described = model.modelDescription.inputDescriptionsByName[key];
    MLImageConstraint* constraint = described.imageConstraint;
    if (constraint == nil)
        throw std::runtime_error("lucid.coreml: the model does not take an image for " +
                                 std::string([key UTF8String]));

    const Shape& shape = tensor->shape();
    if (shape.size() != 4 || shape[0] != 1)
        throw std::invalid_argument(
            "lucid.coreml: an image input must be a (1, C, H, W) tensor");
    const std::int64_t channels = shape[1];
    const std::int64_t height = shape[2];
    const std::int64_t width = shape[3];
    if (width != static_cast<std::int64_t>(constraint.pixelsWide) ||
        height != static_cast<std::int64_t>(constraint.pixelsHigh))
        throw std::invalid_argument(
            "lucid.coreml: the image is the wrong size for this model");

    CVPixelBufferRef buffer = nullptr;
    const CVReturn created =
        CVPixelBufferCreate(kCFAllocatorDefault, constraint.pixelsWide, constraint.pixelsHigh,
                            constraint.pixelFormatType, nullptr, &buffer);
    if (created != kCVReturnSuccess || buffer == nullptr)
        throw std::runtime_error("lucid.coreml: could not allocate a pixel buffer");

    CVPixelBufferLockBaseAddress(buffer, 0);
    auto* base = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(buffer));
    const std::size_t stride = CVPixelBufferGetBytesPerRow(buffer);
    const auto* source = reinterpret_cast<const float*>(
        std::get<CpuStorage>(tensor->storage()).ptr.get());
    const std::size_t plane = static_cast<std::size_t>(height * width);

    // In a 32-bit buffer the bytes are B, G, R, A.  Which tensor channel
    // is which depends on the colour space the package declared, so the
    // mapping is chosen rather than assumed.
    const bool bgra = channels == 3;
    const int slot_for[3] = {
        bgra && color_space == pb::ImageFeatureType_ColorSpace::kBGR ? 0 : 2,
        1,
        bgra && color_space == pb::ImageFeatureType_ColorSpace::kBGR ? 2 : 0,
    };
    for (std::int64_t y = 0; y < height; ++y) {
        std::uint8_t* row = base + static_cast<std::size_t>(y) * stride;
        for (std::int64_t x = 0; x < width; ++x) {
            const std::size_t at = static_cast<std::size_t>(y * width + x);
            if (channels == 1) {
                const float v = source[at];
                row[x] = static_cast<std::uint8_t>(std::clamp(v, 0.0f, 255.0f) + 0.5f);
                continue;
            }
            std::uint8_t* pixel = row + static_cast<std::size_t>(x) * 4;
            for (int c = 0; c < 3; ++c) {
                const float v = source[static_cast<std::size_t>(c) * plane + at];
                pixel[slot_for[c]] =
                    static_cast<std::uint8_t>(std::clamp(v, 0.0f, 255.0f) + 0.5f);
            }
            pixel[3] = 255;
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer, 0);

    MLFeatureValue* value = [MLFeatureValue featureValueWithPixelBuffer:buffer];
    CVPixelBufferRelease(buffer);
    return value;
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
    // Core ML keeps a stateful model's carried values here, and every
    // prediction reads and writes the same one — that is the whole point.
    // It outlives the prediction, so the handle owns it.
    MLState* state = nil;  // ARC strong, nil unless the model declares state
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

CoreMLModel* load_model(const std::string& path, ComputeUnits units,
                        const std::string& function_name) {
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
        if (!function_name.empty())
            config.functionName = [NSString stringWithUTF8String:function_name.c_str()];

        MLModel* model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:config
                                                   error:&error];
        if (model == nil) {
            const std::string detail = describe(error);
            // Core ML answers a planning failure with "Error in building
            // plan", which says nothing about whether the package is
            // malformed or the accelerator simply could not take this
            // graph.  Those need opposite responses from the caller, so
            // ask: does it load on the CPU?  If it does, the bytes are
            // fine and what failed is the planner.
            bool cpu_only_loads = false;
            if (units != ComputeUnits::CpuOnly) {
                MLModelConfiguration* probe = [[MLModelConfiguration alloc] init];
                probe.computeUnits = MLComputeUnitsCPUOnly;
                if (!function_name.empty())
                    probe.functionName = [NSString stringWithUTF8String:function_name.c_str()];
                NSError* probe_error = nil;
                MLModel* on_cpu = [MLModel modelWithContentsOfURL:compiled
                                                    configuration:probe
                                                            error:&probe_error];
                cpu_only_loads = (on_cpu != nil);
            }
            NSError* cleanup = nil;
            [[NSFileManager defaultManager] removeItemAtURL:compiled error:&cleanup];
            if (cpu_only_loads)
                throw std::runtime_error(
                    "lucid.coreml: " + path +
                    " is a well-formed package that Core ML will not plan for the requested "
                    "compute units (" +
                    detail +
                    "). It loads with ComputeUnits.CPU_ONLY, so the graph is translatable and "
                    "the accelerator planner is what refused it — export again with "
                    "CPU_ONLY to run it, at CPU speed.");
            throw std::runtime_error("lucid.coreml: failed to load " + path + ": " + detail);
        }

        auto* handle = new CoreMLModel();
        handle->model = model;
        handle->compiled_url = compiled;
        if (model.modelDescription.stateDescriptionsByName.count > 0)
            handle->state = [model newState];
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

void reset_state(CoreMLModel* model) {
    if (model == nullptr || model->model == nil)
        throw std::invalid_argument("lucid.coreml: null model handle");
    if (model->state == nil)
        throw std::invalid_argument(
            "lucid.coreml: this model carries no state, so there is nothing to reset");
    @autoreleasepool {
        model->state = [model->model newState];
    }
}

bool carries_state(const CoreMLModel* model) {
    return model != nullptr && model->state != nil;
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

    // The output's element type is the package's to choose, not ours.  A
    // package Lucid wrote casts its outputs to float32, but one written
    // elsewhere need not: the reference stateful model returns float16,
    // and reading that as float32 asks for twice the bytes it has.
    Dtype dtype = Dtype::F32;
    switch (out.dataType) {
        case MLMultiArrayDataTypeFloat32:
            dtype = Dtype::F32;
            break;
        case MLMultiArrayDataTypeFloat16:
            dtype = Dtype::F16;
            break;
        case MLMultiArrayDataTypeDouble:
            dtype = Dtype::F64;
            break;
        case MLMultiArrayDataTypeInt32:
            dtype = Dtype::I32;
            break;
        default:
            throw std::runtime_error(
                "lucid.coreml: output " + output_name +
                " has an element type Lucid has no equivalent for");
    }
    const std::size_t element = dtype_size(dtype);

    Shape out_shape;
    out_shape.reserve(out.shape.count);
    std::size_t count = 1;
    for (NSNumber* dim in out.shape) {
        const auto d = static_cast<std::int64_t>([dim longLongValue]);
        out_shape.push_back(d);
        count *= static_cast<std::size_t>(d);
    }

    const std::size_t nbytes = count * element;
    auto bytes = std::shared_ptr<std::byte[]>(new std::byte[nbytes]);

    // ``getBytesWithHandler`` hands over the array's backing store, not a
    // normalised view of it: Core ML pads the innermost dimension for
    // alignment on the paths that permit the Neural Engine, and copying
    // that buffer as if it were packed interleaves padding with data — the
    // output has the right shape and the wrong values, which is the
    // failure this whole package exists to make impossible.  The size
    // check does not catch it either, since a padded buffer is larger.
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
      const auto* src = static_cast<const std::byte*>(raw);
      auto* dst = bytes.get();
      const auto available = static_cast<std::size_t>(size) / element;

      if (contiguous) {
          if (available < count)
              return;
          std::memcpy(dst, src, nbytes);
          copied = true;
          return;
      }
      if (available < static_cast<std::size_t>(span))
          return;
      // Walk the logical index space rather than the buffer.  Byte-wise,
      // so the same walk serves every element type.
      std::vector<std::int64_t> index(out_shape.size(), 0);
      for (std::size_t linear = 0; linear < count; ++linear) {
          std::int64_t offset = 0;
          for (std::size_t d = 0; d < index.size(); ++d)
              offset += index[d] * out_strides[d];
          std::memcpy(dst + linear * element,
                      src + static_cast<std::size_t>(offset) * element, element);
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
    cpu.dtype = dtype;
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, out_shape, dtype,
                                        Device::CPU, false);
}

}  // namespace

namespace {

// Everything both entry points do before the prediction itself.
id<MLFeatureProvider>
run(CoreMLModel* model,
    const std::vector<std::pair<std::string, TensorImplPtr>>& inputs,
    const std::vector<std::pair<std::string, int>>& images) {
    if (model == nullptr || model->model == nil)
        throw std::invalid_argument("lucid.coreml: null model handle");
    if (inputs.empty())
        throw std::invalid_argument("lucid.coreml: no inputs given");

    for (const auto& [name, tensor] : inputs) {
        if (!tensor)
            throw std::invalid_argument("lucid.coreml: null input tensor for " + name);
        if (tensor->device() != Device::CPU)
            throw std::invalid_argument(
                "lucid.coreml: input " + name +
                " must be a CPU tensor — Core ML reads host memory, and moving it "
                "here would hide a copy the caller did not ask for");
        // The same three types the output side reads.  A package Lucid
        // wrote takes float32 and casts inside, but one written elsewhere
        // may ask for float16 directly.
        if (tensor->dtype() != Dtype::F32 && tensor->dtype() != Dtype::F16 &&
            tensor->dtype() != Dtype::I32)
            throw std::invalid_argument(
                "lucid.coreml: input " + name +
                " must be float32, float16 or int32 — Core ML's multi-array has no "
                "int64, so token ids are narrowed by the caller");
        if (!std::get<CpuStorage>(tensor->storage()).ptr)
            throw std::runtime_error("lucid.coreml: input " + name + " has no host storage");
    }

    NSMutableDictionary<NSString*, MLFeatureValue*>* feature_map =
        [NSMutableDictionary dictionaryWithCapacity:inputs.size()];
    NSError* error = nil;

    for (const auto& [name, tensor] : inputs) {
        NSString* key = [NSString stringWithUTF8String:name.c_str()];
        const auto image = std::find_if(images.begin(), images.end(),
                                        [&](const auto& entry) { return entry.first == name; });
        if (image != images.end()) {
            feature_map[key] = make_image_feature(model->model, key, tensor, image->second);
            continue;
        }
        const auto& storage = std::get<CpuStorage>(tensor->storage());
        const Shape& shape = tensor->shape();
        NSMutableArray<NSNumber*>* ns_shape = [NSMutableArray arrayWithCapacity:shape.size()];
        for (std::int64_t dim : shape)
            [ns_shape addObject:@(dim)];

        std::vector<std::int64_t> strides(shape.size(), 1);
        for (std::size_t i = shape.size(); i-- > 1;)
            strides[i - 1] = strides[i] * shape[i];
        NSMutableArray<NSNumber*>* ns_strides = [NSMutableArray arrayWithCapacity:strides.size()];
        for (std::int64_t stride : strides)
            [ns_strides addObject:@(stride)];

        MLMultiArrayDataType in_type = MLMultiArrayDataTypeFloat32;
        if (tensor->dtype() == Dtype::I32)
            in_type = MLMultiArrayDataTypeInt32;
        else if (tensor->dtype() == Dtype::F16)
            in_type = MLMultiArrayDataTypeFloat16;
        MLMultiArray* array =
            [[MLMultiArray alloc] initWithDataPointer:static_cast<void*>(storage.ptr.get())
                                                shape:ns_shape
                                             dataType:in_type
                                              strides:ns_strides
                                          deallocator:nil
                                                error:&error];
        if (array == nil)
            throw std::runtime_error("lucid.coreml: cannot wrap the input " + name + ": " +
                                     describe(error));
        feature_map[key] = [MLFeatureValue featureValueWithMultiArray:array];
    }

    MLDictionaryFeatureProvider* features =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:feature_map error:&error];
    if (features == nil)
        throw std::runtime_error("lucid.coreml: cannot build the input features: " +
                                 describe(error));

    id<MLFeatureProvider> result =
        model->state != nil
            ? [model->model predictionFromFeatures:features usingState:model->state error:&error]
            : [model->model predictionFromFeatures:features error:&error];
    if (result == nil)
        throw std::runtime_error("lucid.coreml: prediction failed: " + describe(error));
    return result;
}

}  // namespace

std::pair<std::string, std::vector<std::pair<std::string, double>>>
classify(CoreMLModel* model,
         const std::vector<std::pair<std::string, TensorImplPtr>>& inputs,
         const std::vector<std::pair<std::string, int>>& images,
         const std::string& label_name,
         const std::string& probabilities_name) {
    @autoreleasepool {
        id<MLFeatureProvider> result = run(model, inputs, images);

        MLFeatureValue* label =
            [result featureValueForName:[NSString stringWithUTF8String:label_name.c_str()]];
        if (label == nil)
            throw std::runtime_error("lucid.coreml: the model produced no label named " +
                                     label_name);

        MLFeatureValue* probabilities = [result
            featureValueForName:[NSString stringWithUTF8String:probabilities_name.c_str()]];
        if (probabilities == nil || probabilities.dictionaryValue == nil)
            throw std::runtime_error("lucid.coreml: the model produced no probabilities "
                                     "named " +
                                     probabilities_name);

        std::vector<std::pair<std::string, double>> scores;
        NSDictionary<id, NSNumber*>* mapping = probabilities.dictionaryValue;
        scores.reserve(mapping.count);
        for (id key in mapping) {
            NSString* text = [key description];
            scores.emplace_back([text UTF8String], [mapping[key] doubleValue]);
        }
        return {std::string([label.stringValue UTF8String]), scores};
    }
}

std::vector<TensorImplPtr> predict(CoreMLModel* model,
                                   const std::vector<std::pair<std::string, TensorImplPtr>>& inputs,
                                   const std::vector<std::string>& output_names,
                                   const std::vector<std::pair<std::string, int>>& images) {
    if (output_names.empty())
        throw std::invalid_argument("lucid.coreml: no outputs requested");
    @autoreleasepool {
        id<MLFeatureProvider> result = run(model, inputs, images);
        std::vector<TensorImplPtr> produced;
        produced.reserve(output_names.size());
        for (const std::string& output_name : output_names)
            produced.push_back(read_output(result, output_name));
        return produced;
    }
}

}  // namespace lucid::coreml
