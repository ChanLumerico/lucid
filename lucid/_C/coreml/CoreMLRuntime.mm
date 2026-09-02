// lucid/_C/coreml/CoreMLRuntime.mm — see CoreMLRuntime.h.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <memory>
#include <stdexcept>
#include <string>
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

// First declared feature name of a description, or empty when absent.
std::string first_name(NSDictionary<NSString*, MLFeatureDescription*>* features) {
    for (NSString* key in features)
        return std::string([key UTF8String]);
    return {};
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
    std::string input_name;
    std::string output_name;

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
        handle->input_name = first_name(model.modelDescription.inputDescriptionsByName);
        handle->output_name = first_name(model.modelDescription.outputDescriptionsByName);
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

std::string input_feature_name(const CoreMLModel* model) {
    return model == nullptr ? std::string{} : model->input_name;
}

std::string output_feature_name(const CoreMLModel* model) {
    return model == nullptr ? std::string{} : model->output_name;
}

TensorImplPtr predict(CoreMLModel* model,
                      const std::string& input_name,
                      const TensorImplPtr& input,
                      const std::string& output_name) {
    if (model == nullptr || model->model == nil)
        throw std::invalid_argument("lucid.coreml: null model handle");
    if (!input)
        throw std::invalid_argument("lucid.coreml: null input tensor");
    if (input->device() != Device::CPU)
        throw std::invalid_argument(
            "lucid.coreml: the input must be a CPU tensor — Core ML reads host "
            "memory, and moving it here would hide a copy the caller did not ask for");
    if (input->dtype() != Dtype::F32 && input->dtype() != Dtype::I32)
        throw std::invalid_argument(
            "lucid.coreml: inputs must be float32 or int32 — Core ML's multi-array "
            "has no int64, so token ids are narrowed by the caller");

    const auto& storage = std::get<CpuStorage>(input->storage());
    if (!storage.ptr)
        throw std::runtime_error("lucid.coreml: the input tensor has no host storage");

    @autoreleasepool {
        const Shape& shape = input->shape();
        NSMutableArray<NSNumber*>* ns_shape = [NSMutableArray arrayWithCapacity:shape.size()];
        for (std::int64_t dim : shape)
            [ns_shape addObject:@(dim)];

        // Row-major strides, in elements, as MLMultiArray counts them.
        std::vector<std::int64_t> strides(shape.size(), 1);
        for (std::size_t i = shape.size(); i-- > 1;)
            strides[i - 1] = strides[i] * shape[i];
        NSMutableArray<NSNumber*>* ns_strides = [NSMutableArray arrayWithCapacity:strides.size()];
        for (std::int64_t stride : strides)
            [ns_strides addObject:@(stride)];

        NSError* error = nil;
        // The array borrows Lucid's buffer; ``deallocator:nil`` says Core
        // ML must not free it.  The tensor outlives this scope because
        // the caller holds it.
        const MLMultiArrayDataType in_type = input->dtype() == Dtype::I32
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
            throw std::runtime_error("lucid.coreml: cannot wrap the input: " + describe(error));

        NSString* in_key = [NSString stringWithUTF8String:input_name.c_str()];
        MLDictionaryFeatureProvider* features = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{in_key : [MLFeatureValue featureValueWithMultiArray:array]}
                         error:&error];
        if (features == nil)
            throw std::runtime_error("lucid.coreml: cannot build the input features: " +
                                     describe(error));

        id<MLFeatureProvider> result = [model->model predictionFromFeatures:features error:&error];
        if (result == nil)
            throw std::runtime_error("lucid.coreml: prediction failed: " + describe(error));

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

        // ``getBytesWithHandler`` is the supported way to read the
        // backing store: the array may be strided or held on another
        // device, and this hands over a contiguous view either way.
        __block bool copied = false;
        [out getBytesWithHandler:^(const void* raw, NSInteger size) {
          if (raw != nullptr && static_cast<std::size_t>(size) >= nbytes) {
              std::memcpy(bytes.get(), raw, nbytes);
              copied = true;
          }
        }];
        if (!copied)
            throw std::runtime_error("lucid.coreml: could not read the output buffer");

        CpuStorage cpu;
        cpu.ptr = std::move(bytes);
        cpu.nbytes = nbytes;
        cpu.dtype = Dtype::F32;
        return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, out_shape, Dtype::F32,
                                            Device::CPU, false);
    }
}

}  // namespace lucid::coreml
