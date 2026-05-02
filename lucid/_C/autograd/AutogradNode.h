#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "Helpers.h"
#include "Node.h"

namespace lucid {

class TensorImpl;

template <class Derived, std::size_t N_IN>
class AutogradNode : public Node {
public:
    static constexpr std::size_t kNumInputs = N_IN;

    std::string_view name() const noexcept { return Derived::schema_v1.name; }

    void validate_versions() override {
        for (std::size_t i = 0; i < N_IN; ++i) {
            ::lucid::check_version_match(input_tensors_[i],
                                         saved_versions_.size() > i ? saved_versions_[i] : 0,
                                         Derived::schema_v1.name, i);
        }
    }

    void release_saved() override {
        for (auto& s : saved_inputs_)
            s = Storage{CpuStorage{}};
        saved_output_ = Storage{CpuStorage{}};
        input_tensors_ = {};
    }

    std::array<std::weak_ptr<TensorImpl>, N_IN> input_tensors_;

    std::array<Shape, N_IN> input_shapes_;

    Shape out_shape_;

    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    std::array<Storage, N_IN> saved_inputs_;

    Storage saved_output_;
};

}  // namespace lucid
