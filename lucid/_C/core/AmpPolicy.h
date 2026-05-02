#pragma once

#include <optional>

#include "../api.h"
#include "Dtype.h"

namespace lucid {

enum class AmpPolicy : std::uint8_t {
    Promote,
    KeepInput,
    ForceFP32,
};

LUCID_API const char* amp_policy_name(AmpPolicy p);

namespace amp {

LUCID_API std::optional<Dtype> active_dtype();

LUCID_API bool is_active();

class LUCID_API AutocastGuard {
public:
    explicit AutocastGuard(Dtype target);
    ~AutocastGuard();

    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;

private:
    bool prev_active_;
    Dtype prev_dtype_;
};

}  // namespace amp
}  // namespace lucid
