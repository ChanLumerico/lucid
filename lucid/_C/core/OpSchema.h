#pragma once

#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

#include "../api.h"
#include "AmpPolicy.h"

namespace lucid {

struct LUCID_API OpSchema {
    std::string_view name;
    int version;

    AmpPolicy amp_policy;
    bool deterministic;
    std::string_view determinism_note;

    int input_arity = -1;

    int output_arity = 1;

    std::vector<int> stable_input_indices;

    bool internal = false;

    constexpr OpSchema(
        std::string_view n, int v, AmpPolicy ap, bool det = true, std::string_view note = "")
        : name(n), version(v), amp_policy(ap), deterministic(det), determinism_note(note) {}

    OpSchema(std::string_view n,
             int v,
             AmpPolicy ap,
             bool det,
             std::string_view note,
             int in_arity,
             int out_arity,
             std::vector<int> stable_ins,
             bool is_internal = false)
        : name(n),
          version(v),
          amp_policy(ap),
          deterministic(det),
          determinism_note(note),
          input_arity(in_arity),
          output_arity(out_arity),
          stable_input_indices(std::move(stable_ins)),
          internal(is_internal) {}
};

LUCID_API std::uint64_t schema_hash(const OpSchema& s);

}  // namespace lucid
