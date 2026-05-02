#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr
einops_rearrange_op(const TensorImplPtr& a,
                    const std::string& pattern,
                    const std::map<std::string, std::int64_t>& axes_lengths);

LUCID_API TensorImplPtr einops_reduce_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         int reduction,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

LUCID_API TensorImplPtr einops_repeat_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

LUCID_API TensorImplPtr einsum_op(const std::string& pattern,
                                  const std::vector<TensorImplPtr>& operands);

}  // namespace lucid
