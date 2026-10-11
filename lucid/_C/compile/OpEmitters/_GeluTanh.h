// lucid/_C/compile/OpEmitters/_GeluTanh.h
//
// The tanh-approximate GELU as an MPSGraph subgraph, shared by the ``gelu``
// emitter and the ``_fused_linear_gelu`` one so both spell the backends'
// formula the same way:
//
//     y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace lucid::compile {

inline MPSGraphTensor* gelu_tanh_graph(MPSGraph* g, MPSGraphTensor* x, NSString* name) {
    MPSDataType dt = x.dataType;
    MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dt];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
    MPSGraphTensor* a044 = [g constantWithScalar:0.044715 dataType:dt];
    MPSGraphTensor* k = [g constantWithScalar:0.7978845608028654 dataType:dt];  // sqrt(2/π)
    MPSGraphTensor* x2 = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
    MPSGraphTensor* x3 = [g multiplicationWithPrimaryTensor:x2 secondaryTensor:x name:nil];
    MPSGraphTensor* ax3 = [g multiplicationWithPrimaryTensor:a044 secondaryTensor:x3 name:nil];
    MPSGraphTensor* inner_add = [g additionWithPrimaryTensor:x secondaryTensor:ax3 name:nil];
    MPSGraphTensor* inner = [g multiplicationWithPrimaryTensor:k
                                               secondaryTensor:inner_add
                                                          name:nil];
    MPSGraphTensor* th = [g tanhWithTensor:inner name:nil];
    MPSGraphTensor* one_plus = [g additionWithPrimaryTensor:one secondaryTensor:th name:nil];
    MPSGraphTensor* half_x = [g multiplicationWithPrimaryTensor:half secondaryTensor:x name:nil];
    return [g multiplicationWithPrimaryTensor:half_x secondaryTensor:one_plus name:name];
}

}  // namespace lucid::compile
