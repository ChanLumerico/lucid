// lucid/_C/kernel/primitives/Im2Col.h
//
// Re-export of the CPU im2col/col2im functions into the kernel primitives
// namespace so that ops/ convolution kernels can include a single short
// path rather than reaching into backend/cpu/ directly. The actual
// function declarations and documentation live in backend/cpu/Im2Col.h.

#pragma once

#include "../../backend/cpu/Im2Col.h"
