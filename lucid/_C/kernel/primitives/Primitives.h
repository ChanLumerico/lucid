// lucid/_C/kernel/primitives/Primitives.h
//
// Aggregated include for all kernel primitive helpers. Including this single
// header pulls in BatchedMatmul, BroadcastReduce, Gather, Im2Col, and Scatter
// so that ops/ nodes do not need to enumerate individual primitive headers.
// This header has no content of its own; it is purely a convenience umbrella.

#pragma once

#include "BatchedMatmul.h"
#include "BroadcastReduce.h"
#include "Gather.h"
#include "Im2Col.h"
#include "Scatter.h"
