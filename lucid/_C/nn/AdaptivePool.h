// lucid/_C/nn/AdaptivePool.h
//
// Adaptive max-pooling and average-pooling for 1-D, 2-D, and 3-D inputs.
//
// Instead of taking ``kernel_size`` / ``stride`` as parameters, the
// adaptive variants take the *target output extent* and infer the window
// hyperparameters from the ratio of the input spatial dimension to the
// requested output dimension.  The classical use case is *global average
// pooling* with ``output_size = 1`` to collapse a feature map to a single
// per-channel value — the canonical head for modern image classifiers.
//
// Current restriction: every input spatial dimension must be evenly
// divisible by its target output extent.  When that holds the inferred
// schedule is simply ``kernel = stride = S_i / O_i`` and ``padding = 0``,
// and the call delegates to the corresponding fixed-stride pool op in
// [[PoolNd.h]].  Non-uniform partitions (which would require per-window
// boundary computation as in the standard adaptive-pool formula
// $\text{start}(i) = \lfloor i\, S / O \rfloor$,
// $\text{end}(i) = \lceil (i+1)\, S / O \rceil$) are not yet implemented;
// a descriptive ``not_implemented`` exception is thrown if the constraint
// is violated.

#pragma once

#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// One-dimensional adaptive max-pool to a fixed output length.
//
// Reduces the length axis of a 3-D input to exactly ``OL`` positions by
// taking the maximum over each contiguous segment.  When ``OL == 1``
// this is *global* max pooling along the sequence.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, L)``.
// OL : int
//     Target output length.  Must satisfy ``L % OL == 0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, OL)``.
//
// Raises
// ------
// ShapeMismatch
//     If ``x`` is not 3-D.
// not_implemented
//     If ``L`` is not evenly divisible by ``OL`` (non-uniform
//     partitioning is not yet supported).
LUCID_API TensorImplPtr adaptive_max_pool1d_op(const TensorImplPtr& x, int OL);

// Two-dimensional adaptive max-pool to a fixed ``(OH, OW)`` grid.
//
// Used in detection / segmentation models that must produce a fixed
// spatial grid regardless of input resolution (e.g. RoIAlign-style
// heads, classification networks fine-tuned on multi-scale data).
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, H, W)``.
// OH, OW : int
//     Target output extents.  Must satisfy ``H % OH == 0`` and
//     ``W % OW == 0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, OH, OW)``.
//
// Raises
// ------
// ShapeMismatch
//     If ``x`` is not 4-D.
// not_implemented
//     If any spatial axis is not evenly divisible by its target.
LUCID_API TensorImplPtr adaptive_max_pool2d_op(const TensorImplPtr& x, int OH, int OW);

// Three-dimensional adaptive max-pool to a fixed ``(OD, OH, OW)`` volume.
//
// Adaptive variant of ``max_pool3d_op`` for video classification and
// volumetric backbones that must accept clips of arbitrary duration.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, D, H, W)``.
// OD, OH, OW : int
//     Target output extents.  All three of ``D``, ``H``, ``W`` must be
//     evenly divisible by the corresponding target.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, OD, OH, OW)``.
//
// Raises
// ------
// ShapeMismatch
//     If ``x`` is not 5-D.
// not_implemented
//     If any spatial axis is not evenly divisible by its target.
LUCID_API TensorImplPtr adaptive_max_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);

// One-dimensional adaptive average-pool to a fixed output length.
//
// Reduces the length axis of a 3-D input to exactly ``OL`` positions by
// taking the arithmetic mean over each contiguous segment.  Setting
// ``OL = 1`` produces *global average pooling* along the sequence.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, L)``.
// OL : int
//     Target output length.  Must satisfy ``L % OL == 0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, OL)``.
//
// Raises
// ------
// ShapeMismatch
//     If ``x`` is not 3-D.
// not_implemented
//     If ``L`` is not evenly divisible by ``OL``.
LUCID_API TensorImplPtr adaptive_avg_pool1d_op(const TensorImplPtr& x, int OL);

// Two-dimensional adaptive average-pool to a fixed ``(OH, OW)`` grid.
//
// The canonical global average pooling head of modern classification
// backbones (ResNet, EfficientNet, ConvNeXt, ...) uses
// ``OH = OW = 1``.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, H, W)``.
// OH, OW : int
//     Target output extents.  Must satisfy ``H % OH == 0`` and
//     ``W % OW == 0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, OH, OW)``.
//
// Raises
// ------
// ShapeMismatch
//     If ``x`` is not 4-D.
// not_implemented
//     If any spatial axis is not evenly divisible by its target.
LUCID_API TensorImplPtr adaptive_avg_pool2d_op(const TensorImplPtr& x, int OH, int OW);

// Three-dimensional adaptive average-pool to a fixed ``(OD, OH, OW)``
// volume.
//
// Used as the final pooling layer in 3-D classification networks
// (3D-ResNet, SlowFast).  ``OD = OH = OW = 1`` yields the volumetric
// global-average-pool head.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C, D, H, W)``.
// OD, OH, OW : int
//     Target output extents.  Each must evenly divide the
//     corresponding input axis.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C, OD, OH, OW)``.
//
// Raises
// ------
// ShapeMismatch
//     If ``x`` is not 5-D.
// not_implemented
//     If any spatial axis is not evenly divisible by its target.
LUCID_API TensorImplPtr adaptive_avg_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);

}  // namespace lucid
