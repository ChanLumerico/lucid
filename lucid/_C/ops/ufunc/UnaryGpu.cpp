// =====================================================================
// Lucid C++ engine — Phase 4.5: all unary GPU kernels have been migrated
// to GpuBackend::* methods (backend/gpu/GpuBackend.h).
//
// Only the activation ops with scalar parameters (LeakyRelu, Elu) still
// own their own GPU forward paths inside Activation.cpp since they cannot
// use the standard single-input dispatch signature.
// =====================================================================
