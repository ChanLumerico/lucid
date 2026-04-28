# Lucid C++ Engine — API Conventions

This document outlines the canonical patterns for C++ API signatures within the Lucid C++ engine.
Adherence to these conventions ensures consistency, predictability, and ease of development
for new operations and features. These rules are enforced by `tools/check_op_api.py` (Phase 1.6)
and serve as a reference for all C++ development.

## General Principles

1.  **Clarity over Brevity**: API signatures should be clear and self-documenting.
2.  **Consistency is Key**: Similar operations should have similar signatures.
3.  **PyTorch Alignment**: Where applicable, conventions align with PyTorch's C++ and Python APIs.

## Specific Conventions

### 1. Tensor Arguments

-   **Input Tensors**: Always passed by `const TensorImplPtr&`. This avoids unnecessary copies and clearly indicates that the function does not take ownership or modify the input tensor in-place (unless explicitly designed as an in-place operation, which should be rare for public APIs).
    ```cpp
    // Good
    TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);
    // Bad (takes by value)
    TensorImplPtr add_op(TensorImplPtr a, TensorImplPtr b);
    ```
-   **Optional Tensor Arguments**: For optional tensor inputs (e.g., `bias` in `Linear`, `weight` in `NLLLoss`), the C++ function should accept `const TensorImplPtr&` and handle `nullptr` gracefully. Python bindings will map `None` to `nullptr`.
    ```cpp
    // Example: Linear operation with optional bias
    TensorImplPtr linear_op(const TensorImplPtr& input, const TensorImplPtr& weight, const TensorImplPtr& bias = nullptr);
    ```

### 2. Scalar Arguments

-   **Floating Point Scalars**: Use `double` for general floating-point scalars (e.g., `lr`, `eps`, `momentum`).
-   **Integer Scalars**: Use `int64_t` for sizes, indices, counts, and other integer parameters that might exceed `int` limits. Use `int` for smaller, fixed-range integer parameters (e.g., `dim`, `reduction` enum values).
-   **Boolean Flags**: Use `bool` for boolean flags (e.g., `keepdims`, `training`, `align_corners`, `is_causal`). Do not use `int` (0 or 1) for boolean semantics.

### 3. Axis and Dimension Arguments

-   **Single Axis/Dimension**: Use `int` for specifying a single axis or dimension. Negative values should be consistently interpreted as indexing from the end.
-   **Multiple Axes/Dimensions**: Use `const std::vector<int>&` for specifying multiple axes or dimensions (e.g., `axes` in reduction ops, `dims` in permutation ops).
-   **Kernel/Stride/Padding (N-D)**: For N-dimensional operations (e.g., ConvNd, PoolNd), use `const std::vector<int>&` for parameters like `kernel_size`, `stride`, `padding`, `dilation` to allow for per-dimension specification.

### 4. Reduction Encoding

-   All loss and reduction operations must standardize on an `int` parameter for `reduction` with the following canonical values:
    -   `0`: `Reduction::None` (no reduction)
    -   `1`: `Reduction::Mean` (mean reduction)
    -   `2`: `Reduction::Sum` (sum reduction)
-   Ops with additional reduction modes extend the same integer field with documented local codes. For `einops_reduce_op`, the extended values are:
    -   `3`: max
    -   `4`: min
    -   `5`: prod
-   Python bindings may still accept strings for compatibility, but must map them to the C++ integer code before calling the op implementation.

### 5. Return Types

-   **Single Tensor Output**: Return `TensorImplPtr`.
-   **Multiple Tensor Outputs**: Return `std::vector<TensorImplPtr>`. The order and meaning of elements in the vector must be clearly documented in the Doxygen block. Avoid `std::pair` or `std::tuple` for multiple tensor returns to maintain consistency and simplify future extensions.
    ```cpp
    // Good (for ops like SVD, QR, Eig, scaled_dot_product_attention_with_weights)
    std::vector<TensorImplPtr> svd_op(const TensorImplPtr& input, bool compute_uv);
    // Bad (for multiple tensor outputs)
    std::pair<TensorImplPtr, TensorImplPtr> svd_op(const TensorImplPtr& input, bool compute_uv);
    ```

### 6. Naming Conventions

-   Follow `docs/STYLE.md` for C++ naming conventions (`CamelCase` for types, `lower_case` for functions/methods, `lower_case_` for member variables, `kCamelCase` for constants).
-   Parameter names should be descriptive and consistent across related operations.

### 7. Doxygen Documentation

-   Every public function and class must have a Doxygen-style block describing its intent, contract, parameters (`@param`), return values (`@returns`), and potential exceptions (`@throws`).

## Enforcement

These conventions are enforced by `tools/check_op_api.py` in CI and through code reviews.
Use `tools/audit_op_api.py` to dump the current signature inventory as CSV when reviewing API churn.
