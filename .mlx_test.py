import mlx.core as mx
import numpy as np  # Only for demonstrating the TypeError
from lucid._backend.metal import parse_mlx_indexing

# Create an MLX tensor
x = mx.array([[10, 20, 30], [40, 50, 60]])

# ✅ Standard Indexing (No Parsing Needed)
print("1:", x[1])  # Expected: array([40, 50, 60])

# ✅ Slice Indexing
print("2:", x[0:2])  # Expected: array([[10, 20, 30], [40, 50, 60]])

# ✅ Fancy Indexing with List
idx_list = [0, 1]
print(
    "3:", x[parse_mlx_indexing(idx_list)]
)  # Expected: array([[10, 20, 30], [40, 50, 60]])

# ✅ Fancy Indexing with MLX Array
idx_mx = mx.array([0, 1])
print(
    "4:", x[parse_mlx_indexing(idx_mx)]
)  # Expected: array([[10, 20, 30], [40, 50, 60]])

# ✅ Multi-dimensional Fancy Indexing
row_idx = mx.array([0, 1])
col_idx = mx.array([1, 2])
print("5:", x[parse_mlx_indexing((row_idx, col_idx))])  # Expected: array([20, 60])

# ✅ Boolean List Indexing
bool_list = [True, False]
print("6:", x[parse_mlx_indexing(bool_list)])  # Expected: array([[10, 20, 30]])

# ✅ Boolean MLX Array Indexing
bool_mask = mx.array([True, False])
print("7:", x[parse_mlx_indexing(bool_mask)])  # Expected: array([[10, 20, 30]])

# ✅ Boolean Masking with Multi-dimension
bool_mask_2d = mx.array([[True, False, True], [False, True, False]])
masked_x = parse_mlx_indexing(bool_mask_2d)
print("8:", x.flatten()[masked_x])  # Expected: array([10, 30, 50])

# ✅ Tuple of Lists (Multi-dimensional Fancy Indexing)
multi_idx = ([0, 1], [1, 2])
print("9:", x[parse_mlx_indexing(multi_idx)])  # Expected: array([20, 60])

# ✅ Single Boolean Indexing
print(
    "10:", x[parse_mlx_indexing(True)]
)  # Expected: array([40, 50, 60])  (Equivalent to x[1])
print(
    "11:", x[parse_mlx_indexing(False)]
)  # Expected: array([10, 20, 30]) (Equivalent to x[0])

# ❌ NumPy Indexing (Should Raise TypeError)
try:
    print("12:", x[parse_mlx_indexing(np.array([0, 1]))])
except TypeError as e:
    print(
        "12: Error Caught:", e
    )  # Expected: "MLX does not support NumPy arrays as indices."
