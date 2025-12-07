# Sparse Matrices

The `sparse` module provides support for sparse matrix formats, efficient for matrices with many zero elements.

## CSRMatrix

Compressed Sparse Row (CSR) matrix format.

### `init`

Initializes an empty CSR matrix.

```zig
pub fn init(allocator: Allocator, rows: usize, cols: usize) !Self
```

### `fromDense`

Converts a dense `NDArray` to CSR format.

```zig
pub fn fromDense(allocator: Allocator, arr: NDArray(T)) !Self
```

**Parameters:**
- `allocator`: Memory allocator.
- `arr`: Input dense `NDArray` (must be rank 2).

**Returns:**
- A new `CSRMatrix`.

### `toDense`

Converts the CSR matrix back to a dense `NDArray`.

```zig
pub fn toDense(self: Self) !NDArray(T)
```

**Returns:**
- A new dense `NDArray`.

## Example

```zig
const std = @import("std");
const num = @import("num");
const CSRMatrix = num.sparse.CSRMatrix;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Create dense matrix
    var dense = try num.NDArray(f64).init(allocator, &.{3, 3});
    try dense.set(&.{0, 0}, 1.0);
    try dense.set(&.{2, 2}, 5.0);

    // Convert to CSR
    var csr = try CSRMatrix(f64).fromDense(allocator, dense);
    defer csr.deinit();

    // Convert back
    var dense2 = try csr.toDense();
    defer dense2.deinit();
}
```
