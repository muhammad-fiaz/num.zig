# Linear Algebra

The `linalg` module provides linear algebra operations.

## Functions

### `dot`

Computes the dot product of two 1D arrays.

```zig
pub fn dot(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !T
```

**Parameters:**
- `T`: Data type.
- `allocator`: Memory allocator (unused for scalar return).
- `a`: First 1D array.
- `b`: Second 1D array.

**Returns:**
- Scalar dot product.

### `matmul`

Performs matrix multiplication.

```zig
pub fn matmul(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T)
```

**Parameters:**
- `T`: Data type.
- `allocator`: Memory allocator.
- `a`: First 2D array (M x K).
- `b`: Second 2D array (K x N).

**Returns:**
- `NDArray(T)` of shape (M, N).

### `trace`

Computes the trace (sum of diagonal elements).

```zig
pub fn trace(comptime T: type, a: *const NDArray(T)) !T
```

### `inverse`

Computes the multiplicative inverse of a matrix.

```zig
pub fn inverse(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T)
```

### `determinant`

Computes the determinant of a matrix.

```zig
pub fn determinant(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !T
```

### `solve`

Solves the linear system Ax = b.

```zig
pub fn solve(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T)
```

### `cholesky`

Computes the Cholesky decomposition.

```zig
pub fn cholesky(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T)
```

### `qr`

Computes the QR decomposition.

```zig
pub fn qr(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !struct { q: NDArray(T), r: NDArray(T) }
```

### `eigvals`

Computes the eigenvalues.

```zig
pub fn eigvals(comptime T: type, allocator: Allocator, a: *const NDArray(T), max_iter: usize) !NDArray(T)
```

### `eig`

Computes the eigenvalues and eigenvectors.

```zig
pub fn eig(comptime T: type, allocator: Allocator, a: *const NDArray(T), max_iter: usize) !EigResult(T)
```

### `svd`

Computes the Singular Value Decomposition.

```zig
pub fn svd(comptime T: type, allocator: Allocator, a: *const NDArray(T), max_iter: usize) !SVDResult(T)
```

### `lu`

Computes the LU decomposition.

```zig
pub fn lu(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !struct { l: NDArray(T), u: NDArray(T) }
```

### `norm`

Computes the matrix or vector norm.

```zig
pub fn norm(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !T
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const linalg = num.linalg;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = try num.NDArray(f32).init(allocator, &.{2, 2});
    // ... fill a ...
    var b = try num.NDArray(f32).init(allocator, &.{2, 2});
    // ... fill b ...

    var result = try linalg.matmul(f32, allocator, &a, &b);
    defer result.deinit();
}
```


