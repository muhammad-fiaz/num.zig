# Linear Algebra

The `num.linalg` module provides standard linear algebra operations.

## Matrix Multiplication

Use `matmul` for matrix multiplication. It handles 2D arrays (matrices).

```zig
const num = @import("num");
// ...
var c = try num.linalg.matmul(f32, allocator, &a, &b);
```

## Dot Product

Use `dot` for the dot product of two 1D arrays (vectors).

```zig
var d = try num.linalg.dot(f32, allocator, &v1, &v2);
```

## Solving Linear Systems

Solve `Ax = b` for `x`.

```zig
var x = try num.linalg.solve(f32, allocator, &A, &b);
```

## Decompositions

`num.zig` supports several matrix decompositions.

### Cholesky Decomposition

Compute the Cholesky decomposition of a positive definite matrix.

```zig
var L = try num.linalg.cholesky(f32, allocator, A);
```

### QR Decomposition

Compute the QR decomposition of a matrix.

```zig
var qr_res = try num.linalg.qr(f32, allocator, A);
// qr_res.q is the orthogonal matrix Q
// qr_res.r is the upper triangular matrix R
```

### Eigenvalues

Compute the eigenvalues of a general matrix.

```zig
var evals = try num.linalg.eigvals(f32, allocator, A, 100);
```

### Singular Value Decomposition (SVD)

Compute the SVD of a matrix.

```zig
var svd_res = try num.linalg.svd(f32, allocator, A, 100);
// svd_res.u, svd_res.s, svd_res.vt
```

## Other Operations

- `trace`: Sum of diagonal elements.
- `determinant`: Compute the determinant of a square matrix.
- `inverse`: Compute the multiplicative inverse of a matrix.
