# Linear Algebra (`num.linalg`)

`num.zig` includes a self-contained, high-performance linear algebra engine for matrix multiplications, vector products, matrix decompositions, solvers, and spectral analysis.

---

## 1. Matrix and Vector Products

### `matmul`
General matrix multiplication $C = A \times B$ with cache tiling and vectorization:

```zig
var a = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 1, 2, 3, 4 },
    .shape = &.{ 2, 2 },
});
defer a.deinit();

var b = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 5, 6, 7, 8 },
    .shape = &.{ 2, 2 },
});
defer b.deinit();

var c = try num.linalg.matmul(a, b, .{});
defer c.deinit();
```

### Products: `dot`, `inner`, `outer`, `kron`
- `dot`: Vector-vector inner product or matrix-vector multiplication.
- `outer`: Outer product of two 1D vectors $A \otimes B^T$.
- `kron`: Kronecker product of two matrices.

---

## 2. Matrix Inversion & Solvers

### `solve`
Solves the linear system $A x = b$:
```zig
var x = try num.linalg.solve(A, b);
defer x.deinit();
```

### `inv` & `pinv`
Computes the matrix inverse $A^{-1}$ or Moore-Penrose pseudo-inverse:
```zig
var a_inv = try num.linalg.inv(a);
defer a_inv.deinit();
```

---

## 3. Matrix Decompositions

- **LU Decomposition (`lu`)**: Computes $P \cdot A = L \cdot U$ with partial pivoting.
- **QR Decomposition (`qr`)**: Computes $A = Q \cdot R$ via Householder reflections.
- **Cholesky Decomposition (`cholesky`)**: Decomposes symmetric positive-definite $A = L \cdot L^T$.
- **Singular Value Decomposition (`svd`)**: Computes $A = U \cdot \Sigma \cdot V^T$.
- **Eigenvalues and Eigenvectors (`eig`, `eigh`)**: Solves $A v = \lambda v$ for general and symmetric/Hermitian matrices.

---

## 4. Matrix Characteristics

- **`det`**: Determinant of a square matrix.
- **`trace`**: Sum along the main diagonal.
- **`matrixRank`**: Numerical rank via singular value thresholding.
- **`norm`**: Matrix and vector norms (Frobenius, L1, L2, Linf).
- **`cond`**: Condition number.
