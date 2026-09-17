# Sparse Matrices & Solvers (`num.sparse`)

For large-scale scientific simulations and graph computations where the vast majority of entries are zero, `num.zig` provides memory-compact compressed sparse matrix representations and iterative Krylov solvers.

---

## 1. Formats: CSR & CSC

- **`CsrMatrix`**: Compressed Sparse Row format, optimal for row-slicing and matrix-vector multiplication $A \cdot x$.
- **`CscMatrix`**: Compressed Sparse Column format, optimal for column-slicing and matrix transpositions.

```zig
const std = @import("std");
const num = @import("num");
const CsrMatrix = num.sparse.CsrMatrix;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create a 4x4 dense matrix and convert to CSR
    var dense = try num.eye(allocator, .{ .n = 4, .dtype = .f64 });
    defer dense.deinit();

    var sp = try CsrMatrix.fromDense(allocator, dense, 1e-12);
    defer sp.deinit();

    std.debug.print("Sparse non-zeros: {d}\n", .{sp.nnz()});
}
```

---

## 2. Sparse Matrix-Vector Multiplication (`dotVector`)

Performs $y = A \cdot x$ in $O(\text{nnz})$ time without expanding zero elements.
Both formats provide `dotVector` with dedicated row (CSR) and column (CSC) traversal:

```zig
var x = try num.ones(allocator, .{ .shape = &.{4}, .dtype = .f64 });
defer x.deinit();

var y = try sp.dotVector(x);
defer y.deinit();
```

## 2b. Transpose

`CsrMatrix.transpose()` returns the `CscMatrix` transpose and vice versa, transferring the compressed structure directly with swapped dimensions:

```zig
var sp_t = try sp.transpose();
defer sp_t.deinit();
```

---

## 3. Conversions between Dense and Sparse

Convert between dense `Array` and sparse `CsrMatrix`:

```zig
// From Dense
var dense = try num.eye(allocator, .{ .n = 100, .dtype = .f64 });
defer dense.deinit();

var sparse_eye = try CsrMatrix.fromDense(allocator, dense, 1e-12);
defer sparse_eye.deinit();

// To Dense
var reconstructed_dense = try sparse_eye.toDense();
defer reconstructed_dense.deinit();
```

---

## 4. Iterative Solvers: CG & GMRES

Solve massive sparse linear systems $A x = b$:
- **Conjugate Gradient (`cg`)**: For symmetric positive-definite sparse systems.
- **Generalized Minimal Residual (`gmres`)**: For general non-symmetric sparse systems.

```zig
var res = try num.sparse.cg(sp, b, .{
    .maxIter = 1000,
    .tol = 1e-6,
});
defer res.deinit();

std.debug.print("Converged: {s}, residual: {d:.2e}\n", .{
    if (res.converged) "yes" else "no",
    res.residualNorm,
});
```
