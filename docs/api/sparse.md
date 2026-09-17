# Sparse Matrix API

Module: `@import("num").sparse`

---

## Types

### `CsrMatrix`
Compressed Sparse Row matrix representation for $O(\text{nnz})$ row-access and matrix-vector products.

```zig
pub const CsrMatrix = struct {
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    data: []f64,
    indices: []usize,
    indptr: []usize,

    pub fn deinit(self: *CsrMatrix) void;
    pub fn nnz(self: CsrMatrix) usize;
    pub fn fromDense(allocator: std.mem.Allocator, dense: Array, tol: f64) !CsrMatrix;
    pub fn toDense(self: CsrMatrix) !Array;
    pub fn dotVector(self: CsrMatrix, x: Array) !Array;
    pub fn transpose(self: CsrMatrix) !CscMatrix;
};
```

### `CscMatrix`
Compressed Sparse Column matrix representation.

```zig
pub const CscMatrix = struct {
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    data: []f64,
    indices: []usize,
    indptr: []usize,

    pub fn deinit(self: *CscMatrix) void;
    pub fn nnz(self: CscMatrix) usize;
    pub fn fromDense(allocator: std.mem.Allocator, dense: Array, tol: f64) !CscMatrix;
    pub fn toDense(self: CscMatrix) !Array;
    pub fn dotVector(self: CscMatrix, x: Array) !Array;
    pub fn transpose(self: CscMatrix) !CsrMatrix;
};
```

---

## Solvers

### `cg` (Conjugate Gradient)
Solves symmetric positive-definite systems $A x = b$:

```zig
pub fn cg(
    A: CsrMatrix,
    b: Array,
    options: struct {
        maxIter: usize = 1000,
        tol: f64 = 1e-6,
        x0: ?Array = null,
    },
) !SparseSolveResult;
```

### `gmres` (Generalized Minimal Residual)
Solves general non-symmetric square systems $A x = b$:

```zig
pub fn gmres(
    A: CsrMatrix,
    b: Array,
    options: struct {
        restart: usize = 30,
        maxIter: usize = 1000,
        tol: f64 = 1e-6,
        x0: ?Array = null,
    },
) !SparseSolveResult;
```

### `SparseSolveResult`

```zig
pub const SparseSolveResult = struct {
    x: Array,
    iterations: usize,
    converged: bool,
    residualNorm: f64,

    pub fn deinit(self: *SparseSolveResult) void;
};
```

