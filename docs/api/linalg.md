# Linear Algebra API

Module: `@import("num").linalg`

---

## Matrix Multiplication & Products

```zig
pub fn matmul(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn dot(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn inner(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn outer(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn kron(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
```

`dot` and `inner` are deliberate aliases of `matmul` with identical semantics.

---

## Solvers and Inverses

```zig
pub fn solve(a: Array, b: Array) !Array;
pub fn solveTriangular(a: Array, b: Array, options: struct { lower: bool = true }) !Array;
pub fn solveSpd(a: Array, b: Array) !Array;
pub fn lstsq(a: Array, b: Array) !Array;
pub fn inv(a: Array) !Array;
pub fn pinv(a: Array, options: struct { rcond: f64 = 1e-15 }) !Array;
pub fn matrixPower(a: Array, n: isize) !Array;
```

---

## Decompositions

```zig
pub const LuResult = struct {
    p: Array,
    l: Array,
    u: Array,
    pub fn deinit(self: *LuResult) void;
};
pub fn lu(a: Array) !LuResult;

pub const QrResult = struct {
    q: Array,
    r: Array,
    pub fn deinit(self: *QrResult) void;
};
pub fn qr(a: Array) !QrResult;

pub fn cholesky(a: Array) !Array;

pub const SvdResult = struct {
    u: Array,
    s: Array,
    vt: Array,
    pub fn deinit(self: *SvdResult) void;
};
pub fn svd(a: Array, options: struct { full_matrices: bool = true, max_iterations: usize = 100, tol: f64 = 1e-12 }) !SvdResult;

pub const EigenResult = struct {
    values: Array,
    vectors: ?Array = null,
    pub fn deinit(self: *EigenResult) void;
};
pub fn eig(a: Array, options: struct { compute_vectors: bool = true, max_iterations: usize = 100, tol: f64 = 1e-12 }) !EigenResult;
pub fn eigvals(a: Array) !Array;
```

---

## Norms and Metrics

```zig
pub fn det(a: Array) !Array;
pub const SlogdetResult = struct {
    sign: Array,
    logabsdet: Array,
    pub fn deinit(self: *SlogdetResult) void;
};
pub fn slogdet(a: Array) !SlogdetResult;
pub fn trace(a: Array) !Array;
pub fn matrixRank(a: Array, options: struct { tol: ?f64 = null }) !usize;
pub fn norm(a: Array, options: struct { ord: NormOrder = .l2, axis: ?isize = null, keepDims: bool = false }) !Array;
```


