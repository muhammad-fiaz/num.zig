<h1 align="center">Num.Zig</h1>

<div align="center">

<a href="https://muhammad-fiaz.github.io/num.zig/"><img src="https://img.shields.io/badge/docs-muhammad--fiaz.github.io-blue" alt="Documentation"></a>
<a href="https://ziglang.org/"><img src="https://img.shields.io/badge/Zig-0.16.0-orange.svg?logo=zig" alt="Zig Version"></a>
<a href="https://github.com/muhammad-fiaz/num.zig"><img src="https://img.shields.io/github/stars/muhammad-fiaz/num.zig" alt="GitHub stars"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/issues"><img src="https://img.shields.io/github/issues/muhammad-fiaz/num.zig" alt="GitHub issues"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/pulls"><img src="https://img.shields.io/github/issues-pr/muhammad-fiaz/num.zig" alt="GitHub pull requests"></a>
<a href="https://github.com/muhammad-fiaz/num.zig"><img src="https://img.shields.io/github/last-commit/muhammad-fiaz/num.zig" alt="GitHub last commit"></a>
<a href="https://github.com/muhammad-fiaz/num.zig"><img src="https://img.shields.io/github/license/muhammad-fiaz/num.zig" alt="License"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/actions/workflows/ci.yml"><img src="https://github.com/muhammad-fiaz/num.zig/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/platforms-linux%20%7C%20windows%20%7C%20macos-blue" alt="Supported Platforms">
<a href="https://github.com/muhammad-fiaz/num.zig/actions/workflows/github-code-scanning/codeql"><img src="https://github.com/muhammad-fiaz/num.zig/actions/workflows/github-code-scanning/codeql/badge.svg" alt="CodeQL"></a>
<a href="https://github.com/muhammad-fiaz/num.zig/releases/latest"><img src="https://img.shields.io/github/v/release/muhammad-fiaz/num.zig?label=Latest%20Release&style=flat-square" alt="Latest Release"></a>
<a href="https://pay.muhammadfiaz.com"><img src="https://img.shields.io/badge/Sponsor-pay.muhammadfiaz.com-ff69b4?style=flat&logo=heart" alt="Sponsor"></a>
<a href="https://github.com/sponsors/muhammad-fiaz"><img src="https://img.shields.io/badge/Sponsor-GitHub-pink?style=social&logo=github" alt="GitHub Sponsors"></a>
<a href="https://hits.sh/muhammad-fiaz/num.zig/"><img src="https://hits.sh/muhammad-fiaz/num.zig.svg?label=Visitors&extraCount=0&color=green" alt="Repo Visitors"></a>

<p><em>A Fast, High-Performance Numerical Computing and N-Dimensional Array Library for Zig.</em></p>

<b><a href="https://muhammad-fiaz.github.io/num.zig/">Documentation</a> |
<a href="https://muhammad-fiaz.github.io/num.zig/api/">API Reference</a> |
<a href="https://muhammad-fiaz.github.io/num.zig/guide/getting-started">Quick Start</a> |
<a href="CONTRIBUTING.md">Contributing</a></b>

</div>

`num.zig` is a modern, native numerical computing library for Zig, providing high-performance N-dimensional arrays, vectorized SIMD math, linear algebra, statistics, pseudo-random number distributions, sorting, polynomial analysis, and portable **NZIG v1.0** binary array serialization.

> [!IMPORTANT]
> **v0.0.3 is the current release.** It completes the production multidimensional array system with robust shape/stride handling, full broadcasting, views, elementwise math, bitwise and complex helpers, reductions, statistics, sorting, linear algebra, FFT, random generation, polynomials with root finding, sparse matrices with inline-configured iterative solvers, parallel execution via `num.parallel.run`, SIMD kernels, and portable NZIG v1.0 serialization. If you are migrating from v0.0.2, review the updated clean namespace usage below (`num.ops`, `num.linalg`, `num.poly`, `num.manip`). The project is in active development and contributions are welcome.

> [!TIP]
> If you build with num.zig, make sure to give it a star!

> [!NOTE]
> **Project maturity:** This project is under active development. It provides an idiomatic, native Zig numerical computation engine with explicit memory allocation discipline, zero hidden allocations, and fast zero-copy views for slicing, reshaping, transposing, and broadcast expansion.

**Related Zig projects:**

- For **HTTP Client & Server** support, check out **[httpx.zig](https://github.com/muhammad-fiaz/httpx.zig)**.
- For **Env.zig** (.env parsing), check out **[env.zig](https://github.com/muhammad-fiaz/env.zig)**.
- For **TUI** support, check out **[tui.zig](https://github.com/muhammad-fiaz/tui.zig)**.
- For **ZON file format** support, check out **[zon.zig](https://github.com/muhammad-fiaz/zon.zig)**.
- For **Spinners/loading/progress bar** support, check out **[loaders.zig](https://github.com/muhammad-fiaz/loaders.zig)**.
- For **MCP** support, check out **[mcp.zig](https://github.com/muhammad-fiaz/mcp.zig)**.
- For **API framework** support, check out **[api.zig](https://github.com/muhammad-fiaz/api.zig)**.
- For **Web framework** support, check out **[zix](https://github.com/muhammad-fiaz/zix)**.
- For **archive/compression** support, check out **[archive.zig](https://github.com/muhammad-fiaz/archive.zig)**.
- For **compression file format** support, check out **[zigx](https://github.com/muhammad-fiaz/zigx)**.
- For **CUDA** support, check out **[cuda.zig](https://github.com/muhammad-fiaz/cuda.zig)**.
- For **Simplified build.zig config** support, check out **[buildx.zig](https://github.com/muhammad-fiaz/buildx.zig)**.
- For **SQLite (zig-native implementation)** support, check out **[sqlite.zig](https://github.com/muhammad-fiaz/sqlite.zig)**.
- For **File downloading** support, check out **[downloader.zig](https://github.com/muhammad-fiaz/downloader.zig)**.
- For **update checker/auto-updater** support, check out **[updater.zig](https://github.com/muhammad-fiaz/updater.zig)**.
- For **Logging** support, check out **[logly.zig](https://github.com/muhammad-fiaz/logly.zig)**.
- For **Data validation and serialization** support, check out **[zigantic](https://github.com/muhammad-fiaz/zigantic)**.
- For **UUID** support, check out **[uuid.zig](https://github.com/muhammad-fiaz/uuid.zig)**.
- For **Key-Value database** support, check out **[zkv.zig](https://github.com/muhammad-fiaz/zkv.zig)**.
- For **Terminal color & text styles** support, check out **[hint.zig](https://github.com/muhammad-fiaz/hint.zig)**.
- For **Brotli compression** support, check out **[brotli.zig](https://github.com/muhammad-fiaz/brotli.zig)**.
- For **Zstd compression** support, check out **[zstd.zig](https://github.com/muhammad-fiaz/zstd.zig)**.
- For **Tree-Sitter** support, check out **[tree-sitter.zig](https://github.com/muhammad-fiaz/tree-sitter.zig)**.

---

<details>
<summary><strong>Features</strong> (click to expand)</summary>

| Feature | Description |
|---|---|
| **N-Dimensional Array (`Array`)** | High-performance unified multidimensional container with Small Buffer Optimization (SBO) up to 8 dimensions inline with zero heap overhead for shapes, strides, and dimension metadata. |
| **Complete DType System** | Full native support across floating-point (`f64`, `f32`, `f16`), complex numbers (`c128`, `c64`), signed integers (`i64`, `i32`, `i16`, `i8`), unsigned integers (`u64`, `u32`, `u16`, `u8`), and `bool` with deterministic numeric type promotion. |
| **Explicit Memory Discipline** | Clean allocator discipline with no hidden global allocations; zero-copy views for slicing, reshaping, transposing, flattening, and broadcasting. |
| **Vectorized Math & SIMD** | Vectorized kernels for arithmetic (`add`, `subtract`/`sub`, `multiply`/`mul`, `divide`/`div`, `pow`/`power`, `remainder`/`mod`, `minimum`, `maximum`), unary math (`negate`/`negative`, `positive`, `abs`/`absolute`, `sqrt`, `square`, `reciprocal`, `sign`, `floor`/`ceil`/`trunc`/`round`), trigonometry (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `hypot`), hyperbolics (`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`), exponentials/logarithms (`exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`), special functions (`gamma`, `lgamma`, `erf`, `erfc`, `cbrt`), clipping, and boolean conditionals (`where`). |
| **Bitwise Integers (`num.ops`)** | Integer-only broadcast operations (`bitwiseAnd`, `bitwiseOr`, `bitwiseXor`, `bitwiseNot`, `leftShift`, `rightShift`) with population utilities (`bitCount`/`popcount`, `clz`/`leadingZeros`, `ctz`/`trailingZeros`); floating-point inputs are rejected. |
| **Complex Helpers (`num.ops`)** | Complex construction via `c64`/`c128` dtypes with `conj`/`conjugate`, `real`, `imag`, `magnitude`, and `phase`; complex arithmetic, matrix operations, FFT, and statistics paths are covered. |
| **Multidimensional Broadcasting** | Automatic broadcast semantics for binary operations following standard dimension-alignment rules, plus explicit zero-copy `broadcastTo` views. |
| **Linear Algebra (`num.linalg`)** | Cache-blocked matrix multiplication (`matmul`), inner/outer products, vector and matrix norms (L1, L2, Linf, Frobenius), LU decomposition, QR decomposition (Householder reflections), Cholesky factorization, linear system solver (`solve`), matrix inverse (`inv`), determinant (`det`), matrix trace (`trace`), and rank (`matrixRank`). |
| **Spectral & Decompositions** | General eigenvalue & eigenvector solver (`eig`, `eigvals`) via Hessenberg reduction and QR iteration, and Singular Value Decomposition (`svd`) via Golub-Kahan bidiagonalization. |
| **Fast Fourier Transform (`num.fft`)** | 1D and multidimensional Fast Fourier Transform (`fft`, `ifft`) with Radix-2 Cooley-Tukey and direct DFT fallback, supporting complex inputs and backward/ortho/forward normalizations. |
| **Sparse Matrices & Solvers (`num.sparse`)** | Compressed Sparse Row (`CsrMatrix`) and Compressed Sparse Column (`CscMatrix`) formats with memory-efficient storage, matrix-vector multiplication, dense roundtripping, and inline-configured iterative solvers: Conjugate Gradient (`cg`) and Restarted GMRES (`gmres`), e.g. `num.sparse.cg(A, b, .{ .tol = 1e-8, .maxIter = 1000 })`. |
| **Parallel CPU Execution (`num.parallel`)** | Deterministic data-parallel engine (`run`) utilizing native Zig 0.16.0 `std.Thread` with automatic CPU core count detection, inline configuration (`.{ .workers, .chunkSize }`), and sequential fallback for small workloads. |
| **Reductions (`num.reduce`)** | Reductions along global and specific axes with `keepDims` and negative axes: `sum`, `prod`, `mean`, `median`, `variance`, `stdDev`, `min`, `max`, `argmin`, `argmax`, `all`, `any`, `countNonzero`, `cumsum`, `cumprod`, `cummin`, `cummax`, and `diff`. |
| **Sorting & Searching (`num.sort`)** | In-place and copy quicksort, `argsort` permutation generation, binary search (`searchSorted`), distinct value extraction (`unique`), linear index filtering (`flatNonzero`, `nonzero`, `argwhere`), set operations (`intersect1d`, `union1d`, `setdiff1d`, `isin`), and coordinate utilities (`ravelIndex`, `unravelIndex`, `indices`). |
| **Polynomial Calculus (`num.poly`)** | Horner's method evaluation (`poly.val`), least-squares polynomial curve fitting (`poly.fit`), polynomial derivatives (`poly.der`), indefinite integration (`poly.integ`), root finding (`poly.roots` with analytic linear/quadratic and Durand-Kerner general solver), and coefficient arithmetic (`poly.add`, `poly.sub`, `poly.mul`). |
| **Descriptive Statistics (`num.stats`)** | Mean, variance, standard deviation, median, quantiles/percentiles (with linear, lower, higher, midpoint, and nearest interpolation), covariance matrices, Pearson correlation matrices, and histogram binning. |
| **Random Number Distributions (`num.random`)** | Seedable pseudo-random engine (`Prng`), uniform float distributions, standard normal distribution (Box-Muller transform), discrete random integers, random choice sampling, and Fisher-Yates array shuffling. |
| **Native NZIG v1.0 Serialization (`num.io`)** | Portable, deterministic, 64-byte aligned, 128-byte header binary file format independent of host compiler ABI, supporting 32-bit and 64-bit systems across Windows, Linux, and macOS. Also includes delimited text I/O (`savetxt`, `loadtxt`). |

</details>

---

<details>
<summary><strong>Prerequisites and Supported Platforms</strong> (click to expand)</summary>

<br>

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Zig** | **0.16.0** (recommended) | Download from [ziglang.org](https://ziglang.org/download/) |
| **Operating System** | Windows 10+, Linux, macOS | Cross-platform numerical computing |

> [!IMPORTANT]
> **Zig 0.16.0 is required.** This project targets the stable Zig 0.16.0 release and uses standard library conventions without deprecated GeneralPurposeAllocator patterns.

---

## Supported Platforms

| Platform | x86_64 (64-bit) | aarch64 (ARM64) | x86 (32-bit) |
|---|---|---|---|
| **Linux** | Yes | Yes | Yes |
| **Windows** | Yes | Yes | Yes |
| **macOS** | Yes | Yes (Apple Silicon) | No |

### Cross-Compilation

```bash
# Build for Linux ARM64 from Windows
zig build -Dtarget=aarch64-linux

# Build for Windows x86_64 from Linux
zig build -Dtarget=x86_64-windows

# Build for macOS Apple Silicon from Linux
zig build -Dtarget=aarch64-macos

# Build for 32-bit Windows
zig build -Dtarget=x86-windows
```

</details>

---

## Installation

### Method 1: Zig Fetch (Recommended)

**Latest Release (v0.0.3)**

```bash
zig fetch --save https://github.com/muhammad-fiaz/num.zig/archive/refs/tags/v0.0.3.tar.gz
```

**Previous Releases (v0.0.2, v0.0.1)**

```bash
zig fetch --save https://github.com/muhammad-fiaz/num.zig/archive/refs/tags/v0.0.2.tar.gz
```

> [!WARNING]
> Zig **0.15** is deprecated. New projects should use **Zig 0.16.0+** with **num.zig v0.0.3**.

### Method 2: Zig Fetch (Latest Development Build)

Use this for the latest development build from the `main` branch:

```bash
zig fetch --save git+https://github.com/muhammad-fiaz/num.zig.git
```

### Method 3: Manual `build.zig.zon` Configuration

```zig
.dependencies = .{
    .num = .{
        .url = "https://github.com/muhammad-fiaz/num.zig/archive/refs/tags/v0.0.3.tar.gz",
        .hash = "...", // Run `zig fetch --save <url>` to generate the hash automatically.
    },
},
```

### Method 4: Local Source Checkout

```bash
git clone https://github.com/muhammad-fiaz/num.zig.git
cd num.zig
zig build test
```

To use a local checkout from another project:

```zig
.dependencies = .{
    .num = .{
        .path = "../num.zig",
    },
},
```

### Wire into `build.zig`

```zig
const num_dep = b.dependency("num", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("num", num_dep.module("num"));
```

---

## Quick Start

### Basic Matrix Operations & Broadcasting

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Create a 2x3 matrix
    var a = try num.arange(allocator, .{ .start = 0, .stop = 6, .dtype = .f64 });
    defer a.deinit();
    var a_2x3 = try num.manip.reshape(a, .{ .shape = &.{ 2, 3 } });
    defer a_2x3.deinit();

    // 2. Create another 2x3 matrix of ones
    var b = try num.ones(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f64 });
    defer b.deinit();

    // 3. Add them together (vectorized elementwise add)
    var c = try num.ops.add(a_2x3, b, .{});
    defer c.deinit();

    // 4. Access elements
    const val = try c.get(f64, &.{ 0, 0 }); // 0.0 + 1.0 = 1.0
    std.debug.print("Result at [0,0]: {d:.1}\n", .{val});
}
```

### Linear Algebra: Systems, Decompositions, and Inversion

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Construct matrix A: [[3.0, 1.0], [1.0, 2.0]]
    const a_data = [_]f64{ 3.0, 1.0, 1.0, 2.0 };
    var A = try num.fromSlice(allocator, f64, .{ .data = &a_data, .shape = &.{ 2, 2 } });
    defer A.deinit();

    // Construct vector b: [9.0, 8.0]
    const b_data = [_]f64{ 9.0, 8.0 };
    var b = try num.fromSlice(allocator, f64, .{ .data = &b_data, .shape = &.{2} });
    defer b.deinit();

    // Solve Ax = b
    var x = try num.linalg.solve(A, b);
    defer x.deinit();
    std.debug.print("Solved x: [{d:.2}, {d:.2}]\n", .{
        try x.get(f64, &.{0}),
        try x.get(f64, &.{1}),
    });

    // Compute determinant
    var det_A = try num.linalg.det(A);
    defer det_A.deinit();
    std.debug.print("Determinant det(A): {d:.2}\n", .{try det_A.itemAsFloat()});

    // Compute inverse
    var inv_A = try num.linalg.inv(A);
    defer inv_A.deinit();
}
```

### Polynomials: Evaluation, Derivatives, and Curve Fitting

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Polynomial coefficients for p(x) = 2x^2 - 3x + 5
    const coeffs = [_]f64{ 2.0, -3.0, 5.0 };
    var p = try num.fromSlice(allocator, f64, .{ .data = &coeffs, .shape = &.{3} });
    defer p.deinit();

    // Evaluate p(2.0) = 2*(4) - 3*(2) + 5 = 7.0
    const xv = [_]f64{2.0};
    var x = try num.fromSlice(allocator, f64, .{ .data = &xv, .shape = &.{} });
    defer x.deinit();
    var pv = try num.poly.val(p, x);
    defer pv.deinit();
    std.debug.print("p(2.0) = {d:.1}\n", .{try pv.get(f64, &.{})});

    // Derivative: p'(x) = 4x - 3
    var dp = try num.poly.der(p, 1);
    defer dp.deinit();

    // Roots: x^2 - 5x + 6 = (x-2)(x-3)
    const qd = [_]f64{ 1.0, -5.0, 6.0 };
    var q = try num.fromSlice(allocator, f64, .{ .data = &qd, .shape = &.{3} });
    defer q.deinit();
    var rts = try num.poly.roots(q);
    defer rts.deinit();

    // Linear fit: y = 2x + 1
    const x_vals = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y_vals = [_]f64{ 1.0, 3.0, 5.0, 7.0 };
    var x_arr = try num.fromSlice(allocator, f64, .{ .data = &x_vals, .shape = &.{4} });
    defer x_arr.deinit();
    var y_arr = try num.fromSlice(allocator, f64, .{ .data = &y_vals, .shape = &.{4} });
    defer y_arr.deinit();

    var line_fit = try num.poly.fit(x_arr, y_arr, 1);
    defer line_fit.deinit();
    std.debug.print("Fit: slope={d:.2}, intercept={d:.2}\n", .{
        try line_fit.get(f64, &.{0}),
        try line_fit.get(f64, &.{1}),
    });
}
```

### Native NZIG v1.0 Binary File Serialization

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var arr = try num.ones(allocator, .{ .shape = &.{ 4, 4 }, .dtype = .f64 });
    defer arr.deinit();

    // Save directly to binary NZIG format
    try num.save(allocator, "weights.nzig", arr);

    // Load back with metadata intact
    var loaded = try num.load(allocator, "weights.nzig");
    defer loaded.deinit();

    std.debug.print("Loaded shape: {any}, dtype: {s}\n", .{
        loaded.shapeSlice(),
        @tagName(loaded.dtype),
    });
}
```

---

## Examples

The `examples/` directory contains runnable examples demonstrating the full suite of `num.zig` capabilities:

**Array Creation & Manipulation:**
- [`array_creation`](examples/array_creation.zig) - Array creation routines (`zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `fromSlice`)
- [`slicing`](examples/slicing.zig) - Zero-copy strided multidimensional slicing
- [`reshape`](examples/reshape.zig) - Reshape, ravel, flatten, squeeze, and expandDims
- [`transpose`](examples/transpose.zig) - Zero-copy matrix transposition, axis swapping, and axis moving
- [`concatenation`](examples/concatenation.zig) - Array concatenation, stacking, splitting, tiling, repeating, and padding

**Vectorized Math & Operations:**
- [`elementwise`](examples/elementwise.zig) - Vectorized arithmetic, trigonometric functions, exponentials, and boolean masking
- [`broadcasting`](examples/broadcasting.zig) - Multidimensional broadcasting rules and explicit `broadcastTo`
- [`reductions`](examples/reductions.zig) - Global and axis-wise reductions (`sum`, `mean`, `median`, `variance`, `stdDev`, `min`, `max`, `argmin`, `argmax`, `cumsum`)
- [`bitwise_ops`](examples/bitwise_ops.zig) - Integer bitwise logic, shifts, and bit counts
- [`complex_ops`](examples/complex_ops.zig) - Complex construction, conjugate, real/imaginary, magnitude, and phase

**Sorting, Searching & Statistics:**
- [`sorting`](examples/sorting.zig) - In-place and copy quicksort, and `argsort` permutation indexing
- [`searching`](examples/searching.zig) - Binary search (`searchSorted`), unique elements (`unique`), and nonzero indices (`flatNonzero`)
- [`random_generation`](examples/random_generation.zig) - Uniform, normal, integer distributions, shuffling, and choice sampling
- [`statistics`](examples/statistics.zig) - Descriptive statistics (`mean`, `median`, `variance`, `stdDev`, `percentile`)
- [`correlation`](examples/correlation.zig) - Covariance matrices, Pearson correlation, and histogram binning

**Linear Algebra & Polynomials:**
- [`matmul`](examples/matmul.zig) - Cache-friendly matrix multiplication, dot products, inner, and outer products
- [`norms`](examples/norms.zig) - Vector norms (L1, L2, Linf) and matrix Frobenius norm
- [`solve`](examples/solve.zig) - Linear system solvers (`Ax = b`), matrix inversion, and determinant computation
- [`decompositions`](examples/decompositions.zig) - LU decomposition, QR decomposition (Householder), and Cholesky factorization
- [`eigenvalues`](examples/eigenvalues.zig) - Eigenvalue and eigenvector computation for square matrices
- [`svd`](examples/svd.zig) - General Singular Value Decomposition (U, S, Vt)
- [`fft`](examples/fft.zig) - 1D Fast Fourier Transform and IFFT with complex spectra
- [`sparse_matrix`](examples/sparse_matrix.zig) - Compressed Sparse Row (CSR) matrix creation, matvec multiplication, dense roundtripping, and Conjugate Gradient (`cg`) solver
- [`polynomials`](examples/polynomials.zig) - Evaluation, derivatives, curve fitting, root finding, and coefficient arithmetic

**Parallel CPU Execution:**
- [`parallel_basic`](examples/parallel_basic.zig) - Automatic multithreaded execution across CPU cores
- [`parallel_config`](examples/parallel_config.zig) - Explicit worker and chunk size configuration
- [`parallel_large_array`](examples/parallel_large_array.zig) - High-throughput parallel processing of $10^6$ element vectors
- [`parallel_f32`](examples/parallel_f32.zig) - Single-precision f32 parallel arrays and typed access
- [`parallel_f64`](examples/parallel_f64.zig) - Double-precision f64 parallel exponential decay calculations
- [`parallel_non_contiguous`](examples/parallel_non_contiguous.zig) - Parallel operations on transposed and sliced views
- [`parallel_reduction`](examples/parallel_reduction.zig) - Multi-threaded tree reduction combining worker accumulators

**Serialization & I/O:**
- [`serialization`](examples/serialization.zig) - Portable NZIG v1.0 binary array serialization, memory streams, and file roundtripping
- [`text_io`](examples/text_io.zig) - Delimited CSV text save/load round-tripping

To run any individual example:
```bash
zig build run-<example-name>
# e.g., zig build run-array_creation
# e.g., zig build run-matmul
# e.g., zig build run-polynomials
# e.g., zig build run-sparse_matrix
# e.g., zig build run-serialization
```

To run all examples sequentially:
```bash
zig build run-all-examples
```

---

## Validation Matrix

```bash
# Run complete test suite (v0.0.3 unit tests)
zig build test

# Run tests, benchmarks, and all runnable examples sequentially
zig build test-all

# Cross-compile test validation
zig build test-check -Dtarget=x86_64-linux-gnu
zig build test-check -Dtarget=aarch64-linux-gnu
zig build test-check -Dtarget=x86_64-macos
zig build test-check -Dtarget=aarch64-macos
zig build test-check -Dtarget=x86-windows
```

---

## Performance

Run benchmarks:

```bash
zig build bench
```

Benchmark target: `x86_64-windows`, `ReleaseFast` (measured with stable Zig 0.16.0):

| Benchmark | Category | Performance / Throughput | Notes |
|:---|:---|:---:|:---|
| `matmul` (200x200 f64) | Linear Algebra | **7.48 GFLOPS** | Tiled/blocked cache-locality optimization |
| `elementwise_add` (1M f64) | Vectorized Math | **484.52 M elements/sec** | Contiguous SIMD vectorization |
| `reduction_sum` (1M f64) | Reductions | **0.55 ms** (1.81 Gelem/s) | Tree-accumulated vector summation |
| `quicksort` (100k f64) | Sorting | **10.16 ms** | In-place dual-pivot partitioning |
| `nzig_write` (3.8 MB f64) | Serialization | **4.08 GB/s** | 64-byte aligned zero-copy raw memory stream |
| `nzig_read` (3.8 MB f64) | Deserialization | **3.72 GB/s** | Validated 128-byte header direct payload ingest |

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for any new functionality
4. Ensure all tests and examples pass:
   ```bash
   zig build test
   zig build run-all-examples
   ```
5. Submit a pull request

---

## Project Structure

```
num.zig/
├── src/
│   ├── num.zig                      # Public API entry point & clean re-exports
│   ├── core/                        # Core data models and memory management
│   │   ├── array.zig                # Array struct, constructors, and SBO
│   │   ├── dtype.zig                # DType enum, traits, and type promotion
│   │   ├── shape.zig                # Shape, Strides, Slice, and index calculations
│   │   ├── buffer.zig               # Raw byte buffer allocation and alignment
│   │   ├── iterator.zig             # Multidimensional strided coordinates iterator
│   │   └── error.zig                # Typed error sets (Shape, DType, Index, Linalg)
│   ├── ops/                         # Vectorized arithmetic, compare, and reduction
│   │   ├── elementwise.zig          # Add, sub, mul, div, pow, trig, log, sqrt, where
│   │   ├── compare.zig              # Equal, less, greater, logical operations
│   │   ├── broadcast.zig            # Multidimensional broadcast rules and views
│   │   └── reduce.zig               # Sum, prod, mean, min, max, argmin, argmax
│   ├── manip/                       # Zero-copy and view transformations
│   │   ├── reshape.zig              # Reshape, ravel, flatten, squeeze, expandDims
│   │   ├── transpose.zig            # Transpose, swapAxes, moveAxis
│   │   ├── slice.zig                # Multi-axis slicing and sub-views
│   │   ├── concat.zig               # Concat, stack, split, tile, repeat
│   │   └── pad.zig                  # Constant, edge, and reflect padding
│   ├── linalg/                      # Linear algebra routines
│   │   ├── matmul.zig               # Tiled matmul, dot, inner, outer
│   │   ├── norm.zig                 # Vector (L1, L2, Linf) and matrix Frobenius norms
│   │   ├── decompose.zig            # LU, QR (Householder), Cholesky decompositions
│   │   └── solve.zig                # Linear system solver, matrix inverse, determinant
│   ├── poly/                        # Polynomial analysis
│   │   ├── eval.zig                 # Horner evaluation (val), derivative (der), integral (integ)
│   │   └── fit.zig                  # Least-squares polynomial fitting (fit)
│   ├── sort/                        # Sorting and searching
│   │   ├── ordering.zig             # Quicksort, sorted copy, and argsort
│   │   └── search.zig               # Binary searchSorted, unique elements, flatNonzero
│   ├── stats/                       # Statistical analysis
│   │   ├── describe.zig             # Mean, median, variance, stdDev, quantiles
│   │   └── correlate.zig            # Covariance, Pearson correlation, histogram
│   ├── random/                      # Pseudo-random generation
│   │   ├── engine.zig               # Prng engine (splitmix64 / xoshiro256++)
│   │   └── distributions.zig        # Uniform, normal, integer, choice, shuffle
│   ├── io/                          # Array serialization
│   │   ├── nzig.zig                 # Portable NZIG v1.0 binary format
│   │   ├── text.zig                 # Delimited text format (savetxt, loadtxt)
│   │   └── stream.zig               # In-memory and file stream helpers
│   └── strops/                      # String and terminal formatting
│       └── format.zig               # Pretty-printing for multidimensional arrays
├── examples/                        # 27 runnable standalone examples
│   ├── array_creation.zig           # Basic array generation routines
│   ├── elementwise.zig              # SIMD arithmetic and mathematical functions
│   ├── broadcasting.zig             # Automatic and manual dimension expansion
│   ├── slicing.zig                  # Strided slicing and views
│   ├── reshape.zig                  # Structural transformations
│   ├── transpose.zig                # Matrix transposition
│   ├── concatenation.zig            # Combining and padding arrays
│   ├── reductions.zig               # Cumulative and axis reductions
│   ├── sorting.zig                  # Ordering and permutation indexing
│   ├── searching.zig                # Binary search and unique extraction
│   ├── random_generation.zig        # Random sampling and distributions
│   ├── statistics.zig               # Descriptive statistics and percentiles
│   ├── correlation.zig              # Covariance and histograms
│   ├── matmul.zig                   # High-performance matrix multiplication
│   ├── norms.zig                    # Mathematical norms
│   ├── solve.zig                    # Inverses and linear systems
│   ├── decompositions.zig           # Factorization algorithms
│   ├── eigenvalues.zig              # Eigenvalues and eigenvectors
│   ├── svd.zig                      # Singular Value Decomposition
│   ├── fft.zig                      # Fast Fourier Transform
│   ├── sparse_matrix.zig            # Compressed Sparse Row and solvers
│   ├── parallel_basic.zig           # Parallel CPU execution
│   ├── parallel_config.zig          # Configured parallel execution
│   ├── parallel_f32.zig             # Single-precision parallel arrays
│   ├── parallel_f64.zig             # Double-precision parallel decay
│   ├── parallel_large_array.zig     # Large array parallel processing
│   ├── parallel_non_contiguous.zig  # Parallel on strided views
│   ├── parallel_reduction.zig       # Multi-threaded tree reduction
│   ├── polynomials.zig              # Curve fitting and polynomial math
│   └── serialization.zig            # NZIG binary file read and write
├── bench/
│   └── benchmark.zig                # Performance benchmark suite
├── build.zig                        # Zig 0.16.0 build system configuration
├── build.zig.zon                    # Zig package metadata
├── README.md                        # Documentation and user guide
├── LICENSE                          # MIT license
├── CONTRIBUTING.md                  # Contribution guidelines
```

---

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Muhammad Fiaz.
