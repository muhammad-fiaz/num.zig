# Introduction to num.zig

**num.zig** is a high-performance, strongly typed numerical computing and N-dimensional array library engineered entirely in **Zig 0.16.0**.

Designed from first principles for performance-critical systems, machine learning primitives, scientific simulations, and numerical analysis, `num.zig` pairs hardware-level control with a clean, expressive, Zig-idiomatic interface.

---

## Why num.zig?

1. **Zero Hidden Allocations**: All memory allocations require an explicit `std.mem.Allocator`. Every allocation path is fully visible and under caller control.
2. **Inline Small-Buffer Optimization (SBO)**: Small arrays up to 64 bytes (such as 3D spatial coordinates, small transformation matrices, or convolution filters) are stored inline without allocating heap memory.
3. **Hardware Acceleration**:
   - SIMD vectorized vector arithmetic (`@Vector(N, T)`).
   - Multi-threaded CPU parallel execution subsystem with automatic cache-conscious chunking.
   - Cache-blocked tiled matrix multiplication.
4. **Rich Scientific Stack**:
   - Comprehensive linear algebra (LU, QR, Cholesky, SVD, Eigenvalues, Inversion, Solvers).
   - Fast Fourier Transforms (1D and multi-dimensional Cooley-Tukey Radix-2 with prime-length DFT fallbacks).
   - Sparse matrix representations (CSR/CSC) and iterative Krylov solvers (Conjugate Gradient, GMRES).
   - Pseudo-random number distributions (Uniform, Normal, Exponential, Poisson, Shuffling).
   - Statistical analysis, polynomial calculus, and set operations.
5. **Portable Binary Serialization**:
   - Includes the self-describing **NZIG v1.0** binary file specification for zero-copy array storage, along with standard delimited text (CSV/TSV) import and export.

---

## Quick Example

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2x2 matrix from existing memory
    const a_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var a = try num.fromSlice(allocator, f64, .{
        .data = &a_data,
        .shape = &.{ 2, 2 },
    });
    defer a.deinit();

    // Create a 2x2 identity matrix
    var eye = try num.identity(allocator, f64, 2);
    defer eye.deinit();

    // Elementwise addition
    var sum_mat = try num.add(allocator, f64, &a, &eye);
    defer sum_mat.deinit();

    // Matrix multiplication
    var prod = try num.linalg.matmul(allocator, f64, &a, &eye);
    defer prod.deinit();
}
```

---

## Next Steps

- Check out the [Installation Guide](/guide/installation) to integrate `num.zig` using the Zig package manager.
- Explore [Architecture & SBO](/guide/architecture) to understand internal memory layouts and optimization strategies.
- Browse the [API Reference](/api/) for exhaustive module and function documentation.
