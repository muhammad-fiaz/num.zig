# Getting Started with num.zig

`num.zig` is a modern, production-ready, pure Zig numerical computing library targeting **Zig 0.16.0**. It provides multidimensional array primitives with Small Buffer Optimization (SBO), vectorized SIMD kernels, comprehensive linear algebra, FFT, statistical routines, pseudo-random distributions, multi-threaded CPU parallel execution, and the portable NZIG v1.0 binary array format.

---

## Key Features

- **Pure Zig 0.16.0**: Zero C/C++ or external dependencies.
- **Small Buffer Optimization (SBO)**: Zero-allocation views for operations altering shapes and strides (e.g., `reshape`, `transpose`, `swapAxes`, `squeeze`, `expandDims`, `slice`).
- **Explicit Memory Discipline**: Clean allocator ownership. `deinit()` frees owned arrays while acting as a safe no-op on borrowed views.
- **Strong Compile-Time & Runtime Typing**: 12 fundamental numeric types plus complex numbers (`c64`, `c128`) with deterministic promotion rules.
- **Multi-Threaded CPU Parallelism**: Native `std.Thread` data parallelism with automatic core detection, inline configuration, and sequential fallback for small arrays.
- **Portable NZIG v1.0**: Deterministic, 64-byte aligned binary array serialization format independent of CPU architecture and ABI.

---

## Quick Example

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Create a 2x3 matrix
    var a = try num.arange(allocator, .{ .start = 0, .stop = 6, .dtype = .f64 });
    defer a.deinit();

    var a_2x3 = try a.reshape(.{ .shape = &.{ 2, 3 } });
    defer a_2x3.deinit();

    // 2. Create another 2x3 matrix of ones
    var b = try num.ones(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f64 });
    defer b.deinit();

    // 3. Add them together with vectorized elementwise addition
    var c = try num.ops.add(a_2x3, b, .{});
    defer c.deinit();

    // 4. Access elements explicitly with typed scalar accessor
    const val = try c.get(f64, &.{ 0, 0 });
    std.debug.print("Result at [0,0]: {d:.1}\n", .{val});
}
```
