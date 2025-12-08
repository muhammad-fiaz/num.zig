# Data Types

`num.zig` supports all standard Zig numeric types, providing a flexible and type-safe foundation for numerical computing.

## Supported Types

### Floating-Point Types

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // 16-bit float (half precision)
    var arr_f16 = try num.NDArray(f16).zeros(allocator, &.{2, 2});
    defer arr_f16.deinit(allocator);

    // 32-bit float (single precision)
    var arr_f32 = try num.NDArray(f32).zeros(allocator, &.{2, 2});
    defer arr_f32.deinit(allocator);

    // 64-bit float (double precision) - most common
    var arr_f64 = try num.NDArray(f64).zeros(allocator, &.{2, 2});
    defer arr_f64.deinit(allocator);

    // 128-bit float (quad precision)
    var arr_f128 = try num.NDArray(f128).zeros(allocator, &.{2, 2});
    defer arr_f128.deinit(allocator);
}
```

**Output:**
```
Array(f16): shape=[2, 2], dtype=f16
Array(f32): shape=[2, 2], dtype=f32
Array(f64): shape=[2, 2], dtype=f64
Array(f128): shape=[2, 2], dtype=f128
```

### Signed Integer Types

```zig
// 8-bit signed integer
var arr_i8 = try num.NDArray(i8).ones(allocator, &.{3});
defer arr_i8.deinit(allocator);
try arr_i8.print();
// Output: [1, 1, 1]

// 16-bit signed integer
var arr_i16 = try num.NDArray(i16).full(allocator, &.{2}, -100);
defer arr_i16.deinit(allocator);
try arr_i16.print();
// Output: [-100, -100]

// 32-bit signed integer
var arr_i32 = try num.NDArray(i32).arange(allocator, -5, 5, 2);
defer arr_i32.deinit(allocator);
try arr_i32.print();
// Output: [-5, -3, -1, 1, 3]

// 64-bit signed integer (default for large integers)
var arr_i64 = try num.NDArray(i64).zeros(allocator, &.{2, 3});
defer arr_i64.deinit(allocator);

// 128-bit signed integer (for very large integers)
var arr_i128 = try num.NDArray(i128).zeros(allocator, &.{2});
defer arr_i128.deinit(allocator);
```

### Unsigned Integer Types

```zig
// 8-bit unsigned integer (0-255)
var arr_u8 = try num.NDArray(u8).arange(allocator, 0, 256, 50);
defer arr_u8.deinit(allocator);
try arr_u8.print();
// Output: [0, 50, 100, 150, 200, 250]

// 16-bit unsigned integer
var arr_u16 = try num.NDArray(u16).full(allocator, &.{3}, 1000);
defer arr_u16.deinit(allocator);
try arr_u16.print();
// Output: [1000, 1000, 1000]

// 32-bit unsigned integer
var arr_u32 = try num.NDArray(u32).ones(allocator, &.{2, 2});
defer arr_u32.deinit(allocator);

// 64-bit unsigned integer
var arr_u64 = try num.NDArray(u64).zeros(allocator, &.{5});
defer arr_u64.deinit(allocator);

// 128-bit unsigned integer
var arr_u128 = try num.NDArray(u128).zeros(allocator, &.{2});
defer arr_u128.deinit(allocator);
```

### Complex Numbers

```zig
const Complex = num.Complex;

// Complex numbers with f32 components
var arr_c32 = try num.NDArray(Complex(f32)).zeros(allocator, &.{2, 2});
defer arr_c32.deinit(allocator);

// Complex numbers with f64 components (most common)
var arr_c64 = try num.NDArray(Complex(f64)).zeros(allocator, &.{3, 3});
defer arr_c64.deinit(allocator);

// Working with complex numbers
const c1 = Complex(f64){ .re = 3.0, .im = 4.0 };
const c2 = Complex(f64){ .re = 1.0, .im = 2.0 };
// Result: c1 + c2 = (4.0, 6.0)
```

## Type Conversion

Convert arrays between different types using `astype`:

```zig
// Create integer array
var int_arr = try num.NDArray(i32).arange(allocator, 0, 5, 1);
defer int_arr.deinit(allocator);
try int_arr.print();
// Output: [0, 1, 2, 3, 4]

// Convert to float
var float_arr = try int_arr.astype(allocator, f64);
defer float_arr.deinit(allocator);
try float_arr.print();
// Output: [0.0, 1.0, 2.0, 3.0, 4.0]

// Convert back to integer (truncates decimals)
var arr_f = try num.NDArray(f64).linspace(allocator, 0.0, 10.0, 5);
defer arr_f.deinit(allocator);
try arr_f.print();
// Output: [0.0, 2.5, 5.0, 7.5, 10.0]

var arr_i = try arr_f.astype(allocator, i32);
defer arr_i.deinit(allocator);
try arr_i.print();
// Output: [0, 2, 5, 7, 10]
```

## Type Properties

### Generic Arrays

The `NDArray` struct is generic over the element type `T`:

```zig
pub fn NDArray(comptime T: type) type {
    return struct {
        data: []T,
        shape: []usize,
        strides: []usize,
        // ...
    };
}
```

### Type Selection Guidelines

| Type | Use Case | Precision | Range |
|------|----------|-----------|-------|
| `f32` | GPU computing, memory-constrained | ~7 decimal digits | ±3.4e38 |
| `f64` | General scientific computing | ~16 decimal digits | ±1.7e308 |
| `f128` | High-precision calculations | ~34 decimal digits | Very large |
| `i32` | General integer operations | Exact | -2³¹ to 2³¹-1 |
| `i64` | Large integer operations | Exact | -2⁶³ to 2⁶³-1 |
| `u8` | Image processing, bytes | Exact | 0 to 255 |
| `Complex(f64)` | Signal processing, quantum | ~16 decimal digits | Complex plane |

## Example: Working with Multiple Types

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Integer computations
    var counts = try num.NDArray(u32).arange(allocator, 0, 10, 1);
    defer counts.deinit(allocator);
    
    // Convert to float for statistical analysis
    var float_counts = try counts.astype(allocator, f64);
    defer float_counts.deinit(allocator);
    
    const mean = try num.stats.mean(allocator, f64, float_counts);
    std.debug.print("Mean: {d}\n", .{mean});
    // Output: Mean: 4.5
    
    // Complex number operations for FFT
    var signal = try num.NDArray(f64).linspace(allocator, 0.0, 1.0, 100);
    defer signal.deinit(allocator);
    
    var spectrum = try num.fft.fft(allocator, f64, signal);
    defer spectrum.deinit(allocator);
    // spectrum is now NDArray(Complex(f64))
}
```
