# Data Types

Num.Zig is designed to be generic and supports all standard Zig primitive types.

## Supported Primitives

### Floating Point
- `f16`: 16-bit floating point
- `f32`: 32-bit floating point (Standard for ML)
- `f64`: 64-bit floating point (Double precision)
- `f80`: 80-bit floating point
- `f128`: 128-bit floating point (Quad precision)

### Integers
- `i8`, `u8`
- `i16`, `u16`
- `i32`, `u32`
- `i64`, `u64`
- `i128`, `u128`
- `isize`, `usize`

### Boolean
- `bool`: Boolean values (supported for masks and logic operations)

## Usage Example

```zig
const std = @import("std");
const num = @import("num");
const NDArray = num.NDArray;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 128-bit Float Array
    var f128_arr = try NDArray(f128).zeros(allocator, &.{2, 2});
    defer f128_arr.deinit();
    try f128_arr.set(&.{0, 0}, 1.23456789012345678901234567890123456789);

    // 128-bit Integer Array
    var i128_arr = try NDArray(i128).arange(allocator, 0, 10, 1);
    defer i128_arr.deinit();
}
```

## Memory Layout

Arrays are stored in contiguous memory blocks (row-major order by default). The total memory usage is `product(shape) * @sizeOf(T)`.

For high-dimensional arrays, the rank is only limited by system memory and the `usize` limits for indexing.
