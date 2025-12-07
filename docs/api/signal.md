# Signal Processing

The `signal` module provides signal processing tools.

## Functions

### `convolve`

Convolve two 1-dimensional arrays.

```zig
pub fn convolve(allocator: Allocator, comptime T: type, a: NDArray(T), v: NDArray(T), mode: ConvolveMode) !NDArray(T)
```

**Parameters:**
- `mode`: `.full`, `.valid`, or `.same`.

### `correlate`

Cross-correlation of two 1-dimensional arrays.

```zig
pub fn correlate(allocator: Allocator, comptime T: type, a: NDArray(T), v: NDArray(T), mode: ConvolveMode) !NDArray(T)
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const signal = num.signal;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = try num.NDArray(f32).init(allocator, &.{3}, &.{1.0, 2.0, 3.0});
    var v = try num.NDArray(f32).init(allocator, &.{3}, &.{0.0, 1.0, 0.5});

    var c = try signal.convolve(allocator, f32, a, v, .full);
    defer c.deinit();
}
```

