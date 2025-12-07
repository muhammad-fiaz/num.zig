# Math

The `math` module provides mathematical operations, including SIMD optimizations.

## SIMD

### `add`

SIMD-accelerated addition.

```zig
pub fn add(comptime T: type, dest: []T, a: []const T, b: []const T) void
```

### `mul`

SIMD-accelerated multiplication.

```zig
pub fn mul(comptime T: type, dest: []T, a: []const T, b: []const T) void
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const simd = num.math.simd;

pub fn main() !void {
    var a = [_]f32{1.0, 2.0, 3.0, 4.0};
    var b = [_]f32{5.0, 6.0, 7.0, 8.0};
    var dest: [4]f32 = undefined;

    simd.add(f32, &dest, &a, &b);
    // dest is {6.0, 8.0, 10.0, 12.0}
}
```

