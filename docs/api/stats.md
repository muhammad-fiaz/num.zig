# Statistics

The `stats` module provides statistical functions.

## Functions

### `sum`

Sum of array elements.

```zig
pub fn sum(comptime T: type, a: *const NDArray(T)) !T
```

### `prod`

Product of array elements.

```zig
pub fn prod(comptime T: type, a: *const NDArray(T)) !T
```

### `min`

Minimum value.

```zig
pub fn min(comptime T: type, a: *const NDArray(T)) !T
```

### `max`

Maximum value.

```zig
pub fn max(comptime T: type, a: *const NDArray(T)) !T
```

### `mean`

Mean value.

```zig
pub fn mean(comptime T: type, a: *const NDArray(T)) !T
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const stats = num.stats;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = try num.NDArray(f32).init(allocator, &.{3}, &.{1.0, 2.0, 3.0});
    
    const s = try stats.sum(f32, &a); // 6.0
    const m = try stats.mean(f32, &a); // 2.0
}
```

