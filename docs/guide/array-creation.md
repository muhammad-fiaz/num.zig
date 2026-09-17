# Array Creation

`num.zig` provides a comprehensive, strongly typed set of factory functions to initialize N-dimensional numerical arrays.

---

## 1. Initializing from Existing Slices

You can create arrays directly from compile-time or runtime Zig slices using `num.fromSlice`. The data is copied into the array's managed buffer (utilizing SBO when possible):

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 1D Vector
    const vec_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var vec = try num.fromSlice(allocator, f64, .{
        .data = &vec_data,
        .shape = &.{4},
    });
    defer vec.deinit();

    // 2D Matrix (2 rows, 3 columns)
    const mat_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var mat = try num.fromSlice(allocator, f32, .{
        .data = &mat_data,
        .shape = &.{ 2, 3 },
    });
    defer mat.deinit();
}
```

---

## 2. Constant Fill Arrays

Create arrays initialized with constant values:

### `zeros`
Initializes all elements to zero:
```zig
var z = try num.zeros(allocator, .{ .shape = &.{ 3, 3 }, .dtype = .f64 });
defer z.deinit();
```

### `ones`
Initializes all elements to one:
```zig
var o = try num.ones(allocator, .{ .shape = &.{ 4, 4 }, .dtype = .i32 });
defer o.deinit();
```

### `full`
Initializes all elements to an arbitrary scalar value:
```zig
var f = try num.full(allocator, .{ .shape = &.{ 2, 5 }, .value = @as(f64, 3.1415926535) });
defer f.deinit();
```

### `empty`
Allocates memory without zeroing elements (for maximum performance before writing values):
```zig
var e = try num.empty(allocator, .{ .shape = &.{ 1024, 1024 }, .dtype = .f32 });
defer e.deinit();
```

---

## 3. Sequences and Ranges

### `arange`
Generates half-open intervals `[start, stop)` with a given step:
```zig
// Numbers from 0.0 to 10.0 in increments of 0.5
var r = try num.arange(allocator, .{ .start = 0.0, .stop = 10.0, .step = 0.5, .dtype = .f64 });
defer r.deinit();
```

### `linspace`
Generates `n` evenly spaced points over a closed interval `[start, stop]`:
```zig
// 50 evenly spaced values between 0.0 and 1.0
var l = try num.linspace(allocator, .{ .start = 0.0, .stop = 1.0, .num = 50, .dtype = .f64 });
defer l.deinit();
```

### `geomspace`
Generates numbers spaced evenly on a log scale (geometric progression):
```zig
// 4 points from 1.0 to 1000.0 (1, 10, 100, 1000)
var g = try num.geomspace(allocator, .{ .start = 1.0, .stop = 1000.0, .num = 4, .dtype = .f64 });
defer g.deinit();
```

---

## 4. Identity and Diagonal Matrices

### `eye`
Creates a 2-D matrix with ones on the diagonal and zeros elsewhere:
```zig
// 3x3 identity matrix
var identity = try num.eye(allocator, .{ .n = 3, .dtype = .f64 });
defer identity.deinit();

// Offset diagonal (k = 1 creates upper sub-diagonal)
var super_diag = try num.eye(allocator, .{ .n = 4, .m = 4, .k = 1, .dtype = .f64 });
defer super_diag.deinit();
```

### `identity`
Convenient square identity matrix generator:
```zig
var id = try num.identity(allocator, .{ .n = 5, .dtype = .f32 });
defer id.deinit();
```
