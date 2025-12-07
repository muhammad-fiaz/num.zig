# Manipulation

The `manipulation` module provides functions to change array structure.

## Functions

### `vstack`

Stack arrays vertically (row-wise).

```zig
pub fn vstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T)
```

### `hstack`

Stack arrays horizontally (column-wise).

```zig
pub fn hstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T)
```

### `dstack`

Stack arrays depth-wise (along third axis).

```zig
pub fn dstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T)
```

### `tile`

Construct an array by repeating A the number of times given by reps.

```zig
pub fn tile(allocator: Allocator, comptime T: type, arr: NDArray(T), reps: []const usize) !NDArray(T)
```

### `repeat`

Repeat elements of an array.

```zig
pub fn repeat(allocator: Allocator, comptime T: type, arr: NDArray(T), repeats: usize, axis: ?usize) !NDArray(T)
```

### `moveaxis`

Move axes of an array to new positions.

```zig
pub fn moveaxis(allocator: Allocator, comptime T: type, arr: NDArray(T), source: []const usize, destination: []const usize) !NDArray(T)
```

### `ravel`

Return a contiguous flattened array.

```zig
pub fn ravel(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(T)
```

### `flip`

Reverse the order of elements in an array along the given axis.

```zig
pub fn flip(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(T)
```

### `roll`

Roll array elements along a given axis.

```zig
pub fn roll(allocator: Allocator, comptime T: type, arr: NDArray(T), shift: isize, axis: usize) !NDArray(T)
```

### `swapaxes`

Interchange two axes of an array.

```zig
pub fn swapaxes(allocator: Allocator, comptime T: type, arr: NDArray(T), axis1: usize, axis2: usize) !NDArray(T)
```

### `delete`

Delete elements at specified indices (flattened).

```zig
pub fn delete(allocator: Allocator, comptime T: type, a: *const NDArray(T), indices: []const usize) !NDArray(T)
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const manipulation = num.manipulation;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = try num.NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.0});
    var b = try num.NDArray(f32).init(allocator, &.{2}, &.{3.0, 4.0});

    var c = try manipulation.vstack(allocator, f32, &.{a, b});
    defer c.deinit();
    // c is [[1, 2], [3, 4]]
}
```
