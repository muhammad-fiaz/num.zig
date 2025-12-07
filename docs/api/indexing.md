# Indexing

The `indexing` module provides advanced slicing and indexing capabilities.

## Structures

### `Slice`

Represents a slice operation on a dimension.

```zig
pub const Slice = union(enum) {
    all: void,
    index: usize,
    range: struct { start: usize, end: usize, step: isize = 1 },
};
```

## Functions

### `slice`

Creates a view of the array using a sequence of slice operations.

```zig
pub fn slice(allocator: Allocator, comptime T: type, arr: NDArray(T), slices: []const Slice) !NDArray(T)
```

**Parameters:**
- `allocator`: Memory allocator.
- `T`: Data type.
- `arr`: Input array.
- `slices`: Slice operations.

**Returns:**
- A new `NDArray` view (shares data).

## Example

```zig
const std = @import("std");
const num = @import("num");
const indexing = num.indexing;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = try num.NDArray(f32).arange(allocator, 0, 10, 1);
    defer a.deinit();

    // Slice: a[2:8:2]
    const slices = &[_]indexing.Slice{
        .{ .range = .{ .start = 2, .end = 8, .step = 2 } },
    };
    var view = try indexing.slice(allocator, f32, a, slices);
    defer view.deinit();
    
    // view is {2, 4, 6}
}
```

