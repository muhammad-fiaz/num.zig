# Input/Output

The `io` module provides functions for saving and loading arrays to disk.

## Functions

### `save`

Saves an `NDArray` to a binary file.

```zig
pub fn save(comptime T: type, arr: NDArray(T), path: []const u8) !void
```

**Parameters:**
- `T`: Data type of the array.
- `arr`: The `NDArray` to save.
- `path`: File path.

**Description:**
Writes a binary file with a header containing magic bytes, version, rank, and shape, followed by the raw data bytes.

### `load`

Loads an `NDArray` from a binary file.

```zig
pub fn load(allocator: Allocator, comptime T: type, path: []const u8) !NDArray(T)
```

**Parameters:**
- `allocator`: Memory allocator.
- `T`: Data type of the array.
- `path`: File path.

**Returns:**
- A new `NDArray` containing the loaded data.

## Example

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Create and save
    var a = try num.NDArray(f64).init(allocator, &.{2, 2});
    try num.io.save(f64, a, "matrix.bin");

    // Load
    var b = try num.io.load(allocator, f64, "matrix.bin");
    defer b.deinit();
}
```
