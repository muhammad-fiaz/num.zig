# Input/Output API Reference

The `io` module provides functions for reading and writing arrays to disk.

## Binary I/O

### save

Save an array to a binary file.

```zig
pub fn save(allocator: Allocator, comptime T: type, arr: NDArray(T), path: []const u8) !void
```

### load

Load an array from a binary file.

```zig
pub fn load(allocator: Allocator, comptime T: type, path: []const u8) !NDArray(T)
```

### writeArray

Write an array to a writer.

```zig
pub fn writeArray(allocator: Allocator, comptime T: type, arr: NDArray(T), writer: anytype) !void
```

### readArray

Read an array from a reader.

```zig
pub fn readArray(allocator: Allocator, comptime T: type, reader: anytype) !NDArray(T)
```

## Text I/O

### writeCSV

Write an array to a CSV file.

```zig
pub fn writeCSV(comptime T: type, arr: NDArray(T), path: []const u8) !void
```

### readCSV

Read an array from a CSV file.

```zig
pub fn readCSV(allocator: Allocator, comptime T: type, path: []const u8) !NDArray(T)
```