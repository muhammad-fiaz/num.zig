# DataFrame API Reference

The `dataframe` module provides data structures for tabular data.

## DataFrame

A 2-dimensional labeled data structure.

```zig
pub const DataFrame = struct {
    // ...
};
```

### init

Initialize an empty DataFrame.

```zig
pub fn init(allocator: Allocator) Self
```

### deinit

Deinitialize the DataFrame.

```zig
pub fn deinit(self: *Self) void
```

### addColumn

Add a column to the DataFrame.

```zig
pub fn addColumn(self: *Self, name: []const u8, col: Column) !void
```

### getColumn

Get a column by name.

```zig
pub fn getColumn(self: Self, name: []const u8) ?Column
```

## Series

A 1-dimensional labeled array.

```zig
pub fn Series(comptime T: type) type
```

### init

Initialize a Series.

```zig
pub fn init(allocator: Allocator, name: []const u8, data: NDArray(T)) !Self
```

### deinit

Deinitialize the Series.

```zig
pub fn deinit(self: *Self) void
```

### print

Print the Series.

```zig
pub fn print(self: Self) !void
```