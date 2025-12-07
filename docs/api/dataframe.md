# DataFrame

The `dataframe` module provides a structure for working with tabular data containing mixed types (Float, Int, Bool).

## Structures

### `Column`

A union representing a column of data.

```zig
pub const Column = union(enum) {
    Float: Series(f64),
    Int: Series(i64),
    Bool: Series(bool),
    // ...
};
```

### `DataFrame`

The main structure for tabular data.

#### `init`

Initializes an empty DataFrame.

```zig
pub fn init(allocator: Allocator) DataFrame
```

#### `addColumn`

Adds a column to the DataFrame.

```zig
pub fn addColumn(self: *DataFrame, name: []const u8, col: Column) !void
```

**Parameters:**
- `name`: Name of the column.
- `col`: The `Column` object to add.

**Returns:**
- `void` or `Error.DimensionMismatch` if the column length doesn't match.

#### `getColumn`

Retrieves a column by name.

```zig
pub fn getColumn(self: DataFrame, name: []const u8) ?Column
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const DataFrame = num.dataframe.DataFrame;
const Series = num.dataframe.Series;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var df = DataFrame.init(allocator);
    defer df.deinit();

    // Create a series
    var s_data = try num.NDArray(f64).init(allocator, &.{3});
    // ... fill data ...
    var series = Series(f64).init(s_data);

    // Add to DataFrame
    try df.addColumn("A", .{ .Float = series });

    // Retrieve
    if (df.getColumn("A")) |col| {
        std.debug.print("Column A len: {}\n", .{col.len()});
    }
}
```

**Output:**
```text
Column A len: 3
```
