const std = @import("std");
const Allocator = std.mem.Allocator;
const Series = @import("series.zig").Series;
const core = @import("../core.zig");

/// Represents a column in a DataFrame, which can hold different types of Series.
pub const Column = union(enum) {
    Float: Series(f64),
    Int: Series(i64),
    Bool: Series(bool),

    /// Frees resources associated with the Column.
    pub fn deinit(self: *Column) void {
        switch (self.*) {
            inline else => |*s| s.deinit(),
        }
    }

    /// Returns the number of elements in the column.
    pub fn len(self: Column) usize {
        switch (self) {
            inline else => |s| return s.data.shape[0],
        }
    }
};

/// A DataFrame structure that supports mixed-type columns.
pub const DataFrame = struct {
    const Self = @This();

    allocator: Allocator,
    columns: std.StringHashMap(Column),
    num_rows: usize,

    /// Initializes a new empty DataFrame.
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .columns = std.StringHashMap(Column).init(allocator),
            .num_rows = 0,
        };
    }

    /// Frees resources associated with the DataFrame.
    pub fn deinit(self: *Self) void {
        var iter = self.columns.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.columns.deinit();
    }

    /// Adds a column to the DataFrame.
    ///
    /// Arguments:
    ///     name: The name of the column.
    ///     col: The Column to add.
    ///
    /// Returns:
    ///     Error.DimensionMismatch if the column length does not match existing columns.
    pub fn addColumn(self: *Self, name: []const u8, col: Column) !void {
        const col_len = col.len();
        if (self.columns.count() > 0) {
            if (col_len != self.num_rows) {
                return core.Error.DimensionMismatch;
            }
        } else {
            self.num_rows = col_len;
        }
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        if (try self.columns.fetchPut(name_copy, col)) |old_entry| {
            self.allocator.free(old_entry.key);
            var old_col = old_entry.value;
            old_col.deinit();
        }
    }

    /// Retrieves a column by name.
    pub fn getColumn(self: Self, name: []const u8) ?Column {
        return self.columns.get(name);
    }
};
