const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

const MAGIC = "NUMZIG";
const VERSION: u8 = 1;

/// Saves an NDArray to a file in a binary format.
///
/// The format includes a magic header, version, rank, shape, and the raw data.
/// If the array is not C-contiguous, a contiguous copy is created and saved.
///
/// Arguments:
///     T: The data type of the array elements.
///     arr: The NDArray to save.
///     path: The file path to save to.
///
/// Returns:
///     Void on success, or an error if file operations fail.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.0});
/// defer a.deinit();
///
/// try io.save(f32, a, "data.bin");
/// ```
pub fn save(comptime T: type, arr: NDArray(T), path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    try file.writeAll(MAGIC);

    const ver = [1]u8{VERSION};
    try file.writeAll(&ver);

    if (arr.rank() > 255) return error.DimensionMismatch;
    const rank_byte = [1]u8{@intCast(arr.rank())};
    try file.writeAll(&rank_byte);

    for (arr.shape) |dim| {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, @intCast(dim), .little);
        try file.writeAll(&buf);
    }

    // Write data.
    const flags = arr.flags();
    if (flags.c_contiguous) {
        const bytes = std.mem.sliceAsBytes(arr.data);
        try file.writeAll(bytes);
    } else {
        // Create contiguous copy
        var temp = try arr.asContiguous();
        defer temp.deinit();
        const bytes = std.mem.sliceAsBytes(temp.data);
        try file.writeAll(bytes);
    }
}

/// Loads an NDArray from a binary file.
///
/// Arguments:
///     allocator: The allocator to use for the array.
///     T: The data type of the array elements.
///     path: The file path to load from.
///
/// Returns:
///     A new NDArray containing the loaded data.
pub fn load(allocator: Allocator, comptime T: type, path: []const u8) !NDArray(T) {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var magic: [6]u8 = undefined;
    if ((try file.readAll(&magic)) != magic.len) return error.EndOfStream;
    if (!std.mem.eql(u8, &magic, MAGIC)) return error.InvalidFormat;

    var ver: [1]u8 = undefined;
    if ((try file.readAll(&ver)) != ver.len) return error.EndOfStream;
    if (ver[0] != VERSION) return error.UnsupportedVersion;

    var rank_byte: [1]u8 = undefined;
    if ((try file.readAll(&rank_byte)) != rank_byte.len) return error.EndOfStream;
    const rank = rank_byte[0];

    const shape = try allocator.alloc(usize, rank);
    defer allocator.free(shape);

    for (0..rank) |i| {
        var buf: [8]u8 = undefined;
        if ((try file.readAll(&buf)) != buf.len) return error.EndOfStream;
        shape[i] = std.mem.readInt(u64, &buf, .little);
    }

    var arr = try NDArray(T).init(allocator, shape);
    errdefer arr.deinit();

    const bytes = std.mem.sliceAsBytes(arr.data);
    if ((try file.readAll(bytes)) != bytes.len) return error.EndOfStream;

    return arr;
}

/// Writes an NDArray to a CSV file.
///
/// Only supports 1D and 2D arrays.
///
/// Arguments:
///     T: The data type of the array elements.
///     arr: The NDArray to save.
///     path: The file path to save to.
///
/// Returns:
///     Void on success.
pub fn writeCSV(comptime T: type, arr: NDArray(T), path: []const u8) !void {
    if (arr.rank() > 2) return core.Error.DimensionMismatch;

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    if (arr.rank() == 1) {
        for (arr.data, 0..) |val, i| {
            var buf: [64]u8 = undefined;
            const str = try std.fmt.bufPrint(&buf, "{any}", .{val});
            try file.writeAll(str);
            if (i < arr.shape[0] - 1) {
                try file.writeAll(",");
            }
        }
        try file.writeAll("\n");
    } else {
        const rows = arr.shape[0];
        const cols = arr.shape[1];
        for (0..rows) |i| {
            for (0..cols) |j| {
                const val = try arr.get(&.{ i, j });
                var buf: [64]u8 = undefined;
                const str = try std.fmt.bufPrint(&buf, "{any}", .{val});
                try file.writeAll(str);
                if (j < cols - 1) {
                    try file.writeAll(",");
                }
            }
            try file.writeAll("\n");
        }
    }
}

/// Reads an NDArray from a CSV file.
///
/// Assumes the file contains numeric data separated by commas.
/// Returns a 2D array.
///
/// Arguments:
///     allocator: The allocator to use.
///     T: The data type (must be float or int).
///     path: The file path to read.
///
/// Returns:
///     A new 2D NDArray.
pub fn readCSV(allocator: Allocator, comptime T: type, path: []const u8) !NDArray(T) {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 1024 * 1024 * 100); // 100MB limit
    defer allocator.free(content);

    var rows = std.ArrayListUnmanaged([]const u8){};
    defer rows.deinit(allocator);

    var iter = std.mem.splitScalar(u8, content, '\n');
    while (iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len > 0) {
            try rows.append(allocator, trimmed);
        }
    }

    if (rows.items.len == 0) return core.Error.DimensionMismatch;

    const num_rows = rows.items.len;
    // Count columns in first row
    var col_count: usize = 0;
    var col_iter = std.mem.splitScalar(u8, rows.items[0], ',');
    while (col_iter.next()) |_| {
        col_count += 1;
    }

    var arr = try NDArray(T).init(allocator, &.{ num_rows, col_count });
    errdefer arr.deinit();

    for (rows.items, 0..) |row, i| {
        var j: usize = 0;
        var val_iter = std.mem.splitScalar(u8, row, ',');
        while (val_iter.next()) |val_str| {
            if (j >= col_count) break;
            const trimmed_val = std.mem.trim(u8, val_str, " \r\t");
            if (trimmed_val.len == 0) continue;

            const val = switch (@typeInfo(T)) {
                .float => try std.fmt.parseFloat(T, trimmed_val),
                .int => try std.fmt.parseInt(T, trimmed_val, 10),
                else => return core.Error.UnsupportedType,
            };
            try arr.set(&.{ i, j }, val);
            j += 1;
        }
    }

    return arr;
}
