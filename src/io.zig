const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

const MAGIC = "NUMZIG";
const VERSION: u8 = 1;

/// Writes an NDArray to a writer in a binary format.
pub fn writeArray(allocator: Allocator, comptime T: type, arr: NDArray(T), writer: anytype) !void {
    try writer.writeAll(MAGIC);

    const ver = [1]u8{VERSION};
    try writer.writeAll(&ver);

    if (arr.rank() > 255) return error.DimensionMismatch;
    const rank_byte = [1]u8{@intCast(arr.rank())};
    try writer.writeAll(&rank_byte);

    for (arr.shape) |dim| {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, @intCast(dim), .little);
        try writer.writeAll(&buf);
    }

    // Write data.
    const flags = arr.flags();
    if (flags.c_contiguous) {
        const bytes = std.mem.sliceAsBytes(arr.data);
        try writer.writeAll(bytes);
    } else {
        // Create contiguous copy
        var temp = try arr.asContiguous(allocator);
        defer temp.deinit(allocator);
        const bytes = std.mem.sliceAsBytes(temp.data);
        try writer.writeAll(bytes);
    }
}

/// Saves an NDArray to a file in a binary format.
///
/// The format includes a magic header, version, rank, shape, and the raw data.
/// If the array is not C-contiguous, a contiguous copy is created and saved.
///
/// Arguments:
///     allocator: The allocator to use for temporary copies.
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
/// defer a.deinit(allocator);
///
/// try io.save(allocator, f32, a, "data.bin");
/// ```
pub fn save(allocator: Allocator, comptime T: type, arr: NDArray(T), path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try writeArray(allocator, T, arr, file);
}

/// Reads an NDArray from a reader.
pub fn readArray(allocator: Allocator, comptime T: type, reader: anytype) !NDArray(T) {
    var magic: [6]u8 = undefined;
    if ((try reader.readAll(&magic)) != magic.len) return error.EndOfStream;
    if (!std.mem.eql(u8, &magic, MAGIC)) return error.InvalidFormat;

    var ver: [1]u8 = undefined;
    if ((try reader.readAll(&ver)) != ver.len) return error.EndOfStream;
    if (ver[0] != VERSION) return error.UnsupportedVersion;

    var rank_byte: [1]u8 = undefined;
    if ((try reader.readAll(&rank_byte)) != rank_byte.len) return error.EndOfStream;
    const rank = rank_byte[0];

    const shape = try allocator.alloc(usize, rank);
    defer allocator.free(shape);

    for (0..rank) |i| {
        var buf: [8]u8 = undefined;
        if ((try reader.readAll(&buf)) != buf.len) return error.EndOfStream;
        shape[i] = std.mem.readInt(u64, &buf, .little);
    }

    var arr = try NDArray(T).init(allocator, shape);
    errdefer arr.deinit(allocator);

    const bytes = std.mem.sliceAsBytes(arr.data);
    if ((try reader.readAll(bytes)) != bytes.len) return error.EndOfStream;

    return arr;
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
    return readArray(allocator, T, file);
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
    errdefer arr.deinit(allocator);

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

test "io save load" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).init(allocator, &.{ 2, 2 });
    defer arr.deinit(allocator);
    arr.fill(1.0);
    try arr.set(&.{ 0, 0 }, 2.0);

    const path = "test_io.bin";
    try save(allocator, f32, arr, path);

    var loaded = try load(allocator, f32, path);
    defer loaded.deinit(allocator);
    defer std.fs.cwd().deleteFile(path) catch {};

    try std.testing.expectEqual(loaded.shape[0], 2);
    try std.testing.expectEqual(loaded.shape[1], 2);
    try std.testing.expectEqual(try loaded.get(&.{ 0, 0 }), 2.0);
    try std.testing.expectEqual(try loaded.get(&.{ 1, 1 }), 1.0);
}

test "io load csv" {
    const allocator = std.testing.allocator;
    const path = "test_io.csv";
    const file = try std.fs.cwd().createFile(path, .{});
    try file.writeAll("1.0, 2.0\n3.0, 4.0");
    file.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    var arr = try readCSV(allocator, f32, path);
    defer arr.deinit(allocator);

    try std.testing.expectEqual(arr.shape[0], 2);
    try std.testing.expectEqual(arr.shape[1], 2);
    try std.testing.expectEqual(try arr.get(&.{ 0, 0 }), 1.0);
    try std.testing.expectEqual(try arr.get(&.{ 1, 1 }), 4.0);
}
