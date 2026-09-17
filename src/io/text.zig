//! Delimited text and CSV array serialization and deserialization.
//!
//! Provides reader and writer implementations for 1D and 2D arrays with
//! configurable delimiters, header/footer skipping, and data type coercion.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IoError = @import("../core/error.zig").IoError;
const IndexError = @import("../core/error.zig").IndexError;

const SaveTxtOptions = struct {
    delimiter: []const u8 = ",",
    header: []const u8 = "",
    footer: []const u8 = "",
};

/// Writes a 1D or 2D array to a writer in delimited text format.
pub fn savetxtWriter(
    arr: Array,
    writer: anytype,
    options: SaveTxtOptions,
) !void {
    if (arr.ndim > 2) return ShapeError.RankExceeded;

    if (options.header.len > 0) {
        try writer.print("{s}\n", .{options.header});
    }

    if (arr.ndim == 1) {
        const n = arr.shape_dims[0];
        for (0..n) |i| {
            try writeScalar(arr, i, writer);
            try writer.writeByte('\n');
        }
    } else if (arr.ndim == 2) {
        const rows = arr.shape_dims[0];
        const cols = arr.shape_dims[1];

        for (0..rows) |r| {
            for (0..cols) |c| {
                const idx = r * cols + c;
                try writeScalar(arr, idx, writer);
                if (c + 1 < cols) {
                    try writer.writeAll(options.delimiter);
                }
            }
            try writer.writeByte('\n');
        }
    }

    if (options.footer.len > 0) {
        try writer.print("{s}\n", .{options.footer});
    }
}

fn writeScalar(arr: Array, flat_idx: usize, writer: anytype) !void {
    if (arr.dtype.isFloat()) {
        const val = if (arr.ndim == 1)
            try arr.get(f64, &.{flat_idx})
        else blk: {
            const cols = arr.shape_dims[1];
            break :blk try arr.get(f64, &.{ flat_idx / cols, flat_idx % cols });
        };
        try writer.print("{d}", .{val});
    } else if (arr.dtype.isInteger()) {
        const val = if (arr.ndim == 1)
            try arr.get(i64, &.{flat_idx})
        else blk: {
            const cols = arr.shape_dims[1];
            break :blk try arr.get(i64, &.{ flat_idx / cols, flat_idx % cols });
        };
        try writer.print("{d}", .{val});
    } else if (arr.dtype == .bool) {
        const val = if (arr.ndim == 1)
            try arr.get(bool, &.{flat_idx})
        else blk: {
            const cols = arr.shape_dims[1];
            break :blk try arr.get(bool, &.{ flat_idx / cols, flat_idx % cols });
        };
        try writer.print("{}", .{val});
    }
}

const LoadTxtOptions = struct {
    delimiter: []const u8 = ",",
    skipRows: usize = 0,
    dtype: DType = .f64,
};

/// Reads a delimited text stream into a 1D or 2D Array.
pub fn loadtxtReader(
    allocator: std.mem.Allocator,
    reader: anytype,
    options: LoadTxtOptions,
) !Array {
    var values: std.ArrayList(f64) = .empty;
    defer values.deinit(allocator);

    var num_rows: usize = 0;
    var num_cols: usize = 0;

    var line_buf: [4096]u8 = undefined;
    var current_line_idx: usize = 0;

    while (true) {
        const line_opt = try readLine(reader, &line_buf);
        if (line_opt == null) break;
        const raw_line = std.mem.trim(u8, line_opt.?, " \r\t\n");
        if (raw_line.len == 0 or raw_line[0] == '#') continue;

        if (current_line_idx < options.skipRows) {
            current_line_idx += 1;
            continue;
        }
        current_line_idx += 1;

        var col_count: usize = 0;
        var token_it = std.mem.splitSequence(u8, raw_line, options.delimiter);

        while (token_it.next()) |token| {
            const trimmed = std.mem.trim(u8, token, " \t\r\n");
            if (trimmed.len == 0) continue;

            const val = std.fmt.parseFloat(f64, trimmed) catch {
                return IoError.ParseError;
            };
            try values.append(allocator, val);
            col_count += 1;
        }

        if (num_rows == 0) {
            num_cols = col_count;
        } else if (col_count != num_cols) {
            return IoError.ShapeMismatch;
        }

        num_rows += 1;
    }

    if (num_rows == 0 or num_cols == 0) {
        return empty(allocator, .{ .shape = &.{0}, .dtype = options.dtype });
    }

    const out_shape: []const usize = if (num_rows == 1)
        &.{num_cols}
    else
        &.{ num_rows, num_cols };

    var out = try empty(allocator, .{
        .shape = out_shape,
        .dtype = options.dtype,
    });
    errdefer out.deinit();

    for (values.items, 0..) |v, i| {
        if (num_rows == 1) {
            try out.set(f64, &.{i}, v);
        } else {
            const r = i / num_cols;
            const c = i % num_cols;
            try out.set(f64, &.{ r, c }, v);
        }
    }

    return out;
}

fn readLine(reader: anytype, buffer: []u8) !?[]const u8 {
    var count: usize = 0;
    while (count < buffer.len) {
        const b = reader.readByte() catch |err| {
            if (err == error.EndOfStream) {
                if (count > 0) return buffer[0..count];
                return null;
            }
            return err;
        };
        if (b == '\n') return buffer[0..count];
        buffer[count] = b;
        count += 1;
    }
    return buffer[0..count];
}

/// Saves an array to a text file.
pub fn savetxt(
    allocator: std.mem.Allocator,
    arr: Array,
    file_path: []const u8,
    options: SaveTxtOptions,
) !void {
    var io_threaded: std.Io.Threaded = .init(allocator, .{});
    defer io_threaded.deinit();
    const io = io_threaded.io();

    var file = try std.Io.Dir.cwd().createFile(io, file_path, .{});
    defer file.close(io);

    const fw = struct {
        file: std.Io.File,
        io: std.Io,

        pub fn writeAll(self: @This(), bytes: []const u8) !void {
            try self.file.writeStreamingAll(self.io, bytes);
        }

        pub fn writeByte(self: @This(), b: u8) !void {
            try self.writeAll(&.{b});
        }

        pub fn print(self: @This(), comptime fmt: []const u8, args: anytype) !void {
            var buf: [512]u8 = undefined;
            const res = try std.fmt.bufPrint(&buf, fmt, args);
            try self.writeAll(res);
        }
    }{ .file = file, .io = io };

    try savetxtWriter(arr, fw, options);
}

/// Loads an array from a text file.
pub fn loadtxt(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    options: LoadTxtOptions,
) !Array {
    var io_threaded: std.Io.Threaded = .init(allocator, .{});
    defer io_threaded.deinit();
    const io = io_threaded.io();

    var file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
    defer file.close(io);

    var fr = struct {
        file: std.Io.File,
        io: std.Io,
        pos: u64 = 0,

        pub fn read(self: *@This(), dest: []u8) !usize {
            const amt = try self.file.readPositionalAll(self.io, dest, self.pos);
            self.pos += amt;
            return amt;
        }

        pub fn readByte(self: *@This()) !u8 {
            var b: [1]u8 = undefined;
            const n = try self.read(&b);
            if (n == 0) return IoError.EndOfStream;
            return b[0];
        }
    }{ .file = file, .io = io };

    return loadtxtReader(allocator, &fr, options);
}

test "savetxt and loadtxt stream roundtrip" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 };
    var orig = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{ 2, 3 } });
    defer orig.deinit();

    const MemoryStream = @import("stream.zig").MemoryStream;
    var buffer: [1024]u8 = undefined;
    var ms = MemoryStream.init(&buffer);

    try savetxtWriter(orig, &ms, .{ .delimiter = "," });

    var loaded = try loadtxtReader(allocator, &ms, .{ .delimiter = "," });
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded.shape_dims[0]);
    try std.testing.expectEqual(@as(usize, 3), loaded.shape_dims[1]);

    for (0..2) |r| {
        for (0..3) |c| {
            try std.testing.expectApproxEqAbs(
                try orig.get(f64, &.{ r, c }),
                try loaded.get(f64, &.{ r, c }),
                1e-5,
            );
        }
    }
}

test "savetxt and loadtxt filesystem roundtrip" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 10.0, 20.0, 30.0, 40.0 };
    var orig = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{ 2, 2 } });
    defer orig.deinit();

    const tmp_path = "test_text_roundtrip.csv";
    try savetxt(allocator, orig, tmp_path, .{ .delimiter = "," });
    defer {
        var io_threaded: std.Io.Threaded = .init(allocator, .{});
        defer io_threaded.deinit();
        const io = io_threaded.io();
        std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
    }

    var loaded = try loadtxt(allocator, tmp_path, .{ .delimiter = "," });
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded.shape_dims[0]);
    try std.testing.expectEqual(@as(usize, 2), loaded.shape_dims[1]);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0), try loaded.get(f64, &.{ 1, 1 }), 1e-5);
}
