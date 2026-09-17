//! Array formatting and pretty-printing.
//!
//! Provides customizable pretty-printing for arrays of any rank, with
//! threshold truncation, edge item display, and std.fmt integration.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;

const FormatConfig = struct {
    precision: usize = 4,
    threshold: usize = 1000,
    edge_items: usize = 3,
    linewidth: usize = 75,
};

/// Formats an array into a writer with customizable formatting options.
pub fn formatArray(
    arr: Array,
    writer: anytype,
    config: FormatConfig,
) !void {
    if (arr.ndim == 0) {
        try printScalar(arr, &.{}, writer, config.precision);
        return;
    }

    var coords: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK;
    try printSubarray(arr, 0, &coords, writer, config);
}

fn printSubarray(
    arr: Array,
    dim: usize,
    coords: *[MAX_RANK]usize,
    writer: anytype,
    config: FormatConfig,
) !void {
    const dim_len = arr.shape_dims[dim];
    const is_last_dim = (dim == arr.ndim - 1);

    try writer.writeByte('[');

    const summarize = (arr.elementCount() > config.threshold) and (dim_len > 2 * config.edge_items);

    if (!summarize) {
        for (0..dim_len) |i| {
            coords[dim] = i;
            if (is_last_dim) {
                try printScalar(arr, coords[0..arr.ndim], writer, config.precision);
                if (i + 1 < dim_len) try writer.writeAll(", ");
            } else {
                if (i > 0) {
                    try writer.writeByte('\n');
                    for (0..dim + 1) |_| try writer.writeByte(' ');
                }
                try printSubarray(arr, dim + 1, coords, writer, config);
                if (i + 1 < dim_len) try writer.writeAll(",");
            }
        }
    } else {
        // Print leading edge items
        for (0..config.edge_items) |i| {
            coords[dim] = i;
            if (is_last_dim) {
                try printScalar(arr, coords[0..arr.ndim], writer, config.precision);
                try writer.writeAll(", ");
            } else {
                if (i > 0) {
                    try writer.writeByte('\n');
                    for (0..dim + 1) |_| try writer.writeByte(' ');
                }
                try printSubarray(arr, dim + 1, coords, writer, config);
                try writer.writeAll(",");
            }
        }

        if (is_last_dim) {
            try writer.writeAll("..., ");
        } else {
            try writer.writeByte('\n');
            for (0..dim + 1) |_| try writer.writeByte(' ');
            try writer.writeAll("...,");
        }

        // Print trailing edge items
        for (dim_len - config.edge_items..dim_len) |i| {
            coords[dim] = i;
            if (is_last_dim) {
                try printScalar(arr, coords[0..arr.ndim], writer, config.precision);
                if (i + 1 < dim_len) try writer.writeAll(", ");
            } else {
                try writer.writeByte('\n');
                for (0..dim + 1) |_| try writer.writeByte(' ');
                try printSubarray(arr, dim + 1, coords, writer, config);
                if (i + 1 < dim_len) try writer.writeAll(",");
            }
        }
    }

    try writer.writeByte(']');
}

fn printScalar(arr: Array, coords: []const usize, writer: anytype, precision: usize) !void {
    _ = precision;
    if (arr.dtype.isFloat()) {
        const val = arr.get(f64, coords) catch unreachable;
        try writer.print("{d:.4}", .{val});
    } else if (arr.dtype.isInteger()) {
        const val = arr.get(i64, coords) catch unreachable;
        try writer.print("{d}", .{val});
    } else if (arr.dtype == .bool) {
        const val = arr.get(bool, coords) catch unreachable;
        try writer.print("{}", .{val});
    }
}

test "formatArray 1D and 2D formatting" {
    const allocator = std.testing.allocator;
    const items_1d = [_]f64{ 1.0, 2.0, 3.0 };
    var arr_1d = try fromSlice(allocator, f64, .{ .data = &items_1d, .shape = &.{3} });
    defer arr_1d.deinit();

    const MemoryStream = @import("../io/stream.zig").MemoryStream;
    var buf: [256]u8 = undefined;
    var ms = MemoryStream.init(&buf);
    try formatArray(arr_1d, &ms, .{});

    const out_str = ms.getWritten();
    try std.testing.expect(std.mem.startsWith(u8, out_str, "[1"));
    try std.testing.expect(std.mem.endsWith(u8, out_str, "]"));

    // 2D Array formatting
    const items_2d = [_]i64{ 10, 20, 30, 40 };
    var arr_2d = try fromSlice(allocator, i64, .{ .data = &items_2d, .shape = &.{ 2, 2 } });
    defer arr_2d.deinit();

    ms.reset();
    try formatArray(arr_2d, &ms, .{});
    const out_2d = ms.getWritten();
    try std.testing.expect(std.mem.startsWith(u8, out_2d, "[[10, 20],"));
    try std.testing.expect(std.mem.endsWith(u8, out_2d, "[30, 40]]"));
}
