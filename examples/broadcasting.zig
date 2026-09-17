//! Demonstrates N-dimensional shape broadcasting in num.zig.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Matrix (2x3)
    const mat_data = [_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var matrix = try num.fromSlice(allocator, f64, .{ .data = &mat_data, .shape = &.{ 2, 3 } });
    defer matrix.deinit();

    // Row vector (1x3)
    const row_data = [_]f64{ 10.0, 20.0, 30.0 };
    var row = try num.fromSlice(allocator, f64, .{ .data = &row_data, .shape = &.{ 1, 3 } });
    defer row.deinit();

    // Broadcast row across matrix rows during addition: (2, 3) + (1, 3) -> (2, 3)
    var result = try num.ops.add(matrix, row, .{});
    defer result.deinit();

    std.debug.print("Broadcasting (2x3) + (1x3) -> (2x3):\n", .{});
    for (0..2) |r| {
        std.debug.print("  row {d}: [{d:.1}, {d:.1}, {d:.1}]\n", .{
            r,
            try result.get(f64, &.{ r, 0 }),
            try result.get(f64, &.{ r, 1 }),
            try result.get(f64, &.{ r, 2 }),
        });
    }

    // Explicit broadcastTo view
    var broadcasted_view = try num.broadcast.broadcastTo(row, &.{ 4, 3 });
    defer broadcasted_view.deinit();
    std.debug.print("Explicit broadcastTo (1x3) -> (4x3):\n  ndim: {d}, shape: [{d}, {d}], isContiguous: {}\n", .{
        broadcasted_view.ndim,
        broadcasted_view.shape_dims[0],
        broadcasted_view.shape_dims[1],
        broadcasted_view.isContiguous(),
    });
}
