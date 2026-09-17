//! Demonstrates multi-dimensional array slicing and zero-copy views in num.zig.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 4x4 grid: numbers 0..15
    const data = [_]i32{
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
        12, 13, 14, 15,
    };
    var grid = try num.fromSlice(allocator, i32, .{ .data = &data, .shape = &.{ 4, 4 } });
    defer grid.deinit();

    // Sub-matrix: rows 1..3, cols 1..3 -> 2x2 view: [[5, 6], [9, 10]]
    const slices = [_]num.Slice{
        .{ .start = 1, .stop = 3, .step = 1 },
        .{ .start = 1, .stop = 3, .step = 1 },
    };
    var sub = try num.manip.slice(grid, &slices);
    defer sub.deinit();

    std.debug.print("Original (4x4) sliced to sub-region (2x2):\n", .{});
    std.debug.print("  sub[0,0] = {d}, sub[0,1] = {d}\n", .{
        try sub.get(i32, &.{ 0, 0 }),
        try sub.get(i32, &.{ 0, 1 }),
    });
    std.debug.print("  sub[1,0] = {d}, sub[1,1] = {d}\n", .{
        try sub.get(i32, &.{ 1, 0 }),
        try sub.get(i32, &.{ 1, 1 }),
    });

    // Strided slice: every second column
    const strided_slices = [_]num.Slice{
        .{ .start = 0, .stop = 4, .step = 1 },
        .{ .start = 0, .stop = 4, .step = 2 },
    };
    var strided = try num.manip.slice(grid, &strided_slices);
    defer strided.deinit();

    std.debug.print("Strided view shape: [{d}, {d}], isContiguous: {}\n", .{
        strided.shape_dims[0],
        strided.shape_dims[1],
        strided.isContiguous(),
    });
}
