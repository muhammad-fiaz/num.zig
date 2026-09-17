//! Demonstrates reductions: sum, prod, mean, min, max, argmin, argmax, all, any, cumsum.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const data = [_]f64{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var mat = try num.fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 3 } });
    defer mat.deinit();

    // 1. Global reductions
    var sum_val = try num.reduce.sum(mat, .{});
    defer sum_val.deinit();
    var mean_val = try num.reduce.mean(mat, .{});
    defer mean_val.deinit();
    var min_val = try num.reduce.min(mat, .{});
    defer min_val.deinit();
    var max_val = try num.reduce.max(mat, .{});
    defer max_val.deinit();

    std.debug.print("Global Reductions (2x3 matrix):\n", .{});
    std.debug.print("  sum: {d:.1}\n", .{try sum_val.get(f64, &.{})});
    std.debug.print("  mean: {d:.2}\n", .{try mean_val.get(f64, &.{})});
    std.debug.print("  min: {d:.1}\n", .{try min_val.get(f64, &.{})});
    std.debug.print("  max: {d:.1}\n", .{try max_val.get(f64, &.{})});

    // 2. Axis-based reduction: sum along columns (axis 0) -> (3,)
    var col_sums = try num.reduce.sum(mat, .{ .axis = 0 });
    defer col_sums.deinit();
    std.debug.print("Column sums (axis 0): [{d:.1}, {d:.1}, {d:.1}]\n", .{
        try col_sums.get(f64, &.{0}),
        try col_sums.get(f64, &.{1}),
        try col_sums.get(f64, &.{2}),
    });

    // 3. Argmax index
    var argmax_idx = try num.reduce.argmax(mat, .{});
    defer argmax_idx.deinit();
    std.debug.print("Global argmax index: {d}\n", .{try argmax_idx.get(i64, &.{})});

    // 4. Cumulative sum along axis 0
    var cs = try num.reduce.cumsum(mat, .{ .axis = 0 });
    defer cs.deinit();
    std.debug.print("Cumsum along axis 0 at [1, 2]: {d:.1}\n", .{try cs.get(f64, &.{ 1, 2 })});
}
