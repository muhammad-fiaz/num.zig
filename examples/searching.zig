//! Demonstrates binary search insertion, unique elements, and non-zero discovery.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. searchSorted: binary search insertion points
    const sorted_data = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    var sorted_arr = try num.fromSlice(allocator, f64, .{ .data = &sorted_data, .shape = &.{5} });
    defer sorted_arr.deinit();

    const query_data = [_]f64{ 5.0, 25.0, 55.0 };
    var query_arr = try num.fromSlice(allocator, f64, .{ .data = &query_data, .shape = &.{3} });
    defer query_arr.deinit();

    var insertion_pts = try num.sort.searchSorted(sorted_arr, query_arr, .{ .side = .left });
    defer insertion_pts.deinit();

    std.debug.print("searchSorted insertion points for [5.0, 25.0, 55.0]:\n  [{d}, {d}, {d}]\n", .{
        try insertion_pts.get(i64, &.{0}),
        try insertion_pts.get(i64, &.{1}),
        try insertion_pts.get(i64, &.{2}),
    });

    // 2. unique: extract sorted unique elements
    const dup_data = [_]i32{ 3, 1, 2, 3, 1, 4, 2, 5 };
    var dup_arr = try num.fromSlice(allocator, i32, .{ .data = &dup_data, .shape = &.{8} });
    defer dup_arr.deinit();

    var u_res = try num.sort.unique(dup_arr, .{});
    defer u_res.deinit();

    std.debug.print("Unique elements:\n  [", .{});
    for (0..u_res.values.elementCount()) |i| {
        std.debug.print("{d} ", .{try u_res.values.get(i32, &.{i})});
    }
    std.debug.print("]\n", .{});

    // 3. flatNonzero: find indices of non-zero elements
    const mask_data = [_]f32{ 0.0, 5.5, 0.0, 8.2, 0.0, 12.1 };
    var mask_arr = try num.fromSlice(allocator, f32, .{ .data = &mask_data, .shape = &.{6} });
    defer mask_arr.deinit();

    var nonzeros = try num.sort.flatNonzero(mask_arr);
    defer nonzeros.deinit();

    std.debug.print("flatNonzero indices:\n  [", .{});
    for (0..nonzeros.elementCount()) |i| {
        std.debug.print("{d} ", .{try nonzeros.get(i64, &.{i})});
    }
    std.debug.print("]\n", .{});

    // 4. argwhere: 2D coordinates of non-zero elements
    const mat2d_data = [_]f64{ 0.0, 1.0, 2.0, 0.0 };
    var mat2d = try num.fromSlice(allocator, f64, .{ .data = &mat2d_data, .shape = &.{ 2, 2 } });
    defer mat2d.deinit();

    var coords = try num.sort.argwhere(mat2d);
    defer coords.deinit();
    std.debug.print("argwhere nonzeros count: {d}\n", .{coords.shapeSlice()[0]});
}
