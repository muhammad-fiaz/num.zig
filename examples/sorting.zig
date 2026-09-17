//! Demonstrates in-place sorting, sorted copies, and index sorting (argsort).

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const values = [_]f64{ 5.5, 1.1, 9.9, 3.3, 7.7 };
    var arr = try num.fromSlice(allocator, f64, .{ .data = &values, .shape = &.{5} });
    defer arr.deinit();

    // 1. Argsort: indices that would sort the array
    var indices = try num.sort.argsort(arr, .{});
    defer indices.deinit();

    std.debug.print("Argsort indices:\n  [", .{});
    for (0..indices.elementCount()) |i| {
        std.debug.print("{d} ", .{try indices.get(i64, &.{i})});
    }
    std.debug.print("]\n", .{});

    // 2. Sorted copy
    var asc = try num.sort.sorted(arr, .{ .order = .asc });
    defer asc.deinit();

    std.debug.print("Sorted ascending:\n  [", .{});
    for (0..asc.elementCount()) |i| {
        std.debug.print("{d:.1} ", .{try asc.get(f64, &.{i})});
    }
    std.debug.print("]\n", .{});

    // 3. In-place sort
    var to_sort = try arr.clone();
    defer to_sort.deinit();
    try num.sort.sort(&to_sort, .{ .order = .desc });

    std.debug.print("In-place sorted descending:\n  [", .{});
    for (0..to_sort.elementCount()) |i| {
        std.debug.print("{d:.1} ", .{try to_sort.get(f64, &.{i})});
    }
    std.debug.print("]\n", .{});
}
